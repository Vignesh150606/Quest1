"""
Phase 2 - ASR track.

Consumes: VideoAsset.audio_path (Phase 1), target phrase, Candidate schema (Phase 1)
Produces: list[Candidate], modality="asr", event_type="speech_onset"

Use faster-whisper with word_timestamps=True (native word-level timestamps) -- not
WhisperX. See CLAUDE.md "Known Gotchas": segment-level timestamps are too coarse for a
~1-2 second line.

No CUDA GPU on the dev machine (Intel Iris Xe integrated only) -- device defaults to
"cpu" with compute_type "int8" accordingly. model_size defaults to "small": "base" is
noticeably weaker on period/accented British dialogue (risks mis-transcribing the exact
target line badly enough that even fuzzy matching misses), "medium" costs ~35-40min on
this CPU for the full ~54min example vs. ~10-15min for "small" -- see PHASES_1_7_PLAN.md.

Chunked decoding (hardening added post-Phase-6): faster-whisper's own transcribe(path)
internally calls decode_audio(), which reads the ENTIRE file into one contiguous
float32 array up front -- for a ~54min file that's a single ~199MB allocation. On a
memory-fragmented machine (many other processes running, gigabytes of free RAM
scattered but no single ~200MB contiguous block available) this raised a real
numpy.core._exceptions._ArrayMemoryError during testing, even though the pipeline's
own actual memory need is trivial. Fix: decode audio.wav ourselves in bounded chunks
(stdlib `wave`, no new dependency -- audio.wav is already 16kHz mono PCM S16LE per
ingest.py's _extract_audio, exactly matching what faster-whisper's own decode_audio()
would produce) and pass each chunk as a numpy ndarray directly to transcribe(), which
skips its internal decode_audio() entirely when given an ndarray (verified directly
against the installed faster-whisper's source, not assumed). This bounds the largest
single allocation to one chunk's size regardless of total audio length.

Batched inference (speed, added post-Phase-6): a real end-to-end run against the full
~54min example (int8/CPU, no GPU on this machine) took well over 20 minutes with the
plain WhisperModel.transcribe() API. faster_whisper.BatchedInferencePipeline wraps a
WhisperModel and batches multiple VAD-detected speech segments through the model
together instead of one at a time -- same model, same weights, no accuracy trade-off
from this change alone. beam_size defaults to 1 (greedy decoding) instead of
faster-whisper's own default of 5: this IS a real accuracy/speed trade-off (a narrower
search than beam_size=5), accepted here because the pipeline fuzzy-matches the result
against a target phrase rather than requiring a word-perfect transcript -- callers
wanting the old wider search can still pass beam_size=5 explicitly.

Verification: pytest tests/test_asr_track.py::test_example_video
(network/model-dependent; verify.py 2 runs the offline word-window-matching suite --
see PHASES_1_7_PLAN.md for why).
"""

import wave
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from src.text_match import DEFAULT_MATCH_THRESHOLD, normalize, window_similarity
from src.types import Candidate, VideoAsset

# Adjacent windows around a true hit all score highly; candidates whose spans overlap
# this closely are treated as the same hit and collapsed to the best-scoring one.
_DEDUP_OVERLAP_S = 0.5

# Bounds peak decode memory to ~(chunk + overlap) seconds of float32 mono audio at
# 16kHz (~39MB for the defaults below) instead of one ~199MB buffer for a full 54min
# file. 10 minutes keeps the number of chunk boundaries low (~5 for the example video)
# since each boundary is a place where transcribe() loses cross-chunk context.
_CHUNK_DURATION_S = 600.0
# Each chunk after the first re-reads the last _CHUNK_OVERLAP_S of the previous chunk,
# so a phrase whose audio spans a chunk boundary is still captured whole within at
# least one chunk. The existing _dedup_overlapping() below collapses the resulting
# duplicate detection in the overlap region for free.
_CHUNK_OVERLAP_S = 10.0


@dataclass
class _Word:
    text: str
    start: float
    end: float
    probability: float


def find_candidates(
    asset: VideoAsset,
    target_phrase: str,
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    chunk_duration_s: float = _CHUNK_DURATION_S,
    chunk_overlap_s: float = _CHUNK_OVERLAP_S,
    batch_size: int = 8,
    beam_size: int = 1,
) -> list[Candidate]:
    """
    Transcribe audio with word-level timestamps, fuzzy-match target_phrase, return
    candidates.

    Language is left auto-detected (not hardcoded to English) -- the evaluator may swap
    in a video in a different language. Detected from the first chunk and assumed
    constant across the file (a reasonable assumption for a single video; not new --
    this was already implicit in a single whole-file transcribe() call before chunking).

    Uses BatchedInferencePipeline (see module docstring) rather than WhisperModel's own
    .transcribe() directly -- same model weights, batches VAD-detected segments through
    the model together for CPU throughput. beam_size=1 (greedy) trades a narrower
    decoding search for speed; pass beam_size=5 to restore faster-whisper's own default
    if accuracy on a harder video matters more than runtime for a given call.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    pipeline = BatchedInferencePipeline(model=model)

    words: list[_Word] = []
    language = None
    language_probability = None
    for offset_s, chunk in _iter_audio_chunks(
        asset.audio_path, chunk_duration_s, chunk_overlap_s
    ):
        segments, info = pipeline.transcribe(
            chunk,
            word_timestamps=True,
            vad_filter=True,
            batch_size=batch_size,
            beam_size=beam_size,
        )
        if language is None:
            language = info.language
            language_probability = info.language_probability
        for segment in segments:
            for w in segment.words or []:
                words.append(
                    _Word(
                        text=w.word,
                        start=w.start + offset_s,
                        end=w.end + offset_s,
                        probability=w.probability,
                    )
                )

    candidates = _match_word_windows(words, target_phrase, threshold)
    for c in candidates:
        c.evidence["model_size"] = model_size
        c.evidence["language"] = language
        c.evidence["language_probability"] = language_probability
        c.evidence["chunk_duration_s"] = chunk_duration_s
        c.evidence["batch_size"] = batch_size
        c.evidence["beam_size"] = beam_size
    return candidates


def _iter_audio_chunks(
    audio_path: str, chunk_duration_s: float, overlap_s: float
) -> Iterator[tuple[float, "np.ndarray"]]:
    """
    Decode audio_path (16kHz mono PCM S16LE) in bounded-size chunks, yielding
    (chunk_start_offset_s, float32_ndarray) pairs -- the offset to add back onto every
    word timestamp faster-whisper reports for that chunk, since each chunk is
    transcribed as if it were its own standalone clip starting at t=0.

    Normalization (int16 -> float32, divide by 32768.0) exactly matches
    faster_whisper.audio.decode_audio()'s own conversion -- verified directly against
    the installed package's source, not assumed, so chunked results are numerically
    equivalent to what a single whole-file decode_audio() call would have produced.
    """
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise RuntimeError(
                f"{audio_path!r} is not 16-bit mono PCM (sampwidth="
                f"{wf.getsampwidth()}, channels={wf.getnchannels()}) -- ingest.py's "
                f"_extract_audio is expected to guarantee this; something upstream "
                f"changed the audio format."
            )
        total_frames = wf.getnframes()

        chunk_frames = int(chunk_duration_s * sample_rate)
        overlap_frames = int(overlap_s * sample_rate)

        start_frame = 0
        while start_frame < total_frames:
            # Extend the read backward by overlap_frames (except for the first chunk,
            # clamped at 0), but stop at this chunk's own nominal end -- NOT nominal
            # end + overlap, which would double the actual shared region between
            # consecutive chunks to ~2x overlap_s instead of overlap_s.
            read_start = max(0, start_frame - overlap_frames)
            read_end = min(total_frames, start_frame + chunk_frames)
            frames_to_read = read_end - read_start
            wf.setpos(read_start)
            raw = wf.readframes(frames_to_read)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            yield read_start / sample_rate, samples
            start_frame += chunk_frames


def _match_word_windows(
    words: list[_Word], target_phrase: str, threshold: float
) -> list[Candidate]:
    """
    Slide word-count windows over `words`, score each against target_phrase with
    window_similarity, keep windows scoring >= threshold, and deduplicate overlapping
    windows around the same hit.

    Factored out from find_candidates() as a pure function: this is the part worth
    testing directly with hand-built word lists, since WhisperModel itself is a
    black-box dependency not worth mocking in a unit test.
    """
    target_word_count = len(normalize(target_phrase).split())
    if target_word_count == 0 or not words:
        return []

    # Windows of N-1 .. N+2 words around the target's word count, so a slightly
    # different transcription word count (e.g. a contraction split differently) still
    # gets a window whose extent roughly matches the target.
    window_sizes = sorted({n for n in range(target_word_count - 1, target_word_count + 3) if n > 0})

    raw: list[Candidate] = []
    for size in window_sizes:
        for i in range(0, len(words) - size + 1):
            window = words[i : i + size]
            window_text = "".join(w.text for w in window).strip()
            score = window_similarity(target_phrase, window_text)
            if score < threshold:
                continue
            confidence = sum(w.probability for w in window) / len(window)
            confidence = max(0.0, min(1.0, confidence))
            raw.append(
                Candidate(
                    modality="asr",
                    event_type="speech_onset",
                    timestamp=window[0].start,
                    end_timestamp=window[-1].end,
                    matched_text=window_text,
                    normalized_text=normalize(window_text),
                    similarity=score,
                    confidence=confidence,
                    evidence={
                        "window_word_count": size,
                        "word_probabilities": [w.probability for w in window],
                    },
                )
            )

    return _dedup_overlapping(raw)


def _dedup_overlapping(candidates: list[Candidate]) -> list[Candidate]:
    """
    Collapse candidates whose [timestamp, end_timestamp] spans overlap into the single
    best-scoring (similarity, then confidence) candidate per overlap group. Without
    this, every window size sliding past a true hit produces its own near-duplicate
    candidate.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.timestamp)
    groups: list[list[Candidate]] = [[ordered[0]]]
    for c in ordered[1:]:
        group_end = max(g.end_timestamp for g in groups[-1])
        if c.timestamp <= group_end + _DEDUP_OVERLAP_S:
            groups[-1].append(c)
        else:
            groups.append([c])

    return [max(g, key=lambda c: (c.similarity, c.confidence)) for g in groups]
