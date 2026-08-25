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
this CPU for the full ~54min example vs. ~10-15min for "small" -- see PHASES_2_7_PLAN.md.

Verification (see PHASES.md): pytest tests/test_asr_track.py::test_example_video
(network/model-dependent; verify.py 2 runs the offline word-window-matching suite --
see PHASES_2_7_PLAN.md for why).
"""

from dataclasses import dataclass

from faster_whisper import WhisperModel

from src.text_match import DEFAULT_MATCH_THRESHOLD, normalize, window_similarity
from src.types import Candidate, VideoAsset

# Adjacent windows around a true hit all score highly; candidates whose spans overlap
# this closely are treated as the same hit and collapsed to the best-scoring one.
_DEDUP_OVERLAP_S = 0.5


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
) -> list[Candidate]:
    """
    Transcribe audio with word-level timestamps, fuzzy-match target_phrase, return
    candidates.

    Language is left auto-detected (not hardcoded to English) -- the evaluator may swap
    in a video in a different language.
    """
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        asset.audio_path, word_timestamps=True, vad_filter=True
    )

    words: list[_Word] = []
    for segment in segments:
        for w in segment.words or []:
            words.append(
                _Word(text=w.word, start=w.start, end=w.end, probability=w.probability)
            )

    candidates = _match_word_windows(words, target_phrase, threshold)
    for c in candidates:
        c.evidence["model_size"] = model_size
        c.evidence["language"] = info.language
        c.evidence["language_probability"] = info.language_probability
    return candidates


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
