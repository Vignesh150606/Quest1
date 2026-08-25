"""
Phase 2 - ASR track tests.

Offline tests (default; see pytest.ini) exercise _match_word_windows and
_dedup_overlapping directly against hand-built word lists -- no model, no network.
This is verify.py 2's target and should run in well under a second. The word-matching
logic is what's actually worth unit-testing; WhisperModel itself is a black-box
dependency not worth mocking here.

Network test (@pytest.mark.network, opt-in) exercises the real model against the real
example video:
    pytest -m network tests/test_asr_track.py -v
"""

import struct
import wave

import numpy as np
import pytest

from src.asr_track import (
    _iter_audio_chunks,
    _Word,
    _dedup_overlapping,
    _match_word_windows,
    find_candidates,
)
from src.ingest import prepare_asset
from src.types import Candidate

TARGET_PHRASE = "My mind rebels at stagnation"
EXAMPLE_URL = "https://ok.ru/video/248244667877"
EXPECTED_ONSET_S = 5 * 60 + 27  # ~05:27, manually verified against the example video
TOLERANCE_S = 2
SIMILARITY_THRESHOLD = 0.85

# A separate, non-graded phrase for the offline pure-function tests, so they're
# independent of the network test's constant and don't conflate the two.
_UNIT_TARGET = "check the mailbox now"


def _w(text, start, end, probability):
    return _Word(text=text, start=start, end=end, probability=probability)


# ---------------------------------------------------------------------------
# _match_word_windows
# ---------------------------------------------------------------------------


def test_match_word_windows_exact_hit():
    words = [
        _w(" Well", 10.0, 10.3, 0.90),
        _w(" so,", 10.3, 10.5, 0.85),
        _w(" check", 20.0, 20.3, 0.95),
        _w(" the", 20.3, 20.45, 0.90),
        _w(" mailbox", 20.45, 21.0, 0.92),
        _w(" now", 21.0, 21.3, 0.88),
        _w(" please,", 21.3, 21.6, 0.85),
        _w(" okay?", 21.6, 21.9, 0.80),
    ]
    candidates = _match_word_windows(words, _UNIT_TARGET, threshold=0.85)
    assert len(candidates) == 1
    best = candidates[0]
    assert best.modality == "asr"
    assert best.event_type == "speech_onset"
    assert best.timestamp == 20.0  # onset -- start of the matched window's first word
    assert best.end_timestamp == 21.3
    assert best.similarity == 1.0
    assert abs(best.confidence - (0.95 + 0.90 + 0.92 + 0.88) / 4) < 1e-9


def test_match_word_windows_one_word_off():
    # ASR heard an extra filler word ("um") inside the target span. Verified against
    # the actual scoring (not assumed): among all overlapping windows here, the full
    # 5-word span genuinely scores highest (0.933) -- a shorter word like "right" in
    # this slot instead lets an incomplete 3-word window ("check the mailbox", missing
    # "now" entirely) win on ratio, which would defeat the point of this test.
    words = [
        _w(" check", 30.0, 30.3, 0.90),
        _w(" the", 30.3, 30.45, 0.90),
        _w(" mailbox", 30.45, 31.0, 0.90),
        _w(" um,", 31.0, 31.2, 0.85),
        _w(" now", 31.2, 31.5, 0.88),
    ]
    candidates = _match_word_windows(words, _UNIT_TARGET, threshold=0.85)
    assert len(candidates) == 1
    best = candidates[0]
    assert best.timestamp == 30.0
    assert best.end_timestamp == 31.5
    assert best.similarity >= 0.85
    assert best.matched_text.strip() == "check the mailbox um, now"


def test_match_word_windows_split_word():
    # ASR tokenized "mailbox" as two separate words ("mail" + "box").
    words = [
        _w(" check", 40.0, 40.3, 0.90),
        _w(" the", 40.3, 40.45, 0.90),
        _w(" mail", 40.45, 40.7, 0.85),
        _w(" box", 40.7, 41.0, 0.85),
        _w(" now", 41.0, 41.3, 0.90),
    ]
    candidates = _match_word_windows(words, _UNIT_TARGET, threshold=0.85)
    assert len(candidates) == 1
    best = candidates[0]
    assert best.timestamp == 40.0
    assert best.end_timestamp == 41.3
    assert best.similarity >= 0.85


def test_match_word_windows_below_threshold_returns_nothing():
    words = [
        _w(" the", 50.0, 50.2, 0.90),
        _w(" weather", 50.2, 50.6, 0.90),
        _w(" today", 50.6, 50.9, 0.90),
        _w(" is", 50.9, 51.0, 0.90),
        _w(" quite", 51.0, 51.3, 0.90),
        _w(" nice", 51.3, 51.6, 0.90),
    ]
    candidates = _match_word_windows(words, _UNIT_TARGET, threshold=0.85)
    assert candidates == []


def test_match_word_windows_empty_input():
    assert _match_word_windows([], _UNIT_TARGET, threshold=0.85) == []


# ---------------------------------------------------------------------------
# _dedup_overlapping
# ---------------------------------------------------------------------------


def _candidate(timestamp, end_timestamp, similarity=0.9, confidence=0.9):
    return Candidate(
        modality="asr",
        event_type="speech_onset",
        timestamp=timestamp,
        end_timestamp=end_timestamp,
        matched_text="x",
        normalized_text="x",
        similarity=similarity,
        confidence=confidence,
    )


def test_dedup_overlapping_collapses_to_best():
    candidates = [
        _candidate(20.0, 21.3, similarity=0.90),
        _candidate(20.0, 21.0, similarity=1.00),  # the exact-match window -- should win
        _candidate(20.3, 21.6, similarity=0.88),
    ]
    result = _dedup_overlapping(candidates)
    assert len(result) == 1
    assert result[0].similarity == 1.00


def test_dedup_overlapping_keeps_separate_non_overlapping_groups():
    candidates = [
        _candidate(20.0, 21.0, similarity=0.90),
        _candidate(100.0, 101.0, similarity=0.92),
    ]
    result = _dedup_overlapping(candidates)
    assert len(result) == 2


def test_dedup_overlapping_empty_input():
    assert _dedup_overlapping([]) == []


# ---------------------------------------------------------------------------
# _iter_audio_chunks -- memory-bounded chunked audio decoding
# ---------------------------------------------------------------------------


def _write_test_wav(path: str, num_samples: int, sample_rate: int = 16000) -> None:
    """
    16-bit mono PCM WAV where sample i has value (i % 30000) -- an index-encoded
    signal, so overlap correctness between chunks can be verified by comparing sample
    VALUES against their expected position in the original signal, not just lengths.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [i % 30000 for i in range(num_samples)]
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))


def test_iter_audio_chunks_offsets_and_bounded_size(tmp_path):
    sample_rate = 16000
    duration_s = 5.0
    wav_path = str(tmp_path / "chunks.wav")
    _write_test_wav(wav_path, int(duration_s * sample_rate), sample_rate)

    chunk_duration_s, overlap_s = 2.0, 0.5
    chunks = list(_iter_audio_chunks(wav_path, chunk_duration_s, overlap_s))

    # 5s of audio at 2s nominal chunk width -> chunk starts at 0, 2, 4 -> 3 chunks.
    # Each chunk after the first is preceded by overlap_s of look-back, hence offsets
    # 1.5 and 3.5 rather than 2.0 and 4.0.
    assert [round(offset, 6) for offset, _ in chunks] == [0.0, 1.5, 3.5]

    # Peak size must never exceed (chunk_duration_s + overlap_s) worth of samples --
    # this bound is the entire point: it's what keeps the largest single allocation
    # small regardless of total audio length.
    max_samples = int((chunk_duration_s + overlap_s) * sample_rate)
    for _, samples in chunks:
        assert len(samples) <= max_samples


def test_iter_audio_chunks_overlap_is_exact_not_doubled(tmp_path):
    # Regression guard: an earlier draft of this function read chunk_duration+overlap
    # for every chunk unconditionally -- including the first, which has nothing before
    # it to overlap with -- making the ACTUAL shared region between consecutive chunks
    # ~2x overlap_s instead of overlap_s. Caught by hand-tracing the arithmetic (and
    # cross-checked with a standalone script) before writing this test, not by the test
    # finding it after the fact -- kept here so a future change can't silently
    # reintroduce it.
    #
    # A first draft of this test asserted offset deltas were uniformly
    # chunk_duration_s - overlap_s, which is wrong: the first chunk's read_start is
    # clamped at 0 (it would otherwise be negative), so only the delta *into* chunk 1
    # equals chunk_duration - overlap; later deltas equal chunk_duration exactly, since
    # they aren't clamped. The real invariant -- and the one that actually catches the
    # doubling bug -- is the INTERSECTION between each consecutive pair's spans, not
    # the gap between their start offsets. That's what's checked below.
    sample_rate = 16000
    wav_path = str(tmp_path / "chunks.wav")
    _write_test_wav(wav_path, int(5.0 * sample_rate), sample_rate)

    chunk_duration_s, overlap_s = 2.0, 0.5
    chunks = list(_iter_audio_chunks(wav_path, chunk_duration_s, overlap_s))
    assert len(chunks) == 3

    # The first chunk has nothing before it to overlap with, so it must be exactly
    # chunk_duration_s long. Under the buggy version this was chunk_duration_s +
    # overlap_s (2.5s of samples instead of 2.0s).
    first_offset, first_samples = chunks[0]
    assert first_offset == 0.0
    assert len(first_samples) == int(chunk_duration_s * sample_rate)

    overlap_samples = int(overlap_s * sample_rate)
    spans = [
        (round(offset * sample_rate), round(offset * sample_rate) + len(samples))
        for offset, samples in chunks
    ]
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
        assert intersection == overlap_samples

    # And the shared region decodes identical underlying samples (the index-encoded
    # test signal makes this an exact value check, not just a length check).
    for (_, samples_a), (_, samples_b) in zip(chunks, chunks[1:]):
        np.testing.assert_array_equal(samples_a[-overlap_samples:], samples_b[:overlap_samples])


def test_iter_audio_chunks_normalization_matches_faster_whisper(tmp_path):
    # int16 max value should map to just under 1.0 (32767/32768), and -32768 to
    # exactly -1.0 -- matching faster_whisper.audio.decode_audio()'s own
    # `astype(np.float32) / 32768.0` conversion exactly (verified against the
    # installed package's source, not assumed).
    sample_rate = 16000
    wav_path = str(tmp_path / "extremes.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<2h", 32767, -32768))

    chunks = list(_iter_audio_chunks(wav_path, chunk_duration_s=10.0, overlap_s=0.0))
    assert len(chunks) == 1
    samples = chunks[0][1]
    assert abs(samples[0] - (32767 / 32768.0)) < 1e-6
    assert samples[1] == -1.0


# ---------------------------------------------------------------------------
# Network test -- opt-in, real model + real download (see pytest.ini)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_example_video():
    asset = prepare_asset(EXAMPLE_URL)
    candidates = find_candidates(asset, TARGET_PHRASE)
    assert candidates, "no ASR candidates found"
    best = max(candidates, key=lambda c: c.similarity)
    assert best.similarity >= SIMILARITY_THRESHOLD
    assert abs(best.timestamp - EXPECTED_ONSET_S) <= TOLERANCE_S
