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

import pytest

from src.asr_track import _Word, _dedup_overlapping, _match_word_windows, find_candidates
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
