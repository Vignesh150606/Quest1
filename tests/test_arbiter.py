"""
Phase 4 - Arbiter tests.

Verification target: pytest tests/test_arbiter.py
"""

import pytest

from src.arbiter import reconcile
from src.types import AmbiguousResult, Candidate


def _candidate(modality, timestamp, similarity=0.9, confidence=0.9):
    return Candidate(
        modality=modality,
        event_type="speech_onset" if modality == "asr" else "visual_text_onset",
        timestamp=timestamp,
        end_timestamp=None,
        matched_text="My mind rebels at stagnation",
        normalized_text="my mind rebels at stagnation",
        similarity=similarity,
        confidence=confidence,
    )


@pytest.mark.parametrize(
    "candidates,expected_outcome",
    [
        # single-track hit -> returns it
        ([_candidate("asr", 327.0)], "single_hit"),
        # agreeing multi-track candidates within tolerance -> higher-confidence one wins
        (
            [_candidate("asr", 327.0, similarity=0.90), _candidate("ocr", 327.4, similarity=0.95)],
            "higher_confidence",
        ),
        # disagreeing candidates outside tolerance -> AmbiguousResult, never a silent pick
        ([_candidate("asr", 100.0), _candidate("ocr", 900.0)], "ambiguous"),
    ],
)
def test_reconciliation_policy(candidates, expected_outcome):
    result = reconcile(candidates)
    if expected_outcome == "single_hit":
        assert isinstance(result, Candidate)
        assert result.modality == "asr"
    elif expected_outcome == "higher_confidence":
        assert isinstance(result, Candidate)
        assert result.modality == "ocr"
    elif expected_outcome == "ambiguous":
        assert isinstance(result, AmbiguousResult)


def test_empty_input_returns_none():
    assert reconcile([]) is None


def test_all_below_threshold_returns_none():
    candidates = [
        # fails on similarity (below SIMILARITY_THRESHOLD=0.85)
        _candidate("subtitle", 50.0, similarity=0.60, confidence=0.95),
        # fails on confidence (below CONFIDENCE_THRESHOLDS["asr"]=0.4)
        _candidate("asr", 200.0, similarity=0.90, confidence=0.20),
    ]
    assert reconcile(candidates) is None


def test_three_separated_clusters_returns_ambiguous_with_three():
    candidates = [
        _candidate("subtitle", 0.0),
        _candidate("asr", 300.0),
        _candidate("ocr", 900.0),
    ]
    result = reconcile(candidates)
    assert isinstance(result, AmbiguousResult)
    assert len(result.candidates) == 3
    assert {c.modality for c in result.candidates} == {"subtitle", "asr", "ocr"}


def test_exact_tie_is_deterministic():
    # Two candidates with identical (confidence, similarity), close enough in time to
    # cluster together -- repeated calls on the same input must pick the same winner.
    candidates = [
        _candidate("asr", 200.0, similarity=0.90, confidence=0.90),
        _candidate("ocr", 200.5, similarity=0.90, confidence=0.90),
    ]
    results = [reconcile(candidates) for _ in range(5)]
    assert all(isinstance(r, Candidate) for r in results)
    assert len({(r.modality, r.timestamp) for r in results}) == 1
