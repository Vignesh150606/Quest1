"""
Phase 4 - Arbiter tests.

Verification target: pytest tests/test_arbiter.py::test_reconciliation_policy
Remove the @pytest.mark.skip below once src/arbiter.py is implemented.
"""

import pytest

from src.types import Candidate, AmbiguousResult
from src.arbiter import reconcile


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


@pytest.mark.skip(reason="Unskip once src/arbiter.py (Phase 4) is implemented")
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
