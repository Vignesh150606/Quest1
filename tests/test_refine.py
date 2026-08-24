"""
Phase 5 - Refine tests.

Verification target: pytest tests/test_refine.py::test_frame_accuracy
Remove the @pytest.mark.skip below once src/refine.py is implemented.
"""

import pytest

from src.refine import to_frame_match  # noqa: F401


@pytest.mark.skip(reason="Unskip once src/refine.py (Phase 5) is implemented")
@pytest.mark.parametrize("event_type", ["speech_onset", "visual_text_onset"])
def test_frame_accuracy(event_type, tmp_path):
    # TODO: wire in real Candidate + VideoAsset fixtures for each event_type once
    # Phase 1-3 fixtures exist. Assert frame_idx == expected_frame_idx (or a documented
    # tolerance, per CLAUDE.md's "Ground Truth / Acceptance Criteria" section, only if
    # PyAV decode proves non-deterministic in testing).
    raise NotImplementedError
