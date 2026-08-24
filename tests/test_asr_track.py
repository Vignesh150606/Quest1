"""
Phase 2 - ASR track tests.

Verification target: pytest tests/test_asr_track.py::test_example_video
Remove the @pytest.mark.skip below once src/asr_track.py is implemented.
"""

import pytest

from src.ingest import prepare_asset
from src.asr_track import find_candidates

TARGET_PHRASE = "My mind rebels at stagnation"
EXAMPLE_URL = "https://ok.ru/video/248244667877"
EXPECTED_ONSET_S = 5 * 60 + 27  # ~05:27, manually verified against the example video
TOLERANCE_S = 2
SIMILARITY_THRESHOLD = 0.85


@pytest.mark.skip(reason="Unskip once src/asr_track.py (Phase 2) is implemented")
def test_example_video():
    asset = prepare_asset(EXAMPLE_URL)
    candidates = find_candidates(asset, TARGET_PHRASE)
    assert candidates, "no ASR candidates found"
    best = max(candidates, key=lambda c: c.similarity)
    assert best.similarity >= SIMILARITY_THRESHOLD
    assert abs(best.timestamp - EXPECTED_ONSET_S) <= TOLERANCE_S
