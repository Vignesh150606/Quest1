"""
Phase 3 - OCR track tests.

Verification target: pytest tests/test_ocr_track.py::test_synthetic_clip
Remove the @pytest.mark.skip below once src/ocr_track.py and the synthetic fixture
are implemented.
"""

import pytest

from src.ingest import prepare_asset
from src.ocr_track import find_candidates
from tests.fixtures.make_synthetic_clip import make_synthetic_clip

SIMILARITY_THRESHOLD = 0.85
SAMPLE_INTERVAL_TOLERANCE_S = 1.0  # should match the OCR track's frame-sampling interval


@pytest.mark.skip(reason="Unskip once src/ocr_track.py (Phase 3) and the synthetic fixture are implemented")
def test_synthetic_clip(tmp_path):
    clip_path = str(tmp_path / "synthetic.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, clip["text"])
    assert candidates, "no OCR candidates found"
    best = max(candidates, key=lambda c: c.similarity)
    assert best.similarity >= SIMILARITY_THRESHOLD
    assert abs(best.timestamp - clip["onset_s"]) <= SAMPLE_INTERVAL_TOLERANCE_S
