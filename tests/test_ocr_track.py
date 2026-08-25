"""
Phase 3 - OCR track tests.

Offline (default; see pytest.ini): a short synthetic captioned clip generated on the
fly via ffmpeg drawtext -- no network required, no real video download. This is
verify.py 3's target.

Verification target: pytest tests/test_ocr_track.py
"""

from src.ingest import prepare_asset
from src.ocr_track import find_candidates
from tests.fixtures.make_synthetic_clip import make_synthetic_clip

SIMILARITY_THRESHOLD = 0.85
SAMPLE_INTERVAL_TOLERANCE_S = 1.0  # should match the OCR track's frame-sampling interval


def test_synthetic_clip(tmp_path):
    clip_path = str(tmp_path / "synthetic.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, clip["text"])
    assert candidates, "no OCR candidates found"
    best = max(candidates, key=lambda c: c.similarity)
    assert best.similarity >= SIMILARITY_THRESHOLD
    assert abs(best.timestamp - clip["onset_s"]) <= SAMPLE_INTERVAL_TOLERANCE_S
    assert best.modality == "ocr"
    assert best.event_type == "visual_text_onset"


def test_synthetic_clip_dedup_single_candidate(tmp_path):
    # Caption visible for 5s at the default 1s sample interval would naively produce
    # ~5 duplicate hits without clustering -- assert exactly one candidate survives.
    clip_path = str(tmp_path / "synthetic_dedup.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=1.0, duration_s=5.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, clip["text"])
    assert len(candidates) == 1


def test_synthetic_clip_negative_control(tmp_path):
    clip_path = str(tmp_path / "synthetic_neg.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, "this phrase never appears on screen anywhere")
    assert candidates == []
