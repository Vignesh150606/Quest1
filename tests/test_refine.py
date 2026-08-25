"""
Phase 5 - Refine tests.

Offline: builds a synthetic captioned clip via the Phase 3 fixture (no network, no
real video download). This is verify.py 5's target.

Verification target: pytest tests/test_refine.py
"""

import os

import pytest

from src.ingest import prepare_asset
from src.refine import to_frame_match
from src.types import Candidate
from tests.fixtures.make_synthetic_clip import make_synthetic_clip


def _make_asset(tmp_path):
    clip_path = str(tmp_path / "refine_synth.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)
    asset = prepare_asset(clip["path"])
    return asset, clip


def _candidate(event_type, timestamp):
    return Candidate(
        modality="ocr" if event_type == "visual_text_onset" else "asr",
        event_type=event_type,
        timestamp=timestamp,
        end_timestamp=None,
        matched_text="Test caption text",
        normalized_text="test caption text",
        similarity=1.0,
        confidence=0.9,
    )


@pytest.mark.parametrize(
    "event_type,timestamp,expected_frame_idx",
    [
        ("visual_text_onset", 2.0, 50),  # exact frame boundary: 25fps * 2.0s
        ("speech_onset", 2.02, 50),  # mid-frame anchor -> last frame at/before it
    ],
)
def test_frame_accuracy(event_type, timestamp, expected_frame_idx, tmp_path):
    asset, clip = _make_asset(tmp_path)
    assert asset.metadata.fps == clip["fps"]  # sanity: probed fps matches the fixed 25fps

    candidate = _candidate(event_type, timestamp)
    output_dir = str(tmp_path / "output")
    match = to_frame_match(candidate, asset, output_dir)

    assert match.frame_idx == expected_frame_idx
    assert os.path.exists(match.image_path)
    assert os.path.getsize(match.image_path) > 0
    assert match.modality == candidate.modality
    assert match.text == candidate.matched_text


def test_frame_accuracy_is_deterministic(tmp_path):
    asset, _ = _make_asset(tmp_path)
    candidate = _candidate("visual_text_onset", 2.0)
    output_dir = str(tmp_path / "output")

    first = to_frame_match(candidate, asset, output_dir)
    second = to_frame_match(candidate, asset, output_dir)

    assert first.frame_idx == second.frame_idx
    assert first.timestamp_s == second.timestamp_s
