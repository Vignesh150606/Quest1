"""
Phase 6 - End-to-end tests.

Offline (default; see pytest.ini): write_report() unit tests for all three statuses
(match/ambiguous/not_found), including exact HH:MM:SS.sss timestamp formatting, plus a
full offline run_pipeline() run against a synthetic captioned clip -- possible because
Phase 3 taught prepare_asset() to accept local file paths, so this exercises the whole
pipeline (ingest -> ASR/OCR -> arbiter -> refine -> report) with zero network. This is
verify.py 6's target.

Network (@pytest.mark.network, opt-in): the two original scaffold tests against the
real ok.ru example video.
    pytest -m network tests/test_end_to_end.py -v
"""

import json
import os

import pytest

from src.report import _format_timestamp, write_report
from src.types import AmbiguousResult, FrameMatch
from tests.fixtures.make_synthetic_clip import make_synthetic_clip

EXAMPLE_URL = "https://ok.ru/video/248244667877"
TARGET_PHRASE = "My mind rebels at stagnation"


def _frame_match(
    timestamp_s=327.041,
    frame_idx=7849,
    modality="asr",
    text="My mind rebels at stagnation",
    score=0.94,
    image_path="output/frame_7849.png",
):
    return FrameMatch(
        frame_idx=frame_idx,
        timestamp_s=timestamp_s,
        text=text,
        image_path=image_path,
        modality=modality,
        match_score=score,
    )


# ---------------------------------------------------------------------------
# write_report -- offline unit tests
# ---------------------------------------------------------------------------


def test_write_report_match(tmp_path):
    fm = _frame_match()
    report_path = write_report(
        fm, str(tmp_path), video_url="https://example.com/v", target_phrase=TARGET_PHRASE
    )

    assert os.path.exists(report_path)
    with open(report_path) as f:
        report = json.load(f)

    assert report["status"] == "match"
    assert report["timestamp"] == "00:05:27.041"
    assert report["timestamp_s"] == 327.041
    assert report["frame"] == 7849
    assert report["extracted_text"] == TARGET_PHRASE
    assert report["modality"] == "asr"
    assert report["match_score"] == 0.94
    assert report["candidates"] == []


def test_write_report_not_found(tmp_path):
    report_path = write_report(
        None,
        str(tmp_path),
        video_url="https://example.com/v",
        target_phrase="nonexistent phrase",
    )

    with open(report_path) as f:
        report = json.load(f)

    assert report["status"] == "not_found"
    assert report["frame"] is None
    assert report["image_path"] is None
    assert report["timestamp"] is None
    assert report["candidates"] == []


def test_write_report_ambiguous(tmp_path):
    alt1 = _frame_match(timestamp_s=100.0, frame_idx=2500, modality="asr", score=0.90)
    alt2 = _frame_match(timestamp_s=900.0, frame_idx=22500, modality="ocr", score=0.95)
    ambiguous = AmbiguousResult(
        candidates=[alt1, alt2], reason="2 candidate clusters disagree"
    )

    report_path = write_report(
        ambiguous, str(tmp_path), video_url="https://example.com/v", target_phrase=TARGET_PHRASE
    )

    with open(report_path) as f:
        report = json.load(f)

    assert report["status"] == "ambiguous"
    # primary is whichever alternative has the highest match_score
    assert report["frame"] == 22500
    assert report["modality"] == "ocr"
    assert len(report["candidates"]) == 2
    assert {c["frame"] for c in report["candidates"]} == {2500, 22500}


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00.000"),
        (327.041, "00:05:27.041"),
        (3661.5, "01:01:01.500"),
        (59.9996, "00:01:00.000"),  # rounds up and carries into minutes
    ],
)
def test_format_timestamp(seconds, expected):
    assert _format_timestamp(seconds) == expected


# ---------------------------------------------------------------------------
# Full pipeline -- offline (synthetic clip, no network)
# ---------------------------------------------------------------------------


def test_full_run_offline_synthetic_clip(tmp_path):
    from src.main import run_pipeline

    clip_path = str(tmp_path / "e2e_synthetic.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)

    output_dir = str(tmp_path / "output")
    result_path = run_pipeline(clip["path"], clip["text"], output_dir)

    with open(result_path) as f:
        report = json.load(f)

    assert report["status"] == "match"
    assert report["frame"] is not None
    assert os.path.exists(report["image_path"])
    assert abs(report["timestamp_s"] - clip["onset_s"]) <= 1.0


def test_full_run_offline_no_match(tmp_path):
    from src.main import run_pipeline

    clip_path = str(tmp_path / "e2e_synthetic_neg.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)

    output_dir = str(tmp_path / "output")
    result_path = run_pipeline(
        clip["path"], "this phrase will never appear anywhere in this clip xyz987", output_dir
    )

    with open(result_path) as f:
        report = json.load(f)

    assert report["status"] == "not_found"


# ---------------------------------------------------------------------------
# Network tests -- opt-in, real download + real model (see pytest.ini)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_full_run(tmp_path):
    from src.main import run_pipeline

    output_dir = str(tmp_path)
    result_path = run_pipeline(EXAMPLE_URL, TARGET_PHRASE, output_dir)

    with open(result_path) as f:
        report = json.load(f)

    assert report["frame"] is not None
    assert os.path.exists(report["image_path"])


@pytest.mark.network
def test_no_candidates_found(tmp_path):
    from src.main import run_pipeline

    output_dir = str(tmp_path)
    result_path = run_pipeline(
        EXAMPLE_URL, "this phrase will never appear in this video xyz123", output_dir
    )

    with open(result_path) as f:
        report = json.load(f)

    assert report["status"] == "not_found"
