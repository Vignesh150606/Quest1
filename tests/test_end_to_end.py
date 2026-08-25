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
from src.types import AmbiguousResult, Candidate, FrameMatch, VideoAsset, VideoMetadata
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
# Staged short-circuit (runtime hardening) -- offline, monkeypatched call spies
# ---------------------------------------------------------------------------
#
# These test run_pipeline()'s CONTROL FLOW (which stages get called, in what
# circumstances) rather than end-to-end correctness (already covered above) --
# monkeypatching the individual track functions in src.main's own namespace lets each
# test isolate exactly one branch of the staged short-circuit without needing to
# fabricate real audio/video content that would organically trigger it.


def _fake_candidate(modality, event_type, text, timestamp=2.0, similarity=0.95, confidence=0.9):
    return Candidate(
        modality=modality,
        event_type=event_type,
        timestamp=timestamp,
        end_timestamp=timestamp + 1.0,
        matched_text=text,
        normalized_text=text.lower(),
        similarity=similarity,
        confidence=confidence,
    )


def test_run_pipeline_confident_subtitle_hit_skips_asr_and_ocr(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    fake_hit = _fake_candidate("subtitle", "speech_onset", clip["text"])

    asr_calls, ocr_calls = [], []
    monkeypatch.setattr(main_module, "try_subtitle_fast_path", lambda asset, text: fake_hit)
    monkeypatch.setattr(
        main_module, "find_asr_candidates", lambda *a, **k: asr_calls.append(1) or []
    )
    monkeypatch.setattr(
        main_module, "find_ocr_candidates", lambda *a, **k: ocr_calls.append(1) or []
    )

    output_dir = str(tmp_path / "output")
    result_path = main_module.run_pipeline(clip["path"], clip["text"], output_dir)

    assert asr_calls == []
    assert ocr_calls == []
    with open(result_path) as f:
        report = json.load(f)
    assert report["status"] == "match"
    assert report["modality"] == "subtitle"


def test_run_pipeline_confident_asr_hit_skips_ocr(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    fake_hit = _fake_candidate("asr", "speech_onset", clip["text"])

    ocr_calls = []
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [fake_hit])
    monkeypatch.setattr(
        main_module, "find_ocr_candidates", lambda *a, **k: ocr_calls.append(1) or []
    )

    output_dir = str(tmp_path / "output")
    result_path = main_module.run_pipeline(clip["path"], clip["text"], output_dir)

    assert ocr_calls == []
    with open(result_path) as f:
        report = json.load(f)
    assert report["status"] == "match"
    assert report["modality"] == "asr"


def test_run_pipeline_inconclusive_asr_still_runs_ocr(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    ocr_hit = _fake_candidate("ocr", "visual_text_onset", clip["text"])

    ocr_calls = []
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [])  # inconclusive

    def fake_ocr(*a, **k):
        ocr_calls.append(1)
        return [ocr_hit]

    monkeypatch.setattr(main_module, "find_ocr_candidates", fake_ocr)

    output_dir = str(tmp_path / "output")
    result_path = main_module.run_pipeline(clip["path"], clip["text"], output_dir)

    assert ocr_calls == [1]  # OCR must actually run when ASR alone is inconclusive
    with open(result_path) as f:
        report = json.load(f)
    assert report["status"] == "match"
    assert report["modality"] == "ocr"


def test_run_pipeline_skip_ocr_forces_disable_even_if_inconclusive(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)

    ocr_calls = []
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [])
    monkeypatch.setattr(
        main_module, "find_ocr_candidates", lambda *a, **k: ocr_calls.append(1) or []
    )

    output_dir = str(tmp_path / "output")
    result_path = main_module.run_pipeline(
        clip["path"], clip["text"], output_dir, skip_ocr=True
    )

    assert ocr_calls == []  # manual override wins even though ASR was inconclusive
    with open(result_path) as f:
        report = json.load(f)
    assert report["status"] == "not_found"


def test_run_pipeline_hq_frame_forces_high_tier_fetch(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    fake_hit = _fake_candidate("asr", "speech_onset", clip["text"])

    real_prepare_asset = main_module.prepare_asset
    tier_calls = []

    def spy_prepare_asset(url, *, tier="low", **kwargs):
        tier_calls.append(tier)
        return real_prepare_asset(url, tier=tier, **kwargs)

    monkeypatch.setattr(main_module, "prepare_asset", spy_prepare_asset)
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [fake_hit])

    output_dir = str(tmp_path / "output")
    main_module.run_pipeline(clip["path"], clip["text"], output_dir, hq_frame=True)

    # Confident ASR hit -> OCR (and its tier escalation) skipped -- but hq_frame=True
    # should still force one more prepare_asset(tier="high") call before refine.
    assert "high" in tier_calls


def test_run_pipeline_no_hq_frame_does_not_escalate(tmp_path, monkeypatch):
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    fake_hit = _fake_candidate("asr", "speech_onset", clip["text"])

    real_prepare_asset = main_module.prepare_asset
    tier_calls = []

    def spy_prepare_asset(url, *, tier="low", **kwargs):
        tier_calls.append(tier)
        return real_prepare_asset(url, tier=tier, **kwargs)

    monkeypatch.setattr(main_module, "prepare_asset", spy_prepare_asset)
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [fake_hit])

    output_dir = str(tmp_path / "output")
    main_module.run_pipeline(clip["path"], clip["text"], output_dir)  # hq_frame=False (default)

    assert tier_calls == ["low"]  # exactly one fetch, never escalated


def test_run_pipeline_escalates_when_low_tier_has_no_video(tmp_path, monkeypatch):
    # Real bug, found by running against a real YouTube video: tier="low" can
    # legitimately be audio-only (a genuine audio-only format exists on that host,
    # unlike ok.ru). Every result path still needs to produce an output frame image,
    # so run_pipeline must escalate to tier="high" even with hq_frame=False (the
    # default) when the low-tier asset has no video -- this is a correctness
    # requirement, not the hq_frame quality preference.
    import src.main as main_module

    clip_path = str(tmp_path / "clip.mp4")
    clip = make_synthetic_clip(clip_path, text="Elementary my dear friend", onset_s=2.0)
    fake_hit = _fake_candidate("asr", "speech_onset", clip["text"])

    audio_only_asset = VideoAsset(
        video_path=clip["path"],  # unused by refine unless this asset is the one passed
        audio_path=clip["path"],
        metadata=VideoMetadata(fps=0.0, duration_s=7.0, width=0, height=0, has_video=False),
        subtitle_paths=[],
    )

    real_prepare_asset = main_module.prepare_asset
    tier_calls = []

    def spy_prepare_asset(url, *, tier="low", **kwargs):
        tier_calls.append(tier)
        if tier == "low":
            return audio_only_asset
        return real_prepare_asset(url, tier=tier, **kwargs)

    monkeypatch.setattr(main_module, "prepare_asset", spy_prepare_asset)
    monkeypatch.setattr(main_module, "find_asr_candidates", lambda *a, **k: [fake_hit])

    output_dir = str(tmp_path / "output")
    result_path = main_module.run_pipeline(clip["path"], clip["text"], output_dir)  # hq_frame=False

    assert tier_calls == ["low", "high"]  # escalated despite hq_frame not being set
    with open(result_path) as f:
        report = json.load(f)
    assert report["status"] == "match"
    assert os.path.exists(report["image_path"])  # a real frame WAS extracted


def test_main_reconfigures_stdout_stderr_to_avoid_encoding_crashes(monkeypatch, tmp_path):
    # Real bug, found by running against a real (non-example) video: a caption track
    # contained a musical note character, and printing the final summary crashed with
    # UnicodeEncodeError on a terminal whose codepage can't encode it (e.g. Windows'
    # default cp1252, not UTF-8) -- right as the tool was about to report a correct
    # match. The underlying report.json/PNG were unaffected (file writes already use
    # UTF-8); only the stdout/stderr summary crashed. main() now reconfigures both
    # streams with errors="replace" up front so this can't happen regardless of the
    # evaluator's console codepage. This locks in that the reconfigure call fires,
    # without depending on the test runner's own console encoding.
    import src.main as main_module

    calls = []

    class _FakeStream:
        def __init__(self, name):
            self.name = name

        def reconfigure(self, **kwargs):
            calls.append((self.name, kwargs))

        def write(self, *args, **kwargs):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(main_module.sys, "stdout", _FakeStream("stdout"))
    monkeypatch.setattr(main_module.sys, "stderr", _FakeStream("stderr"))
    monkeypatch.setattr(
        main_module.sys,
        "argv",
        ["prog", "--url", "http://example.com/video", "--dialogue-text", "hello"],
    )
    monkeypatch.setattr(
        main_module, "run_pipeline", lambda *a, **k: os.path.join(str(tmp_path), "report.json")
    )

    main_module.main()

    assert ("stdout", {"errors": "replace"}) in calls
    assert ("stderr", {"errors": "replace"}) in calls


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
