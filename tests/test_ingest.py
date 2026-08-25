"""
Phase 1 - Ingest tests.

Offline tests (default; see pytest.ini) exercise text_match, subtitle parsing, the
subtitle fast-path, and ffprobe-based metadata probing against local fixtures -- no
network required. This is verify.py 1's target and should run in seconds.

Network tests (@pytest.mark.network, opt-in via `pytest -m network`) exercise
prepare_asset() against the real ok.ru example video and a second domain
(archive.org). These are slow and depend on remote hosts; run them explicitly with:
    pytest -m network tests/test_ingest.py -v
"""

import os
import subprocess

import pytest

from src.ingest import _ffprobe_streams, _parse_subtitle_cues, prepare_asset, try_subtitle_fast_path
from src.text_match import normalize
from src.types import VideoAsset, VideoMetadata

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
VTT_FIXTURE = os.path.join(FIXTURES_DIR, "sample.en.vtt")
SRT_FIXTURE = os.path.join(FIXTURES_DIR, "sample.srt")
DISTINCTIVE_PHRASE = "The quick brown fox jumps over the lazy dog."

OK_RU_URL = "https://ok.ru/video/248244667877"
ARCHIVE_URL = "https://archive.org/details/BigBuckBunny_124"


def _asset_with_subs(*paths: str) -> VideoAsset:
    return VideoAsset(
        video_path="unused.mp4",
        audio_path="unused.wav",
        metadata=VideoMetadata(fps=25.0, duration_s=20.0, width=320, height=240),
        subtitle_paths=list(paths),
    )


# ---------------------------------------------------------------------------
# text_match.normalize
# ---------------------------------------------------------------------------


def test_normalize():
    assert normalize("Hello, World!") == "hello world"
    assert normalize("  Multiple   Spaces  ") == "multiple spaces"
    assert normalize("‘Curly’ “Quotes”") == "curly quotes"
    assert normalize("well—actually") == "well actually"


# ---------------------------------------------------------------------------
# Subtitle cue parsing
# ---------------------------------------------------------------------------


def test_parse_vtt_cues():
    cues = _parse_subtitle_cues(VTT_FIXTURE)
    assert len(cues) == 5
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[0].text == "Good evening, and welcome to the show."
    assert cues[-1].start == 14.0
    assert cues[-1].end == 16.75
    assert cues[-1].text == "Thank you all for watching, goodnight."
    assert cues[2].text == DISTINCTIVE_PHRASE


def test_parse_srt_cues():
    cues = _parse_subtitle_cues(SRT_FIXTURE)
    assert len(cues) == 5
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[0].text == "Good evening, and welcome to the show."
    assert cues[-1].start == 14.0
    assert cues[-1].end == 16.75
    assert cues[-1].text == "Thank you all for watching, goodnight."
    assert cues[2].text == DISTINCTIVE_PHRASE


# ---------------------------------------------------------------------------
# try_subtitle_fast_path
# ---------------------------------------------------------------------------


def test_subtitle_fast_path_hit():
    asset = _asset_with_subs(VTT_FIXTURE)
    cand = try_subtitle_fast_path(asset, DISTINCTIVE_PHRASE)
    assert cand is not None
    assert cand.modality == "subtitle"
    assert cand.event_type == "speech_onset"
    assert cand.timestamp == 7.0
    assert cand.end_timestamp == 9.8
    assert cand.similarity >= 85.0


def test_subtitle_fast_path_fuzzy_hit():
    asset = _asset_with_subs(VTT_FIXTURE)
    # Case change, trailing punctuation change, and a one-character typo (jumbs/jumps).
    cand = try_subtitle_fast_path(asset, "THE QUICK BROWN FOX JUMBS OVER THE LAZY DOG!!!")
    assert cand is not None
    assert cand.timestamp == 7.0
    assert cand.similarity >= 85.0


def test_subtitle_fast_path_miss():
    asset = _asset_with_subs(VTT_FIXTURE)
    cand = try_subtitle_fast_path(asset, "this phrase does not appear anywhere nearby")
    assert cand is None


def test_subtitle_fast_path_no_tracks():
    # subtitle_paths=[] -- the graded example video's actual situation (no CC track).
    asset = _asset_with_subs()
    cand = try_subtitle_fast_path(asset, DISTINCTIVE_PHRASE)
    assert cand is None


def test_subtitle_fast_path_srt_also_matches():
    asset = _asset_with_subs(SRT_FIXTURE)
    cand = try_subtitle_fast_path(asset, DISTINCTIVE_PHRASE)
    assert cand is not None
    assert cand.timestamp == 7.0


# ---------------------------------------------------------------------------
# ffprobe-based metadata probing
# ---------------------------------------------------------------------------


def test_probe_metadata(tmp_path):
    video_path = str(tmp_path / "sample.mp4")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    probe = _ffprobe_streams(video_path)
    assert probe["fps"] == 25.0
    assert probe["width"] == 320
    assert probe["height"] == 240
    assert abs(probe["duration_s"] - 2.0) < 0.1
    assert probe["has_audio"] is True


# ---------------------------------------------------------------------------
# Network tests -- opt-in, real downloads (see pytest.ini)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_prepare_asset_okru(tmp_path):
    asset = prepare_asset(OK_RU_URL, work_dir=str(tmp_path))
    assert asset.metadata.fps == 24.0
    assert 3255 < asset.metadata.duration_s < 3270
    assert asset.metadata.height <= 720
    assert asset.subtitle_paths == []  # the example video ships no subtitle track
    assert os.path.exists(asset.audio_path)
    assert os.path.getsize(asset.audio_path) > 0


@pytest.mark.network
def test_prepare_asset_archive(tmp_path):
    asset = prepare_asset(ARCHIVE_URL, work_dir=str(tmp_path))
    assert abs(asset.metadata.duration_s - 596.46) < 2
    assert asset.metadata.height <= 720
    assert os.path.exists(asset.audio_path)
    assert os.path.getsize(asset.audio_path) > 0


@pytest.mark.network
def test_prepare_asset_uses_cache(tmp_path):
    first = prepare_asset(ARCHIVE_URL, work_dir=str(tmp_path))
    first_mtime = os.path.getmtime(first.video_path)

    second = prepare_asset(ARCHIVE_URL, work_dir=str(tmp_path))
    second_mtime = os.path.getmtime(second.video_path)

    assert second.video_path == first.video_path
    assert second_mtime == first_mtime  # not re-downloaded
