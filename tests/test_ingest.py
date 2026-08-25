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

import src.ingest as ingest_module
from src.ingest import (
    _extract_audio,
    _extract_info_with_subtitle_fallback,
    _ffprobe_streams,
    _format_selector_for_tier,
    _looks_like_subtitle_failure,
    _migrate_legacy_meta,
    _parse_subtitle_cues,
    _parse_tagged_words,
    prepare_asset,
    try_subtitle_fast_path,
)
from src.text_match import normalize, similarity, window_similarity
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


def test_normalize_strips_bracketed_sound_annotations():
    # Real bug, found by running against a real (non-example) music video: auto
    # captions insert bracketed sound-event annotations ("[singing]", "[music]") as
    # literal words in the caption stream. A whole bracketed span must be dropped
    # (not just its brackets, which would leave "singing" behind as a real word).
    assert normalize("they make [singing] me feel sad") == "they make me feel sad"
    assert normalize("[music] don't want no happy pills") == "don t want no happy pills"


def test_similarity_scale_is_0_to_1():
    # Locks the scale so a future change can't silently reintroduce the 0-100 vs 0-1
    # mismatch that made test_asr_track.py / test_ocr_track.py's original
    # `similarity >= 0.85` assertions vacuously true against a 0-100 value.
    assert similarity("hello world", "hello world") == 1.0
    assert 0.0 <= similarity("hello world", "completely unrelated text") <= 1.0
    assert 0.0 <= window_similarity("hello world", "hello world there") <= 1.0
    assert window_similarity("hello world", "hello world") == 1.0


def test_window_similarity_penalizes_partial_overlap():
    # partial_ratio (similarity) would score this a perfect match since "my mind" is a
    # substring hit; ratio (window_similarity) must not, since the extents differ.
    target = "my mind rebels at stagnation"
    assert similarity(target, "my mind") == 1.0
    assert window_similarity(target, "my mind") < 0.85


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


def test_parse_cues_keeps_cue_with_stray_space_filler_line(tmp_path):
    # Real bug, found by running against a real (non-example) video: user reported a
    # dialogue line ("Do I look civilized to you?") as genuinely present -- confirmed
    # directly with grep that it WAS in the raw .vtt file, yet the pipeline reported
    # not_found. Root cause: this auto-caption file uses a line containing a single
    # stray space character as filler WITHIN some cues' own multi-line content
    # (verified against the real file: "TIMING\n \nACTUAL TEXT", where the " " line is
    # not a separator). The old `\n\s*\n` block-split pattern let \s match that space,
    # silently splitting the cue into a timing-only fragment (discarded: no text) and a
    # text-only fragment (discarded: no timing line) -- the entire line vanished from
    # every parse with no error. Reproduces the exact real structure.
    vtt_path = str(tmp_path / "stray_space.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(
            "WEBVTT\n\n"
            "00:02:40.000 --> 00:02:41.000\n"
            "previous line\n\n"
            "00:02:42.000 --> 00:03:04.710\n"
            " \n"
            "Do I look civilized to you?\n\n"
            "00:03:05.000 --> 00:03:06.000\n"
            "next line\n"
        )

    cues = _parse_subtitle_cues(vtt_path)
    texts = [c.text for c in cues]
    assert "Do I look civilized to you?" in texts
    civilized_cue = next(c for c in cues if c.text == "Do I look civilized to you?")
    assert civilized_cue.start == 162.0

    # And genuinely bare blank lines between DIFFERENT cues must still separate them
    # correctly (not accidentally merged into one giant cue) -- the fix narrows what
    # counts as a separator, it must not stop separating genuine ones.
    assert "previous line" in texts
    assert "next line" in texts
    assert len(cues) == 3


def test_parse_tagged_words_recovers_word_after_stray_space_filler_line(tmp_path):
    # Companion to the cue-level test above: the same real bug also silently dropped
    # this cue's karaoke <c> word tags entirely (not just its plain text), which is
    # what try_subtitle_fast_path's word-level onset refinement depends on.
    vtt_path = str(tmp_path / "stray_space.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(
            "WEBVTT\n\n"
            "00:02:42.000 --> 00:03:04.710\n"
            " \n"
            "Do<00:02:42.240><c> I</c><00:02:42.400><c> look</c><00:02:42.720><c> civilized</c>"
            "<00:02:43.320><c> to</c><00:02:43.480><c> you?</c>\n"
        )

    words = _parse_tagged_words(vtt_path)
    word_texts = [w.strip() for _, w in words]
    assert "civilized" in word_texts
    civilized_ts = next(t for t, w in words if w.strip() == "civilized")
    assert civilized_ts == 162.72


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
    assert cand.similarity >= 0.85


def test_subtitle_fast_path_fuzzy_hit():
    asset = _asset_with_subs(VTT_FIXTURE)
    # Case change, trailing punctuation change, and a one-character typo (jumbs/jumps).
    cand = try_subtitle_fast_path(asset, "THE QUICK BROWN FOX JUMBS OVER THE LAZY DOG!!!")
    assert cand is not None
    assert cand.timestamp == 7.0
    assert cand.similarity >= 0.85


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


# Real karaoke-tagged VTT fragment, reproduced verbatim from a real (non-example)
# video's actual auto-caption file: a continuously scrolling 2-line window, where cue
# block boundaries do NOT correspond to true word onsets. "approval" is truly first
# spoken/tagged at 140.319s; the cue block containing the best cue-level text match
# starts at 142.070s -- 1.75s later. See the module comment above try_subtitle_fast_path
# for the two rejected cue-level fixes and why only per-word <c> tags solve this.
_ROLLING_CAPTION_VTT = (
    "WEBVTT\n\n"
    "00:02:15.000 --> 00:02:20.000 align:start position:0%\n"
    "narrating<00:02:15.500><c> how</c><00:02:16.000><c> spend</c><00:02:16.500><c> currently</c>"
    "<00:02:17.000><c> moves</c><00:02:17.500><c> from</c><00:02:18.000><c> the</c>"
    "<00:02:18.500><c> request</c>\n\n"
    "00:02:20.000 --> 00:02:22.070 align:start position:0%\n"
    "spend currently moves from the request\n"
    "and<00:02:20.319><c> approval</c><00:02:20.720><c> through</c><00:02:20.959><c> a</c>"
    "<00:02:21.120><c> payments</c><00:02:21.760><c> and</c>\n\n"
    "00:02:22.070 --> 00:02:22.080 align:start position:0%\n"
    "and approval through a payments and\n\n"
    "00:02:22.080 --> 00:02:24.710 align:start position:0%\n"
    "and approval through a payments and\n"
    "accounting<00:02:22.720><c> and</c><00:02:23.200><c> identify</c><00:02:23.599><c> the</c>"
    "<00:02:23.840><c> actual</c><00:02:24.239><c> gap</c>\n"
)


def test_subtitle_fast_path_word_level_onset_for_rolling_caption(tmp_path):
    # The fix: when the target phrase's words are all captured by real <c> tags (plus
    # the one boundary word recovered per cue, see _parse_tagged_words), onset comes
    # from the true word timestamp (140.319s), not the winning cue block's own later
    # start (142.070s) or the video's very first cue (the rejected clustering attempt's
    # failure mode).
    vtt_path = str(tmp_path / "rolling.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(_ROLLING_CAPTION_VTT)

    asset = _asset_with_subs(vtt_path)
    cand = try_subtitle_fast_path(asset, "approval through a payments and accounting")

    assert cand is not None
    assert cand.evidence["match_method"] == "word_window"
    assert cand.timestamp == 140.319
    assert cand.similarity == 1.0


def test_subtitle_fast_path_falls_back_to_cue_level_when_word_match_is_weak(tmp_path):
    # Real behavior, found by running against a real video: a target phrase containing
    # content genuinely absent from the video (here "reconciliation", which was never
    # actually said) scores just under threshold at the word level (0.8491 -- ratio
    # correctly penalizes the missing word, unlike partial_ratio) even though the
    # phrase is "close". The fast path must fall back to the pre-existing cue-level
    # match rather than reporting no match at all -- never worse than the original
    # (pre-investigation) behavior, even when the more precise method can't confirm.
    vtt_path = str(tmp_path / "rolling.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(_ROLLING_CAPTION_VTT)

    asset = _asset_with_subs(vtt_path)
    cand = try_subtitle_fast_path(asset, "And approval through a payment,reconciliation and accounting")

    assert cand is not None
    assert cand.evidence["match_method"] == "cue"
    assert cand.timestamp == 142.07  # the cue-level answer, same as before this fix


def test_subtitle_fast_path_plain_track_still_uses_cue_matching():
    # Regression guard: a track with no <c> tags at all (sample.en.vtt, and every real
    # non-example video tested so far except the rolling-caption one) must be entirely
    # unaffected by this change -- _parse_tagged_words returns [] and the original,
    # untouched cue-level path handles it exactly as before.
    asset = _asset_with_subs(VTT_FIXTURE)
    cand = try_subtitle_fast_path(asset, DISTINCTIVE_PHRASE)
    assert cand is not None
    assert cand.evidence["match_method"] == "cue"
    assert cand.timestamp == 7.0


def test_subtitle_fast_path_rejects_degenerate_short_cue(tmp_path):
    # Real bug, found by running against a real (non-example) video: rapidfuzz's
    # partial_ratio scored a standalone one-character cue ("I") as a PERFECT 1.0 match
    # against the unrelated target "Do I look civilized to you?", purely because "i"
    # trivially appears in it -- the exact same root cause as an earlier OCR-track bug
    # (ocr_track.py's _MIN_CANDIDATE_LENGTH_RATIO), now confirmed in the subtitle
    # cue-matching path too. Reproduces the real cue structure: a short "I" cue sitting
    # right next to the actual (unrelated) dialogue.
    vtt_path = str(tmp_path / "degenerate.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(
            "WEBVTT\n\n"
            "00:00:22.88 --> 00:00:23.59\n"
            "I promise.\n\n"
            "00:00:23.59 --> 00:00:23.60\n"
            "I\n\n"
            "00:02:35.67 --> 00:02:41.99\n"
            "I was hoping to have a civilized conversation.\n"
        )

    asset = _asset_with_subs(vtt_path)
    cand = try_subtitle_fast_path(asset, "Do I look civilized to you?")

    # The degenerate "I" cue must never win -- either no match at all, or (since a
    # longer, topically-related cue exists) that longer cue, never the single character.
    assert cand is None or cand.matched_text != "I"


def test_subtitle_fast_path_still_matches_short_target_short_cue(tmp_path):
    # Regression guard: the length-ratio gate is relative to the TARGET's own length, so
    # a short target genuinely matching a comparably-short cue must still work --
    # mirrors ocr_track.py's own equivalent guard-doesn't-overreach test
    # (test_score_lines_still_matches_genuine_short_target).
    vtt_path = str(tmp_path / "short.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nStop!\n")

    asset = _asset_with_subs(vtt_path)
    cand = try_subtitle_fast_path(asset, "Stop")
    assert cand is not None
    assert cand.matched_text == "Stop!"


# ---------------------------------------------------------------------------
# Two-tier fetch (runtime hardening)
# ---------------------------------------------------------------------------


def test_format_selector_for_tier_low():
    # "bestaudio/worst": true audio-only where a host provides one, degrades to the
    # cheapest available muxed stream where it doesn't (verified: ok.ru has no
    # audio-only format at all -- see ingest.py's module docstring).
    assert _format_selector_for_tier("low") == "bestaudio/worst"


def test_format_selector_for_tier_high():
    # Unchanged from the original single-tier selector.
    assert _format_selector_for_tier("high") == "bestvideo[height<=720]+bestaudio/best[height<=720]/best"


def test_format_selector_for_tier_unknown_raises():
    with pytest.raises(ValueError):
        _format_selector_for_tier("ultra-hd")


def test_migrate_legacy_meta_upgrades_flat_shape():
    # Pre-tiering cache entries stored a flat {video_path, audio_path, metadata,
    # subtitle_paths}. Must upgrade to {tiers: {high: ...}, audio_path,
    # subtitle_paths} without discarding an already-downloaded video.
    legacy = {
        "video_path": ".cache/abc/video.mp4",
        "audio_path": ".cache/abc/audio.wav",
        "subtitle_paths": [],
        "metadata": {"fps": 24.0, "duration_s": 100.0, "width": 960, "height": 720},
    }
    migrated = _migrate_legacy_meta(legacy)
    assert migrated["tiers"]["high"]["video_path"] == ".cache/abc/video.mp4"
    assert migrated["tiers"]["high"]["metadata"]["fps"] == 24.0
    assert migrated["audio_path"] == ".cache/abc/audio.wav"
    assert migrated["subtitle_paths"] == []


def test_looks_like_subtitle_failure_matches_known_pattern():
    # Real observed failure: YouTube rate-limiting the subtitle endpoint specifically
    # (HTTP 429), which must not be allowed to abort the whole video/audio download.
    exc = RuntimeError(
        "Failed to extract/download 'x' after 3 attempts (last error: ERROR: Unable "
        "to download video subtitles for 'en': HTTP Error 429: Too Many Requests)"
    )
    assert _looks_like_subtitle_failure(exc) is True


def test_looks_like_subtitle_failure_does_not_match_unrelated_error():
    exc = RuntimeError("Failed to extract/download 'x' after 3 attempts (last error: connection reset)")
    assert _looks_like_subtitle_failure(exc) is False


class _FakeYDL:
    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_extract_info_with_subtitle_fallback_retries_without_subtitles(monkeypatch):
    created_opts = []

    def fake_youtube_dl(opts):
        created_opts.append(opts)
        return _FakeYDL(opts)

    monkeypatch.setattr(ingest_module.yt_dlp, "YoutubeDL", fake_youtube_dl)

    call_count = [0]

    def fake_retry(ydl, video_url, attempts=3):
        call_count[0] += 1
        if ydl.opts.get("writesubtitles"):
            raise RuntimeError(
                "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"
            )
        return {"fake": "info"}

    monkeypatch.setattr(ingest_module, "_extract_info_with_retry", fake_retry)

    ydl_opts = {"writesubtitles": True, "writeautomaticsub": True, "format": "x"}
    info, subtitles_requested = _extract_info_with_subtitle_fallback(
        ydl_opts, "http://example.com/video"
    )

    assert subtitles_requested is False
    assert info == {"fake": "info"}
    assert call_count[0] == 2  # first (failed) attempt + fallback (succeeded)
    assert created_opts[0]["writesubtitles"] is True
    assert created_opts[1]["writesubtitles"] is False
    assert created_opts[1]["writeautomaticsub"] is False


def test_extract_info_with_subtitle_fallback_succeeds_without_retry_when_subtitles_work(monkeypatch):
    monkeypatch.setattr(ingest_module.yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(opts))

    call_count = [0]

    def fake_retry(ydl, video_url, attempts=3):
        call_count[0] += 1
        return {"fake": "info"}

    monkeypatch.setattr(ingest_module, "_extract_info_with_retry", fake_retry)

    info, subtitles_requested = _extract_info_with_subtitle_fallback(
        {"writesubtitles": True}, "http://example.com/video"
    )

    assert subtitles_requested is True
    assert call_count[0] == 1  # no fallback needed


def test_extract_info_with_subtitle_fallback_reraises_non_subtitle_error(monkeypatch):
    monkeypatch.setattr(ingest_module.yt_dlp, "YoutubeDL", lambda opts: _FakeYDL(opts))

    def fake_retry(ydl, video_url, attempts=3):
        raise RuntimeError("Failed to extract/download: connection reset")

    monkeypatch.setattr(ingest_module, "_extract_info_with_retry", fake_retry)

    with pytest.raises(RuntimeError, match="connection reset"):
        _extract_info_with_subtitle_fallback({"writesubtitles": True}, "http://example.com/video")


def test_prepare_asset_recovers_subtitles_written_before_fallback_fired(monkeypatch, tmp_path):
    # Real bug, found by running against a real (non-example) video: yt-dlp's subtitle
    # request list ("en", "en.*") pulled in auto-translated variants (e.g. "en-hi",
    # "en-fil") alongside genuine English captions. Those translated variants got
    # rate-limited (429), which made the whole extract_info() call raise -- but only
    # AFTER yt-dlp had already written the real 'en' .vtt file to disk. The subtitle
    # fallback then retried without subtitles and returned subtitles_requested=False,
    # and prepare_asset used to trust that flag as "don't bother checking disk",
    # silently discarding a genuine, already-downloaded English caption track. Verified
    # live: this exact scenario reported not_found for a phrase that WAS in the caption
    # file the whole time. This test reproduces it offline by pre-writing a real .vtt
    # to the cache dir before prepare_asset runs, with subtitles_requested=False.
    cache_dir = os.path.join(str(tmp_path), "abc123")
    os.makedirs(cache_dir)

    # Simulate yt-dlp's partial-success-then-raise: a real subtitle file already sitting
    # in the cache dir by the time _extract_info_with_subtitle_fallback returns.
    vtt_path = os.path.join(cache_dir, "video_low.en.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(
            "WEBVTT\n\n"
            "00:00:57.182 --> 00:00:58.934\n"
            f"{DISTINCTIVE_PHRASE}\n"
        )

    # A dummy "video" file so _find_downloaded_video succeeds without a real download.
    video_path = os.path.join(cache_dir, "video_low.mp4")
    with open(video_path, "wb") as f:
        f.write(b"not a real video, just needs to exist")

    monkeypatch.setattr(ingest_module, "_cache_dir_for", lambda video_url, work_dir: cache_dir)
    monkeypatch.setattr(
        ingest_module,
        "_extract_info_with_subtitle_fallback",
        lambda ydl_opts, video_url: ({"fake": "info"}, False),
    )
    monkeypatch.setattr(
        ingest_module,
        "_ffprobe_streams",
        lambda video_path: {
            "fps": 25.0,
            "duration_s": 60.0,
            "width": 320,
            "height": 240,
            "has_audio": True,
            "has_video": True,
        },
    )
    monkeypatch.setattr(
        ingest_module,
        "_extract_audio",
        lambda video_path, audio_path: open(audio_path, "wb").close(),
    )

    asset = prepare_asset("http://example.com/video", work_dir=str(tmp_path))

    assert asset.subtitle_paths == [vtt_path]
    hit = try_subtitle_fast_path(asset, DISTINCTIVE_PHRASE)
    assert hit is not None
    assert hit.matched_text == DISTINCTIVE_PHRASE


def test_migrate_legacy_meta_is_noop_on_already_tiered_shape():
    already_tiered = {
        "tiers": {"low": {"video_path": "x", "metadata": {}}},
        "audio_path": "a",
        "subtitle_paths": [],
    }
    assert _migrate_legacy_meta(already_tiered) == already_tiered


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
    assert probe["has_video"] is True


def test_ffprobe_streams_audio_only_reports_has_video_false(tmp_path):
    # Real bug found by running against a real YouTube video: tier="low"'s
    # "bestaudio/worst" selector can legitimately pick a genuine audio-only format on
    # a host that exposes one (verified: YouTube does; ok.ru does not, which is why
    # this was never hit against the ok.ru example). _ffprobe_streams must NOT raise
    # in that case -- it's a real, expected outcome, not an error.
    audio_path = str(tmp_path / "audio_only.m4a")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:duration=2",
            "-c:a", "aac", audio_path,
        ],
        capture_output=True, text=True,
    )

    probe = _ffprobe_streams(audio_path)
    assert probe["has_video"] is False
    assert probe["has_audio"] is True
    assert probe["width"] == 0
    assert probe["height"] == 0
    assert probe["fps"] == 0.0
    assert abs(probe["duration_s"] - 2.0) < 0.2  # duration still comes from format-level data


def test_ffprobe_streams_timeout_raises_clear_error(tmp_path, monkeypatch):
    # Runtime-audit finding: subprocess.run for ffprobe/ffmpeg had no timeout= at all --
    # a hung subprocess would block indefinitely with no way to notice. This forces a
    # REAL subprocess.TimeoutExpired against a real ffprobe call (not just checking the
    # parameter exists) and asserts it surfaces as a clear RuntimeError.
    import src.ingest as ingest_module

    video_path = str(tmp_path / "sample.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", video_path,
        ],
        capture_output=True, text=True,
    )

    monkeypatch.setattr(ingest_module, "_FFPROBE_TIMEOUT_S", 0.0001)
    with pytest.raises(RuntimeError, match="timed out"):
        _ffprobe_streams(video_path)


def test_extract_audio_timeout_raises_clear_error(tmp_path, monkeypatch):
    import src.ingest as ingest_module

    video_path = str(tmp_path / "sample.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-shortest",
            "-c:v", "libx264", "-c:a", "aac", video_path,
        ],
        capture_output=True, text=True,
    )

    audio_path = str(tmp_path / "audio.wav")
    monkeypatch.setattr(ingest_module, "_FFMPEG_TIMEOUT_S", 0.0001)
    with pytest.raises(RuntimeError, match="timed out"):
        _extract_audio(video_path, audio_path)


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
