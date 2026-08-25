"""
Phase 1 - Ingest + subtitle fast-path.

Consumes: video URL (str), target phrase (str, CLI arg)
Produces: VideoAsset; optional Candidate(modality="subtitle") on fast-path hit

Verification (see PHASES.md): pytest tests/test_ingest.py
(Phase 1's real network-dependent check is `pytest -m network tests/test_ingest.py`;
verify.py 1 runs the offline suite -- see PHASE1_PLAN.md for why.)
"""

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import yt_dlp

from src.text_match import DEFAULT_MATCH_THRESHOLD, normalize, similarity
from src.types import Candidate, VideoAsset, VideoMetadata

_SUBTITLE_KIND_CONFIDENCE = {"manual": 1.0, "auto": 0.7, "unknown": 0.85}

_CUE_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# prepare_asset
# ---------------------------------------------------------------------------


def prepare_asset(
    video_url: str, *, work_dir: Optional[str] = None, use_cache: bool = True
) -> VideoAsset:
    """
    Resolve a video URL to a local VideoAsset: download, probe metadata, extract audio.

    Downloads into a cache dir keyed by a hash of video_url (under work_dir, or
    $QUEST1_WORK_DIR, or ./.cache). If a complete cached result exists and use_cache is
    True, reuses it instead of re-downloading -- Phases 2/3/5/6 all re-consume the same
    video, and re-downloading a 54-minute source on every run is impractical.
    """
    cache_dir = _cache_dir_for(video_url, work_dir)
    meta_path = os.path.join(cache_dir, "meta.json")

    if use_cache and os.path.exists(meta_path):
        cached = _load_cached_asset(meta_path)
        if cached is not None:
            return cached

    os.makedirs(cache_dir, exist_ok=True)
    outtmpl = os.path.join(cache_dir, "video.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": outtmpl,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en.*"],
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = _extract_info_with_retry(ydl, video_url)

    video_path = _find_downloaded_video(cache_dir)
    if video_path is None:
        raise RuntimeError(
            f"yt-dlp reported success for {video_url!r} but no video file was found "
            f"in {cache_dir!r}"
        )

    probe = _ffprobe_streams(video_path)
    if not probe["has_audio"]:
        raise RuntimeError(
            f"Downloaded video {video_path!r} has no audio stream. This most likely "
            f"means the format selector "
            f"'bestvideo[height<=720]+bestaudio/best[height<=720]/best' dropped the "
            f"audio track for this source (some HLS formats report audio_ext=none in "
            f"their format list) -- the selector needs rework rather than proceeding, "
            f"since Phase 2 (ASR) has no audio to transcribe."
        )

    audio_path = os.path.join(cache_dir, "audio.wav")
    _extract_audio(video_path, audio_path)

    subtitle_paths = _find_subtitle_files(cache_dir)
    _write_subtitle_kinds(cache_dir, subtitle_paths, info)

    metadata = VideoMetadata(
        fps=probe["fps"],
        duration_s=probe["duration_s"],
        width=probe["width"],
        height=probe["height"],
    )

    _save_meta(meta_path, video_path, audio_path, subtitle_paths, metadata)

    return VideoAsset(
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
        subtitle_paths=subtitle_paths,
    )


def _cache_dir_for(video_url: str, work_dir: Optional[str]) -> str:
    base = work_dir or os.environ.get("QUEST1_WORK_DIR") or ".cache"
    key = hashlib.sha256(video_url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, key)


def _load_cached_asset(meta_path: str) -> Optional[VideoAsset]:
    with open(meta_path, encoding="utf-8") as f:
        data = json.load(f)
    if not (os.path.exists(data["video_path"]) and os.path.exists(data["audio_path"])):
        return None
    metadata = VideoMetadata(**data["metadata"])
    return VideoAsset(
        video_path=data["video_path"],
        audio_path=data["audio_path"],
        metadata=metadata,
        subtitle_paths=data.get("subtitle_paths", []),
    )


def _save_meta(
    meta_path: str,
    video_path: str,
    audio_path: str,
    subtitle_paths: list[str],
    metadata: VideoMetadata,
) -> None:
    data = {
        "video_path": video_path,
        "audio_path": audio_path,
        "subtitle_paths": subtitle_paths,
        "metadata": {
            "fps": metadata.fps,
            "duration_s": metadata.duration_s,
            "width": metadata.width,
            "height": metadata.height,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _extract_info_with_retry(ydl: "yt_dlp.YoutubeDL", video_url: str, attempts: int = 3):
    """
    Retry extract_info itself, not just the download.

    yt-dlp's own `retries`/`fragment_retries` options cover fragment and format-URL HTTP
    downloads, but NOT the initial webpage/API extraction step. Empirically, ok.ru resets
    the connection intermittently during exactly that step (its extractor's mobile-webpage
    fallback) -- the identical command failed, then succeeded, then failed again across
    consecutive runs during Phase 1 planning. Without this separate retry loop,
    prepare_asset() fails nondeterministically on the graded example video.
    """
    delay = 2.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return ydl.extract_info(video_url, download=True)
        except Exception as exc:  # yt-dlp wraps most failures as DownloadError, but not all
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(
        f"Failed to extract/download {video_url!r} after {attempts} attempts "
        f"(last error: {last_exc})"
    ) from last_exc


def _find_downloaded_video(cache_dir: str) -> Optional[str]:
    non_video_ext = {".vtt", ".srt", ".json", ".wav", ".part", ".ytdl"}
    candidates = []
    for name in os.listdir(cache_dir):
        if not name.startswith("video."):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in non_video_ext:
            continue
        candidates.append(os.path.join(cache_dir, name))
    if not candidates:
        return None
    # Prefer the largest file in case yt-dlp left a stray partial/merge artifact behind.
    return max(candidates, key=os.path.getsize)


def _find_subtitle_files(cache_dir: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(cache_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in (".vtt", ".srt"):
            paths.append(os.path.join(cache_dir, name))
    return paths


def _parse_frame_rate(value: str) -> float:
    """Parse an ffprobe frame-rate fraction like '24/1' or '24000/1001'. '0/0' means unset."""
    try:
        num_s, _, den_s = value.partition("/")
        num = float(num_s)
        den = float(den_s) if den_s else 1.0
        if den == 0:
            return 0.0
        return num / den
    except (ValueError, TypeError):
        return 0.0


def _ffprobe_streams(video_path: str) -> dict:
    """
    Probe the downloaded file directly rather than trusting yt-dlp's manifest-reported
    fps -- HLS-muxed output can differ from what the m3u8 advertised, and Phase 5's
    frame-accuracy math depends on this number being right.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path!r}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not video_streams:
        raise RuntimeError(f"No video stream found in {video_path!r}")
    vs = video_streams[0]

    # Prefer avg_frame_rate (measured over the whole stream); fall back to r_frame_rate
    # (the container's nominal rate) when avg_frame_rate is unset ("0/0").
    fps = _parse_frame_rate(vs.get("avg_frame_rate", "0/0"))
    if fps <= 0:
        fps = _parse_frame_rate(vs.get("r_frame_rate", "0/0"))

    duration_s = float(
        data.get("format", {}).get("duration") or vs.get("duration") or 0.0
    )

    return {
        "fps": fps,
        "duration_s": duration_s,
        "width": int(vs.get("width", 0)),
        "height": int(vs.get("height", 0)),
        "has_audio": bool(audio_streams),
    }


def _extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract mono 16kHz PCM audio via ffmpeg CLI.

    ffmpeg CLI (not PyAV) is correct here: CLAUDE.md's "PyAV only" rule is scoped to
    frame-accurate seeking inside the pipeline, where ffmpeg -ss and PyAV decode-forward
    genuinely disagree on which frame you land on. Audio extraction has no such ambiguity
    -- ffmpeg's own resampler is the simplest reliable tool for it, and 16kHz mono is
    exactly faster-whisper's expected input, so Phase 2 does no resampling of its own.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed for {video_path!r}: "
            f"{result.stderr.strip()[-2000:]}"
        )
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError(f"ffmpeg produced an empty/missing audio file at {audio_path!r}")


def _lang_from_subtitle_filename(path: str) -> Optional[str]:
    """yt-dlp names subtitle files '<base>.<lang>.<ext>' (e.g. 'video.en.vtt')."""
    parts = os.path.basename(path).split(".")
    if len(parts) >= 3:
        return parts[-2]
    return None


def _classify_subtitle_kind(info: dict, lang: Optional[str]) -> str:
    """
    yt-dlp standardizes 'subtitles' (manually authored) vs 'automatic_captions'
    (platform-generated) as separate top-level keys in the extracted info dict, keyed by
    language, across extractors -- this is more reliable than trying to infer manual vs.
    auto from the downloaded filename, which has no consistent convention.
    """
    if lang is None:
        return "unknown"
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if lang in manual:
        return "manual"
    if lang in auto:
        return "auto"
    return "unknown"


def _write_subtitle_kinds(cache_dir: str, subtitle_paths: list[str], info: dict) -> None:
    """
    Persist manual-vs-auto classification alongside the subtitle files, since VideoAsset
    only carries subtitle_paths (not per-file metadata) -- try_subtitle_fast_path reads
    this sidecar back to set Candidate.confidence and evidence["subtitle_kind"].
    """
    kinds = {}
    for path in subtitle_paths:
        lang = _lang_from_subtitle_filename(path)
        kinds[os.path.basename(path)] = _classify_subtitle_kind(info, lang)
    kinds_path = os.path.join(cache_dir, "subtitle_kinds.json")
    with open(kinds_path, "w", encoding="utf-8") as f:
        json.dump(kinds, f, indent=2)


def _subtitle_kind_for(sub_path: str) -> str:
    kinds_path = os.path.join(os.path.dirname(sub_path), "subtitle_kinds.json")
    if not os.path.exists(kinds_path):
        return "unknown"
    try:
        with open(kinds_path, encoding="utf-8") as f:
            kinds = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return kinds.get(os.path.basename(sub_path), "unknown")


# ---------------------------------------------------------------------------
# Subtitle parsing (WebVTT + SRT, one parser -- see PHASE1_PLAN.md)
# ---------------------------------------------------------------------------


@dataclass
class _Cue:
    start: float
    end: float
    text: str


def _parse_timestamp(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_subtitle_cues(path: str) -> list[_Cue]:
    """
    Parse a WebVTT or SRT file into cues.

    Both formats share the 'HH:MM:SS[.,]mmm --> HH:MM:SS[.,]mmm' cue-timing line; the
    differences are VTT's 'WEBVTT' header, comma-vs-dot decimal separator, and SRT's
    numeric index line before each cue. Rather than two format-specific parsers, this
    finds the timing line within each blank-line-delimited block and treats everything
    before it (a 'WEBVTT' header, a NOTE/STYLE block, or an SRT index) as non-cue content
    to skip, and everything after it as cue text.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    cues: list[_Cue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        timing_idx = None
        match = None
        for i, line in enumerate(lines):
            m = _CUE_TIME_RE.search(line)
            if m:
                timing_idx, match = i, m
                break
        if timing_idx is None:
            continue  # header / NOTE / STYLE / cue-id-only block

        start = _parse_timestamp(match.group(1))
        end = _parse_timestamp(match.group(2))
        text_lines = lines[timing_idx + 1 :]
        text = " ".join(_TAG_RE.sub("", ln).strip() for ln in text_lines).strip()
        if text:
            cues.append(_Cue(start=start, end=end, text=text))
    return cues


# ---------------------------------------------------------------------------
# try_subtitle_fast_path
# ---------------------------------------------------------------------------


def try_subtitle_fast_path(
    asset: VideoAsset,
    target_phrase: str,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Optional[Candidate]:
    """
    Check for an existing subtitle/CC track and fuzzy-match target_phrase against it.

    Returns None if no track exists or no match clears threshold -- callers fall through
    to the ASR/OCR tracks in that case. Never raises: a malformed subtitle file is a
    reason to fall through, not to fail the whole pipeline. This is the common outcome
    on the graded example video, which ships no subtitle track at all.

    event_type is "speech_onset", not "visual_text_onset": a platform subtitle cue is
    timed to speech, not to a pixel change, so Phase 5 must apply its ASR-to-frame
    policy to subtitle candidates too, not the direct visual mapping used for OCR.

    confidence distinguishes manually authored captions (1.0) from platform
    auto-captions (0.7, since they're themselves an ASR guess) via the subtitle_kinds.json
    sidecar written by prepare_asset; files with no such sidecar (e.g. hand-provided
    fixtures) get "unknown" (0.85). Flag for Phase 4: manual subtitles at 1.0 will
    automatically outrank ASR/OCR candidates within an agreement cluster -- that's
    intended, but the arbiter should document it rather than inherit it silently.
    """
    if not asset.subtitle_paths:
        return None

    best: Optional[Candidate] = None
    for sub_path in asset.subtitle_paths:
        kind = _subtitle_kind_for(sub_path)
        confidence = _SUBTITLE_KIND_CONFIDENCE[kind]
        try:
            cues = _parse_subtitle_cues(sub_path)
        except OSError:
            continue

        for i, cue in enumerate(cues):
            score = similarity(target_phrase, cue.text)
            if score < threshold:
                continue
            if best is not None and score <= best.similarity:
                continue
            best = Candidate(
                modality="subtitle",
                event_type="speech_onset",
                timestamp=cue.start,
                end_timestamp=cue.end,
                matched_text=cue.text,
                normalized_text=normalize(cue.text),
                similarity=score,
                confidence=confidence,
                evidence={
                    "source_file": sub_path,
                    "cue_index": i,
                    "subtitle_kind": kind,
                    "lang": _lang_from_subtitle_filename(sub_path),
                },
            )

    return best
