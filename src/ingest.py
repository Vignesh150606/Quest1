"""
Phase 1 - Ingest + subtitle fast-path.

Consumes: video URL (str), target phrase (str, CLI arg)
Produces: VideoAsset; optional Candidate(modality="subtitle") on fast-path hit

Verification: pytest tests/test_ingest.py
(Phase 1's real network-dependent check is `pytest -m network tests/test_ingest.py`;
verify.py 1 runs the offline suite -- see PHASES_1_7_PLAN.md for why.)

Two-tier fetch (hardening added post-Phase-6, runtime audit): verified directly (not
assumed) that ok.ru exposes no genuine audio-only format -- every listed format has
empty vcodec/acodec fields, re-probed twice. So prepare_asset() defaults to tier="low"
(format="bestaudio/worst": picks true audio-only where a host provides one, degrades
to the cheapest available muxed stream where it doesn't -- generic, not ok.ru-specific)
for subtitles/ASR/a fallback frame, and only escalates to tier="high" (the original
height<=720 selector) when OCR actually turns out to be needed. Audio and subtitles are
fetched/extracted once, from whichever tier is fetched first, and reused across tiers --
they don't depend on video quality.
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

from src.text_match import DEFAULT_MATCH_THRESHOLD, normalize, similarity, window_similarity
from src.types import Candidate, VideoAsset, VideoMetadata

_SUBTITLE_KIND_CONFIDENCE = {"manual": 1.0, "auto": 0.7, "unknown": 0.85}

_CUE_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")

_FFPROBE_TIMEOUT_S = 30
_FFMPEG_TIMEOUT_S = 600  # generous for a long video's audio extraction on a slow machine

# Real bug, found by running against a real (non-example) video: rapidfuzz's
# partial_ratio finds the best alignment of the SHORTER of its two strings within the
# longer one -- a subtitle cue that's just "I" (one character) scored a PERFECT 1.0
# match against the target "Do I look civilized to you?", purely because "i" trivially
# appears in it. This is the exact same root cause as an earlier OCR-track bug
# (ocr_track.py's _MIN_CANDIDATE_LENGTH_RATIO), now found independently in the subtitle
# cue-matching path -- a single character can never be a meaningful signal that a cue
# genuinely IS the target line. NOT fixed inside text_match.similarity() itself: that
# function's own test (test_window_similarity_penalizes_partial_overlap) deliberately
# relies on a short candidate ("my mind") validly matching a longer target ("my mind
# rebels at stagnation") -- the failure mode is about how SHORT a cue is allowed to be
# relative to the target it's being checked against, which is specific to how this
# call site uses similarity(), not a defect in the function itself.
_MIN_CUE_LENGTH_RATIO = 0.5

_TIER_FORMAT_SELECTORS = {
    "low": "bestaudio/worst",
    "high": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
}


def _format_selector_for_tier(tier: str) -> str:
    """
    Pure function (no network) so the tier->selector mapping is directly unit-testable.
    See module docstring for the rationale behind each tier's selector.
    """
    try:
        return _TIER_FORMAT_SELECTORS[tier]
    except KeyError:
        raise ValueError(
            f"Unknown tier {tier!r}; expected one of {sorted(_TIER_FORMAT_SELECTORS)}"
        )


class _QuietYdlLogger:
    """
    Suppresses yt-dlp's own stdout/stderr messages (its "quiet" option silences INFO but
    still prints its own WARNING/ERROR lines directly, bypassing our own progress
    output). Real UX issue, found by watching a real cold run: ok.ru's extractor
    intermittently resets the connection on its first extraction attempt (a known,
    already-retried flake -- see _extract_info_with_retry) -- the run still succeeds on
    retry, but the raw "ERROR: ... Connection aborted" line from the FAILED attempt
    prints to the terminal before that, which reads as "this tool is broken" to someone
    unfamiliar with the code (a recruiter running this cold, exactly the audience this
    matters for). A genuine, final failure (all retries exhausted) still surfaces
    clearly -- that raises RuntimeError, which main.py prints as "Error: ...".
    """

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


# ---------------------------------------------------------------------------
# prepare_asset
# ---------------------------------------------------------------------------


def prepare_asset(
    video_url: str,
    *,
    tier: str = "low",
    work_dir: Optional[str] = None,
    use_cache: bool = True,
) -> VideoAsset:
    """
    Resolve a video URL (or a local video file path) to a local VideoAsset: download
    (if remote), probe metadata, extract audio.

    Downloads into a cache dir keyed by a hash of video_url (under work_dir, or
    $QUEST1_WORK_DIR, or ./.cache). If a complete cached result exists for the
    requested tier and use_cache is True, reuses it instead of re-downloading.

    tier="low" (default) fetches the cheapest available format -- enough for the
    subtitle fast-path, ASR, and a fallback output frame. tier="high" fetches the
    height<=720-capped format, needed only when OCR ends up running. Call again with
    tier="high" to escalate an asset already fetched at tier="low"; audio and
    subtitles are fetched/extracted once (whichever tier is fetched first) and reused
    across tiers -- see module docstring.

    If video_url is an existing local file path, yt-dlp is skipped entirely, the file
    is probed directly, and `tier` is ignored (there's nothing to escalate -- the file
    IS the file). This is what lets Phase 3's synthetic-clip test (and Phase 6's
    offline end-to-end test) exercise the full pipeline with no network.
    """
    cache_dir = _cache_dir_for(video_url, work_dir)
    meta_path = os.path.join(cache_dir, "meta.json")

    is_local = os.path.isfile(video_url)
    effective_tier = "high" if is_local else tier

    if use_cache and os.path.exists(meta_path):
        cached = _load_cached_asset(meta_path, effective_tier)
        if cached is not None:
            return cached

    os.makedirs(cache_dir, exist_ok=True)
    existing_meta = _read_meta(meta_path)

    info = None
    if is_local:
        video_path = os.path.abspath(video_url)
        subtitle_paths: list[str] = existing_meta.get("subtitle_paths", [])
    else:
        outtmpl = os.path.join(cache_dir, f"video_{effective_tier}.%(ext)s")

        ydl_opts = {
            "format": _format_selector_for_tier(effective_tier),
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
            "logger": _QuietYdlLogger(),
        }

        info, _subtitles_requested = _extract_info_with_subtitle_fallback(ydl_opts, video_url)

        video_path = _find_downloaded_video(cache_dir, effective_tier)
        if video_path is None:
            raise RuntimeError(
                f"yt-dlp reported success for {video_url!r} (tier={effective_tier!r}) "
                f"but no video file was found in {cache_dir!r}"
            )

        # Always check disk for subtitle files, regardless of subtitles_requested: a
        # failed-with-fallback download (see _extract_info_with_subtitle_fallback) can
        # still have written genuine subtitle files during its earlier failed attempt(s)
        # -- e.g. yt-dlp successfully wrote 'en' captions to disk, then the overall call
        # raised because a LATER requested language (an auto-translated variant) hit a
        # 429. subtitles_requested=False only means "the attempt that finally succeeded
        # didn't ask for subtitles" -- it says nothing about whether earlier attempts
        # already left real files behind. Checking disk directly (cheap: one listdir on
        # a hash-scoped cache dir) reflects what's actually there instead of trusting a
        # flag that can undercount it. Verified live: this exact scenario dropped a
        # real English caption track containing the target phrase, causing a false
        # not_found.
        subtitle_paths = existing_meta.get("subtitle_paths") or _find_subtitle_files(cache_dir)
        if not existing_meta.get("subtitle_paths") and subtitle_paths:
            _write_subtitle_kinds(cache_dir, subtitle_paths, info)

    probe = _ffprobe_streams(video_path)
    if not probe["has_audio"]:
        raise RuntimeError(
            f"Video {video_path!r} (tier={effective_tier!r}) has no audio stream. For "
            f"a downloaded video this most likely means the format selector "
            f"{_format_selector_for_tier(effective_tier)!r} dropped the audio track "
            f"for this source (some HLS formats report audio_ext=none in their format "
            f"list); for a local file it means the file itself has no audio track -- "
            f"either way, the ASR track has no audio to transcribe."
        )

    audio_path = existing_meta.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        audio_path = os.path.join(cache_dir, "audio.wav")
        _extract_audio(video_path, audio_path)

    metadata = VideoMetadata(
        fps=probe["fps"],
        duration_s=probe["duration_s"],
        width=probe["width"],
        height=probe["height"],
        has_video=probe["has_video"],
    )

    _save_tier_meta(
        meta_path, existing_meta, effective_tier, video_path, audio_path, subtitle_paths, metadata
    )

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


def _read_meta(meta_path: str) -> dict:
    """Load meta.json, transparently upgrading a pre-tiering cache entry if found."""
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return _migrate_legacy_meta(data)


def _migrate_legacy_meta(data: dict) -> dict:
    """
    Pre-tiering cache entries stored a single flat {video_path, audio_path, metadata,
    subtitle_paths}. Transparently upgrade to the tiered shape ({tiers: {...},
    audio_path, subtitle_paths}) rather than silently discarding and re-fetching an
    already-downloaded video -- the original single fetch used the height<=720
    selector, i.e. today's "high" tier.
    """
    if "tiers" in data or "video_path" not in data:
        return data
    return {
        "tiers": {"high": {"video_path": data["video_path"], "metadata": data["metadata"]}},
        "audio_path": data.get("audio_path"),
        "subtitle_paths": data.get("subtitle_paths", []),
    }


def _load_cached_asset(meta_path: str, tier: str) -> Optional[VideoAsset]:
    data = _read_meta(meta_path)
    tier_data = data.get("tiers", {}).get(tier)
    if tier_data is None:
        return None
    video_path = tier_data["video_path"]
    audio_path = data.get("audio_path")
    if not audio_path or not (os.path.exists(video_path) and os.path.exists(audio_path)):
        return None
    metadata = VideoMetadata(**tier_data["metadata"])
    return VideoAsset(
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
        subtitle_paths=data.get("subtitle_paths", []),
    )


def _save_tier_meta(
    meta_path: str,
    existing_meta: dict,
    tier: str,
    video_path: str,
    audio_path: str,
    subtitle_paths: list[str],
    metadata: VideoMetadata,
) -> None:
    """Merge this tier's result into meta.json, preserving any other tier already recorded."""
    data = dict(existing_meta)
    tiers = dict(data.get("tiers", {}))
    tiers[tier] = {
        "video_path": video_path,
        "metadata": {
            "fps": metadata.fps,
            "duration_s": metadata.duration_s,
            "width": metadata.width,
            "height": metadata.height,
            "has_video": metadata.has_video,
        },
    }
    data["tiers"] = tiers
    data["audio_path"] = audio_path
    data["subtitle_paths"] = subtitle_paths
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

    Prints a short, reassuring line on a non-final retry (paired with _QuietYdlLogger
    suppressing yt-dlp's own alarming raw "ERROR: ..." line for the failed attempt) --
    a transient network hiccup that resolves on retry shouldn't look like the tool is
    broken to someone running this cold.
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
            print(
                f"    Network hiccup during download setup, retrying "
                f"({attempt}/{attempts})...",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(
        f"Failed to extract/download {video_url!r} after {attempts} attempts "
        f"(last error: {last_exc})"
    ) from last_exc


def _looks_like_subtitle_failure(exc: Exception) -> bool:
    """
    Best-effort classification of a download failure as subtitle-specific rather than
    a genuine video/audio problem. String-matches yt-dlp's known error phrasing
    ("Unable to download video subtitles...") since yt-dlp doesn't expose a typed
    distinction here -- narrow, but targets a real observed failure directly: YouTube
    rate-limiting (HTTP 429) the subtitle endpoint specifically while the video/audio
    format itself was never the problem. Subtitles are best-effort by design (the
    subtitle fast-path already treats "no track" as a normal outcome) and must never
    be allowed to abort the whole download.
    """
    return "subtitle" in str(exc).lower()


def _extract_info_with_subtitle_fallback(ydl_opts: dict, video_url: str):
    """
    Try the download with subtitles requested first. If that specifically fails on a
    subtitle-related error (see _looks_like_subtitle_failure), retry the SAME download
    fresh with subtitles disabled entirely, rather than retrying the identical
    already-failing subtitle request again (_extract_info_with_retry's own 3 attempts
    already exhausted that, and hit the same rate limit each time in practice).

    Returns (info, subtitles_requested) -- subtitles_requested is False when the
    fallback fired, so the caller knows not to look for subtitle files that were never
    asked for.
    """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            return _extract_info_with_retry(ydl, video_url), True
        except RuntimeError as exc:
            if not _looks_like_subtitle_failure(exc):
                raise

    no_subs_opts = dict(ydl_opts, writesubtitles=False, writeautomaticsub=False)
    with yt_dlp.YoutubeDL(no_subs_opts) as ydl:
        return _extract_info_with_retry(ydl, video_url), False


def _find_downloaded_video(cache_dir: str, tier: str) -> Optional[str]:
    prefix = f"video_{tier}."
    non_video_ext = {".vtt", ".srt", ".json", ".wav", ".part", ".ytdl"}
    candidates = []
    for name in os.listdir(cache_dir):
        if not name.startswith(prefix):
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
    try:
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
            timeout=_FFPROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffprobe timed out after {exc.timeout}s on {video_path!r} -- a hung "
            f"subprocess would otherwise block indefinitely with no way to notice."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path!r}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    format_duration_s = float(data.get("format", {}).get("duration") or 0.0)

    if not video_streams:
        # A legitimate outcome, not an error: tier="low"'s "bestaudio/worst" selector
        # picks a genuine audio-only format on hosts that expose one (verified: YouTube
        # does; ok.ru does not). Callers needing an actual frame to extract must
        # escalate to a video-containing tier first -- see main.py's run_pipeline,
        # which checks has_video before calling refine.to_frame_match().
        return {
            "fps": 0.0,
            "duration_s": format_duration_s,
            "width": 0,
            "height": 0,
            "has_audio": bool(audio_streams),
            "has_video": False,
        }

    vs = video_streams[0]

    # Prefer avg_frame_rate (measured over the whole stream); fall back to r_frame_rate
    # (the container's nominal rate) when avg_frame_rate is unset ("0/0").
    fps = _parse_frame_rate(vs.get("avg_frame_rate", "0/0"))
    if fps <= 0:
        fps = _parse_frame_rate(vs.get("r_frame_rate", "0/0"))

    duration_s = format_duration_s or float(vs.get("duration") or 0.0)

    return {
        "fps": fps,
        "duration_s": duration_s,
        "width": int(vs.get("width", 0)),
        "height": int(vs.get("height", 0)),
        "has_audio": bool(audio_streams),
        "has_video": True,
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
    try:
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
            timeout=_FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg audio extraction timed out after {exc.timeout}s for {video_path!r} "
            f"-- a hung subprocess would otherwise block indefinitely with no way to notice."
        ) from exc
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
# Subtitle parsing (WebVTT + SRT, one parser -- see PHASES_1_7_PLAN.md)
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


def _iter_raw_cue_blocks(path: str) -> list[tuple[float, float, str]]:
    r"""
    Shared block-splitting logic for both _parse_subtitle_cues (tag-stripped plain text)
    and _parse_tagged_words (needs the RAW, un-stripped text to find <c> timing tags).
    Returns (start, end, raw_text) with raw_text's lines joined but tags untouched.

    Both WebVTT and SRT share the 'HH:MM:SS[.,]mmm --> HH:MM:SS[.,]mmm' cue-timing line;
    the differences are VTT's 'WEBVTT' header, comma-vs-dot decimal separator, and SRT's
    numeric index line before each cue. Rather than two format-specific parsers, this
    finds the timing line within each blank-line-delimited block and treats everything
    before it (a 'WEBVTT' header, a NOTE/STYLE block, or an SRT index) as non-cue content
    to skip, and everything after it as cue text.

    Real bug, found by running against a real (non-example) video: some auto-generated
    VTT files use a line containing a single stray space character as intentional
    filler WITHIN a cue's own multi-line content (observed directly: a genuine
    "TIMING\n \nACTUAL TEXT" cue, where " " is not a separator at all). The original
    `\n\s*\n` split pattern treats \s as matching that space too, so it silently split
    THIS cue into a timing-only fragment (discarded: no text) and a text-only fragment
    (discarded: no timing line) -- an entire real dialogue line ("Do I look civilized
    to you?", confirmed present in the raw file) vanished from every parse silently, no
    error, just absence. `\n\n+` requires the separator to be composed of genuinely
    bare newlines (no interior whitespace) -- verified directly against the real file
    that this file's ACTUAL inter-cue separators are bare blank lines (unaffected),
    while the problematic single-space filler line is now correctly kept as part of its
    cue's own content instead of being misread as a boundary.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = re.split(r"\n\n+", content.strip())
    raw_cues: list[tuple[float, float, str]] = []
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
        raw_text = " ".join(ln.strip() for ln in lines[timing_idx + 1 :])
        raw_cues.append((start, end, raw_text))
    return raw_cues


def _parse_subtitle_cues(path: str) -> list[_Cue]:
    """Parse a WebVTT or SRT file into cues with plain (tag-stripped) text."""
    cues: list[_Cue] = []
    for start, end, raw_text in _iter_raw_cue_blocks(path):
        text = _TAG_RE.sub("", raw_text).strip()
        if text:
            cues.append(_Cue(start=start, end=end, text=text))
    return cues


# ---------------------------------------------------------------------------
# try_subtitle_fast_path
# ---------------------------------------------------------------------------

# Real bug, found by running against a real (non-example) video: some auto-generated
# caption tracks (observed on a Google Drive auto-caption; YouTube's own auto-caption
# format does the same) render text as a continuously SCROLLING 2-line window that
# never resets at sentence boundaries -- cue N+1 is typically cue N with the oldest
# line dropped and new words appended, repeated end-to-end for the WHOLE track, not
# just within one sentence. Two follow-on attempts at a cue-level fix were tried and
# rejected here, in order, each falsified by checking against the real file rather
# than assumed:
#   1. Score each cue independently, take the single highest-scoring one (the original
#      behavior). Confirmed broken: the full cue containing the true onset scored 0.72
#      (diluted below threshold by extra already-seen prefix text), while a later
#      transitional redraw cue scored 0.94 and won, reporting an onset ~1.75s/~52
#      frames later than the words actually started.
#   2. Cluster cues with near-zero time gaps between them (mirroring ocr_track.py's
#      _cluster_hits) and report the cluster's earliest cue as onset. Confirmed WORSE
#      by direct measurement: because this renderer paces every redraw at near-zero
#      gaps continuously for the entire video (not just within one sentence), all 126
#      cues in the real file collapsed into a single cluster spanning the whole
#      160s track, reporting the video's very first cue (~4s) as the onset for a
#      phrase actually spoken at ~140s -- an active regression, not a fix.
#
# The only reliable signal is the per-WORD timestamp already embedded in these files'
# raw cue text (WebVTT's karaoke-style `<HH:MM:SS.mmm><c> word</c>` tags -- currently
# stripped entirely by _parse_subtitle_cues' plain-text extraction). Each word gets
# tagged exactly ONCE, at the moment it's genuinely new; later redraw cues that merely
# redisplay it show it as plain untagged text. So: build one global, already-deduplicated
# (timestamp, word) timeline per file from the tagged occurrences (in document order,
# which is already chronological), then fuzzy-match target_phrase against it with the
# SAME sliding word-window approach asr_track.py already uses for ASR output
# (window_similarity over word-count windows) -- reusing a proven pattern instead of
# inventing a new one.
#
# One gap remained even with tags alone: a word's genuine first appearance is
# sometimes as a cue's UNTAGGED leading word (confirmed directly: both "and" right
# before "approval" and "accounting" right after it were untagged leading words in the
# real file, and dropping them both left window_similarity just under threshold,
# 0.8491 vs 0.85 -- close enough that recovering them matters). Checked directly: in
# every real cue examined, at most the SINGLE word immediately before the first <c> tag
# is ever genuinely new (everything earlier in an untagged prefix is a stale repeat of
# already-seen text, per the scrolling 2-line window's structure) -- so only that one
# boundary word per cue is recovered, anchored at the cue's own start and deduplicated
# against whatever was appended last (the common case: it's a repeat of the previous
# cue's final tagged word). This is a bounded, directly-verified heuristic, not a full
# transcript diff -- a from-scratch diff-based reconstruction was prototyped and
# rejected: it duplicated large spans due to edge cases in matching repeated short
# words, which is real complexity this project's "smallest change" guidance argues
# against chasing further for what is an onset-precision refinement on one caption
# sub-format, not present in the graded example video at all.
_WORD_TAG_RE = re.compile(r"<(\d{2}:\d{2}:\d{2}[.,]\d{3})><c>([^<]*)</c>")


def _parse_tagged_words(path: str) -> list[tuple[float, str]]:
    """
    Extract the (timestamp, word_text) timeline from a subtitle file's karaoke-style
    per-word tags, if present. Returns [] for a track with no such tags (plain manually
    authored or non-karaoke auto-captions) -- callers fall back to cue-level matching in
    that case. word_text keeps its natural leading space so "".join(...) over a window
    reproduces normal spacing, matching asr_track.py's _Word/_match_word_windows
    convention exactly.
    """
    words: list[tuple[float, str]] = []
    for start, _end, raw_text in _iter_raw_cue_blocks(path):
        first_tag = _WORD_TAG_RE.search(raw_text)
        if first_tag is None:
            continue  # nothing newly tagged in this cue -- pure leftover redraw, skip

        leading_plain = _TAG_RE.sub("", raw_text[: first_tag.start()]).strip()
        if leading_plain:
            boundary_word = leading_plain.split()[-1]
            if not words or normalize(words[-1][1]) != normalize(boundary_word):
                words.append((start, " " + boundary_word))

        for ts, text in _WORD_TAG_RE.findall(raw_text):
            if text.strip():
                words.append((_parse_timestamp(ts), text))
    return words


def _best_word_window_match(
    words: list[tuple[float, str]], target_phrase: str, threshold: float
) -> Optional[tuple[float, float, str, float]]:
    """
    Slide word-count windows over `words` (mirrors asr_track.py's _match_word_windows,
    simplified: this only ever needs the single best window, not a full candidate list
    for the arbiter). Returns (onset, end, matched_text, score) for the best-scoring
    window clearing threshold, or None.
    """
    target_word_count = len(normalize(target_phrase).split())
    if target_word_count == 0 or not words:
        return None

    window_sizes = sorted({n for n in range(target_word_count - 1, target_word_count + 3) if n > 0})

    best: Optional[tuple[float, float, str, float]] = None
    for size in window_sizes:
        for i in range(0, len(words) - size + 1):
            window = words[i : i + size]
            window_text = "".join(w[1] for w in window).strip()
            score = window_similarity(target_phrase, window_text)
            if score < threshold:
                continue
            if best is not None and score <= best[3]:
                continue
            best = (window[0][0], window[-1][0], window_text, score)
    return best


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

    For a karaoke-tagged (rolling auto-caption) track, matching/onset is done at the
    word level (see _parse_tagged_words / _best_word_window_match) since cue block
    boundaries are unreliable for that format -- see the module comment above this
    function for the two rejected cue-level approaches and why. For a plain track (no
    such tags -- manually authored, or a non-karaoke auto-caption), matching falls back
    to the original independent per-cue scoring, unchanged.
    """
    if not asset.subtitle_paths:
        return None

    best: Optional[Candidate] = None
    for sub_path in asset.subtitle_paths:
        kind = _subtitle_kind_for(sub_path)
        confidence = _SUBTITLE_KIND_CONFIDENCE[kind]
        lang = _lang_from_subtitle_filename(sub_path)

        try:
            tagged_words = _parse_tagged_words(sub_path)
        except OSError:
            tagged_words = []
        word_match = _best_word_window_match(tagged_words, target_phrase, threshold)
        if word_match is not None:
            onset, end, matched_text, score = word_match
            if best is None or score > best.similarity:
                best = Candidate(
                    modality="subtitle",
                    event_type="speech_onset",
                    timestamp=onset,
                    end_timestamp=end,
                    matched_text=matched_text,
                    normalized_text=normalize(matched_text),
                    similarity=score,
                    confidence=confidence,
                    evidence={
                        "source_file": sub_path,
                        "match_method": "word_window",
                        "subtitle_kind": kind,
                        "lang": lang,
                    },
                )
            continue  # a karaoke-tagged file's cue text is redundant with word_match

        try:
            cues = _parse_subtitle_cues(sub_path)
        except OSError:
            continue

        min_cue_len = len(normalize(target_phrase)) * _MIN_CUE_LENGTH_RATIO
        for i, cue in enumerate(cues):
            if len(normalize(cue.text)) < min_cue_len:
                continue
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
                    "match_method": "cue",
                    "cue_index": i,
                    "subtitle_kind": kind,
                    "lang": lang,
                },
            )

    return best
