"""
Phase 1 - Ingest + subtitle fast-path.

Consumes: video URL (str), target phrase (str, CLI arg)
Produces: VideoAsset; optional Candidate(modality="subtitle") on fast-path hit

Verification (see PHASES.md): pytest tests/test_ingest.py::test_prepare_asset
"""

from typing import Optional

from src.types import VideoAsset, Candidate


def prepare_asset(video_url: str) -> VideoAsset:
    """Resolve a video URL to a local VideoAsset: download, probe metadata, extract audio."""
    raise NotImplementedError


def try_subtitle_fast_path(asset: VideoAsset, target_phrase: str) -> Optional[Candidate]:
    """
    Check for an existing subtitle/CC track (yt-dlp --write-subs --write-auto-subs) and
    fuzzy-match target_phrase against it. Returns None if no track exists or no match
    clears threshold -- callers should fall through to the ASR/OCR tracks in that case.
    """
    raise NotImplementedError
