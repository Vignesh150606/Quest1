"""
Phase 3 - OCR track.

Consumes: VideoAsset.video_path + metadata (Phase 1), target phrase, Candidate schema (Phase 1)
Produces: list[Candidate], modality="ocr", event_type="visual_text_onset"

Pipeline interface (see CLAUDE.md, not a prescribed implementation):
  frame sampler -> candidate-region detector -> candidate windows -> OCR -> text matching

Choose the simplest reliable candidate-region detector (contour/edge heuristics,
connected components, frame differencing, or PaddleOCR's own detection-only mode at low
sampling frequency) and document the choice in APPROACH.md. Don't reach for a heavier
detector unless testing shows the simple version is insufficient.

Verification (see PHASES.md): pytest tests/test_ocr_track.py::test_synthetic_clip
"""

from src.types import VideoAsset, Candidate


def find_candidates(asset: VideoAsset, target_phrase: str) -> list[Candidate]:
    """Sample frames, detect candidate text regions, OCR them, fuzzy-match target_phrase."""
    raise NotImplementedError
