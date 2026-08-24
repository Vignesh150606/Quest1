"""
Phase 5 - Refine.

Consumes: winning Candidate(s) (Phase 4), VideoAsset (Phase 1)
Produces: FrameMatch(es)

Use PyAV exclusively for all frame-accurate extraction here -- never mix with raw
ffmpeg -ss seeking inside pipeline code (keyframe-snapped ffmpeg seeks and
decode-forward PyAV seeks can return different frames for the identical timestamp).

ASR timestamps are temporal anchors, not inherently exact frame boundaries -- this
stage must decode the frames surrounding the anchor and map it to a specific frame
number per an explicit, documented policy. For OCR candidates the mapping is more
direct (the event is itself a visual frame property).

Verification (see PHASES.md): pytest tests/test_refine.py::test_frame_accuracy
"""

from src.types import Candidate, VideoAsset, FrameMatch


def to_frame_match(candidate: Candidate, asset: VideoAsset, output_dir: str) -> FrameMatch:
    """Map a Candidate's timestamp to an exact frame via PyAV and save it as a PNG."""
    raise NotImplementedError
