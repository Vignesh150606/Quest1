"""
Phase 5 - Refine.

Consumes: winning Candidate(s) (Phase 4), VideoAsset (Phase 1)
Produces: FrameMatch(es)

Use PyAV exclusively for all frame-accurate extraction here -- never mix with raw
ffmpeg -ss seeking inside pipeline code (keyframe-snapped ffmpeg seeks and
decode-forward PyAV seeks can return different frames for the identical timestamp).

Verification: pytest tests/test_refine.py::test_frame_accuracy
"""

import os

import av

from src.types import Candidate, FrameMatch, VideoAsset


def to_frame_match(candidate: Candidate, asset: VideoAsset, output_dir: str) -> FrameMatch:
    """
    Map a Candidate's timestamp to an exact frame via PyAV and save it as a PNG.

    Both event types map to the same rule: the frame being displayed at the anchor
    timestamp -- the last frame with pts_time <= candidate.timestamp. What differs
    between them is the epistemic status of the anchor, not the arithmetic:
      - visual_text_onset: the anchor IS a decoded frame's own presentation time (OCR
        found text starting at that sampled frame), so this mapping is exact and
        self-consistent by construction.
      - speech_onset: the anchor is an audio-domain estimate (faster-whisper's word
        start time, or a subtitle cue's start time). The frame returned is whichever
        one is on screen when the audio says the line starts; it inherits whatever
        timing error the ASR/subtitle estimate carries -- the frame mapping itself
        cannot remove that error, it just applies the same rule consistently to both
        cases rather than needing two different arithmetic paths.

    Seeking: PyAV's container.seek(pts, backward=True) always lands at the keyframe at
    or before the target -- verified directly (not assumed) against a real GOP=50 clip:
    a single seek to a given timestamp decodes forward only as far as the next frame
    past the target before stopping, and returns the same frame on repeated calls. No
    extra seek margin is needed or added.

    frame_idx is computed as round(pts_time * fps) using the ffprobe-measured fps from
    Phase 1's VideoAsset.metadata -- exact for constant-frame-rate video. Decoding from
    frame 0 to count frames exactly would be more robust to VFR sources but costs a
    full linear scan of a potentially 54-minute video for one frame; this is a
    documented assumption, not a silent one -- if a real VFR source is encountered,
    that's a stop-and-flag case per CLAUDE.md, not something to silently keep counting
    on top of a wrong fps.

    FrameMatch.timestamp_s is the returned frame's own pts_time, not
    candidate.timestamp -- these can differ slightly (the frame is at-or-before the
    anchor, never after), and the frame's own time is the truthful answer to "when is
    this frame", which is what gets reported to the user.
    """
    os.makedirs(output_dir, exist_ok=True)

    with av.open(asset.video_path) as container:
        stream = container.streams.video[0]
        frame = _decode_frame_at_or_before(container, stream, candidate.timestamp)
        if frame is None:
            raise RuntimeError(
                f"No frame found at or before timestamp {candidate.timestamp}s in "
                f"{asset.video_path!r} -- the timestamp may be beyond the video's "
                f"duration or before its start."
            )
        pts_time = float(frame.pts * stream.time_base)
        image = frame.to_image()

    frame_idx = round(pts_time * asset.metadata.fps)
    image_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
    image.save(image_path)

    return FrameMatch(
        frame_idx=frame_idx,
        timestamp_s=pts_time,
        text=candidate.matched_text,
        image_path=image_path,
        modality=candidate.modality,
        match_score=candidate.similarity,
    )


def _decode_frame_at_or_before(container, stream, timestamp_s: float):
    """
    Seek to timestamp_s (PyAV snaps backward to the preceding keyframe) and decode
    forward, returning the last frame with pts_time <= timestamp_s -- i.e. the frame
    actually on screen at that moment. Stops as soon as it sees a frame past the
    target, so it never decodes further into the video than necessary.
    """
    target_pts = int(timestamp_s / stream.time_base)
    container.seek(target_pts, stream=stream, backward=True, any_frame=False)

    best = None
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * stream.time_base)
        if frame_time <= timestamp_s:
            best = frame
        else:
            break
    return best
