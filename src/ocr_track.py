"""
Phase 3 - OCR track.

Consumes: VideoAsset.video_path + metadata (Phase 1), target phrase, Candidate schema (Phase 1)
Produces: list[Candidate], modality="ocr", event_type="visual_text_onset"

Pipeline: frame sampler (PyAV seek+decode, per CLAUDE.md -- never ffmpeg -ss inside the
pipeline) -> PaddleOCR full-frame -> text matching -> clustering -> optional
backward-walk onset refinement.

Candidate-region detector: deliberately NONE. CLAUDE.md's own menu of options for this
stage includes "PaddleOCR's own detection-only mode at low sampling frequency" -- that
IS this implementation: PaddleOCR already runs text detection internally before
recognition, so a separate contour/connected-component/frame-diff pre-filter would only
add a second detector on top of the one already running, for no measured benefit at the
sample rate used here (default 1 sample/sec). See APPROACH.md for the actual
measurement that would justify reaching for something heavier.

Verification (see PHASES.md): pytest tests/test_ocr_track.py::test_synthetic_clip
"""

from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
from paddleocr import PaddleOCR

from src.text_match import DEFAULT_MATCH_THRESHOLD, normalize, similarity
from src.types import Candidate, VideoAsset

_ocr_instance: Optional[PaddleOCR] = None

# Hits are grouped into one cluster (one physical on-screen event) if consecutive
# sample times are within this many sampling intervals of each other.
_CLUSTER_GAP_FACTOR = 1.5


@dataclass
class _Sample:
    time_s: float
    lines: list  # list[tuple[text: str, confidence: float]]


def _get_ocr() -> PaddleOCR:
    """
    Lazily construct a process-wide PaddleOCR instance. Model load (downloading +
    initializing the detection/recognition/classification models) is expensive and
    happens once regardless of how many frames find_candidates() samples.
    """
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _ocr_instance


def find_candidates(
    asset: VideoAsset,
    target_phrase: str,
    *,
    sample_interval_s: float = 1.0,
    refine: bool = True,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> list[Candidate]:
    """
    Sample frames at sample_interval_s, OCR each with PaddleOCR full-frame, fuzzy-match
    target_phrase against the OCR'd lines, cluster consecutive hits into one candidate
    per on-screen event, and (if refine=True) backward-walk each cluster's onset from
    the coarse sample grid to the exact frame where the text first appears.

    refine is a flag (not always-on) because it's CLAUDE.md's cutting-list item #1 --
    if OCR proves too slow in practice, this is the first thing to disable.
    """
    with av.open(asset.video_path) as container:
        stream = container.streams.video[0]

        hits: list[tuple[_Sample, float, str]] = []
        for sample in _sample_frames(container, stream, sample_interval_s):
            line_texts = [text for text, _ in sample.lines]
            score, matched_text = _score_lines(target_phrase, line_texts)
            if score >= threshold:
                hits.append((sample, score, matched_text))

        candidates = []
        for group in _cluster_hits(hits, sample_interval_s):
            best_sample, best_score, best_text = max(group, key=lambda h: h[1])
            onset_time = group[0][0].time_s
            end_time = group[-1][0].time_s

            if refine:
                onset_time = _backward_walk_refine(
                    container, stream, asset.metadata.fps, onset_time, target_phrase, threshold
                )

            confidence = _confidence_for_text(best_sample.lines, best_text)
            candidates.append(
                Candidate(
                    modality="ocr",
                    event_type="visual_text_onset",
                    timestamp=onset_time,
                    end_timestamp=end_time,
                    matched_text=best_text,
                    normalized_text=normalize(best_text),
                    similarity=best_score,
                    confidence=confidence,
                    evidence={
                        "sample_interval_s": sample_interval_s,
                        "frame_time": best_sample.time_s,
                        "ocr_lines": [t for t, _ in best_sample.lines],
                        "refined": refine,
                    },
                )
            )

    return candidates


def _sample_frames(container: av.container.InputContainer, stream, interval_s: float):
    """Yield one OCR'd _Sample per interval_s across the whole video."""
    duration_s = _stream_duration_s(container, stream)
    t = 0.0
    while t < duration_s:
        frame = _decode_frame_at(container, stream, t)
        if frame is not None:
            bgr = frame.to_ndarray(format="bgr24")
            lines = _ocr_frame(bgr)
            yield _Sample(time_s=t, lines=lines)
        t += interval_s


def _stream_duration_s(container: av.container.InputContainer, stream) -> float:
    if stream.duration is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0


def _decode_frame_at(container: av.container.InputContainer, stream, time_s: float):
    """Seek near time_s and decode forward to the first frame at/after it (PyAV only,
    per CLAUDE.md -- never mixed with ffmpeg -ss inside the pipeline)."""
    target_pts = int(time_s / stream.time_base)
    container.seek(target_pts, stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * stream.time_base)
        if frame_time >= time_s:
            return frame
    return None


def _ocr_frame(frame_bgr: np.ndarray) -> list:
    """Run PaddleOCR full-frame; return [(text, confidence), ...] for each detected line."""
    result = _get_ocr().ocr(frame_bgr, cls=False)
    if not result or result[0] is None:
        return []
    return [(text, float(conf)) for _, (text, conf) in result[0]]


def _score_lines(target_phrase: str, line_texts: list[str]) -> tuple[float, str]:
    """
    Score target against each OCR'd line individually AND against all lines joined
    (captions commonly wrap across two lines) -- keep whichever scores higher. Uses
    similarity (partial_ratio), not window_similarity: a line legitimately containing
    the target as a substring (surrounded by other on-screen text) is a real match,
    unlike the ASR word-window case that window_similarity exists to guard against.
    """
    if not line_texts:
        return 0.0, ""
    candidate_texts = list(line_texts) + [" ".join(line_texts)]
    scored = [(similarity(target_phrase, t), t) for t in candidate_texts]
    return max(scored, key=lambda st: st[0])


def _confidence_for_text(lines: list, matched_text: str) -> float:
    """
    Mean PaddleOCR recognition confidence of the line(s) making up matched_text.
    matched_text is either one line's own text (exact match in `lines`) or the
    all-lines-joined text, in which case confidence is averaged across every line.
    """
    if not lines:
        return 0.0
    for text, conf in lines:
        if text == matched_text:
            return conf
    confs = [c for _, c in lines]
    return sum(confs) / len(confs)


def _cluster_hits(
    hits: list[tuple[_Sample, float, str]], sample_interval_s: float
) -> list[list[tuple[_Sample, float, str]]]:
    """
    Group hits whose sample times are close together into one cluster -- a caption
    visible for several seconds at a 1s sample interval otherwise yields several
    duplicate candidates for the same on-screen event.
    """
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h[0].time_s)
    gap_tolerance = sample_interval_s * _CLUSTER_GAP_FACTOR
    groups = [[ordered[0]]]
    for h in ordered[1:]:
        if h[0].time_s - groups[-1][-1][0].time_s <= gap_tolerance:
            groups[-1].append(h)
        else:
            groups.append([h])
    return groups


def _backward_walk_refine(
    container: av.container.InputContainer,
    stream,
    fps: float,
    onset_time: float,
    target_phrase: str,
    threshold: float,
) -> float:
    """
    Step backward frame-by-frame from onset_time while the target phrase still
    OCR-matches, to find the true onset frame rather than reporting the coarse
    sample_interval_s grid point that happened to hit it. CLAUDE.md cutting-list item
    #1 -- this is exactly what the `refine` flag disables if OCR proves too slow.
    """
    if fps <= 0:
        return onset_time
    frame_period = 1.0 / fps
    t = onset_time
    while True:
        prev_t = t - frame_period
        if prev_t < 0:
            break
        frame = _decode_frame_at(container, stream, prev_t)
        if frame is None:
            break
        bgr = frame.to_ndarray(format="bgr24")
        lines = _ocr_frame(bgr)
        score, _ = _score_lines(target_phrase, [text for text, _ in lines])
        if score < threshold:
            break
        t = prev_t
    return t
