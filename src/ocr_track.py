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

Frame sampling (hardening added post-Phase-6, runtime audit): a real measurement against
the ~54min example video found per-timestamp seeking (`_decode_frame_at` called 3,261
times) cost MORE than the OCR inference itself -- 0.484s/sample seek+decode vs.
0.334s/sample OCR (26.3min vs. 18.2min total). A single sequential forward decode pass
(`stream.thread_type = "AUTO"`, PyAV's own frame/slice threading) measured 686fps --
the whole video decoded in 1.9min. `_sample_frames` below uses that sequential pass
instead of seeking, selecting the identical frames (same fixed time grid) at a fraction
of the cost -- a pure throughput fix, zero accuracy impact.

Verification (see PHASES.md): pytest tests/test_ocr_track.py::test_synthetic_clip
"""

import contextlib
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np

# paddle's cpp_extension helper warns "No ccache found..." on import, unconditionally,
# regardless of whether anything actually needs to JIT-compile a C++ extension -- we
# never do (PaddleOCR is used purely through its inference API). Scoped to this exact
# message (not a blanket warnings.filterwarnings("ignore")) so it doesn't hide an
# unrelated, genuinely useful warning from the same or another library.
warnings.filterwarnings("ignore", message="No ccache found.*")


@contextlib.contextmanager
def _suppress_native_stderr():
    """
    Temporarily redirects the OS-level stderr file descriptor to devnull.

    That ccache probe above shells out to `where ccache` (and paddle's cuda probe does
    the same for `nvcc`) at import time. When nothing is found, Windows' `where.exe`
    itself writes "INFO: Could not find files for the given pattern(s)." directly to
    the OS stderr file descriptor -- confirmed directly (`where nonexistent 1>/dev/null`
    still shows it; `2>/dev/null` hides it) that this bypasses Python's warnings/logging
    entirely, since it's a separate child process's own stderr, not anything raised in
    this interpreter. Redirecting sys.stderr alone would not reach it -- only the
    underlying OS file descriptor does, which is what a child process actually inherits
    and writes to.
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


with _suppress_native_stderr():
    from paddleocr import PaddleOCR  # noqa: E402 -- must follow the filter/suppress above

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
    max_backward_s: float = 30.0,
) -> list[Candidate]:
    """
    Sample frames at sample_interval_s, OCR each with PaddleOCR full-frame, fuzzy-match
    target_phrase against the OCR'd lines, cluster consecutive hits into one candidate
    per on-screen event, and (if refine=True) backward-walk each cluster's onset from
    the coarse sample grid to the exact frame where the text first appears.

    refine is a flag (not always-on) because it's CLAUDE.md's cutting-list item #1 --
    if OCR proves too slow in practice, this is the first thing to disable. max_backward_s
    bounds that walk (see _backward_walk_refine's docstring) -- it was previously
    unbounded.
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
                    container,
                    stream,
                    asset.metadata.fps,
                    onset_time,
                    target_phrase,
                    threshold,
                    max_backward_s,
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


_PROGRESS_INTERVAL_S = 30.0  # print an OCR progress line at most this often (video time)


def _sample_frames(container: av.container.InputContainer, stream, interval_s: float):
    """
    Yield one OCR'd _Sample per interval_s across the whole video, via a single
    sequential forward decode pass -- NOT a seek-per-timestamp loop (see module
    docstring for the measurement that motivated this).

    thread_type = "AUTO" enables PyAV's own frame/slice-level decode threading, which is
    what produced the measured 686fps sequential-decode throughput.

    Fixed-grid semantics preserved exactly from the original seek-based version: target
    query points are 0, interval_s, 2*interval_s, ... regardless of which decode
    strategy reaches them, so the set of frames selected is unchanged -- this is a
    throughput fix, not a behavior change.
    """
    duration_s = _stream_duration_s(container, stream)
    stream.thread_type = "AUTO"

    next_target = 0.0
    last_progress_at = 0.0
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * stream.time_base)
        if frame_time < next_target:
            continue

        bgr = frame.to_ndarray(format="bgr24")
        lines = _ocr_frame(bgr)
        yield _Sample(time_s=frame_time, lines=lines)

        if frame_time - last_progress_at >= _PROGRESS_INTERVAL_S:
            if duration_s > 0:
                pct = 100 * frame_time / duration_s
                print(f"    OCR: {frame_time:.0f}s / {duration_s:.0f}s ({pct:.0f}%)", flush=True)
            else:
                print(f"    OCR: {frame_time:.0f}s processed", flush=True)
            last_progress_at = frame_time

        # Next point on the fixed grid strictly after this frame.
        next_target = (int(frame_time / interval_s) + 1) * interval_s


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


# Real bug, found by running against a real (non-example) video: rapidfuzz's
# partial_ratio finds the best alignment of the SHORTER of its two strings within the
# longer one -- a 1-character OCR misread (e.g. a stray logo/watermark artifact)
# scores a PERFECT 1.0 against almost any target phrase, since that single character
# is almost always present somewhere in it. Verified directly:
# similarity("I am the one who shall decide", "O") == 1.0. Guard against this by
# requiring an OCR candidate text to be at least this fraction of the target phrase's
# own (normalized) length before it's even scored -- a genuine on-screen match for the
# full target phrase is never drastically shorter than the phrase itself.
_MIN_CANDIDATE_LENGTH_RATIO = 0.5


def _score_lines(target_phrase: str, line_texts: list[str]) -> tuple[float, str]:
    """
    Score target against each OCR'd line individually AND against all lines joined
    (captions commonly wrap across two lines) -- keep whichever scores higher. Uses
    similarity (partial_ratio), not window_similarity: a line legitimately containing
    the target as a substring (surrounded by other on-screen text) is a real match,
    unlike the ASR word-window case that window_similarity exists to guard against.

    Candidates shorter than _MIN_CANDIDATE_LENGTH_RATIO of the target's length are
    excluded before scoring -- see that constant's comment for why.
    """
    if not line_texts:
        return 0.0, ""
    target_len = len(normalize(target_phrase))
    min_len = target_len * _MIN_CANDIDATE_LENGTH_RATIO
    candidate_texts = list(line_texts) + [" ".join(line_texts)]
    scored = [
        (similarity(target_phrase, t), t)
        for t in candidate_texts
        if len(normalize(t)) >= min_len
    ]
    if not scored:
        return 0.0, ""
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
    max_backward_s: float = 30.0,
) -> float:
    """
    Step backward frame-by-frame from onset_time while the target phrase still
    OCR-matches, to find the true onset frame rather than reporting the coarse
    sample_interval_s grid point that happened to hit it. CLAUDE.md cutting-list item
    #1 -- this is exactly what the `refine` flag disables if OCR proves too slow.

    Bounded to max_backward_s (default 30s) of walk-back -- runtime-audit finding: this
    loop's only exits were previously a below-threshold score, t=0, or a decode failure,
    with no cap on distance/steps. A slowly-fading caption or a permissive threshold
    could otherwise walk arbitrarily far back with no bound. 30s is generous relative to
    any real on-screen caption's duration while guaranteeing termination.
    """
    if fps <= 0:
        return onset_time
    frame_period = 1.0 / fps
    earliest_allowed = max(0.0, onset_time - max_backward_s)
    t = onset_time
    while True:
        prev_t = t - frame_period
        if prev_t < earliest_allowed:
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
