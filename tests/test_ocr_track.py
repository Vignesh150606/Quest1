"""
Phase 3 - OCR track tests.

Offline (default; see pytest.ini): a short synthetic captioned clip generated on the
fly via ffmpeg drawtext -- no network required, no real video download. This is
verify.py 3's target.

Verification target: pytest tests/test_ocr_track.py
"""

import av

from src.ingest import prepare_asset
from src.ocr_track import _backward_walk_refine, _sample_frames, _score_lines, find_candidates
from tests.fixtures.make_synthetic_clip import make_synthetic_clip

SIMILARITY_THRESHOLD = 0.85
SAMPLE_INTERVAL_TOLERANCE_S = 1.0  # should match the OCR track's frame-sampling interval


def test_synthetic_clip(tmp_path):
    clip_path = str(tmp_path / "synthetic.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, clip["text"])
    assert candidates, "no OCR candidates found"
    best = max(candidates, key=lambda c: c.similarity)
    assert best.similarity >= SIMILARITY_THRESHOLD
    assert abs(best.timestamp - clip["onset_s"]) <= SAMPLE_INTERVAL_TOLERANCE_S
    assert best.modality == "ocr"
    assert best.event_type == "visual_text_onset"


def test_synthetic_clip_dedup_single_candidate(tmp_path):
    # Caption visible for 5s at the default 1s sample interval would naively produce
    # ~5 duplicate hits without clustering -- assert exactly one candidate survives.
    clip_path = str(tmp_path / "synthetic_dedup.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=1.0, duration_s=5.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, clip["text"])
    assert len(candidates) == 1


def test_synthetic_clip_negative_control(tmp_path):
    clip_path = str(tmp_path / "synthetic_neg.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0)

    asset = prepare_asset(clip["path"])
    candidates = find_candidates(asset, "this phrase never appears on screen anywhere")
    assert candidates == []


def test_score_lines_rejects_short_noise_false_positive():
    # Real bug, found by running against a real (non-example) video: rapidfuzz's
    # partial_ratio scored a single stray OCR'd character ("O") as a PERFECT 1.0
    # match against an unrelated 30-character target phrase, purely because that one
    # character happens to appear somewhere in the target. Confirmed the raw function
    # would score this 1.0 before the length gate was added; this asserts the gate
    # actually rejects it now.
    target = "I am the one who shall decide"
    score, matched = _score_lines(target, ["O"])
    assert score == 0.0
    assert matched == ""


def test_score_lines_still_matches_genuine_short_target():
    # The length gate is relative to the target's own length, so a short target
    # phrase matched against a comparably-short genuine caption must still work.
    score, matched = _score_lines("Stop", ["Stop!"])
    assert score >= 0.85
    assert matched == "Stop!"


def test_score_lines_still_matches_full_length_ocr_line():
    # Regression guard: the length gate must not break the normal, intended case --
    # a genuine full-length OCR'd line matching the target.
    target = "Test caption text"
    score, matched = _score_lines(target, [target])
    assert score == 1.0
    assert matched == target


def test_sample_frames_fixed_grid(tmp_path):
    # Runtime-hardening regression guard: _sample_frames was rewritten from a
    # seek-per-timestamp loop to a single sequential decode pass (measured: seeking
    # cost MORE than the OCR inference itself on the real example video -- see
    # ocr_track.py's module docstring). This locks in that the fixed query grid
    # (0, interval_s, 2*interval_s, ...) is preserved exactly by the new decode
    # strategy, not just "close enough".
    clip_path = str(tmp_path / "grid_test.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=2.0, duration_s=5.0)

    with av.open(clip["path"]) as container:
        stream = container.streams.video[0]
        samples = list(_sample_frames(container, stream, interval_s=1.0))

    # 7.0s clip @ 25fps, 1.0s grid -> samples at t=0,1,2,3,4,5,6 (last frame is 6.96s,
    # short of the t=7.0 grid point).
    times = [round(s.time_s) for s in samples]
    assert times == [0, 1, 2, 3, 4, 5, 6]


def test_backward_walk_refine_is_bounded(tmp_path):
    # Runtime-audit finding: _backward_walk_refine's only exits were previously a
    # below-threshold score, t=0, or a decode failure -- no distance/step cap. Caption
    # visible from t=0.0 for the whole clip means an UNBOUNDED walk from onset_time=5.0
    # would reach ~0.0; this asserts it instead stops within max_backward_s of onset_time.
    clip_path = str(tmp_path / "long_caption.mp4")
    clip = make_synthetic_clip(clip_path, text="Test caption text", onset_s=0.0, duration_s=15.0)

    with av.open(clip["path"]) as container:
        stream = container.streams.video[0]
        onset_time = 5.0
        max_backward_s = 1.0
        frame_period = 1.0 / clip["fps"]
        result = _backward_walk_refine(
            container,
            stream,
            clip["fps"],
            onset_time,
            clip["text"],
            threshold=0.85,
            max_backward_s=max_backward_s,
        )

    # Stopped at (or within one frame of) the bound, not walked all the way to ~0.0.
    assert result >= onset_time - max_backward_s - frame_period
    # Sanity: it did walk back some distance, not a no-op.
    assert result < onset_time - 0.5
