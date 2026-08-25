# Phases 2–7 — Implementation Plan

## Context

Phases 0 (environment) and 1 (ingest + subtitle fast-path) are done and pushed;
`verify.py 0` and `verify.py 1` both pass. Everything from `asr_track.py` onward is still
a stub raising `NotImplementedError`.

This plan covers the remaining six phases. Sonnet implements them **one at a time**,
commits and pushes each, then **stops and waits** for instructions before starting the
next.

### Environment facts that constrain the design

Measured on this machine, not assumed:

- **No CUDA GPU** — Intel Iris Xe integrated only. i5-1240P (12C/16T), 15.7 GB RAM,
  41.7 GB free on D:. faster-whisper and PaddleOCR both run **CPU-only**, which is the
  dominant runtime constraint for Phases 2, 3 and 6.
- Deadline is tight (26 Aug). CLAUDE.md's cutting list is live, not hypothetical — see
  *If time runs short* at the end.

### Four decisions already confirmed

| Decision | Choice |
|---|---|
| Score scale | Normalize `similarity` **and** `confidence` to 0–1 |
| ASR model | `small`, `int8`, `vad_filter=True`, exposed as `--model` |
| OCR cost | Implement unconditional, **measure**, then decide on short-circuiting |
| OCR library | **PaddleOCR directly**; drop the `videocr` wrapper |

### Two defects in the existing scaffold this plan fixes

1. **Vacuous ground-truth assertions.** `tests/test_asr_track.py` and
   `tests/test_ocr_track.py` both assert `similarity >= 0.85`, but Phase 1 stores
   similarity on rapidfuzz's **0–100** scale — so those assertions are *always true* and
   currently test nothing. Phase 1 is also internally inconsistent: `similarity` is 0–100
   while `_SUBTITLE_KIND_CONFIDENCE` is 0–1. Fixed at the top of Phase 2.
2. **`tests/test_refine.py`'s body is `raise NotImplementedError`** — it is a stub, not a
   test. Phase 5 must write it from scratch, not just remove a skip marker.

---

## Cross-cutting conventions (apply to every phase)

- **`verify.py` targets become whole files.** Phase 1 already changed its entry to
  `tests/test_ingest.py` so the `-m "not network"` default doesn't deselect everything to
  zero collected tests. Do the same for phases 2–6 as each is implemented.
- **Every phase gets offline tests that actually run under `verify.py N`.** Network- or
  model-dependent checks are additionally marked `@pytest.mark.network`.
- **Negative controls are mandatory.** Every matching test must include a deliberately
  wrong phrase asserted to score *below* threshold. This is what would have caught defect
  #1 above, and it is cheap.
- **Per phase: commit, push, stop.** One commit per phase on `main`, message naming the
  phase and any deviation from PHASES.md. Then wait.
- **Stop-and-flag** per CLAUDE.md rather than working around: a broken assumption, a
  needed architecture change, or an assertion that only passes when loosened.

---

## Phase 2 — ASR track

**Files:** `src/text_match.py`, `src/ingest.py`, `tests/test_ingest.py` (scale fix);
`src/asr_track.py`, `tests/test_asr_track.py`; `verify.py`

### 2a. Scale normalization (do this first, it touches completed Phase 1)

- `text_match.similarity()` returns `fuzz.partial_ratio(...) / 100.0`;
  `DEFAULT_MATCH_THRESHOLD = 0.85`.
- `tests/test_ingest.py`: assertions become `>= 0.85`; add one asserting
  `0.0 <= similarity <= 1.0` to lock the invariant so this cannot silently regress.
- `_SUBTITLE_KIND_CONFIDENCE` in `ingest.py` is already 0–1 — no change.

### 2b. Add `window_similarity` — and understand why

`similarity()` is `partial_ratio`, which scores the *best matching substring*. That is
correct for "does this long subtitle cue contain the target", but it is a **false-positive
generator** for ASR word windows: `partial_ratio("my mind rebels at stagnation", "my mind")`
scores **1.0**, so a 2-word window would look like a perfect match.

Add alongside it:

```python
def window_similarity(target: str, window_text: str) -> float:   # fuzz.ratio / 100
```

`ratio` penalizes both missing and extra content, so the window whose extent actually
matches the target wins. Document both functions' intended use in the module docstring —
the arbiter treats them as comparable 0–1 "match quality" scores.

### 2c. `src/asr_track.py`

```python
def find_candidates(asset, target_phrase, *, model_size="small",
                    device="cpu", compute_type="int8") -> list[Candidate]:
```

1. `WhisperModel(model_size, device=device, compute_type=compute_type)`.
2. `model.transcribe(asset.audio_path, word_timestamps=True, vad_filter=True)`.
   Leave language auto-detected — the eval video may not be English.
3. Flatten all segments into one list of `(text, start, end, probability)` words.
4. **Factor the matching into a pure function** — this is the part worth unit-testing;
   the model itself is a black box:
   ```python
   def _match_word_windows(words, target_phrase, threshold) -> list[Candidate]
   ```
   Slide windows of `N-1 … N+2` words (N = word count of the normalized target), score
   each with `window_similarity`, keep those ≥ threshold.
5. `timestamp` = **start of the window's first word** (onset — CLAUDE.md's
   "first appears"); `end_timestamp` = end of the last word.
6. **Deduplicate overlapping windows.** Adjacent windows around a true hit all score high;
   cluster overlapping hits and keep the best per cluster, or the pipeline returns dozens
   of near-identical candidates.
7. `confidence` = mean `word.probability` across the matched window, clamped to [0, 1].
8. `evidence`: `{model_size, window_word_count, word_probabilities, language,
   language_probability}`.

### Tests

**Offline** (these are `verify.py 2`'s real content):
`_match_word_windows` against hand-built word lists with exact expected onsets — exact
match, one-word-off match, split-word case, a below-threshold negative control, and
overlapping-window dedup asserting exactly one candidate is returned.

**Network** (`@pytest.mark.network`): the existing `test_example_video`, unskipped, with
`SIMILARITY_THRESHOLD` corrected to 0–1, asserting onset within ±2s of 05:27.

### Verify → commit → **stop**

```bash
python verify.py 2
pytest -m network tests/test_asr_track.py -v
```

Record the measured wall-clock of the full 54-min transcription — it feeds Phase 6 and
APPROACH.md.

---

## Phase 3 — OCR track

**Files:** `src/__init__.py`, `src/ingest.py`, `src/ocr_track.py`,
`tests/fixtures/make_synthetic_clip.py`, `tests/test_ocr_track.py`, `requirements.txt`,
`tests/test_environment.py`, `verify.py`

### 3a. Resolve the Phase 0 `libiomp5md.dll` flag — required before paddle enters

`paddle` and `ctranslate2` each ship their own Intel OpenMP runtime. Whichever loads first
wins; **paddle-first makes ctranslate2 fail with `OSError: [WinError 127]`** (verified in
Phase 0). Phase 6 runs both in one process, so this must be fixed now.

Put the guard in `src/__init__.py` so it runs for *any* `from src.X import Y`:

```python
import ctranslate2  # noqa: F401  -- MUST precede any paddle/paddleocr import
```

Add a test asserting the safe order holds: a subprocess doing
`import src; import paddleocr; import faster_whisper` exits 0.

### 3b. `prepare_asset` must accept local file paths

`tests/test_ocr_track.py` already calls `prepare_asset(clip["path"])` with a local `.mp4`.
Phase 1 only handles URLs, so this is a **pre-existing gap the scaffold depends on**. Add:
if the input is an existing local file, skip download and probe it directly. This also
unlocks a fully offline end-to-end test in Phase 6.

### 3c. Drop `videocr`

Remove from `requirements.txt` and from `tests/test_environment.py`'s import list. Note in
the commit message that this reverses part of Phase 0's pinned stack, per CLAUDE.md's
"prefer a library's direct API over a wrapper" — and that dropping it also removes the
constraint that forced `paddleocr==2.10.0`.

### 3d. `tests/fixtures/make_synthetic_clip.py`

ffmpeg `drawtext` burning `text` in from `onset_s`, on a fixed 25 fps clip so onset lands
on an exact frame boundary (2.0s → frame 50), returning
`{path, text, onset_s, fps}`.

> **Gotcha:** `drawtext` needs a font. Try `font=Arial` first (the Gyan build bundles
> fontconfig); if that fails fall back to an explicit
> `fontfile=C\\:/Windows/Fonts/arial.ttf` — Windows path escaping inside a filter string
> is fiddly. If neither works, flag rather than silently switching to a different fixture
> strategy.

### 3e. `src/ocr_track.py`

```python
def find_candidates(asset, target_phrase, *, sample_interval_s=1.0,
                    refine=True) -> list[Candidate]:
```

1. **Sampler** — PyAV seek + decode at `sample_interval_s` (PyAV, per CLAUDE.md, not
   ffmpeg CLI).
2. **Region detector** — deliberately **none**: run PaddleOCR full-frame
   (`use_angle_cls=False, lang="en"`) at the sample interval. This *is* CLAUDE.md's
   "PaddleOCR's own detection-only mode at low sampling frequency" option and its
   "simplest reliable" instruction. Record ROI-cropping and frame-differencing in
   APPROACH.md as measured-and-deferred, not as unexplored.
3. **Match** — score the target against each OCR line *and* against all lines joined
   (captions commonly wrap across two lines); take the max. Use `similarity`
   (partial_ratio) — correct here, since a line legitimately contains the target.
4. **Cluster** consecutive matching samples into one candidate at the earliest time. A
   caption on screen for 3s at 1s sampling otherwise yields 3 duplicate candidates.
5. **Backward-walk refinement** (`refine=True`): from the first matching sample, step back
   frame-by-frame until the text stops matching — that frame+1 is the true onset. This is
   **cutting-list item #1**; keep it behind the flag so it can be disabled in one place.
6. `confidence` = mean PaddleOCR recognition confidence of the matched line(s) (already
   0–1). `evidence`: `{sample_interval_s, frame_time, ocr_lines, box, refined}`.

### Tests

**Offline:** `test_synthetic_clip` unskipped (threshold corrected to 0–1) — the synthetic
clip is local, so this needs no network; a negative control asserting an absent phrase
returns no candidate; a dedup test asserting a multi-second caption yields exactly one
candidate.

### Measurement (the reason this phase is instrumented)

Time `find_candidates` over the **full 54-min example** and write the number into
APPROACH.md. That measurement is what CLAUDE.md requires before any short-circuit
decision in Phase 6.

### Verify → commit → **stop**

---

## Phase 4 — Arbiter

**Files:** `src/arbiter.py`, `tests/test_arbiter.py`, `verify.py`

```python
CONFIDENCE_THRESHOLDS = {"subtitle": 0.5, "asr": 0.4, "ocr": 0.5}
SIMILARITY_THRESHOLD  = 0.85
CLUSTER_TOLERANCE_S   = 2.0

def reconcile(candidates, *, tolerance_s=CLUSTER_TOLERANCE_S): ...
```

1. Drop candidates below their modality's confidence threshold or below the similarity
   threshold → none left returns `None`.
2. Sort by timestamp; cluster candidates within `tolerance_s`.
3. Within a cluster, rank by **`(confidence, similarity)` descending**.
4. More than one surviving cluster → `AmbiguousResult(candidates=[best per cluster],
   reason=...)`. Never a silent pick.

> **Read the scaffold test carefully.** Its `"higher_confidence"` case gives both
> candidates `confidence=0.9` and differs only in `similarity` (0.90 asr vs 0.95 ocr),
> expecting **ocr** to win. So the label is misleading but the assertion is right — the
> `(confidence, similarity)` ordering above satisfies it. Do not "fix" the test to match
> its name.

Two things to document rather than silently absorb:
- **Cross-`event_type` clusters.** ASR `speech_onset` and OCR `visual_text_onset` can
  cluster together; the winner's `event_type` drives Phase 5's frame policy.
- **Winner-by-score vs. earliest-in-cluster.** CLAUDE.md says prefer higher confidence,
  which can shift the reported onset by up to `tolerance_s` versus the earliest candidate.
  Follow the spec; note the trade-off in APPROACH.md.

**Tests:** scaffold cases unskipped, plus empty input → `None`, all-below-threshold →
`None`, three separated clusters → `AmbiguousResult` with 3 candidates, and an exact tie →
deterministic result across repeated calls.

### Verify → commit → **stop**

---

## Phase 5 — Refine

**Files:** `src/refine.py`, `tests/test_refine.py`, `verify.py`

### The frame policy — the crux of the whole task

**One mapping, two justifications.** Both event types map to *the frame being displayed at
the anchor timestamp* — the **last frame with `pts_time <= timestamp`**. What differs is
the epistemic status of the anchor, and that is what APPROACH.md must say:

- `visual_text_onset` — the anchor *is* a decoded frame's own presentation time, so the
  mapping is exact and self-consistent.
- `speech_onset` — the anchor is an audio-domain estimate from Whisper (or a subtitle cue
  timed to speech). The frame returned is the one on screen when the line starts; it
  carries the ASR's timing error, which the frame mapping cannot remove.

This "same mapping, different confidence in the input" framing is the honest answer and
the defensible one in interview.

### Implementation

```python
def to_frame_match(candidate, asset, output_dir) -> FrameMatch:
```

- PyAV only. Seek ~1s *before* the anchor (`backward=True`), decode forward, keep the last
  frame with `pts_time <= timestamp`, stop at the first one past it.
- `frame_idx = round(pts_time * fps)` using the **ffprobe-measured fps from Phase 1**.
  Exact for CFR; document the assumption rather than decoding from frame 0 (which would
  cost a full 54-min decode).
- `FrameMatch.timestamp_s` = **the frame's actual pts**, not the candidate's anchor. These
  differ slightly and the frame's own time is the truthful answer to "when is this frame".
- Save the PNG via `frame.to_image().save(...)` (Pillow is already installed).

### Tests (write from scratch — the current body is a stub)

Build a synthetic clip at 25 fps via the Phase 3 fixture, then assert **exact** frame
equality per CLAUDE.md, falling back to a documented tolerance only if decode actually
proves unstable:

- `visual_text_onset` at t=2.0 → `frame_idx == 50`; PNG exists and is non-empty.
- `speech_onset` at t=2.02 (mid-frame) → `frame_idx == 50` (last frame at or before).
- Determinism: two identical calls produce an identical `frame_idx`.

### Verify → commit → **stop**

---

## Phase 6 — Report + CLI integration

**Files:** `src/report.py`, `src/main.py`, `tests/test_end_to_end.py`, `verify.py`

### `report.py`

```python
def write_report(result, output_dir, *, video_url=None,
                 target_phrase=None) -> str:
```

`report.json` must contain at minimum the keys the scaffold e2e test asserts — `status`,
`frame`, `image_path` — plus the problem statement's required outputs:

```json
{
  "status": "match | ambiguous | not_found",
  "video_url": "...", "dialogue_text": "...",
  "timestamp": "00:05:27.041", "timestamp_s": 327.041,
  "frame": 7849, "extracted_text": "...",
  "image_path": "output/frame_7849.png",
  "modality": "asr", "match_score": 0.94,
  "candidates": []
}
```

- `timestamp` in **HH:MM:SS.sss** — explicitly required by the problem statement.
- `not_found` → `frame: null`, `image_path: null`, no crash.
- `ambiguous` → emit the top candidate's frame **and** list the alternatives. This
  satisfies "never silently hide the disagreement" while still giving the evaluator an
  answer. (CLAUDE.md cutting-list #2 collapses this to a single guess + `low_confidence`
  flag only if time forces it.)
- Also print the Timestamp / Frame / Text summary to stdout.

### `main.py`

Wire `prepare_asset → subtitle fast-path → ASR + OCR → reconcile → to_frame_match →
write_report`. Add `--model` and `--work-dir` to the existing flags; resist adding more.

**Decide the short-circuit here**, using Phase 3's measurement: if OCR over the example is
prohibitively slow, add `--skip-ocr` and/or a documented high-confidence short-circuit,
recording the measured latency that justified it — exactly the evidence CLAUDE.md demands.

Error handling: download failure → clear message, non-zero exit. Zero candidates →
`not_found` report and **exit 0** (the tool ran correctly and found nothing; that is not a
crash). Document that choice.

### Tests

**Offline:** `write_report` unit tests for all three statuses, including exact
`HH:MM:SS.sss` formatting. Then a **full offline end-to-end run on the synthetic clip** —
possible because Phase 3 taught `prepare_asset` to accept local paths, and valuable
because it exercises the whole pipeline with zero network for the evaluator.

**Network:** the two scaffold tests against the real ok.ru URL, unskipped and marked.

### Verify → commit → **stop**

---

## Phase 7 — Packaging

**Files:** `Dockerfile`, `README.md`, `APPROACH.md`, `prompts.txt`

- **Dockerfile → `python:3.12-slim`** to match the version actually verified here (the
  skeleton says 3.11). Keep `ffmpeg` + `libgl1`. Drop any `videocr` remnants.
- **Pre-download the Whisper model in the image**, or document loudly that the first run
  pulls ~500 MB. A container that needs an unexpected network fetch on first run is
  exactly the "runs without errors on someone else's machine" failure CLAUDE.md warns
  about.
- **README** — the local-Python fallback is graded as heavily as Docker: exact ffmpeg
  install per OS, venv steps, how `pytest.ini`'s `network` marker works, and the
  `verify.py N` workflow.
- **APPROACH.md** — the most heavily graded document. It must cover: the dual-track
  rationale; both CLAUDE.md interpretations (`--dialogue-text` as input, onset-vs-completion)
  presented *as interpretations*; the Phase 5 frame policy and its two justifications; the
  arbiter policy and ambiguity handling; the measured ASR/OCR runtimes and any
  short-circuit decision they drove; and the real findings from this build —
  the `libiomp5md.dll` collision, ok.ru's intermittent resets, dropping `videocr`, and the
  0–1 scale normalization that fixed two vacuous assertions.
- **Correct CLAUDE.md's "~491×275"** claim — yt-dlp reports formats up to 960×720. Leaving
  a trivially disprovable claim in a graded doc is worse than the error itself.
- **prompts.txt** — final human review pass across all tools, per CLAUDE.md.

**Verify:** `docker build` + a containerized run writing `report.json` + PNG to a mounted
dir, then confirm the README's local path reproduces it. Commit → **stop**.

---

## If time runs short

Follow CLAUDE.md's cutting list in order — 1) drop OCR backward-walk refinement (it is
already behind a flag), 2) collapse `AmbiguousResult` to a single guess plus a
`low_confidence` flag, 3) OCR sampling 1s → 2s, 4) drop the archive.org ingestion check
and verify against ok.ru only.

Phases 5 and 6 are load-bearing for the deliverable and must not be cut. **Phase 7's
`APPROACH.md` is graded directly** — reserve real time for it rather than treating it as
cleanup, and keep CLAUDE.md's rule that the final stretch is packaging and docs only, not
functionality changes.
