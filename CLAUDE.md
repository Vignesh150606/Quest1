# Quest1 Campus Placement 2027 — Project Context

Read this fully before writing any code. This is the current engineering baseline —
not an unquestionable spec. Implement incrementally, validate assumptions, and if
testing reveals a correctness, compatibility, performance, or reliability problem
with a decision below, **stop and flag it with evidence before deviating**. Don't
silently work around a broken assumption, and don't silently keep following one
that's been disproven.

## Problem Statement (condensed)

**Title:** Find the Exact Frame Where a Dialogue Appears in a Media URL

Given a video URL, identify:
1. The exact frame where an on-screen/spoken dialogue **first appears**
2. The text of that dialogue

**Example target:** ok.ru video of Granada TV's *Sherlock Holmes: A Scandal in Bohemia*
(Jeremy Brett, ~54 min). Example line: **"My mind rebels at stagnation."** This exact
string is for validation only — never hardcode it or the example URL into runtime logic.

**Required output (minimum):**
- Timestamp (HH:MM:SS.sss)
- Frame number
- Extracted dialogue text
- The corresponding frame as an image

**Hard requirements:**
- Must run without the evaluator manually inspecting the video
- Robust to normal variance in quality/resolution/frame rate/text appearance
- AI/ML tools, APIs, local models all allowed
- **The evaluator may swap in a different video/dialogue at eval time** — the solution
  must generalize, not be tuned to the example.

**Graded explicitly on:**
- Prompts documented (`prompts.txt`)
- Design/approach doc (separate file)
- How the solution decides where to look, which frame, how it extracts text
- How ambiguity/uncertainty is handled

**Interview:** they may modify the implementation live, change a requirement, or ask you
to defend a choice. Explicit distinction they're grading: "wrote it yourself" vs "can
actually engineer with AI" — drive the AI, don't proxy for it.

## Submission Requirements

- Deadline: **26 Aug, 23:59:59 IST**
- Public GitHub repo containing:
  - `/src` — source code
  - `README.md` — how to run
  - `APPROACH.md` — design, choices, algorithms, assumptions, trade-offs
  - `prompts.txt` — every AI prompt used, including iterations that influenced the
    solution, **across every tool used** (Claude Code, chat sessions, any other LLM) —
    not just what a coding agent captures automatically. Treat this as a human-verified
    audit record: append as you go, then review the final file yourself before
    submission rather than trusting any single tool to have logged everything.
- **Unit tests required**
- A hosted/deployed solution is welcomed **in addition to** the setup doc, never instead
  of it — the setup doc itself is graded as evidence of understanding
- Innovation is allowed; **over-engineering is explicitly penalized**. Solve for the ask,
  make it extensible, don't build for imaginary future requirements.
- **Minimize dependencies.** Prefer a library's direct API over a wrapper package when
  the wrapper adds little value or introduces compatibility risk. Every dependency is a
  way for the "runs without errors on someone else's machine" requirement to fail.
- **Prefer the smallest change that solves the current problem.** Don't refactor
  unrelated code, don't introduce abstractions before there's a demonstrated need,
  don't rewrite a working implementation for stylistic reasons.

## Facts Established by Manual Investigation (not in the PDF)

- Example video: Granada TV *Sherlock Holmes: A Scandal in Bohemia*, Jeremy Brett, ~54 min,
  low native resolution (~491×275)
- Manually reviewed the frame at the target quote (~05:27–05:29): Brett is speaking
  on camera with **no visible on-screen caption** at that timestamp → for this specific
  video, it's a **spoken-dialogue case, not a caption-overlay case**
- Because the eval video may differ, the solution still needs to handle the
  caption-overlay case too — hence the dual-track design below

## Interpretation vs Fact — read this distinction carefully

The items below are **documented interpretations of an underspecified problem**, not
requirements stated verbatim in the PDF. Each is defensible and each was checked against
the actual document text — but they are still assumptions, and should be presented as
such in `APPROACH.md`, not as if the PDF demanded them.

- **`--dialogue-text` as an explicit input parameter.** The PDF's "Input" section lists
  only the video URL; the example quote is given separately as the expected answer for
  self-verification. In a long video with continuous dialogue, there is no algorithmic
  way to identify "the important line" without being told what to look for — this is a
  deliberate response to a genuine ambiguity in the input contract, checked against the
  literal PDF text, not a shortcut or an assumption made without basis.
- **Return the onset frame, not the completion frame** — the spec says "the exact video
  frame in which the dialogue **first appears**," which points to the start of the event.

If, during implementation or interview prep, evidence surfaces that either interpretation
doesn't match what's actually being tested, flag it rather than continuing to build on it.

## Agreed Architecture

Shared pipeline as the default: both tracks run unconditionally on every video, and an
arbiter decides after. This is a deliberate simplicity choice for the time remaining —
adding confidence-threshold short-circuiting is a valid future optimization, but it's
itself a new tunable decision requiring its own calibration, and isn't worth the risk
this close to deadline unless the OCR track proves to be an actual bottleneck in
practice. If it does, document the measured latency/reliability trade-off before adding
a short-circuit.

```
                    VIDEO URL
                       |
                       v
                 +-----------+
                 |  Ingest   |
                 +-----+-----+
                       |
             +---------+---------+
             v                   v
       Subtitle path        VideoAsset
             |                   |
             |          +--------+--------+
             |          v                 v
             |        ASR               OCR
             |          |                 |
             +----------+-----------------+
                        |
                        v
                 Candidate set
                        |
                        v
                    Arbiter
                        |
              +---------+---------+
              v                   v
            Match             Ambiguous
              |
              v
           PyAV refine
              |
              v
          FrameMatch
              |
              v
      JSON + PNG + stdout
```

1. **Ingest** — `URL → VideoAsset(video_path, audio_path, metadata)`. Includes a subtitle
   fast-path: `yt-dlp --write-subs --write-auto-subs` first; if a track exists, fuzzy-match
   the target phrase against it before running ASR or OCR at all — cheap, and skips two
   expensive stages when it hits.
2. **ASR Track** — `VideoAsset, target phrase → List[Candidate]`. Use `faster-whisper`
   with **native word-level timestamps** (`word_timestamps=True`) — not WhisperX.
   Segment-level timestamps are too coarse for a ~1-2 second line; word-level solves that
   without an extra alignment-model dependency. Event type: `speech_onset` — this is a
   speech-recognition estimate of when a word starts, not inherently a video-frame
   boundary (see Refine, below).
3. **OCR Track** — `VideoAsset, target phrase → List[Candidate]`. Pipeline **interface**,
   not a prescribed implementation:
   `frame sampler -> candidate-region detector -> candidate windows -> OCR -> text matching`.
   Choose the simplest reliable implementation for the candidate-region detector (could be
   contour/edge heuristics, connected components, frame differencing, or PaddleOCR's own
   detection-only mode at low sampling frequency) and document why in `APPROACH.md`. Don't
   reach for a heavier detector unless testing shows the simple version is insufficient.
   `PaddleOCR` pinned to a version compatible with `videocr-PaddleOCR` (that wrapper breaks
   on PaddleOCR 3.x — pin to `2.10.0`/`2.7.0.2` in requirements). Event type:
   `visual_text_onset` — the frame where text becomes visually recognizable, a distinct
   concept from speech onset.
4. **Arbiter** — `ASR candidates, OCR candidates -> winning Candidate(s) | AmbiguousResult | None`.
   Deterministic policy, no ML calibration model:
   1. Reject candidates below a per-modality confidence threshold.
   2. Cluster remaining candidates that fall within a temporal tolerance window.
   3. Within a cluster, prefer the higher-confidence candidate.
   4. If modalities disagree beyond the tolerance window (i.e. point to different
      moments, not just noisy variants of the same one), return `AmbiguousResult` —
      never silently pick one and hide the disagreement.
5. **Refine** — `winning Candidate(s), VideoAsset -> FrameMatch(es)`. Use **PyAV
   exclusively** for all in-pipeline frame-accurate extraction — never mix with raw
   `ffmpeg -ss` seeking inside pipeline code (keyframe-snapped ffmpeg seeks and
   decode-forward PyAV seeks can return different frames for the identical timestamp;
   raw ffmpeg CLI is fine for manual debugging only). Explicitly acknowledge: ASR
   timestamps are temporal anchors, not inherently exact frame boundaries. This stage
   decodes the frames surrounding the anchor and maps it to a specific frame number
   according to the timestamp/frame policy below — for OCR candidates this mapping is
   comparatively direct (the event *is* a visual frame property); for ASR candidates it
   requires an explicit, documented policy for translating an audio-timestamp anchor
   into a specific video frame.
6. **Report** — `FrameMatch(es)/AmbiguousResult, VideoAsset -> report.json + PNG + stdout`

### Candidate schema (define this before writing the arbiter)

```
Candidate
  modality          # "subtitle" | "asr" | "ocr"
  event_type        # "speech_onset" | "visual_text_onset"
  timestamp
  end_timestamp
  matched_text
  normalized_text
  similarity         # match score vs target phrase
  confidence         # modality-native confidence, if available
  evidence           # anything needed to justify this candidate later
```

## Ground Truth / Acceptance Criteria

A pipeline that "produces a result" is not the same as one that produces the *correct*
result. Every test fixture needs explicit expected values, not just a smoke check that
the pipeline runs:

```
Fixture:
  known video
  known target phrase
  known expected onset timestamp
  known expected frame number (or a documented tolerance if exact frame identity
    proves unstable across decode/encode environments)

Assertions:
  ASR track:  expected phrase match, expected onset within tolerance
  OCR track:  expected visible text match, expected onset within tolerance
  End-to-end: expected frame/timestamp in final report
```

Decide and document the tolerance policy explicitly — don't quietly redefine "exact
frame" as "approximately around the timestamp" without saying so. If PyAV decoding
proves deterministic for a given source file (it should), prefer asserting exact frame
equality and only fall back to a documented tolerance if testing shows real instability.

## Tech Stack (baseline)

| Purpose | Tool |
|---|---|
| URL resolution / download | `yt-dlp` |
| Metadata / audio / subtitle extraction | `ffmpeg` / `ffprobe` |
| ASR with word-level timestamps | `faster-whisper` |
| Fuzzy text matching | `rapidfuzz` |
| On-screen text detection/OCR | `PaddleOCR` + `videocr-PaddleOCR` (version-pinned) |
| Candidate-region pre-filter | simplest reliable option that passes testing (see OCR Track above) |
| In-pipeline frame-accurate seeking | `PyAV` |
| Tests | `pytest` (per-module + one offline smoke test, no network dependency) |
| Packaging | Docker (primary) + documented local-Python fallback (see below) |

## Docker: primary path, not the only path

Dockerize as planned — but the README should also document a plain local-Python setup
(exact system dependencies: ffmpeg, any OCR system libs) as a fallback. The evaluator
may hit Docker Desktop, GPU, networking, or image-size friction; the project's
correctness shouldn't hinge on Docker working perfectly on an unknown machine.

## Time Plan

Recompute this against the actual time remaining when you resume work — don't trust
fixed "Day 1/2/2.5" labels if the calendar has moved since this file was written.
Structure sessions around what needs to be true by the end of each, not a specific
clock time:

**Session 1** — project skeleton, environment, `ingest.py` (yt-dlp download, ffprobe
metadata, audio extraction, subtitle fast-path), validate yt-dlp against ok.ru + one
other domain, `asr_track.py` (transcribe + fuzzy match). Goal: ASR track locates the
target phrase on the example video, with ground-truth assertions passing.

**Session 2** — `ocr_track.py`, a synthetic captioned test clip (the example video
doesn't exercise the OCR path), `refine.py` (PyAV-based frame-accurate extraction for
both event types), Candidate/arbiter implementation. Goal: both tracks produce a
`FrameMatch` + PNG on their respective fixtures, arbiter reconciles them correctly on a
constructed dual-signal test case.

**Final session** — `main.py` + `report.py` wiring, end-to-end smoke test on the example
video, error handling (download failure, zero candidates -> `NotFound`), Dockerfile +
local fallback docs, `README.md`, `APPROACH.md`, final `prompts.txt` review across all
tools used. Reserve real time at the end for: **do not touch functionality, only test
and package.**

## Cutting List (if behind schedule, drop in this order)

1. OCR backward-walk refinement -> report the coarse sample hit instead
2. `AmbiguousResult` top-k list -> collapse to single best guess + `low_confidence` flag
3. OCR sample interval: 1s -> 2s
4. Cross-source ingestion validation -> verify against ok.ru only

## Explicitly NOT Building (documented scope boundary)

- Scene-cut-triggered OCR sampling (shot-detection dependency for marginal gain)
- LLM/VLM verification pass (second opaque source of truth, unneeded)
- Cross-modal score calibration model (a simple deterministic arbiter policy, above, is
  sufficient and more explainable in an interview)
- Multi-ASR-size retry, distributed processing, web UI, language-specific handling
  (hypothetical future requirements, not this task)

## Known Gotchas

- `ffmpeg -ss` before `-i` snaps to the nearest keyframe (fast, imprecise); after `-i` is
  slow but precise; PyAV decode-forward is a third distinct behavior. Use PyAV
  consistently inside the pipeline; ffmpeg CLI for manual debugging only.
- PaddleOCR 3.x breaks `videocr-PaddleOCR`'s wrapper — pin the compatible version.
- `faster-whisper` segment-level timestamps are too coarse for a ~1-2 second line — use
  word-level timestamps, not segment-level.

## Instructions for This Claude Code Session

- Build incrementally per the time plan above; commit as you go.
- Append every prompt given to you (and meaningful follow-ups) to `prompts.txt` as the
  session progresses — not reconstructed at the end. This file will also be reviewed and
  supplemented by hand with prompts from other tools used outside this session.
- Write `pytest` tests alongside each module, not after, with explicit expected values
  (see Ground Truth section) — not just "did it run without crashing."
- Never hardcode "My mind rebels at stagnation" or the example video URL into runtime
  logic — test fixtures only.
- **Stop and explain before proceeding** if you hit any of: an ambiguous requirement, a
  conflicting constraint, a major dependency incompatibility, a needed architecture
  change, an inability to meet the "exact frame" correctness bar, or an interpretation of
  the task that differs materially from what's documented above. Don't silently work
  around a broken assumption or silently keep following one that testing has disproven —
  flag it with evidence and propose the alternative before making the change.
