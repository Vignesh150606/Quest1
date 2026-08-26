# Approach

## My Approach (in my own words)

The first time I read the problem — the example being "My mind rebels at stagnation" —
what's given is a video URL, and at some point an on-screen dialogue appears. I
initially read "on-screen dialogue" as meaning someone speaks the dialogue, so either I
could use the subtitles already available (in the case of a YouTube-style video) to
save time, or I'd need another way to get the text.

The specification doesn't define an algorithm for selecting an important line from
arbitrary dialogue, so I treated the target dialogue as an explicit input, and
documented that as an interpretation rather than something the spec demanded.

Then I looked at the actual video given in the problem statement — it had no
subtitles, so speech recognition was required. I also noticed there could be on-screen
text involved, which is usually handled with OCR, so I added that as a third path too.

At a high level, the flow I landed on was: Video → Subtitle / ASR / OCR → Candidates →
Arbiter → Best candidate → Refine → Exact frame.

The reasoning for why all three, not just one: suppose subtitles are available — then
you don't need to run Whisper at all. Whisper is expensive because it's processing
audio. OCR is also expensive because it's processing frames. So subtitles are the
cheapest path, and the other two are fallbacks, only used when needed.

For checking similarity between the target phrase and whatever text was found, I used
fuzzy matching — if the similarity score is greater than a threshold, that's the
output; otherwise, not found.

For audio-to-text, I looked at Whisper and then faster-whisper. I used word-level
timestamps to note the start of a particular word, since that's what I thought I
should map back to the frame image. For faster-whisper, I chose the "small" model as a
time-vs-accuracy trade-off. OCR became the third fallback.

I tested this against real videos (not just the example) and uncovered several real
problems in the process — details in the elaboration below.

---

*Everything above is written as I originally understood and approached the problem.
Everything below was elaborated with AI assistance (Claude Code) — expanding the same
decisions into full technical detail, adding the parts of the design I hadn't
articulated yet (the arbiter's exact policy, the frame-mapping nuance, the testing
process), and documenting the real bugs found along the way.*

---

## Problem Interpretation

The task: given a video URL, find the exact frame where a target line of dialogue
first appears (spoken or on-screen), the timestamp, the extracted text, and the frame
itself as an image.

Two things in the brief needed an explicit interpretation, since the PDF doesn't state
either verbatim:

- **The target dialogue as an explicit input (`--dialogue-text`)**. The PDF's "Input"
  section lists only the video URL; the example quote is given separately as the
  expected answer for self-verification. In a long video with continuous dialogue,
  there's no algorithmic way to identify "the important line" without being told what
  to look for — so the target phrase became a required parameter, not something the
  pipeline guesses at.
- **Return the onset frame, not the completion frame.** The spec says "the exact video
  frame in which the dialogue *first appears*" — that phrasing points at the start of
  the event, not its end.

A third constraint shaped the whole design: **the evaluator may swap in a different
video/dialogue at eval time.** The solution can't be tuned to the example — it has to
generalize to a video with different characteristics (captioned vs. spoken, different
resolution/frame rate, different language patterns).

## Architecture

```
                    VIDEO URL
                       │
                       ▼
                    INGEST
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          Subtitles           Video
              │                 │
              ▼                 │
       Fuzzy text match         │
              │                 │
       confident match?         │
          │         │           │
         YES        NO          │
          │         │           │
          │         ▼           │
          │        ASR          │
          │         │           │
          │    confident?       │
          │      │      │       │
          │     YES     NO      │
          │      │      │       │
          │      │      ▼       │
          │      │     OCR ◄────┘
          │      │      │
          └──────┴──────┘
                 │
                 ▼
             CANDIDATES
                 │
                 ▼
               ARBITER
                 │
          ┌──────┴──────┐
          │             │
       confident      disagree
          │             │
          ▼             ▼
       REFINE       AMBIGUOUS
          │
          ▼
    EXACT VIDEO FRAME
          │
          ▼
     JSON + PNG + stdout
```

Each stage's job:

1. **Ingest** (`ingest.py`) — resolves the URL to a local `VideoAsset` (video path,
   audio path, metadata, subtitle paths), via `yt-dlp`. Includes the subtitle
   fast-path: if a subtitle/CC track exists, fuzzy-match the target phrase against it
   *before* running anything expensive.
2. **ASR track** (`asr_track.py`) — `faster-whisper` transcription with word-level
   timestamps, runs only if the subtitle path didn't produce a confident match.
3. **OCR track** (`ocr_track.py`) — on-screen text detection via PaddleOCR, runs only
   if ASR also didn't produce a confident match.
4. **Arbiter** (`arbiter.py`) — reconciles whatever candidates exist across modalities
   into a single answer, or flags disagreement explicitly.
5. **Refine** (`refine.py`) — maps the winning candidate's timestamp to an exact frame
   number and extracts it, via PyAV.
6. **Report** (`report.py`) — writes `report.json`, the frame PNG, and a stdout
   summary.

**One deliberate evolution from the original design worth calling out**: the initial
baseline (documented in this project's own planning notes) ran all three tracks
unconditionally on every video, with the arbiter deciding after — a simplicity choice
made under time pressure, with short-circuiting flagged as "a valid future
optimization... if OCR proves an actual bottleneck in practice." It did. A later
measurement against the real example video found that per-sample OCR seeking cost more
than the OCR inference itself (26.3 min vs. 18.2 min for the full video), and that the
subtitle-fast-path's own documented benefit ("skips two expensive stages when it hits")
had never actually been wired into the pipeline's control flow — a confident subtitle
hit was still followed by a full ASR run regardless. Both were fixed together: the
staged short-circuit above (skip ASR when subtitle is confident; skip OCR when ASR is
confident) is real code, reusing the arbiter's own threshold/clustering logic as the
one definition of "confident," not a second ad hoc check.

## How the Solution Decides Where to Look

- **Subtitle fast-path**: `yt-dlp --write-subs --write-auto-subs`. If a track exists,
  parse cues (VTT/SRT, including karaoke-style word-timing tags for rolling
  auto-captions), fuzzy-match each cue against the target phrase. Cheapest possible
  path — no audio/video processing at all.
- **ASR track**: only reached if the subtitle path found nothing confident (no track,
  or no cue scored above threshold). Transcribes the audio with word-level timestamps,
  then slides a window across the transcribed words looking for the best fuzzy match
  against the target phrase.
- **OCR track**: only reached if ASR also found nothing confident. Samples frames at a
  fixed interval, runs PaddleOCR (which does its own text-region detection internally
  — a separate contour/connected-component pre-filter was considered and deliberately
  not added, since it would just be a second detector running on top of one already
  doing the same job, with no measured benefit at this sampling rate), fuzzy-matches
  any detected text against the target phrase.

Dual/triple-track exists because the video's actual content dictates which signal is
even present — a caption-only video has no useful ASR signal if the audio is music or
narration-free; a purely spoken video with no captions has no OCR signal at all. Since
the evaluator's video is unknown in advance, the pipeline can't assume which case it's
in.

## How the Relevant Frame Is Determined

**Arbiter** — a deterministic policy, not a trained model, so every decision is fully
explainable:
1. Reject any candidate below its modality's confidence floor.
2. Cluster the remaining candidates that fall within a temporal tolerance window of
   each other.
3. Within a cluster, keep the higher-confidence candidate.
4. If clusters disagree *beyond* the tolerance window — i.e. they point at genuinely
   different moments in the video, not just noisy variants of the same one — return an
   `AmbiguousResult` rather than silently picking one and hiding the disagreement.

**Refine** — this is where a real subtlety lives: an ASR word timestamp is *when
Whisper estimates a word starts*, not inherently a video frame boundary — it's a
temporal anchor, not a frame property. An OCR timestamp, by contrast, *is* a direct
visual frame property (the frame the text was detected in). So mapping an ASR
candidate to an exact frame needs an explicit policy (decode forward from the anchor,
map to the nearest frame at-or-before it, using the video's own measured fps), while an
OCR candidate's mapping is comparatively direct.

All in-pipeline frame-accurate seeking uses **PyAV exclusively**, never mixed with raw
`ffmpeg -ss` — a keyframe-snapped `ffmpeg -ss` seek and a PyAV decode-forward seek can
land on genuinely different frames for the identical timestamp. Consistency of
seeking method matters more than either method being independently "more correct" —
mixing them would make the frame number non-deterministic depending on which stage
touched it last.

## How the Text Is Extracted

- **ASR**: `faster-whisper`, "small" model, CPU/int8 (no GPU on the dev machine),
  `word_timestamps=True`. Batched inference (`BatchedInferencePipeline`) for real
  throughput on a full-length video, with `beam_size=1` (greedy) rather than the
  library's own default of 5 — a real accuracy/speed trade-off, justified by the fact
  that the result is fuzzy-matched against a target phrase rather than needing a
  word-perfect transcript. Audio is decoded in bounded chunks (not read entirely into
  one array) to avoid a single large allocation on a memory-fragmented machine.
- **OCR**: PaddleOCR run directly (not through the `videocr-PaddleOCR` wrapper that
  was originally pinned — dropped once the wrapper's own frame-sampling/SRT output
  proved a poor fit for "find the onset of one target phrase," per the project's own
  "prefer a library's direct API over a wrapper" principle). Frames are sampled via a
  single sequential PyAV decode pass (not per-timestamp seeking — seeking repeatedly
  measured *slower* than the OCR inference itself), at a fixed grid interval.
- **Shared text normalization** (`text_match.py`): casefold, unify unicode
  quotes/dashes to ASCII, strip bracketed annotations (e.g. `[music]`, `[singing]`),
  strip punctuation, collapse whitespace — applied identically across all three
  modalities so a subtitle cue, an ASR transcript window, and an OCR line are compared
  on equal footing.

## Ambiguity & Uncertainty Handling

- Per-modality confidence floors reject low-signal candidates before clustering even
  happens (subtitle 0.5, ASR 0.4 — lower, since Whisper's mean word probability runs
  lower than a clean OCR line or authored subtitle even on a correct match — OCR 0.5).
- A similarity threshold (0.85) on top of confidence — a candidate has to *both* be
  confident in its own modality *and* actually resemble the target text.
- Genuinely disagreeing candidates produce `status: "ambiguous"` with every
  alternative listed in `candidates`, not a silent best-guess.
- No match above threshold anywhere → `status: "not_found"`. This is a legitimate,
  non-error outcome (the tool ran correctly and the phrase genuinely isn't there, or
  didn't clear the confidence bar) — the process still exits 0.

## Trade-offs & What Was Cut

Deliberately not built, to avoid over-engineering a task that explicitly penalizes it:

- **Scene-cut-triggered OCR sampling** — a shot-detection dependency for marginal gain
  over fixed-interval sampling.
- **An LLM/VLM verification pass** — would add a second, opaque source of truth on top
  of an already-explainable deterministic arbiter, for no clear benefit.
- **A learned cross-modal calibration model** — the deterministic arbiter policy above
  is simpler, fully explainable, and there was no labeled dataset to train one against
  anyway.
- **Multi-ASR-size retry, distributed processing, language-specific handling** —
  hypothetical future requirements not implied by this task.

What *was* added beyond the original baseline, backed by measurement rather than
guessing: the staged short-circuit and the two-tier video fetch (cheapest-available
format for subtitle/ASR/a fallback frame; escalate to a higher-quality fetch only if
OCR is actually needed) — both real, measured runtime fixes, not speculative
optimization.

## Known Limitations

- **ASR mis-transcription of unfamiliar proper nouns/brand names.** Tested live: a
  video containing the spoken word "Volopay" (a company name) was transcribed by
  Whisper "small" as "Wallopay" — a plausible phonetic guess for a word outside its
  training vocabulary. Querying the correct spelling alone (`"Volopay"`) returns
  `not_found`, since `rapidfuzz` scores `"volopay"` vs `"wallopay"` at ~67% similarity,
  below the threshold — but querying a longer phrase containing the word (e.g. "a
  potential fit for Volopay") succeeds, because the rest of the phrase transcribed
  correctly and dilutes the one wrong word's impact on the overall score. This is
  correct, non-hallucinating behavior (it doesn't guess), but it means short,
  single-word queries are more fragile to ASR error than full-sentence queries — worth
  knowing when choosing what to search for.
- **Real bugs found and fixed via testing against non-example videos** (none of these
  were caught by the example video or synthetic fixtures alone):
  - A single-character subtitle cue ("I") scored a perfect fuzzy-match score against a
    much longer target, because `rapidfuzz`'s `partial_ratio` trivially finds a
    one-character string inside any longer one. Fixed with a minimum-cue-length
    guard, scoped to that call site only.
  - A real caption file used a stray whitespace-only line as intentional filler
    *inside* one cue's content; the cue-block parser's splitting regex treated it as a
    cue separator and silently dropped an entire caption from consideration.
  - `[singing]`/`[music]`-style bracketed annotations in auto-generated captions were
    being scored as literal words, diluting genuine matches below threshold. Fixed by
    stripping bracketed annotations in the shared normalization step.
- **OCR crashes specifically inside Docker Desktop's WSL2 environment** (`could not
  create a primitive descriptor for a reorder primitive`) — an apparently unresolved
  upstream PaddlePaddle/oneDNN issue; the identical code runs correctly natively on the
  same machine, and outside WSL2-based Docker generally. The CLI's local Python path is
  unaffected.
- **YouTube blocks requests from cloud/datacenter IP ranges** with a bot-detection
  wall — relevant only if the optional web layer is ever hosted on a cloud platform,
  not for local/CLI use, where it works normally from a residential IP.

---

*The optional web application (`api/`, `web/`) that wraps this same pipeline behind a
browser UI is documented separately in `README.md`'s "Local Web App" section — it's an
additive convenience layer, not a replacement for the CLI, and calls the identical
`run_pipeline()` function.*
