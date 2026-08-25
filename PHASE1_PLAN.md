# Phase 1 — Ingest + Shared Schema

## Context

Phase 0 is done: the toolchain installs and `verify.py 0` passes. Every `src/` module is
still a stub raising `NotImplementedError`.

Phase 1 turns a video URL into a local `VideoAsset` (downloaded video, extracted audio,
probed metadata) and adds the subtitle fast-path: if the platform already ships a
subtitle/CC track, fuzzy-match the target phrase against it and skip the two expensive
tracks entirely. It also lands the `Candidate` schema that Phases 2–4 all depend on.

**Four facts verified against the live example video** (`yt-dlp -J`, run 2026-08-25) that
change the design and are not in CLAUDE.md:

1. **`"subtitles": {}`** — the example video has **no subtitle track**. The fast-path can
   never hit on it. It must be tested positively against an offline fixture instead, and
   its no-track path must return `None` cleanly rather than throw.
2. **ok.ru resets connections intermittently.** The identical command failed → succeeded →
   failed across three runs. The reset lands on the extractor's *mobile webpage* fallback,
   i.e. **during info extraction**, which yt-dlp's own `retries` option does *not* cover.
   Retry must wrap the `extract_info` call.
3. **Real metadata: `duration=3261`s, `fps=24.0`, best format 960×720 HLS.** Use these as
   the ground-truth values in assertions.
4. **CLAUDE.md's "~491×275" native resolution is wrong** — yt-dlp reports formats up to
   960×720 (4:3). Correct it in CLAUDE.md/APPROACH.md rather than leaving a claim that an
   interviewer can trivially disprove.

Second-domain fixture verified working: `https://archive.org/details/BigBuckBunny_124`
(596.46s, progressive HTTPS — a useful contrast to ok.ru's HLS path).

---

## Decisions taken (already confirmed)

| Decision | Choice |
|---|---|
| Test policy | Offline unit tests run by default; real-network test marked `network`, opt-in |
| Download cache | Work dir keyed by URL hash; reuse if present; `use_cache=False` to force |
| Format selection | `height<=720` cap, fall back to best |

---

## Work

### 1. `src/types.py` — one field

Add to `VideoAsset` only:

```python
subtitle_paths: list[str] = field(default_factory=list)
```

`try_subtitle_fast_path(asset, ...)` takes the asset, not a path, so downloaded subtitle
files must be reachable from it. Defaulted so existing construction sites don't break.

**Do not** add anything else to `VideoMetadata` — Phase 5 reads the container directly via
PyAV and shouldn't inherit guesses from here.

### 2. `src/text_match.py` — NEW module

**This is a deliberate deviation from PHASES.md**, which lists only `ingest.py`,
`types.py`, `test_ingest.py` for Phase 1. Justification to state in the commit message:
the arbiter (Phase 4) compares `similarity` scores *across* modalities, which is only
meaningful if subtitle/ASR/OCR normalize text identically. Three private copies of a
normalizer would guarantee drift. Phase 1 is simply the first consumer.

Keep the surface minimal — resist adding what Phase 2/3 haven't asked for yet:

```python
DEFAULT_MATCH_THRESHOLD = 85.0          # rapidfuzz 0-100 scale

def normalize(text: str) -> str: ...
def similarity(target: str, candidate: str) -> float: ...
```

- `normalize`: casefold; unify unicode quotes/dashes to ASCII; strip punctuation; collapse
  whitespace. Nothing language-specific (explicitly out of scope per CLAUDE.md).
- `similarity`: `rapidfuzz.fuzz.partial_ratio` over the normalized pair. `partial_ratio`,
  not `ratio` — the target phrase is a *substring* of a longer cue/transcript, so `ratio`
  would penalize the surrounding words.

### 3. `src/ingest.py` — `prepare_asset`

Keep the scaffold signature, extend with keyword-only optionals:

```python
def prepare_asset(video_url: str, *, work_dir: str | None = None,
                  use_cache: bool = True) -> VideoAsset:
```

Steps:

1. `key = sha256(video_url.encode()).hexdigest()[:16]`; work dir =
   `work_dir or $QUEST1_WORK_DIR or ".cache"` / `key`.
2. If cached `video.*` + `audio.wav` + `meta.json` all exist and `use_cache` → rebuild
   `VideoAsset` from `meta.json` and return.
3. Download via the **`yt_dlp.YoutubeDL` Python API, not subprocess** — CLAUDE.md prefers a
   library's direct API, and it hands back a structured info dict instead of scraped stdout.
   Options: `format="bestvideo[height<=720]+bestaudio/best[height<=720]/best"`,
   `writesubtitles=True`, `writeautomaticsub=True`, `subtitleslangs=["en", "en.*"]`,
   `retries=5`, `fragment_retries=5`, `socket_timeout=30`, `quiet=True`, `noprogress=True`.
4. **Wrap `extract_info` in its own retry loop** (3 attempts, exponential backoff). This is
   the fix for finding #2 — yt-dlp's `retries` covers fragment/HTTP download, *not*
   extraction. Without this, Phase 1 fails randomly on the graded example video.
5. Probe the **downloaded file** with `ffprobe -of json` — do not trust the manifest's
   reported fps. HLS-muxed output can differ from what the m3u8 advertised, and Phase 5's
   frame math depends on this number being right.
   - fps: prefer `avg_frame_rate` when it parses to > 0, else `r_frame_rate`; both are
     fractions like `"24/1"`. Document the choice in a comment — Phase 5 inherits it.
6. Extract audio: `ffmpeg -i <video> -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`.
   16 kHz mono is exactly faster-whisper's expected input, so Phase 2 does no resampling.
   - **Note in a comment that ffmpeg CLI is correct here.** CLAUDE.md's "PyAV only" rule is
     scoped to *frame-accurate seeking*, not audio extraction. Say so, so neither the next
     implementer nor an interviewer reads it as a violated constraint.
7. Write `meta.json` (cache marker + the probed metadata + subtitle file list), return
   `VideoAsset`.

**Assert audio actually exists**: every ok.ru format reported `"audio_ext": "none"`. That
is probably just yt-dlp not knowing codecs from the m3u8, but it is unverified. If
`audio.wav` comes out empty or ffprobe shows no audio stream, raise a clear error naming
the format selector as the likely cause rather than letting Phase 2 fail mysteriously.

### 4. `src/ingest.py` — `try_subtitle_fast_path`

```python
def try_subtitle_fast_path(asset, target_phrase,
                           threshold: float = DEFAULT_MATCH_THRESHOLD) -> Optional[Candidate]:
```

- **Parse WebVTT and SRT with a small local parser (~40 lines), no new dependency.** Both
  formats share the `HH:MM:SS[.,]mmm --> HH:MM:SS[.,]mmm` cue-timing line; the differences
  are the `WEBVTT` header, comma-vs-dot decimal separator, and SRT's numeric index lines.
  Strip VTT inline tags (`<c>`, `<00:00:01.000>`) and HTML tags from cue text.
- Score every cue with `similarity(target_phrase, cue.text)`; keep the best.
- Below `threshold` or no cues → return `None`. **Never raise** — the caller falls through
  to ASR/OCR, and the example video takes this path every time.

Two schema decisions to make explicitly and document:

- **`event_type="speech_onset"`.** A platform subtitle cue is timed to *speech*, not to a
  pixel change, so Phase 5 must apply its ASR→frame policy to subtitle candidates too —
  not the direct visual mapping. Getting this wrong silently shifts the reported frame.
- **`confidence` distinguishes manual from auto captions.** `writesubtitles` yields
  human-authored tracks; `writeautomaticsub` yields the platform's own ASR, which can be
  wrong. Use `1.0` for manual, `0.7` for auto, and record which in
  `evidence["subtitle_kind"]`. Flag for Phase 4: manual subtitles at `1.0` will
  automatically outrank ASR/OCR inside a cluster. That is intended, but the arbiter must
  document it rather than inherit it by accident.

`evidence` payload: `{source_file, cue_index, subtitle_kind, lang}`.

### 5. `pytest.ini` — NEW

```ini
[pytest]
markers =
    network: hits real remote hosts; excluded by default
addopts = -m "not network"
```

### 6. `verify.py` — one-line change

Phase 1's entry is currently `tests/test_ingest.py::test_prepare_asset`, which is the
network test. Under `addopts = -m "not network"` that deselects everything and pytest
exits 5 ("no tests collected"), failing the phase. Change the value to
`tests/test_ingest.py` — it runs all of Phase 1's offline tests, which is more correct
anyway.

> Still open from Phase 0, out of scope here: `verify.py` shells out to bare `pytest`, so
> it only works with the venv activated. `sys.executable -m pytest` would fix it.

### 7. Fixtures + tests

New: `tests/fixtures/sample.en.vtt`, `tests/fixtures/sample.srt`. Hand-written, a handful
of cues, one of which carries a distinctive phrase to search for. Do **not** use the
graded example line in runtime-reachable code — fixtures only, per CLAUDE.md.

Rewrite `tests/test_ingest.py`. Every assertion uses **explicit expected values**, not
`> 0` smoke checks (CLAUDE.md's ground-truth rule — the existing stub's
`assert duration_s > 0` is exactly what it warns against):

**Offline (default):**

| Test | Asserts |
|---|---|
| `test_normalize` | exact expected output strings across case/punctuation/unicode-quote inputs |
| `test_parse_vtt_cues` / `test_parse_srt_cues` | exact cue count, exact first+last timestamps, exact text |
| `test_subtitle_fast_path_hit` | `modality=="subtitle"`, `event_type=="speech_onset"`, `timestamp ==` the exact cue start, `similarity >= 85` |
| `test_subtitle_fast_path_fuzzy_hit` | same cue still matches with a typo + case + punctuation change |
| `test_subtitle_fast_path_miss` | nonsense phrase → `None`, no exception |
| `test_subtitle_fast_path_no_tracks` | `subtitle_paths=[]` → `None` (the example video's real situation) |
| `test_probe_metadata` | tiny `ffmpeg -f lavfi` clip at known fps/size/duration → `fps == 25.0`, `width == 320`, `height == 240`, `abs(duration_s - 2.0) < 0.1` |

**Network (`@pytest.mark.network`, opt-in):**

| Test | Asserts |
|---|---|
| `test_prepare_asset[okru]` | `fps == 24.0`, `3255 < duration_s < 3270`, `height <= 720`, `subtitle_paths == []`, audio exists and non-empty |
| `test_prepare_asset[archive]` | `abs(duration_s - 596.46) < 2`, `height <= 720`, audio exists and non-empty |

### 8. Housekeeping

- `.gitignore`: add `.cache/`.
- `PHASE_CHECKLIST.md`: tick Phase 0 and Phase 1.
- `prompts.txt`: append the Phase 1 prompt + outcome, in the existing BUILD PHASE format.

---

## Verification

```bash
python verify.py 1
```

Offline suite, seconds, no network. This is the gate for calling Phase 1 done.

```bash
pytest -m network tests/test_ingest.py -v
```

Real downloads against ok.ru + archive.org. Slow. **Run it at least once** — it is the
only thing that proves the retry loop, the format selector, and audio extraction actually
work on the graded video. Expect it to need a retry or two on ok.ru; that is the bug being
defended against, not a failure.

Manual smoke, confirming the cache short-circuits on the second call:

```bash
python -c "from src.ingest import prepare_asset; a = prepare_asset('https://archive.org/details/BigBuckBunny_124'); print(a.metadata, a.audio_path, a.subtitle_paths)"
```

---

## Stop-and-flag triggers

Per CLAUDE.md, escalate rather than working around:

- **Downloaded video has no audio stream.** Means the `height<=720` selector dropped audio
  on HLS; the selector needs rework and Phase 2 is blocked until it is.
- **ffprobe fps disagrees with yt-dlp's reported 24.0.** Phase 5's frame math depends on
  this; resolve which is authoritative before building on it.
- **ok.ru resets persist through 3 retries.** Then it is a block, not flakiness, and the
  fallback (`--impersonate` + `curl_cffi`, a new dependency) needs an explicit decision.
- **A test only passes with a loosened assertion.** Report the real number instead of
  quietly widening the tolerance.
