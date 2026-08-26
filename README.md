# Find the Exact Frame Where a Dialogue Appears in a Media URL

Given a video URL and a target line of dialogue, this tool finds the exact frame where
that line first appears (spoken or on-screen), the timestamp, the extracted text, and
saves that frame as an image. It works from a subtitle/CC track when the platform ships
one, speech recognition (faster-whisper) when it doesn't, and on-screen text detection
(PaddleOCR) as a further fallback — see `APPROACH.md` for the full design rationale.

## What This Does

1. Downloads the video (`yt-dlp` — works with YouTube, Google Drive, ok.ru, and any
   other `yt-dlp`-supported host).
2. Checks for an existing subtitle/CC track first — cheapest path, skipped if none
   exists or none scores a confident match.
3. Falls back to speech-to-text (word-level timestamps) if no confident subtitle hit.
4. Falls back further to on-screen text detection (OCR) if neither of the above is
   confident.
5. Reconciles whatever candidates were found into a single answer — or reports
   ambiguity/no-match explicitly rather than guessing.
6. Extracts the exact frame via frame-accurate decoding and writes `report.json` + a
   PNG of that frame.

## Quick Start (Docker)

```bash
docker build -t quest1 .
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/.cache:/app/.cache" \
  -v "$(pwd)/.hf-cache:/root/.cache/huggingface" \
  -v "$(pwd)/.paddle-cache:/root/.paddleocr" \
  quest1 --url <video_url> --dialogue-text "<target phrase>" --output /app/output
```

The three cache mounts (`.cache/`, `.hf-cache/`, `.paddle-cache/`) are optional but
strongly recommended: without them, every fresh `docker run` re-downloads the video and
both ML models (faster-whisper, PaddleOCR) from scratch, since a container's filesystem
is ephemeral by default. With them, only the first run pays that cost; later runs
(including against the same video with different dialogue text) reuse everything.

**Known limitation — OCR crashes inside this containerized environment.** Verified by
actually building and running the image (not assumed): the subtitle fast-path, ASR
track, arbiter, refine, and report stages all run correctly in Docker (confirmed with a
real end-to-end run, exit 0, correct match). The OCR track specifically crashes with
`could not create a primitive descriptor for a reorder primitive` — a PaddlePaddle/oneDNN
issue that only reproduces inside this virtualized environment (Docker Desktop's WSL2
VM); the identical code runs correctly natively on the same physical machine. This
appears to be an unresolved upstream PaddlePaddle/PaddleOCR bug (multiple GitHub issues
report the same error with no confirmed fix as of this writing); two targeted
environment-variable workarounds were tried and verified not to resolve it (see
`Dockerfile`'s comments for what was tried). **If a video needs the OCR track
specifically** (on-screen/burned-in text, no usable subtitle track, and ASR doesn't
confidently match), use `--skip-ocr` to get a fast `not_found` in Docker rather than a
crash, or run that case via the local Python fallback below, where OCR works normally.

## Quick Start (Local Fallback)

Docker is the primary path, but this runs identically without it.

**1. System dependency: ffmpeg** (needed for audio extraction/probing; PyAV bundles its
own decoder for frame extraction, but `ffmpeg`/`ffprobe` on `PATH` are still required):

- **Windows**: `winget install ffmpeg` or download a build from
  [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add it to `PATH`.
- **macOS**: `brew install ffmpeg`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install ffmpeg`

Verify with `ffmpeg -version` and `ffprobe -version`.

**2. Python environment** (Python 3.12 — the version `requirements.txt`'s pins were
verified against):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run it:**

```bash
python -m src.main --url <video_url> --dialogue-text "<target phrase>" --output ./output
```

## Usage

```
python -m src.main [--url URL] [--dialogue-text TEXT] [--output DIR]
                    [--model SIZE] [--work-dir DIR] [--skip-ocr] [--hq-frame]
```

If `--url` or `--dialogue-text` are omitted, the tool prompts for them interactively —
so `python -m src.main` with no arguments also works, and asks:

```
Video URL: <paste it here>
Dialogue text to find: <type it here>
```

| Flag | Default | Meaning |
|---|---|---|
| `--url` | *(prompted)* | Video URL to search |
| `--dialogue-text` | *(prompted)* | Target dialogue text to locate |
| `--output` | `./output` | Where `report.json` and the frame PNG are written |
| `--model` | `small` | faster-whisper model size for the ASR track |
| `--work-dir` | `./.cache` (or `$QUEST1_WORK_DIR`) | Download/cache directory |
| `--skip-ocr` | off | Force-skip the OCR track regardless of confidence |
| `--hq-frame` | off | Re-fetch at higher quality just for a sharper output frame, if the fast low-quality fetch already answered |

## Output Format

`report.json` in the output directory:

```json
{
  "status": "match",
  "video_url": "...",
  "dialogue_text": "...",
  "timestamp": "00:05:27.041",
  "timestamp_s": 327.041,
  "frame": 7849,
  "extracted_text": "...",
  "image_path": "output/frame_7849.png",
  "modality": "asr",
  "match_score": 0.94,
  "candidates": []
}
```

- `status` is one of `match`, `ambiguous`, or `not_found`.
- `not_found` → `frame`/`image_path` are `null`; the tool still exits 0 (it ran
  correctly and found nothing — that isn't a crash).
- `ambiguous` → `frame`/`image_path` are the best guess, and `candidates` lists every
  disagreeing alternative rather than silently picking one.
- The same summary (status, timestamp, frame, text) prints to stdout as well.

## Running Tests

```bash
python verify.py <phase_number>   # or "all"
```

Runs the offline suite for one phase (or all of them) — no network required, seconds to
run. Real-network/model-dependent tests are marked `@pytest.mark.network` and excluded
by default; run them explicitly with:

```bash
pytest -m network tests/ -v
```

See `PHASE_CHECKLIST.md` for the phase-by-phase build order.

## Web App (Vercel + Render)

The CLI above is the graded core; this is an optional web wrapper around the *same*
pipeline (`src/main.run_pipeline`) — the API layer never duplicates pipeline logic, it
only calls it. Long jobs (a 54-minute video can take many minutes for ASR/OCR) run in
a background thread so the browser never blocks on a single HTTP request:

```
Browser --(POST /api/jobs)--> FastAPI (Render) --submits to a background thread-->
   run_pipeline() [same code the CLI calls]
Browser --(poll GET /api/jobs/{id} every ~2.5s)--> FastAPI --> queued/processing/completed/not_found/failed
```

- **Frontend** (`web/`): Next.js (App Router), deployed to Vercel. One page: URL +
  phrase inputs, a status area, a result card (timestamp, matched text, frame image),
  a "Search again" reset. No UI framework, no state library — plain `fetch` + `useState`.
- **Backend** (`api/`): FastAPI, deployed to Render as a Docker web service. `api/jobs.py`
  is an in-memory `dict` + `ThreadPoolExecutor` — deliberately the simplest thing that's
  reliable for a first deployment (see its module docstring for the tradeoffs and how to
  swap in a real queue like RQ/Celery later, if a single Render instance ever proves
  insufficient). This is also why the API **must run with a single worker process**
  (`--workers 1`, already set in `Dockerfile.api`) — a second worker would have its own,
  disconnected job dict.
- **Job states**: `queued` → `processing` → one of `completed` / `not_found` / `failed`.
  `not_found` means the pipeline ran correctly and the phrase genuinely wasn't found
  (same meaning as the CLI's `status: "not_found"`) — not an error. `failed` carries a
  short, human-readable `error` message (the same text the CLI prints as `Error: ...`);
  full tracebacks are logged server-side only, never sent to the browser.
- **Files**: each job writes to `output/<job_id>/` on the Render instance's own disk
  (report.json + the winning frame PNG). The frontend never receives raw files from
  Vercel — it fetches the frame through `GET /api/jobs/{id}/frame`, which streams the
  PNG from the backend. This disk is **not** treated as permanent — it only needs to
  survive long enough for one browser session to poll and view its own result.

### Local development

Two processes, run in separate terminals from the repo root:

```bash
# Terminal 1 -- backend (same venv as the CLI; see Quick Start (Local Fallback) above)
pip install -r requirements.txt   # now also installs fastapi/uvicorn
uvicorn api.app:app --reload --port 8000
```

```bash
# Terminal 2 -- frontend
cd web
cp .env.example .env.local        # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

Open `http://localhost:3000`. The backend's default `ALLOWED_ORIGINS` already includes
`http://localhost:3000`, so no CORS setup is needed locally.

### Deploying the backend to Render

1. Push this repo to GitHub (already required for submission).
2. In Render: **New → Blueprint**, point it at the repo — it picks up `render.yaml`
   automatically (it builds `Dockerfile.api`, not the CLI's `Dockerfile`).
   - No Blueprint? New → Web Service → Docker → set **Dockerfile Path** to
     `Dockerfile.api` manually.
3. Set the `ALLOWED_ORIGINS` env var to your Vercel URL once you have it (step 3 below)
   — until then CORS will reject the browser's requests, which is expected.
4. **Free tier, by design** (`render.yaml` sets `plan: free`): 512MB RAM, 0.1 CPU,
   spins down after 15min idle (~1min cold start on the next request), 750 free
   instance-hours/month. To fit real ML models into that, `render.yaml` also sets:
   - `QUEST1_SKIP_OCR=true` — PaddleOCR is lazy-loaded (`src/ocr_track.py`'s
     `_get_ocr()`), so skipping it means its model weights never get allocated at
     all, leaving the 512MB budget to faster-whisper alone. **Trade-off**: a video
     whose answer is on-screen/caption text only (no subtitle track, no confident ASR
     hit) will report `not_found` on this free deployment. The full three-track
     pipeline is still available via a paid Render plan (`QUEST1_SKIP_OCR=false`) or
     the local Python fallback above.
   - `QUEST1_MODEL_SIZE=small` is left as-is, not silently downgraded, even though
     it's the larger of the two remaining memory consumers (`src/asr_track.py`
     documents "small" as a deliberate accuracy choice over "base"). If it OOMs in
     practice on Render's actual infra (unverified from here — no Render account
     access in this environment), `QUEST1_MODEL_SIZE=base` is the documented lever to
     trade accuracy for headroom, applied explicitly, not by default.
   - `QUEST1_MAX_CONCURRENT_JOBS=1` — 0.1 CPU can't usefully run two ML jobs at once.
   Want the full pipeline with more headroom instead? Change `plan: free` to
   `plan: standard` and `QUEST1_SKIP_OCR` to `false` in `render.yaml` before deploying.
5. **OCR-in-Docker risk**: the CLI's own Dockerfile hit a real, unresolved
   PaddlePaddle/oneDNN crash specifically under Docker Desktop's WSL2 on the dev
   machine (see that Dockerfile's comments) — whether Render's container host hits the
   same issue is *unverified* until actually deployed there. Moot while
   `QUEST1_SKIP_OCR=true` (the free-tier default above), relevant again if you switch
   to a paid plan with OCR enabled.

### Deploying the frontend to Vercel

1. In Vercel: **New Project**, import this repo, set **Root Directory** to `web`.
2. Add the environment variable `NEXT_PUBLIC_API_URL` = your Render backend's URL
   (e.g. `https://quest1-api.onrender.com`), no trailing slash.
3. Deploy. Copy the resulting `https://....vercel.app` URL back into Render's
   `ALLOWED_ORIGINS` (step 3 above) and redeploy the backend so CORS allows it.

### Environment variables

| Var | Where | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `web/.env.example` (Vercel) | Backend base URL the browser calls |
| `ALLOWED_ORIGINS` | `.env.example` (Render) | Comma-separated frontend origins allowed by CORS |
| `QUEST1_SKIP_OCR` | `.env.example` (Render) | Force-disable OCR (default `true` on the free-tier `render.yaml` to fit 512MB RAM; also a mitigation for the Docker/oneDNN issue above) |
| `QUEST1_MODEL_SIZE` | `.env.example` (Render) | faster-whisper model size (default `small`; lower to `base` only if you hit real OOM on a constrained plan) |
| `QUEST1_MAX_CONCURRENT_JOBS` | `.env.example` (Render) | Background thread-pool size (default 1 on free tier, since 0.1 CPU can't usefully run two ML jobs at once) |
| `QUEST1_OUTPUT_ROOT` | `.env.example` (Render) | Where per-job `report.json`/PNGs are written |
| `QUEST1_WORK_DIR` | `.env.example` (Render) | yt-dlp/audio cache dir (same meaning as the CLI's `--work-dir`) |

No secrets/API keys are required anywhere in this project — nothing above should ever
hold a real credential.

### Limitations (documented, not silently ignored)

- **No process restart durability**: a job's status lives in memory; if the Render
  instance restarts mid-job, that job's progress is lost (the browser would see the
  poll requests start failing/timing out). Acceptable for a first deployment per the
  brief; the fix is a real queue + persistent store, deliberately not built now (see
  `api/jobs.py`'s docstring).
- **Single Render instance only** (`--workers 1`) — the in-memory job store does not
  span multiple processes or instances. Scaling out requires the Redis/RQ swap noted
  above.
- **OCR-in-container risk is unverified on Render** — see step 5 above.
- **Free-tier deployment runs a reduced pipeline, on purpose**: OCR is off by default
  to fit 512MB RAM (see step 4 above) — subtitle + ASR only. A video that genuinely
  needs OCR (burned-in/on-screen text, no subtitle track, no confident speech match)
  will report `not_found` on the free deployment specifically, not on the CLI/local
  Python path, which still runs all three tracks.
- **Cold starts**: free tier spins down after 15min idle; the first request after that
  takes up to ~1min to wake the instance before a job even starts processing.
- **Whisper's actual memory footprint on Render's free tier is unmeasured** — the
  512MB budget is tight even with OCR off; `QUEST1_MODEL_SIZE=base` is the documented
  fallback if it OOMs in practice.

## Repository Structure

```
src/
  main.py         CLI entry point + pipeline orchestration
  ingest.py       URL -> VideoAsset (download, probe, subtitle fast-path)
  asr_track.py    Speech-to-text candidate generation (faster-whisper)
  ocr_track.py    On-screen text candidate generation (PaddleOCR)
  arbiter.py      Deterministic reconciliation of candidates across modalities
  refine.py       Candidate timestamp -> exact video frame (PyAV)
  report.py       Final report.json + PNG + stdout summary
  text_match.py   Shared text normalization/fuzzy-matching used by every track
  types.py        Shared dataclasses (VideoAsset, Candidate, FrameMatch, ...)
api/
  app.py          FastAPI routes (POST/GET /api/jobs, GET /api/health, frame serving)
  jobs.py         In-memory job store + background ThreadPoolExecutor worker
web/
  app/page.js     The whole frontend: form, polling, result rendering
tests/            pytest suite, one file per src/ module above, plus api/ and fixtures/
verify.py         Runs one phase's (or all phases') offline test suite
Dockerfile        CLI image (primary submission artifact)
Dockerfile.api    Backend API image (Render)
render.yaml       Render Blueprint for the backend
APPROACH.md       Design rationale, trade-offs, and known limitations
PHASES.md         Original phase specification
PHASES_1_7_PLAN.md  Detailed per-phase implementation plan (historical)
prompts.txt       Full AI prompt log, including real bugs found via generalization testing
```
