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

## Local Web App

The CLI above is the graded core; this is an optional browser UI around the *same*
pipeline (`src/main.run_pipeline`) — the API layer never duplicates pipeline logic, it
only calls it, exactly like the CLI does. Everything — `yt-dlp`, ffmpeg, faster-whisper,
PaddleOCR — runs on **your own machine**, using its CPU/RAM/storage, the same as running
the CLI directly. There is no remote processing service. Long jobs (a 54-minute video
can take many minutes for ASR/OCR) run in a background thread on your machine so the
browser never blocks on a single HTTP request:

```
Browser (localhost:3000) --(POST /api/jobs)--> FastAPI (localhost:8000, your machine)
   --submits to a background thread--> run_pipeline() [same code the CLI calls]
Browser --(poll GET /api/jobs/{id} every ~2.5s)--> FastAPI --> queued/processing/completed/not_found/failed
```

- **Frontend** (`web/`): Next.js (App Router). One page: URL + phrase inputs, a status
  area, a result card (timestamp, matched text, frame image), a "Search again" reset.
  No UI framework, no state library — plain `fetch` + `useState`.
- **Backend** (`api/`): FastAPI, run locally with `uvicorn` — no Docker, no cloud
  service. `api/jobs.py` is an in-memory `dict` + `ThreadPoolExecutor`, deliberately
  the simplest thing that's reliable for one machine with no other jobs to coordinate
  across (see its module docstring). This is also why the API **must run as a single
  `uvicorn` worker** (the default — don't pass `--workers N`) — a second worker
  process would have its own, disconnected job dict.
- **Job states**: `queued` → `processing` → one of `completed` / `not_found` / `failed`.
  `not_found` means the pipeline ran correctly and the phrase genuinely wasn't found
  (same meaning as the CLI's `status: "not_found"`) — not an error. `failed` carries a
  short, human-readable `error` message (the same text the CLI prints as `Error: ...`);
  full tracebacks are logged server-side (your own terminal) only, never sent to the
  browser. If the backend process restarts while a job is running (you stopped it,
  it crashed), that job's in-memory state is gone — the frontend detects the resulting
  404 on its next poll and shows "Lost track of this job... Please try again" instead
  of polling forever.
- **Files**: each job writes to `output/<job_id>/` on your machine (report.json + the
  winning frame PNG) — the same output directory the CLI already uses, just one
  subfolder per job instead of one shared directory. The frontend never receives raw
  image bytes as JSON — it fetches the frame through `GET /api/jobs/{id}/frame`, which
  streams the PNG file directly.
- **Cache**: unchanged from the CLI — `prepare_asset()` in `ingest.py` still keys its
  download cache by a hash of the video URL under `./.cache` (or `$QUEST1_WORK_DIR`),
  and a job re-run against an already-cached video reuses it instead of re-downloading,
  exactly as before. The API doesn't touch this logic at all.

### Local development

Two processes, run in separate terminals from the repo root:

```bash
# Terminal 1 -- backend (same venv as the CLI; see Quick Start (Local Fallback) above)
pip install -r requirements.txt   # now also installs fastapi/uvicorn
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# Terminal 2 -- frontend
cd web
cp .env.example .env.local        # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Open `http://localhost:3000`. The backend's default `ALLOWED_ORIGINS` already includes
`http://localhost:3000`, so no CORS setup is needed for local use.

### About deploying the frontend to Vercel

The frontend *can* still be deployed to Vercel as a static Next.js site — nothing stops
that. But be clear about what that does and doesn't get you:

- **Vercel = frontend only.** The actual video processing (`yt-dlp`, ffmpeg, Whisper,
  PaddleOCR) is not something that belongs in a Vercel serverless function — it's a
  long-running, CPU-heavy local process, and Vercel's functions aren't built for that.
- **A Vercel-hosted frontend cannot reach a backend running on your own computer.**
  `http://127.0.0.1:8000` means "this machine" to whoever's browser is loading the
  page — for someone else (or even you, on a different device) visiting your Vercel
  URL, that address resolves to *their* machine, not yours, and there's nothing
  running there. This isn't a bug to fix; it's what "local backend" means. Making a
  deployed frontend reach your machine would require exposing it to the internet
  (a tunnel like `ngrok`, port forwarding, or hosting the backend somewhere reachable
  — which is what the earlier Render-based deployment did, now removed per this
  project's current direction).
- **For local development and actually running searches**, run both pieces on the same
  machine as shown above — that's the supported path.

### Environment variables

| Var | Where | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `web/.env.example` | Backend base URL the browser calls (`http://127.0.0.1:8000` locally) |
| `ALLOWED_ORIGINS` | `.env.example` | Comma-separated frontend origins allowed by CORS (`http://localhost:3000` by default) |
| `QUEST1_SKIP_OCR` | `.env.example` | Force-disable OCR regardless of confidence (off by default — your machine runs the full pipeline) |
| `QUEST1_MODEL_SIZE` | `.env.example` | faster-whisper model size (default `small`; lower to `base` only if your machine is slow enough that it matters) |
| `QUEST1_MAX_CONCURRENT_JOBS` | `.env.example` | Background thread-pool size (default 2) |
| `QUEST1_OUTPUT_ROOT` | `.env.example` | Where per-job `report.json`/PNGs are written |
| `QUEST1_WORK_DIR` | `.env.example` | yt-dlp/audio cache dir (same meaning as the CLI's `--work-dir`) |

No secrets/API keys are required anywhere in this project — nothing above should ever
hold a real credential.

### Limitations (documented, not silently ignored)

- **No process restart durability**: a job's status lives in memory; if you stop or
  restart the `uvicorn` process mid-job, that job's progress is lost. The frontend
  handles this gracefully (see "Job states" above) rather than hanging, but the job
  itself has to be re-submitted.
- **Single process only** — the in-memory job store does not span multiple `uvicorn`
  worker processes. Not a real constraint for one person on one machine.
- **A Vercel-hosted frontend can't reach your local backend** — see above. This is a
  fundamental property of "local backend," not a missing feature.

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
APPROACH.md       Design rationale, trade-offs, and known limitations
PHASES.md         Original phase specification
PHASES_1_7_PLAN.md  Detailed per-phase implementation plan (historical)
prompts.txt       Full AI prompt log, including real bugs found via generalization testing
```
