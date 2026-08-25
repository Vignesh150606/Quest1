# Find the Exact Frame Where a Dialogue Appears in a Media URL

<!-- One or two sentences: what this does, restated in your own words. -->

## What This Does

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
is ephemeral by default -- runtime-audit finding. With them, only the first run pays
that cost; later runs (including against the same video with different dialogue text)
reuse everything.

## Quick Start (Local Fallback)

<!-- Exact system dependencies (ffmpeg, OCR system libs) and pip install steps --
     this path matters as much as Docker; see CLAUDE.md's "Docker: primary path,
     not the only path". -->

## Usage

```
python -m src.main --url <video_url> --dialogue-text "<target phrase>" --output <dir>
```

## Output Format

<!-- report.json shape, PNG naming, stdout format -->

## Running Tests

```
python verify.py <phase_number>   # or "all"
```

<!-- See PHASE_CHECKLIST.md for the phase-by-phase order. -->

## Repository Structure
