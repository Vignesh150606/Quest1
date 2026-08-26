"""
In-process job store + background worker for the local web API.

Deliberately the simplest thing that's reliable for a single personal computer: a dict
guarded by a lock, and a small ThreadPoolExecutor. No Redis, no Celery/RQ, no Docker --
this is a single Python process running on the same machine as the pipeline itself, so
there's nothing to coordinate across. The one hard constraint this design implies: the
API process must run as a single `uvicorn` worker (the default when you just run
`uvicorn api.app:app`, no `--workers N`), since a second worker process would have its
own, disconnected copy of _JOBS and never see jobs created by the first.

If this ever needed to become a multi-machine or multi-process service, swapping to a
real queue would mean replacing create_job()'s executor.submit() call with e.g. an RQ
enqueue() and _JOBS's dict with a Redis hash -- get_job()'s and the API layer's shape
(job_id -> status dict) doesn't need to change. Not needed for local use.
"""

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.main import run_pipeline

logger = logging.getLogger("api.jobs")

# Small on purpose: each job is itself CPU-heavy (ASR/OCR), so a handful of concurrent
# jobs on one machine would just thrash for CPU rather than finish any faster. 2 lets
# one job process while another finishes queueing/downloading without the whole app
# feeling wedged. Raise via QUEST1_MAX_CONCURRENT_JOBS if your machine has cores to spare.
_MAX_CONCURRENT_JOBS = int(os.environ.get("QUEST1_MAX_CONCURRENT_JOBS", "2"))
_executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_JOBS, thread_name_prefix="job")

_OUTPUT_ROOT = os.environ.get("QUEST1_OUTPUT_ROOT", "./output")
_WORK_DIR = os.environ.get("QUEST1_WORK_DIR")  # None -> ingest.py's own default
_SKIP_OCR = os.environ.get("QUEST1_SKIP_OCR", "false").lower() in ("1", "true", "yes")
# faster-whisper model size (same meaning as the CLI's --model). Left at the pipeline's
# own "small" default unless overridden -- asr_track.py's own docstring documents
# "small" as a deliberate accuracy choice over "base". QUEST1_MODEL_SIZE is a real,
# explicit lever if you want to trade accuracy for speed on a slower machine -- not
# lowered by default.
_MODEL_SIZE = os.environ.get("QUEST1_MODEL_SIZE", "small")

_lock = threading.Lock()
_JOBS: dict[str, dict] = {}

_TERMINAL_STATUSES = {"completed", "not_found", "failed"}


def create_job(video_url: str, query: str) -> str:
    """Register a queued job and submit it to the background executor. Returns job_id."""
    job_id = uuid.uuid4().hex
    with _lock:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "video_url": video_url,
            "query": query,
            "timestamp": None,
            "timestamp_formatted": None,
            "frame_number": None,
            "frame_url": None,
            "matched_text": None,
            "source": None,
            "match_score": None,
            "candidates": None,
            "error": None,
            "_image_path": None,
        }
    _executor.submit(_run_job, job_id, video_url, query)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Public job view -- excludes internal (`_`-prefixed) bookkeeping fields."""
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return {k: v for k, v in job.items() if not k.startswith("_")}


def _set(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def _run_job(job_id: str, video_url: str, query: str) -> None:
    _set(job_id, status="processing")
    output_dir = os.path.join(_OUTPUT_ROOT, job_id)

    try:
        import json

        report_path = run_pipeline(
            video_url,
            query,
            output_dir,
            model_size=_MODEL_SIZE,
            work_dir=_WORK_DIR,
            skip_ocr=_SKIP_OCR,
        )
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        # Full traceback server-side only -- CLAUDE.md/spec: never expose a stack trace
        # to the caller. str(exc) is safe to surface: every exception the pipeline
        # itself raises (ingest.py's RuntimeErrors, etc.) is already an intentional,
        # human-readable message -- the same text main.py's CLI path prints as
        # "Error: {exc}" -- not an internal implementation detail.
        logger.exception("Job %s failed", job_id)
        _set(job_id, status="failed", error=str(exc))
        return

    status = report["status"]  # "match" | "ambiguous" | "not_found"
    if status == "not_found":
        _set(job_id, status="not_found")
        return

    frame_number = report["frame"]
    _set(
        job_id,
        status="completed",
        timestamp=report["timestamp_s"],
        timestamp_formatted=report["timestamp"],
        frame_number=frame_number,
        frame_url=f"/api/jobs/{job_id}/frame" if frame_number is not None else None,
        matched_text=report["extracted_text"],
        source=report["modality"],
        match_score=report["match_score"],
        candidates=report["candidates"] or None,
        _image_path=report["image_path"],
    )


def frame_image_path(job_id: str) -> Optional[str]:
    """Path to the winning frame's PNG for this job, if the job produced one."""
    with _lock:
        job = _JOBS.get(job_id)
        return job["_image_path"] if job is not None else None
