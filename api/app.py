"""
Thin FastAPI wrapper around the existing pipeline (src/main.run_pipeline). See api/jobs.py
for the job store/background-worker design. This layer's only job is: validate input,
create/read jobs, serve the winning frame image -- it must never duplicate pipeline logic.

Run locally:
    uvicorn api.app:app --reload --port 8000
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.jobs import create_job, frame_image_path, get_job

app = FastAPI(title="Quest1 Video Dialogue Finder API")

# Comma-separated list of allowed frontend origins, e.g. "https://myapp.vercel.app".
# Defaults to the local Next.js dev server so `npm run dev` + `uvicorn` work together
# out of the box; production deployments must set ALLOWED_ORIGINS explicitly (see
# render.yaml / README) -- allow_origins=["*"] is deliberately not used here since a
# real origin can be configured.
_allowed_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class JobCreateRequest(BaseModel):
    video_url: str = Field(min_length=1)
    query: str = Field(min_length=1)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/jobs", status_code=201)
def post_job(body: JobCreateRequest):
    video_url = body.video_url.strip()
    query = body.query.strip()
    if not video_url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="video_url must be an http(s) URL")
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    job_id = create_job(video_url, query)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/jobs/{job_id}/frame")
def get_job_frame(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = frame_image_path(job_id)
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no frame available for this job")
    return FileResponse(path, media_type="image/png")
