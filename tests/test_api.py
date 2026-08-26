"""
FastAPI layer tests (api/app.py, api/jobs.py).

Offline only -- run_pipeline() is monkeypatched so these never touch the network or a
real ASR/OCR model, consistent with this project's existing test philosophy (see
tests/test_end_to_end.py's synthetic-clip tests for the equivalent at the pipeline
layer; this file tests the HTTP/job-store layer sitting on top of it, not the pipeline
itself again).
"""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

import api.jobs as jobs_module
from api.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path, monkeypatch):
    # Each test gets its own output dir and a clean in-memory job store, so tests can't
    # see each other's jobs or files.
    monkeypatch.setattr(jobs_module, "_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setattr(jobs_module, "_JOBS", {})
    yield


def _wait_for_terminal(job_id, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = jobs_module.get_job(job_id)
        if job["status"] in ("completed", "not_found", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal status in {timeout_s}s")


def _fake_match_report(output_dir, **overrides):
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "frame_739.png")
    with open(image_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nfake")
    report = {
        "status": "match",
        "video_url": "http://example.com/v",
        "dialogue_text": "hello there",
        "timestamp": "00:00:24.633",
        "timestamp_s": 24.633,
        "frame": 739,
        "extracted_text": "hello there, general",
        "image_path": image_path,
        "modality": "asr",
        "match_score": 0.94,
        "candidates": [],
    }
    report.update(overrides)
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return report_path


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/jobs -- validation
# ---------------------------------------------------------------------------


def test_create_job_rejects_non_http_url():
    resp = client.post("/api/jobs", json={"video_url": "not-a-url", "query": "hi"})
    assert resp.status_code == 400


def test_create_job_rejects_empty_query():
    resp = client.post("/api/jobs", json={"video_url": "http://example.com/v", "query": "  "})
    assert resp.status_code == 400


def test_create_job_rejects_missing_fields():
    resp = client.post("/api/jobs", json={"video_url": "http://example.com/v"})
    assert resp.status_code == 422  # pydantic: query is required


# ---------------------------------------------------------------------------
# End-to-end job lifecycle -- run_pipeline monkeypatched, real thread pool
# ---------------------------------------------------------------------------


def test_job_completes_with_match(monkeypatch):
    def fake_run_pipeline(video_url, query, output_dir, **kwargs):
        return _fake_match_report(output_dir)

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    resp = client.post(
        "/api/jobs", json={"video_url": "http://example.com/v", "query": "hello there"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]

    job = _wait_for_terminal(job_id)
    assert job["status"] == "completed"
    assert job["timestamp"] == 24.633
    assert job["timestamp_formatted"] == "00:00:24.633"
    assert job["frame_number"] == 739
    assert job["frame_url"] == f"/api/jobs/{job_id}/frame"
    assert job["matched_text"] == "hello there, general"
    assert job["source"] == "asr"
    assert job["match_score"] == 0.94

    # Same shape via the actual HTTP endpoint, not just the in-process job store.
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
    assert "_image_path" not in resp.json()  # internal field must not leak

    resp = client.get(f"/api/jobs/{job_id}/frame")
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")


def test_job_reports_not_found(monkeypatch):
    def fake_run_pipeline(video_url, query, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "not_found",
                    "video_url": video_url,
                    "dialogue_text": query,
                    "timestamp": None,
                    "timestamp_s": None,
                    "frame": None,
                    "extracted_text": None,
                    "image_path": None,
                    "modality": None,
                    "match_score": None,
                    "candidates": [],
                },
                f,
            )
        return report_path

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    resp = client.post(
        "/api/jobs", json={"video_url": "http://example.com/v", "query": "nope"}
    )
    job_id = resp.json()["job_id"]

    job = _wait_for_terminal(job_id)
    assert job["status"] == "not_found"
    assert job["frame_number"] is None

    # No frame was ever produced -- the frame endpoint must 404, not error.
    resp = client.get(f"/api/jobs/{job_id}/frame")
    assert resp.status_code == 404


def test_job_reports_failed_without_leaking_a_traceback(monkeypatch):
    def fake_run_pipeline(video_url, query, output_dir, **kwargs):
        raise RuntimeError("yt-dlp could not resolve this URL")

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    resp = client.post(
        "/api/jobs", json={"video_url": "http://example.com/v", "query": "hi"}
    )
    job_id = resp.json()["job_id"]

    job = _wait_for_terminal(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "yt-dlp could not resolve this URL"
    assert "Traceback" not in job["error"]


def test_get_unknown_job_is_404():
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_frame_for_unknown_job_is_404():
    resp = client.get("/api/jobs/does-not-exist/frame")
    assert resp.status_code == 404
