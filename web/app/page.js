"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLL_INTERVAL_MS = 2500;

const STATUS_MESSAGES = {
  queued: "Queued...",
  processing: "Processing video... this can take a while for longer videos.",
};

export default function Home() {
  const [videoUrl, setVideoUrl] = useState("");
  const [query, setQuery] = useState("");
  const [job, setJob] = useState(null); // latest /api/jobs/{id} response
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  const isBusy = submitting || (job && !["completed", "not_found", "failed"].includes(job.status));

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitError(null);
    setJob(null);
    clearInterval(pollRef.current);

    if (!videoUrl.trim() || !query.trim()) {
      setSubmitError("Please enter both a video URL and a dialogue phrase.");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/api/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_url: videoUrl.trim(), query: query.trim() }),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(body?.detail || `Request failed (${res.status})`);
      }
      setJob(body);
      pollRef.current = setInterval(() => pollJob(body.job_id), POLL_INTERVAL_MS);
    } catch (err) {
      setSubmitError(
        err instanceof TypeError
          ? "Could not reach the backend. Is the API server running and reachable?"
          : err.message
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(`${API_URL}/api/jobs/${jobId}`);
      if (res.status === 404) {
        // Unlike a 500 or a network blip, a 404 here means this exact job_id will
        // never come back (e.g. a free-tier backend instance restarted mid-job and
        // lost its in-memory job store -- see api/jobs.py's docstring). Retrying
        // forever would leave the user staring at "Processing..." indefinitely, so
        // stop polling and surface it as a real failure instead of a silent hang.
        clearInterval(pollRef.current);
        setJob({
          status: "failed",
          error: "Lost track of this job -- the server may have restarted while it was running. Please try again.",
        });
        return;
      }
      if (!res.ok) return; // other transient network blip -- try again on the next tick
      const body = await res.json();
      setJob(body);
      if (["completed", "not_found", "failed"].includes(body.status)) {
        clearInterval(pollRef.current);
      }
    } catch {
      // transient network blip -- next poll tick will retry
    }
  }

  function handleReset() {
    clearInterval(pollRef.current);
    setJob(null);
    setSubmitError(null);
    setVideoUrl("");
    setQuery("");
  }

  return (
    <main className="page">
      <div className="card">
        <h1>Video Dialogue Finder</h1>
        <p className="subtitle">
          Paste a video URL and the line of dialogue you're looking for.
        </p>

        <form onSubmit={handleSubmit} className="form">
          <label className="field">
            <span>Video URL</span>
            <input
              type="text"
              placeholder="https://youtube.com/..."
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              disabled={isBusy}
            />
          </label>

          <label className="field">
            <span>What are you looking for?</span>
            <input
              type="text"
              placeholder="My mind rebels at stagnation"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isBusy}
            />
          </label>

          <button type="submit" disabled={isBusy}>
            {isBusy ? "Searching..." : "Find Dialogue"}
          </button>
        </form>

        {submitError && <p className="error">{submitError}</p>}

        {job && <JobStatus job={job} onReset={handleReset} />}
      </div>
    </main>
  );
}

function JobStatus({ job, onReset }) {
  if (job.status === "queued" || job.status === "processing") {
    return (
      <div className="status">
        <div className="spinner" />
        <p>{STATUS_MESSAGES[job.status]}</p>
      </div>
    );
  }

  if (job.status === "not_found") {
    return (
      <div className="status">
        <p>No match found for that phrase in this video.</p>
        <button className="secondary" onClick={onReset}>
          Search again
        </button>
      </div>
    );
  }

  if (job.status === "failed") {
    return (
      <div className="status">
        <p className="error">{job.error || "Something went wrong while processing this video."}</p>
        <button className="secondary" onClick={onReset}>
          Search again
        </button>
      </div>
    );
  }

  // completed
  return (
    <div className="result">
      <h2>Match found</h2>
      <p className="timestamp">{job.timestamp_formatted}</p>
      <p className="matched-text">&ldquo;{job.matched_text}&rdquo;</p>
      <p className="meta">
        source: {job.source} &middot; frame {job.frame_number} &middot; score{" "}
        {job.match_score != null ? job.match_score.toFixed(2) : "n/a"}
      </p>
      {job.frame_url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="frame" src={`${API_URL}${job.frame_url}`} alt="Matched frame" />
      )}
      <button className="secondary" onClick={onReset}>
        Search again
      </button>
    </div>
  );
}
