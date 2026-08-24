# Phase Specification

Produced by a separate Claude session (the "Phase Architect" prompt) working from
CLAUDE.md and the problem statement PDF. Kept here verbatim as ongoing reference
alongside CLAUDE.md.

## Phase 0 - Environment sanity
Goal: Verify the full toolchain (deps, ffmpeg, PaddleOCR/videocr-PaddleOCR version pin) installs and imports cleanly before pipeline code exists.
Consumes: nothing.
Produces: requirements.txt with pinned versions; passing environment-check test.
Files touched: requirements.txt, tests/test_environment.py
Verification: `pytest tests/test_environment.py::test_toolchain` — imports `faster_whisper`, `paddleocr`, `videocr` (pinned wrapper), `av`, `rapidfuzz` without error; `ffmpeg -version` and `ffprobe -version` exit 0.
Est. effort: S

## Phase 1 - Ingest + shared schema
Note: Candidate schema moved here from the draft's Phase 4 — Phase 1's subtitle fast-path must construct a Candidate(modality="subtitle"), and Phases 2/3 both produce List[Candidate], so the schema must exist before any phase that uses it, not after.
Goal: Resolve a video URL into a local VideoAsset, and short-circuit via a platform subtitle/CC track when one exists and matches.
Consumes: video URL (str), target phrase (str, CLI arg).
Produces: VideoAsset(video_path, audio_path, metadata); optional Candidate(modality="subtitle") on fast-path hit; src/types.py defining VideoAsset, VideoMetadata, Candidate (modality, event_type, timestamp, end_timestamp, matched_text, normalized_text, similarity, confidence, evidence).
Files touched: src/ingest.py, src/types.py, tests/test_ingest.py
Verification: `pytest tests/test_ingest.py::test_prepare_asset` (parametrized over the ok.ru URL + one other domain) — asserts `metadata.fps > 0`, `duration_s` within expected range, `audio_path` exists and is non-empty.
Est. effort: M

## Phase 2 - ASR track
Goal: Transcribe audio with word-level timestamps and fuzzy-match the target phrase into speech_onset candidates.
Consumes: VideoAsset.audio_path (Phase 1), target phrase, Candidate schema (Phase 1).
Produces: List[Candidate], modality="asr", event_type="speech_onset".
Files touched: src/asr_track.py, tests/test_asr_track.py
Verification: `pytest tests/test_asr_track.py::test_example_video` — asserts a returned Candidate has `similarity >= threshold` against "My mind rebels at stagnation" and `timestamp` within ±2s of the manually-verified ~05:27–05:29 window.
Est. effort: M

## Phase 3 - OCR track
Goal: Sample frames, detect candidate text regions, OCR them, and fuzzy-match the target phrase into visual_text_onset candidates.
Consumes: VideoAsset.video_path + metadata (Phase 1), target phrase, Candidate schema (Phase 1).
Produces: List[Candidate], modality="ocr", event_type="visual_text_onset"; synthetic captioned test clip fixture.
Files touched: src/ocr_track.py, tests/fixtures/make_synthetic_clip.py, tests/test_ocr_track.py
Verification: `pytest tests/test_ocr_track.py::test_synthetic_clip` — asserts the returned Candidate's `similarity >= threshold` and `timestamp` falls within sample-interval tolerance of the clip's known text-onset time.
Est. effort: L

## Phase 4 - Arbiter
Note: scoped to reconciliation logic only — Candidate schema moved to Phase 1; this phase consumes the schema rather than defining it.
Goal: Deterministically reconcile subtitle/ASR/OCR candidates into a winning result or an explicit ambiguous/none outcome.
Consumes: List[Candidate] from Phase 1 (subtitle), Phase 2 (ASR), Phase 3 (OCR).
Produces: winning Candidate(s) | AmbiguousResult | None.
Files touched: src/arbiter.py, tests/test_arbiter.py
Verification: `pytest tests/test_arbiter.py::test_reconciliation_policy` (parametrized: single-track hit → returns it; agreeing multi-track candidates within tolerance → returns higher-confidence one; disagreeing candidates outside tolerance → returns AmbiguousResult, never a silent pick).
Est. effort: M

## Phase 5 - Refine
Goal: Map the arbiter's winning Candidate(s) to an exact, decode-accurate video frame via PyAV, per event-type-specific policy.
Consumes: winning Candidate(s) (Phase 4), VideoAsset (Phase 1).
Produces: FrameMatch(frame_idx, timestamp_s, text, image_path, modality, match_score) per winning candidate.
Files touched: src/refine.py, tests/test_refine.py
Verification: `pytest tests/test_refine.py::test_frame_accuracy` (parametrized over one speech_onset and one visual_text_onset fixture) — asserts `frame_idx == expected_frame_idx` (or within documented tolerance if PyAV decode proves non-deterministic).
Est. effort: M

## Phase 6 - Report + CLI integration
Goal: Wire ingest → ASR/OCR → arbiter → refine into one runnable end-to-end command emitting the required output formats.
Consumes: FrameMatch(es) or AmbiguousResult/None (Phase 4/5), VideoAsset (Phase 1).
Produces: report.json, extracted frame PNG(s), formatted stdout (Timestamp/Frame/Text); src/main.py CLI entry point.
Files touched: src/main.py, src/report.py, tests/test_end_to_end.py
Verification: `pytest tests/test_end_to_end.py::test_full_run` (parametrized: example URL + phrase → report.json has expected frame/timestamp within tolerance and PNG exists on disk; zero-candidate input → report outcome is "not_found", no crash).
Est. effort: M

## Phase 7 - Packaging
Goal: Make the solution runnable by someone else via Docker or a documented local fallback, with all required repo artifacts finalized.
Consumes: working src/ pipeline (Phase 6).
Produces: Dockerfile, README.md, APPROACH.md, final prompts.txt.
Files touched: Dockerfile, README.md, APPROACH.md, prompts.txt
Verification: `docker build . && docker run <image> --url <ok.ru URL> --dialogue-text "..."` exits 0 and writes report.json + PNG to the mounted output dir; README's local-fallback steps reproduce the same result without Docker.
Est. effort: S
