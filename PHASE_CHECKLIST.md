# Phase Checklist

- [x] **Phase 0** - Environment sanity — `python verify.py 0`
- [x] **Phase 1** - Ingest + shared schema — `src/ingest.py`, `python verify.py 1`
- [x] **Phase 2** - ASR track — `src/asr_track.py`, `python verify.py 2`
- [x] **Phase 3** - OCR track — `src/ocr_track.py` + `tests/fixtures/make_synthetic_clip.py`, `python verify.py 3`
- [x] **Phase 4** - Arbiter — `src/arbiter.py`, `python verify.py 4`
- [x] **Phase 5** - Refine — `src/refine.py`, `python verify.py 5`
- [x] **Phase 6** - Report + CLI integration — `src/report.py` + `run_pipeline()` in `src/main.py`, `python verify.py 6`
- [x] **Phase 7** - Packaging — `docker build -t quest1-solver . && docker run --rm -v $(pwd)/output:/output quest1-solver --url <ok.ru URL> --dialogue-text "..."`, confirmed README's local-fallback steps reproduce the same result without Docker

All phases verified with `python verify.py all`. See `prompts.txt` for the real bugs
found and fixed after each phase's initial "done" (subtitle-track edge cases, OCR/subtitle
false-positive matching, runtime hardening) -- generalization testing against real,
non-example videos kept surfacing issues no synthetic fixture or the single graded
example could have caught.
