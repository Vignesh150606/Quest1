# Phase Checklist

- [x] **Phase 0** - Environment sanity — `python verify.py 0`
- [x] **Phase 1** - Ingest + shared schema — implement `src/ingest.py`, remove skip in `tests/test_ingest.py`, then `python verify.py 1`
- [ ] **Phase 2** - ASR track — implement `src/asr_track.py`, remove skip in `tests/test_asr_track.py`, then `python verify.py 2`
- [ ] **Phase 3** - OCR track — implement `src/ocr_track.py` + `tests/fixtures/make_synthetic_clip.py`, remove skip in `tests/test_ocr_track.py`, then `python verify.py 3`
- [ ] **Phase 4** - Arbiter — implement `src/arbiter.py`, remove skip in `tests/test_arbiter.py`, then `python verify.py 4`
- [ ] **Phase 5** - Refine — implement `src/refine.py`, remove skip in `tests/test_refine.py`, then `python verify.py 5`
- [ ] **Phase 6** - Report + CLI integration — implement `src/report.py` + `run_pipeline()` in `src/main.py`, remove skips in `tests/test_end_to_end.py`, then `python verify.py 6`
- [ ] **Phase 7** - Packaging — `docker build -t quest1-solver . && docker run --rm -v $(pwd)/output:/output quest1-solver --url <ok.ru URL> --dialogue-text "..."`, then confirm README's local-fallback steps reproduce the same result without Docker
