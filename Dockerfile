# Phase 7 - Packaging skeleton. Fill in / adjust as the real dependency set is proven
# out during Phases 0-6 -- e.g. paddleocr/paddlepaddle may need extra system libs
# beyond libgl1 depending on the final OCR approach chosen in Phase 3.
#
# python:3.12-slim (not 3.11): requirements.txt's own header states its pins were
# verified against Python 3.12.10 -- runtime-audit finding, previously mismatched.

FROM python:3.12-slim

# System dependencies: ffmpeg for audio/frame extraction, libgl1 for OpenCV/PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Known limitation, found by actually building and running this image (not assumed):
# PaddleOCR inference crashes inside this containerized environment (Docker Desktop's
# WSL2 VM) with "could not create a primitive descriptor for a reorder primitive" --
# the identical code runs correctly natively on the same physical machine, and the
# rest of the pipeline (subtitle fast-path, ASR, arbiter, refine, report) runs
# correctly in Docker too -- verified with a real end-to-end run, exit 0, correct
# match. Multiple GitHub issues report this exact error against PaddlePaddle/PaddleOCR
# with no confirmed fix as of this writing. Two targeted env-var workarounds below were
# tried and verified NOT to fix it (rebuilt + re-ran after each, identical crash) --
# kept anyway since they're harmless and may help on a different host's virtualization
# stack even though they didn't here: DNNL_MAX_CPU_ISA/ONEDNN_MAX_CPU_ISA cap the CPU
# instruction set oneDNN is allowed to detect/use (its own documented mechanism for
# exactly this class of problem); FLAGS_use_mkldnn is PaddlePaddle's own, separate flag
# for the same underlying acceleration. See README.md's "Known limitation" note for the
# user-facing workaround (--skip-ocr in Docker, or the local Python fallback for a case
# that genuinely needs OCR).
ENV DNNL_MAX_CPU_ISA=AVX2
ENV ONEDNN_MAX_CPU_ISA=AVX2
ENV FLAGS_use_mkldnn=0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.main"]
