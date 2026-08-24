# Phase 7 - Packaging skeleton. Fill in / adjust as the real dependency set is proven
# out during Phases 0-6 -- e.g. paddleocr/paddlepaddle may need extra system libs
# beyond libgl1 depending on the final OCR approach chosen in Phase 3.

FROM python:3.11-slim

# System dependencies: ffmpeg for audio/frame extraction, libgl1 for OpenCV/PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.main"]
