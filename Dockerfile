FROM oven/bun:latest AS frontend-builder

WORKDIR /frontend

COPY frontend/ .

RUN bun install
RUN bun run build

# Use Python 3.11 slim base image
FROM python:3.11-slim

# Install system dependencies for audio processing and PyTorch
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch optimized for CPU (since MPS is not available in Docker on macOS)
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Pre-download the faster-whisper 'tiny' model to avoid runtime downloads
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', local_files_only=False)"

# Verify the model is cached
RUN ls -lh /root/.cache/faster_whisper || echo "Model cache directory not found"

# Copy application code
COPY main.py .
COPY models/ ./models
COPY nltk_data/ ./nltk_data
COPY celery_worker.py .
COPY .env .
COPY env .
COPY models.py .
# Copy the built frontend
COPY --from=frontend-builder /frontend/build/ ./static

# Default command (will be overridden in docker-compose.yml for Celery worker)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]