FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FFMPEG_BIN=ffmpeg \
    FFPROBE_BIN=ffprobe

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Сервис запускается в собственном виртуальном окружении, чтобы изолировать зависимости WhisperX.
RUN python -m venv /opt/whisperx_venv \
    && /opt/whisperx_venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools \
    && /opt/whisperx_venv/bin/pip install --no-cache-dir whisperx grpcio protobuf python-dotenv

EXPOSE 50051

CMD ["/opt/whisperx_venv/bin/python", "/app/grpc_server.py"]
