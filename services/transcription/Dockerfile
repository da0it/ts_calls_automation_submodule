FROM python:3.11-slim

ARG WHISPER_DIARIZATION_REPO=https://github.com/MahmoudAshraf97/whisper-diarization
ARG WHISPER_DIARIZATION_REF=main

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_REPO_DIR=/opt/whisper-diarization \
    WHISPER_VENV_PYTHON=/opt/whisper-diarization/whisper_venv/bin/python \
    WHISPERX_VENV_PYTHON=/opt/whisper-diarization/whisper_venv/bin/python \
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

# Service code
COPY . /app

# Whisper + diarization toolchain in isolated venv
RUN git clone --depth 1 --branch "${WHISPER_DIARIZATION_REF}" "${WHISPER_DIARIZATION_REPO}" /opt/whisper-diarization \
    && python -m venv /opt/whisper-diarization/whisper_venv \
    && /opt/whisper-diarization/whisper_venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools \
    && /opt/whisper-diarization/whisper_venv/bin/pip install --no-cache-dir -r /opt/whisper-diarization/requirements.txt \
    && /opt/whisper-diarization/whisper_venv/bin/pip install --no-cache-dir whisperx grpcio protobuf python-dotenv

EXPOSE 50051

CMD ["/opt/whisper-diarization/whisper_venv/bin/python", "/app/grpc_server.py"]
