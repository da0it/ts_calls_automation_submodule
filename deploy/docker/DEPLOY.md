# Docker Deployment

## Prerequisites

- Linux host with Docker Engine + Docker Compose plugin
- At least 16 GB RAM recommended (WhisperX + NeMo on CPU)
- Enough disk for model caches

## 1. Configure env

Edit these files:

- `configs/transcription.env`
- `configs/routing.env`
- `configs/ticket.env`
- `configs/orchestrator.env`
- `configs/notification.env`
- `configs/entity.env`

Important settings:

- `HF_TOKEN` for whisperx/pyannote path (if needed)
- `ASR_BACKEND` (`whisperx` or `faster`)
- `WHISPERX_DIARIZATION_BACKEND` (`nemo` or `pyannote`)

## 2. Build

```bash
docker compose build
```

## 3. Start

```bash
docker compose up -d
```

## 4. Verify

```bash
docker compose ps
curl http://localhost:8000/health
```

## 5. Test call processing

```bash
curl -s -X POST http://localhost:8000/api/v1/process-call \
  -F "audio=@services/transcription/dengi.mp3" | jq .
```

## 6. Logs

```bash
docker compose logs -f orchestrator
docker compose logs -f transcription
docker compose logs -f entity_extraction
```

## 7. Stop

```bash
docker compose down
```

With persistent postgres + model cache volumes:

```bash
docker compose down
docker volume ls | grep ts_calls_automation_submodule
```
