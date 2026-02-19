# Linux Deployment

## 1. Prepare machine (Ubuntu example)

```bash
sudo bash scripts/install_deps_ubuntu.sh
```

Install Docker separately if you want postgres from `docker-compose.yml`.

## 2. Clone project

```bash
sudo mkdir -p /opt
sudo chown -R "$USER":"$USER" /opt
cd /opt
git clone <YOUR_REPO_URL> ts_calls_automation_submodule
cd ts_calls_automation_submodule
```

## 3. Configure env files

Edit:

- `configs/transcription.env`
- `configs/routing.env`
- `configs/ticket.env`
- `configs/orchestrator.env`
- `configs/notification.env`
- `configs/entity.env`

Important:

- Set real `HF_TOKEN` for whisperx/pyannote mode.
- Ensure `DATABASE_URL` points to postgres.
- If not using docker postgres, set `START_POSTGRES_WITH_DOCKER=0`.

## 4. Bootstrap runtimes

```bash
./scripts/bootstrap_linux.sh
```

This script creates:

- `.venv` for entity extraction
- `services/router/venv`
- `~/whisperx_venv` (WhisperX runtime)
- optional `~/whisper-diarization` only if `INSTALL_NEMO_BACKEND=1`
- installs Python/Go dependencies

## 5. Smoke run

```bash
./scripts/start_linux_stack.sh
```

Health check:

```bash
curl http://localhost:8000/health
```

## 6. Run with systemd

1. Edit `deploy/linux/ts-calls.service`:
   - set `User`
   - set `WorkingDirectory`
   - set `HOME`
   - adjust `START_POSTGRES_WITH_DOCKER`

2. Install and enable:

```bash
sudo cp deploy/linux/ts-calls.service /etc/systemd/system/ts-calls.service
sudo systemctl daemon-reload
sudo systemctl enable --now ts-calls.service
```

3. Logs:

```bash
journalctl -u ts-calls.service -f
tail -f .run/logs/orchestrator.log
```

## 7. Update deployment

```bash
cd /opt/ts_calls_automation_submodule
git pull
./scripts/bootstrap_linux.sh
sudo systemctl restart ts-calls.service
```
