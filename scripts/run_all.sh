#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

declare -a SERVICE_NAMES=()
declare -a SERVICE_PIDS=()
CLEANED_UP=0

if [[ -x "$HOME/whisperx_venv/bin/python" ]]; then
  DEFAULT_TRANSCRIPTION_PYTHON="$HOME/whisperx_venv/bin/python"
elif [[ -x "$HOME/whisper-diarization/whisper_venv/bin/python" ]]; then
  # Backward-compatible fallback for existing local setups.
  DEFAULT_TRANSCRIPTION_PYTHON="$HOME/whisper-diarization/whisper_venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_TRANSCRIPTION_PYTHON="$ROOT_DIR/.venv/bin/python"
else
  DEFAULT_TRANSCRIPTION_PYTHON="python3"
fi

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_ENTITY_PYTHON="$ROOT_DIR/.venv/bin/python"
else
  DEFAULT_ENTITY_PYTHON="python3"
fi

TRANSCRIPTION_PYTHON="${TRANSCRIPTION_PYTHON:-$DEFAULT_TRANSCRIPTION_PYTHON}"
ENTITY_PYTHON="${ENTITY_PYTHON:-$DEFAULT_ENTITY_PYTHON}"
ROUTER_PYTHON="${ROUTER_PYTHON:-$ROOT_DIR/services/router/venv/bin/python}"
DATABASE_URL="${DATABASE_URL:-postgres://postgres:postgres@localhost:5432/tickets?sslmode=disable}"
SKIP_DB_MIGRATION="${SKIP_DB_MIGRATION:-0}"

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Command not found: $cmd"
    exit 1
  fi
}

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

wait_for_entity_ready() {
  local timeout_sec="${1:-120}"
  local url="${2:-http://localhost:5001/health}"
  local start_ts
  start_ts="$(date +%s)"

  if ! command -v curl >/dev/null 2>&1; then
    log "curl not found; skipping entity readiness probe."
    return
  fi

  while true; do
    if curl -sf "$url" | grep -q '"ready":[[:space:]]*true'; then
      log "Entity extraction is ready."
      return
    fi

    local now
    now="$(date +%s)"
    if (( now - start_ts >= timeout_sec )); then
      log "[WARN] Entity extraction readiness timeout after ${timeout_sec}s; continuing anyway."
      return
    fi

    sleep 2
  done
}

register_service() {
  local name="$1"
  local pid="$2"
  SERVICE_NAMES+=("$name")
  SERVICE_PIDS+=("$pid")
  echo "$pid" >"$PID_DIR/$name.pid"
}

cleanup() {
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  log "Stopping services..."
  local i
  for i in "${!SERVICE_PIDS[@]}"; do
    local pid="${SERVICE_PIDS[$i]}"
    local name="${SERVICE_NAMES[$i]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      log "Stopping $name (pid=$pid)"
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done

  sleep 1
  for i in "${!SERVICE_PIDS[@]}"; do
    local pid="${SERVICE_PIDS[$i]}"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done

  rm -f "$PID_DIR"/*.pid >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

load_env() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    # shellcheck disable=SC1090
    set -a && source "$env_file" && set +a
  fi
}

start_service() {
  local name="$1"
  local cwd="$2"
  local env_file="$3"
  local cmd="$4"
  local log_file="$LOG_DIR/$name.log"

  log "Starting $name..."
  (
    cd "$cwd"
    load_env "$env_file"
    exec bash -lc "$cmd"
  ) >"$log_file" 2>&1 &

  local pid=$!
  register_service "$name" "$pid"
  sleep 2

  if ! kill -0 "$pid" >/dev/null 2>&1; then
    log "[ERROR] $name failed to start. Tail of log:"
    tail -n 40 "$log_file" || true
    exit 1
  fi

  log "$name started (pid=$pid, log=$log_file)"
}

run_migration() {
  local migration_file="$ROOT_DIR/services/ticket_creation/migrations/001_create_tickets.up.sql"
  if [[ "$SKIP_DB_MIGRATION" == "1" ]]; then
    log "Skipping DB migration (SKIP_DB_MIGRATION=1)."
    return
  fi
  if ! command -v psql >/dev/null 2>&1; then
    log "Skipping DB migration: psql not found."
    return
  fi
  if [[ ! -f "$migration_file" ]]; then
    log "Skipping DB migration: migration file not found."
    return
  fi

  log "Applying DB migration..."
  psql "$DATABASE_URL" -f "$migration_file" >/dev/null
  log "DB migration applied."
}

check_python() {
  local python_bin="$1"
  local label="$2"
  if [[ ! -x "$python_bin" && "$python_bin" != "python3" ]]; then
    log "[ERROR] $label python not found: $python_bin"
    exit 1
  fi
}

main() {
  require_cmd go
  check_python "$TRANSCRIPTION_PYTHON" "transcription"
  check_python "$ENTITY_PYTHON" "entity_extraction"
  check_python "$ROUTER_PYTHON" "router"

  run_migration

  start_service \
    "transcription" \
    "$ROOT_DIR" \
    "$ROOT_DIR/configs/transcription.env" \
    "$TRANSCRIPTION_PYTHON $ROOT_DIR/services/transcription/grpc_server.py"

  start_service \
    "router" \
    "$ROOT_DIR" \
    "$ROOT_DIR/configs/routing.env" \
    "$ROUTER_PYTHON $ROOT_DIR/services/router/grpc_server.py"

  start_service \
    "entity_extraction" \
    "$ROOT_DIR/services/entity_extraction" \
    "$ROOT_DIR/configs/entity.env" \
    "$ENTITY_PYTHON $ROOT_DIR/services/entity_extraction/main.py"
  wait_for_entity_ready 120 "http://localhost:5001/health"

  start_service \
    "ticket_creation" \
    "$ROOT_DIR/services/ticket_creation" \
    "$ROOT_DIR/configs/ticket.env" \
    "DATABASE_URL='$DATABASE_URL' GOCACHE='$ROOT_DIR/.gocache' go run cmd/server/main.go"

  start_service \
    "notification_sender" \
    "$ROOT_DIR/services/notification_sender" \
    "$ROOT_DIR/configs/notification.env" \
    "GOCACHE='$ROOT_DIR/.gocache' go run cmd/server/main.go"

  start_service \
    "orchestrator" \
    "$ROOT_DIR/services/orchestrator" \
    "$ROOT_DIR/configs/orchestrator.env" \
    "GOCACHE='$ROOT_DIR/.gocache' go run cmd/server/main.go"

  log "All services are running."
  log "HTTP check: curl http://localhost:8000/health"
  log "E2E check: curl -X POST http://localhost:8000/api/v1/process-call -F \"audio=@$ROOT_DIR/services/transcription/dengi.mp3\""
  log "Press Ctrl+C to stop all services."

  while true; do
    local i
    for i in "${!SERVICE_PIDS[@]}"; do
      local pid="${SERVICE_PIDS[$i]}"
      local name="${SERVICE_NAMES[$i]}"
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        log "[ERROR] $name exited unexpectedly. Check $LOG_DIR/$name.log"
        exit 1
      fi
    done
    sleep 2
  done
}

main "$@"
