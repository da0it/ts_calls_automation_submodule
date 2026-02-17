#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_POSTGRES_WITH_DOCKER="${START_POSTGRES_WITH_DOCKER:-1}"

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

if [[ "$START_POSTGRES_WITH_DOCKER" == "1" ]]; then
  if command -v docker >/dev/null 2>&1; then
    log "Starting postgres via docker compose..."
    docker compose -f "$ROOT_DIR/docker-compose.yml" up -d postgres
  else
    log "[WARN] docker not found, skipping postgres startup."
  fi
fi

exec "$ROOT_DIR/scripts/run_all.sh"
