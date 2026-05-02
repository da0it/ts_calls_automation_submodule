#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/proto"
PROTO_FILE="call_processing.proto"
GO_PLUGIN_BIN="${GO_PLUGIN_BIN:-$ROOT_DIR/.bin}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-$PYTHON_BIN -m pip}"

GO_OUT_DIRS=(
  "$ROOT_DIR/services/orchestrator/internal/gen"
  "$ROOT_DIR/services/ticket_creation/internal/gen"
)

PY_OUT_DIRS=(
  "$ROOT_DIR/services/router/grpc_gen"
  "$ROOT_DIR/services/transcription/grpc_gen"
)

log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $cmd" >&2
    exit 1
  fi
}

ensure_python_grpc_tools() {
  if "$PYTHON_BIN" -c "import grpc_tools.protoc" >/dev/null 2>&1; then
    return 0
  fi
  if command -v grpc_python_plugin >/dev/null 2>&1; then
    log "grpcio-tools is unavailable; falling back to protoc + grpc_python_plugin"
    return 1
  fi
  log "grpcio-tools and grpc_python_plugin are unavailable; Python gRPC stubs will not be regenerated"
  return 2
}

ensure_go_plugin() {
  local binary="$1"
  local package_ref="$2"
  if command -v "$binary" >/dev/null 2>&1; then
    return
  fi
  mkdir -p "$GO_PLUGIN_BIN"
  log "Installing $binary into $GO_PLUGIN_BIN..."
  GOBIN="$GO_PLUGIN_BIN" go install "$package_ref"
}

generate_go_stubs() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  log "Generating Go stubs -> $out_dir"
  (
    cd "$PROTO_DIR"
    protoc \
      --go_out=paths=source_relative:"$out_dir" \
      --go-grpc_out=paths=source_relative:"$out_dir" \
      "$PROTO_FILE"
  )
}

generate_python_stubs() {
  local out_dir="$1"
  mkdir -p "$out_dir"
  : >"$out_dir/__init__.py"
  log "Generating Python stubs -> $out_dir"
  if "$PYTHON_BIN" -c "import grpc_tools.protoc" >/dev/null 2>&1; then
    (
      cd "$PROTO_DIR"
      "$PYTHON_BIN" -m grpc_tools.protoc \
        -I . \
        --python_out="$out_dir" \
        --grpc_python_out="$out_dir" \
        "$PROTO_FILE"
    )
    return
  fi
  if command -v grpc_python_plugin >/dev/null 2>&1; then
    (
      cd "$PROTO_DIR"
      protoc \
        -I . \
        --python_out="$out_dir" \
        --grpc_python_out="$out_dir" \
        --plugin=protoc-gen-grpc_python="$(command -v grpc_python_plugin)" \
        "$PROTO_FILE"
    )
    return
  fi
  (
    cd "$PROTO_DIR"
    protoc \
      -I . \
      --python_out="$out_dir" \
      "$PROTO_FILE"
  )
}

main() {
  require_cmd "$PYTHON_BIN"
  require_cmd go
  require_cmd protoc

  export PATH="$GO_PLUGIN_BIN:$PATH"

  ensure_python_grpc_tools || true
  ensure_go_plugin "protoc-gen-go" "google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11"
  ensure_go_plugin "protoc-gen-go-grpc" "google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1"

  if [[ ! -f "$PROTO_DIR/$PROTO_FILE" ]]; then
    echo "[ERROR] Proto file not found: $PROTO_DIR/$PROTO_FILE" >&2
    exit 1
  fi

  log "Step 1/3: generating Go protobuf code"
  for out_dir in "${GO_OUT_DIRS[@]}"; do
    generate_go_stubs "$out_dir"
  done

  log "Step 2/3: generating Python protobuf code"
  for out_dir in "${PY_OUT_DIRS[@]}"; do
    generate_python_stubs "$out_dir"
  done

  log "Step 3/3: done"
  log "Generated files are ready. Next step: git add generated dirs or rerun docker compose build."
}

main "$@"
