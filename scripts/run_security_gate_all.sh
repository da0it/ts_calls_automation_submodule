#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATUS=0

echo "[security-gate] SAST"
python3 "${ROOT_DIR}/security_gate.py" \
  --threshold high \
  --bandit "${ROOT_DIR}/reports/sast/bandit-report.json" \
  --gosec "${ROOT_DIR}/reports/sast/gosec-orchestrator.json" \
  --gosec "${ROOT_DIR}/reports/sast/gosec-ticket_creation.json" \
  --gosec "${ROOT_DIR}/reports/sast/gosec-notification_sender.json" || STATUS=1

echo "[security-gate] SCA"
python3 "${ROOT_DIR}/security_gate.py" \
  --threshold critical \
  --policy "${ROOT_DIR}/security_policy.yaml" \
  --trivy "${ROOT_DIR}/reports/sca/trivy-report.json" || STATUS=1

exit "${STATUS}"
