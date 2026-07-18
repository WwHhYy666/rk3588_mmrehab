#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SERVICE_NAME="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
DROPIN_DIR="/etc/systemd/system/${SERVICE_NAME}.d"
DROPIN_FILE="${DROPIN_DIR}/20-async-pipeline.conf"
PROJECT_DIR="${PROJECT_DIR:-/home/elf/project}"

case "$MODE" in
  async|1)
    VALUE=1
    ;;
  sync|0)
    VALUE=0
    ;;
  *)
    echo "Usage: $0 async|sync" >&2
    exit 2
    ;;
esac

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT
printf '[Service]\nEnvironment=RKNN_ASYNC_PIPELINE=%s\n' "$VALUE" >"$TMP_FILE"

sudo mkdir -p "$DROPIN_DIR"
sudo install -m 0644 "$TMP_FILE" "$DROPIN_FILE"
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE_NAME"

cd "$PROJECT_DIR"
./scripts/check_npu_rehab_8085.sh
echo "[OK] RKNN_ASYNC_PIPELINE=${VALUE} (${MODE})"
