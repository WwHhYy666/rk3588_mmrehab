#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! curl -fsS http://127.0.0.1:8085/status >/dev/null 2>&1; then
  echo "[FAIL] NPU rehab 8085 is not running." >&2
  echo "Start it first with: ./scripts/start_npu_rehab_8085.sh" >&2
  exit 2
fi

export REHAB_STATION_URL="http://127.0.0.1:8085/npu-debug"
export REHAB_BROWSER_PROFILE_DIR="${REHAB_BROWSER_PROFILE_DIR:-${HOME}/.cache/npu-debug-8085-browser}"
exec bash scripts/open_npu_rehab_8085_kiosk.sh
