#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode_file="${REHAB_MODE_FILE:-runtime/selected_rehab_mode}"
mode="$(tr -d ' \t\r\n' < "$mode_file" 2>/dev/null || true)"
case "$mode" in
  cpu_8082|npu_8085) ;;
  *) mode="npu_8085" ;;
esac

REHAB_SWITCH_SKIP_DISPLAY=1 exec bash scripts/switch_rehab_mode.sh "$mode"
