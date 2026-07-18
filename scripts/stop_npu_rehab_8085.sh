#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="runtime/npu/pids/npu_rehab_8085.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "[OK] No recorded NPU 8085 process."
  exit 0
fi

pid="$(tr -cd '0-9' < "$PID_FILE")"
if [[ -z "$pid" ]]; then
  echo "[WARN] Invalid PID file: $PID_FILE" >&2
  exit 1
fi

if [[ ! -r "/proc/$pid/cmdline" ]]; then
  echo "[OK] NPU 8085 process is already stopped."
  rm -f "$PID_FILE"
  exit 0
fi

if ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Eq 'rehab_app/server/npu_rehab_server.py|rehab_app\.server\.npu_rehab_server'; then
  echo "[FAIL] PID $pid is not the NPU 8085 entrypoint; refusing to stop it." >&2
  exit 2
fi

kill "$pid"
for _ in $(seq 1 50); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done

if kill -0 "$pid" 2>/dev/null; then
  echo "[FAIL] NPU 8085 did not stop in time." >&2
  exit 3
fi

rm -f "$PID_FILE"
echo "[OK] NPU 8085 stopped. Optional Qwen services were not modified."
