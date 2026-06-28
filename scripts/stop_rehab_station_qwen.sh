#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PID_DIR="$(pwd)/runtime/pids"

stop_pid_file() {
  local name="$1"
  local file="$2"
  if [[ ! -f "$file" ]]; then
    return
  fi
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "[STOP] ${name} pid ${pid}"
    kill "$pid" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      echo "[KILL] ${name} pid ${pid}"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "$file"
}

stop_pattern() {
  local name="$1"
  local pattern="$2"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "[STOP] ${name}"
    pkill -f "$pattern" >/dev/null 2>&1 || true
  fi
}

stop_pid_file "Qwen Flask server" "${PID_DIR}/qwen_flask.pid"
stop_pid_file "RKLLM proxy" "${PID_DIR}/qwen_proxy.pid"

stop_pattern "Rehab 8082 service" "prescription/banzi/record_prescription_http.py"
stop_pattern "RKLLM proxy fallback" "llm/rkllm_proxy_server.py"
stop_pattern "Qwen Flask fallback" "/home/elf/qwen_server/rkllm_server/flask_server.py"

echo "[OK] Rehab station Qwen services stopped."
