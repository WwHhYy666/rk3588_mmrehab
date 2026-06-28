#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_DIR="${PROJECT_ROOT}/runtime/pids"
MODEL_PATH="${QWEN_RKLLM_MODEL:-/home/elf/models/qwen/qwen1_5b.rkllm}"
QWEN_SERVER_DIR="${QWEN_SERVER_DIR:-/home/elf/qwen_server/rkllm_server}"
QWEN_FLASK_PY="${QWEN_FLASK_PY:-${QWEN_SERVER_DIR}/flask_server.py}"
QWEN_RUNTIME_LIB="${QWEN_RUNTIME_LIB:-${QWEN_SERVER_DIR}/lib/librkllmrt.so}"
QWEN_FIX_FREQ="${QWEN_FIX_FREQ:-${QWEN_SERVER_DIR}/fix_freq_rk3588.sh}"
QWEN_FLASK_PORT="${QWEN_FLASK_PORT:-8080}"
RKLLM_PROXY_PORT="${RKLLM_PROXY_PORT:-18080}"
REHAB_PORT="${REHAB_PORT:-8082}"

mkdir -p "$LOG_DIR" "$PID_DIR"

load_llm_env() {
  local env_file
  for env_file in "${PROJECT_ROOT}/runtime/llm.env" "${PROJECT_ROOT}/.env.llm"; do
    if [[ -f "$env_file" ]]; then
      echo "[ENV] loading LLM env from ${env_file}"
      set -a
      # shellcheck disable=SC1090
      source "$env_file"
      set +a
    fi
  done
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[FAIL] Missing required file: $1" >&2
    exit 2
  fi
}

port_open() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

wait_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local tries="${4:-90}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if port_open "$host" "$port"; then
      echo "[OK] ${name} is listening on ${host}:${port}"
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] ${name} did not listen on ${host}:${port}" >&2
  return 1
}

wait_proxy_health() {
  local tries="${1:-30}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -fsS "http://127.0.0.1:${RKLLM_PROXY_PORT}/health" >/dev/null 2>&1; then
      echo "[OK] RKLLM proxy health is ok"
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] RKLLM proxy health is not ok" >&2
  return 1
}

start_qwen_flask() {
  if port_open 127.0.0.1 "$QWEN_FLASK_PORT"; then
    echo "[SKIP] Qwen Flask server already uses port ${QWEN_FLASK_PORT}"
    return
  fi
  echo "[START] Qwen Flask server -> ${LOG_DIR}/qwen_flask.log"
  (
    cd "$QWEN_SERVER_DIR"
    python3 "$QWEN_FLASK_PY" \
      --rkllm_model_path "$MODEL_PATH" \
      --target_platform rk3588
  ) >"${LOG_DIR}/qwen_flask.log" 2>&1 &
  echo "$!" >"${PID_DIR}/qwen_flask.pid"
  wait_port "Qwen Flask server" 127.0.0.1 "$QWEN_FLASK_PORT" 120
}

start_proxy() {
  if port_open 127.0.0.1 "$RKLLM_PROXY_PORT"; then
    echo "[SKIP] RKLLM proxy already uses port ${RKLLM_PROXY_PORT}"
    wait_proxy_health 5
    return
  fi
  echo "[START] RKLLM proxy -> ${LOG_DIR}/qwen_proxy.log"
  export REHAB_RKLLM_CHAT_ENDPOINT="${REHAB_RKLLM_CHAT_ENDPOINT:-http://127.0.0.1:${QWEN_FLASK_PORT}/rkllm_chat}"
  python3 llm/rkllm_proxy_server.py \
    --host 127.0.0.1 \
    --port "$RKLLM_PROXY_PORT" \
    --chat-endpoint "$REHAB_RKLLM_CHAT_ENDPOINT" \
    >"${LOG_DIR}/qwen_proxy.log" 2>&1 &
  echo "$!" >"${PID_DIR}/qwen_proxy.pid"
  wait_proxy_health 30
}

need_file "$MODEL_PATH"
need_file "$QWEN_FLASK_PY"
need_file "$QWEN_RUNTIME_LIB"
need_file "$PROJECT_ROOT/llm/rkllm_proxy_server.py"
if [[ -f "$QWEN_FIX_FREQ" ]]; then
  chmod +x "$QWEN_FIX_FREQ" >/dev/null 2>&1 || true
fi
export LD_LIBRARY_PATH="${QWEN_SERVER_DIR}/lib:${LD_LIBRARY_PATH:-}"
load_llm_env

start_qwen_flask
start_proxy

export REHAB_LLM_PROVIDER="${REHAB_LLM_PROVIDER:-auto}"
export REHAB_LLM_ONLINE_PROVIDER="${REHAB_LLM_ONLINE_PROVIDER:-glm4v_api}"
export REHAB_LLM_OFFLINE_PROVIDER="${REHAB_LLM_OFFLINE_PROVIDER:-local_qwen_rkllm}"
export REHAB_LOCAL_QWEN_ENDPOINT="${REHAB_LOCAL_QWEN_ENDPOINT:-http://127.0.0.1:${RKLLM_PROXY_PORT}/generate}"
export REHAB_LOCAL_QWEN_HEALTH_ENDPOINT="${REHAB_LOCAL_QWEN_HEALTH_ENDPOINT:-http://127.0.0.1:${RKLLM_PROXY_PORT}/health}"
export REHAB_LOCAL_QWEN_TIMEOUT="${REHAB_LOCAL_QWEN_TIMEOUT:-120}"
export REHAB_ASR_PROVIDER="${REHAB_ASR_PROVIDER:-sherpa_paraformer}"
export REHAB_ASR_MODEL_DIR="${REHAB_ASR_MODEL_DIR:-/home/elf/models/sherpa-onnx-paraformer-zh}"

export POSE_BACKEND="${POSE_BACKEND:-mediapipe}"
export RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-auto}"
export RK_CAMERA_OPEN_MODE="${RK_CAMERA_OPEN_MODE:-opencv}"
export RK_CAMERA_WIDTH="${RK_CAMERA_WIDTH:-640}"
export RK_CAMERA_HEIGHT="${RK_CAMERA_HEIGHT:-360}"

echo "[START] Rehab 8082 service -> ${LOG_DIR}/rehab_8082.log"
echo "[URL] http://127.0.0.1:${REHAB_PORT}/train"
if [[ -n "${ZHIPUAI_API_KEY:-${GLM_API_KEY:-}}" ]]; then
  echo "[LLM] GLM API key is configured for this 8082 process"
else
  echo "[LLM] GLM API key is not configured; auto mode will use local Qwen if available"
fi
echo "[TIP] Text Q&A is ready. Microphone/ASR can be tested later."

python3 prescription/banzi/record_prescription_http.py 2>&1 | tee "${LOG_DIR}/rehab_8082.log"
