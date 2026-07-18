#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
RUNTIME_DIR="${PROJECT_ROOT}/runtime/npu"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DET_MODEL="${RKNN_DET_MODEL:-models/vision/yolov5n_raw_fp.rknn}"
POSE_MODEL="${RKNN_RTMPOSE_MODEL:-models/vision/rtmpose_m_256x192_fp.rknn}"
QWEN_MODEL="${QWEN_RKLLM_MODEL:-models/language/qwen/qwen1_5b.rkllm}"
QWEN_SERVER_DIR="${QWEN_SERVER_DIR:-/home/elf/qwen_server/rkllm_server}"
QWEN_FLASK_PY="${QWEN_FLASK_PY:-${QWEN_SERVER_DIR}/flask_server.py}"
QWEN_RUNTIME_LIB="${QWEN_RUNTIME_LIB:-${QWEN_SERVER_DIR}/lib/librkllmrt.so}"
QWEN_FLASK_PORT="${QWEN_FLASK_PORT:-8080}"
RKLLM_PROXY_PORT="${RKLLM_PROXY_PORT:-18080}"

mkdir -p "$LOG_DIR" "$PID_DIR"

port_open() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1.0):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

wait_port() {
  local name="$1" host="$2" port="$3" tries="${4:-60}"
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

load_llm_env() {
  local env_file
  for env_file in "${PROJECT_ROOT}/runtime/llm.env" "${PROJECT_ROOT}/.env.llm"; do
    if [[ -f "$env_file" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "$env_file"
      set +a
    fi
  done
}

if port_open 127.0.0.1 8082; then
  echo "[STOP] CPU 8082 is still running and may own the USB camera." >&2
  echo "Stop it manually with the existing CPU stop script, then rerun this command." >&2
  exit 4
fi
if port_open 127.0.0.1 8085; then
  echo "[FAIL] Port 8085 is already in use." >&2
  exit 5
fi
if [[ ! -f "$DET_MODEL" ]]; then
  echo "[FAIL] Missing detector model: $DET_MODEL" >&2
  exit 2
fi
if [[ ! -f "$POSE_MODEL" ]]; then
  echo "[FAIL] Missing pose model: $POSE_MODEL" >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("cv2", "numpy", "rknnlite", "yaml") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY

load_llm_env
export LD_LIBRARY_PATH="${QWEN_SERVER_DIR}/lib:${LD_LIBRARY_PATH:-}"

if ! port_open 127.0.0.1 "$QWEN_FLASK_PORT"; then
  if [[ -f "$QWEN_MODEL" && -f "$QWEN_FLASK_PY" && -f "$QWEN_RUNTIME_LIB" ]]; then
    echo "[START] Qwen RKLLM server"
    (
      cd "$QWEN_SERVER_DIR"
      "$PYTHON_BIN" "$QWEN_FLASK_PY" --rkllm_model_path "$QWEN_MODEL" --target_platform rk3588
    ) >"${LOG_DIR}/qwen_flask.log" 2>&1 &
    echo "$!" >"${PID_DIR}/qwen_flask.pid"
    wait_port "Qwen RKLLM server" 127.0.0.1 "$QWEN_FLASK_PORT" 120
  else
    echo "[WARN] Qwen runtime files are incomplete; NPU training can start, but local Qwen will be unavailable." >&2
  fi
fi

if port_open 127.0.0.1 "$QWEN_FLASK_PORT" && ! port_open 127.0.0.1 "$RKLLM_PROXY_PORT"; then
  export REHAB_RKLLM_CHAT_ENDPOINT="${REHAB_RKLLM_CHAT_ENDPOINT:-http://127.0.0.1:${QWEN_FLASK_PORT}/rkllm_chat}"
  "$PYTHON_BIN" llm/rkllm_proxy.py \
    --host 127.0.0.1 \
    --port "$RKLLM_PROXY_PORT" \
    --chat-endpoint "$REHAB_RKLLM_CHAT_ENDPOINT" \
    >"${LOG_DIR}/qwen_proxy.log" 2>&1 &
  echo "$!" >"${PID_DIR}/qwen_proxy.pid"
  wait_port "RKLLM proxy" 127.0.0.1 "$RKLLM_PROXY_PORT" 30
fi

export REHAB_LLM_PROVIDER="${REHAB_LLM_PROVIDER:-auto}"
export REHAB_LLM_ONLINE_PROVIDER="${REHAB_LLM_ONLINE_PROVIDER:-glm4v_api}"
export REHAB_LLM_OFFLINE_PROVIDER="${REHAB_LLM_OFFLINE_PROVIDER:-local_qwen_rkllm}"
export REHAB_LOCAL_QWEN_ENDPOINT="${REHAB_LOCAL_QWEN_ENDPOINT:-http://127.0.0.1:${RKLLM_PROXY_PORT}/generate}"
export REHAB_LOCAL_QWEN_HEALTH_ENDPOINT="${REHAB_LOCAL_QWEN_HEALTH_ENDPOINT:-http://127.0.0.1:${RKLLM_PROXY_PORT}/health}"
export REHAB_LOCAL_QWEN_TIMEOUT="${REHAB_LOCAL_QWEN_TIMEOUT:-120}"
export REHAB_AUDIO_OUTPUT_DEVICE="${REHAB_AUDIO_OUTPUT_DEVICE:-plughw:1,0}"
export REHAB_ASSISTANT_TTS_GAIN="${REHAB_ASSISTANT_TTS_GAIN:-1.35}"

export POSE_BACKEND="rknn"
export RKNN_POSE_PIPELINE="yolov5n_rtmpose"
export RKNN_DET_MODEL="$DET_MODEL"
export RKNN_RTMPOSE_MODEL="$POSE_MODEL"
export RKNN_CORE_MASK="${RKNN_CORE_MASK:-NPU_CORE_0_1_2}"
export RKNN_DET_CORE_MASK="${RKNN_DET_CORE_MASK:-${RKNN_CORE_MASK}}"
export RKNN_POSE_CORE_MASK="${RKNN_POSE_CORE_MASK:-${RKNN_CORE_MASK}}"
export RKNN_POSE_KEYPOINT_THRES="${RKNN_POSE_KEYPOINT_THRES:-0.18}"
export RKNN_FIXED_LEG_VISIBILITY_THRESHOLD="${RKNN_FIXED_LEG_VISIBILITY_THRESHOLD:-0.20}"
export RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD="${RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD:-0.20}"
export RKNN_STABILIZER_ALPHA="${RKNN_STABILIZER_ALPHA:-0.55}"
export RKNN_STABILIZER_LOW_CONF_ALPHA="${RKNN_STABILIZER_LOW_CONF_ALPHA:-0.32}"
export RKNN_STABILIZER_JUMP_SCALE="${RKNN_STABILIZER_JUMP_SCALE:-0.55}"
export RKNN_STABILIZER_MAX_HOLD_FRAMES="${RKNN_STABILIZER_MAX_HOLD_FRAMES:-8}"
export RKNN_DISPLAY_ALPHA="${RKNN_DISPLAY_ALPHA:-0.50}"
export RKNN_DISPLAY_LOW_CONF_ALPHA="${RKNN_DISPLAY_LOW_CONF_ALPHA:-0.30}"
export RKNN_DISPLAY_JUMP_SCALE="${RKNN_DISPLAY_JUMP_SCALE:-0.35}"
export RKNN_DISPLAY_MAX_HOLD_FRAMES="${RKNN_DISPLAY_MAX_HOLD_FRAMES:-4}"
export RKNN_DISPLAY_JUMP_CONFIRM_FRAMES="${RKNN_DISPLAY_JUMP_CONFIRM_FRAMES:-2}"
export RKNN_DISPLAY_BBOX_ALPHA="${RKNN_DISPLAY_BBOX_ALPHA:-0.45}"
export RKNN_DISPLAY_BBOX_HOLD_FRAMES="${RKNN_DISPLAY_BBOX_HOLD_FRAMES:-6}"
export RKNN_DISPLAY_HOLD_SECONDS="${RKNN_DISPLAY_HOLD_SECONDS:-0.25}"
export RKNN_DISPLAY_BBOX_HOLD_SECONDS="${RKNN_DISPLAY_BBOX_HOLD_SECONDS:-0.35}"
export RKNN_DISPLAY_JUMP_CONFIRM_SECONDS="${RKNN_DISPLAY_JUMP_CONFIRM_SECONDS:-0.20}"
export RKNN_DISPLAY_DISAPPEAR_RATIO="${RKNN_DISPLAY_DISAPPEAR_RATIO:-0.65}"
export RKNN_DISPLAY_BBOX_IOU_JUMP="${RKNN_DISPLAY_BBOX_IOU_JUMP:-0.35}"
export RKNN_DISPLAY_MAX_STALE_SECONDS="${RKNN_DISPLAY_MAX_STALE_SECONDS:-0.50}"
export RKNN_YOLOV5_INPUT_LAYOUT="nhwc"
export RKNN_RTMPOSE_INPUT_LAYOUT="nhwc"
export RKNN_YOLOV5_INPUT_SIZE="${RKNN_YOLOV5_INPUT_SIZE:-640}"
export RKNN_YOLOV5_INPUT_MODE="${RKNN_YOLOV5_INPUT_MODE:-rgb_0_1}"
export RKNN_DET_SCORE_THRES="${RKNN_DET_SCORE_THRES:-0.80}"
export RKNN_DET_NMS_THRES="${RKNN_DET_NMS_THRES:-0.65}"
export RKNN_RTMPOSE_INPUT_WIDTH="192"
export RKNN_RTMPOSE_INPUT_HEIGHT="256"
export RKNN_RTMPOSE_BBOX_EXPAND="${RKNN_RTMPOSE_BBOX_EXPAND:-1.25}"
export RKNN_RTMPOSE_BBOX_TOP_EXPAND="${RKNN_RTMPOSE_BBOX_TOP_EXPAND:-0.10}"
export RKNN_RTMPOSE_WIDE_BBOX_RATIO="${RKNN_RTMPOSE_WIDE_BBOX_RATIO:-0.65}"
export RKNN_RTMPOSE_WIDE_BBOX_EXPAND="${RKNN_RTMPOSE_WIDE_BBOX_EXPAND:-1.50}"
export RKNN_RTMPOSE_DRAW="0"
export RK_CAMERA_SOURCE="device"
export RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
export RK_CAMERA_OPEN_MODE="${RK_CAMERA_OPEN_MODE:-auto}"
export RK_CAMERA_GST_FORMAT="${RK_CAMERA_GST_FORMAT:-MJPG}"
export RK_CAMERA_GST_JPEG_DECODER="${RK_CAMERA_GST_JPEG_DECODER:-auto}"
export RK_CAMERA_GST_BACKEND="${RK_CAMERA_GST_BACKEND:-gi}"
unset RK_CAMERA_STREAM_URL RTM_POSE_STREAM_URL
export RK_CAMERA_WIDTH="${RK_CAMERA_WIDTH:-1280}"
export RK_CAMERA_HEIGHT="${RK_CAMERA_HEIGHT:-720}"
export RK_CAMERA_FPS="${RK_CAMERA_FPS:-30}"
export RK_CAMERA_FIXED_FPS="${RK_CAMERA_FIXED_FPS:-1}"
export RKNN_PROCESS_WIDTH="${RKNN_PROCESS_WIDTH:-1280}"
export RKNN_PROCESS_HEIGHT="${RKNN_PROCESS_HEIGHT:-720}"
export RKNN_STREAM_WIDTH="${RKNN_STREAM_WIDTH:-960}"
export RKNN_STREAM_HEIGHT="${RKNN_STREAM_HEIGHT:-540}"
export RKNN_STREAM_FPS="${RKNN_STREAM_FPS:-20}"
export RKNN_DIAGNOSTIC_SAMPLE_INTERVAL="${RKNN_DIAGNOSTIC_SAMPLE_INTERVAL:-5}"
export RKNN_FAST_PREVIEW="${RKNN_FAST_PREVIEW:-1}"
export RKNN_FAST_FRAME_DATA="${RKNN_FAST_FRAME_DATA:-1}"
export REHAB_KEYFRAME_EVERY_N="${REHAB_KEYFRAME_EVERY_N:-8}"
export RKNN_DET_INTERVAL="${RKNN_DET_INTERVAL:-3}"
export RKNN_DET_CACHE_SECONDS="${RKNN_DET_CACHE_SECONDS:-1.5}"
export RKNN_ADAPTIVE_DETECTOR="${RKNN_ADAPTIVE_DETECTOR:-1}"
export RKNN_DET_REFRESH_SECONDS="${RKNN_DET_REFRESH_SECONDS:-0.75}"
export RKNN_DET_RETRY_SECONDS="${RKNN_DET_RETRY_SECONDS:-0.25}"
export RKNN_DET_BAD_POSE_FRAMES="${RKNN_DET_BAD_POSE_FRAMES:-2}"
export RKNN_TRACKER_MARGIN="${RKNN_TRACKER_MARGIN:-0.20}"
export RKNN_TRACKER_ALPHA="${RKNN_TRACKER_ALPHA:-0.35}"
export RKNN_TRACKER_MIN_POINTS="${RKNN_TRACKER_MIN_POINTS:-5}"
export RKNN_MAX_POSE_PADDING_RATIO="${RKNN_MAX_POSE_PADDING_RATIO:-0.55}"
export RKNN_ASYNC_PIPELINE="${RKNN_ASYNC_PIPELINE:-1}"
export REHAB_SERVICE_MODE="${REHAB_SERVICE_MODE:-npu_rehab}"
export RKNN_YOLOV5_BACKEND_DRAW="${RKNN_YOLOV5_BACKEND_DRAW:-0}"
export RKNN_YOLOV5_PERSON_ONLY_FAST="${RKNN_YOLOV5_PERSON_ONLY_FAST:-1}"
export RKNN_RTMPOSE_DEBUG_CROP_EVERY="${RKNN_RTMPOSE_DEBUG_CROP_EVERY:-0}"
export RK_JPEG_QUALITY="${RK_JPEG_QUALITY:-72}"

if [[ "$RK_CAMERA_FIXED_FPS" == "1" ]] && command -v v4l2-ctl >/dev/null 2>&1; then
  if v4l2-ctl --device="$RK_CAMERA_DEVICE" --list-ctrls 2>/dev/null | grep -q 'exposure_auto_priority'; then
    if v4l2-ctl --device="$RK_CAMERA_DEVICE" --set-ctrl=exposure_auto_priority=0 >/dev/null 2>&1; then
      echo "[CAMERA] exposure_auto_priority=0 (fixed FPS preferred)"
    else
      echo "[WARN] Failed to disable camera exposure auto-priority; continuing with camera defaults." >&2
    fi
  fi
fi

echo "[START] NPU rehab 8085"
echo "[TRAIN] http://127.0.0.1:8085/train"
echo "[DOCTOR] http://127.0.0.1:8085/doctor"
echo "[DEBUG] http://127.0.0.1:8085/npu-debug"
echo "[POSE] detector=${DET_MODEL} pose=${POSE_MODEL} core=${RKNN_CORE_MASK} det_score=${RKNN_DET_SCORE_THRES}"
echo "[CAMERA] source=direct_device device=${RK_CAMERA_DEVICE} ${RK_CAMERA_WIDTH}x${RK_CAMERA_HEIGHT} mode=${RK_CAMERA_OPEN_MODE}"
echo "[AUDIO] fixed WAV output=${REHAB_AUDIO_OUTPUT_DEVICE}; post-training TTS gain=${REHAB_ASSISTANT_TTS_GAIN}"
echo "[PERF] process=${RKNN_PROCESS_WIDTH}x${RKNN_PROCESS_HEIGHT} stream=${RKNN_STREAM_WIDTH}x${RKNN_STREAM_HEIGHT} det_interval=${RKNN_DET_INTERVAL} fast_preview=${RKNN_FAST_PREVIEW} fast_frame_data=${RKNN_FAST_FRAME_DATA} person_only=${RKNN_YOLOV5_PERSON_ONLY_FAST} backend_draw=${RKNN_YOLOV5_BACKEND_DRAW} jpeg_quality=${RK_JPEG_QUALITY} keyframe_every_n=${REHAB_KEYFRAME_EVERY_N}"

"$PYTHON_BIN" -u rehab_app/server/npu_rehab_server.py > >(tee -a "${LOG_DIR}/npu_rehab_8085.log") 2>&1 &
npu_pid=$!
echo "$npu_pid" >"${PID_DIR}/npu_rehab_8085.pid"

cleanup() {
  if kill -0 "$npu_pid" 2>/dev/null; then
    kill "$npu_pid" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

wait_port "NPU rehab" 127.0.0.1 8085 120
wait "$npu_pid"
