#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODE_FILE="${REHAB_MODE_FILE:-runtime/selected_rehab_mode}"
DISPLAY_REFRESH_FILE="${REHAB_DISPLAY_REFRESH_FILE:-runtime/display_refresh}"
MODE_SWITCH_FILE="${REHAB_MODE_SWITCH_FILE:-runtime/rehab_mode_switching}"
CPU_SERVICE="${REHAB_CPU_SERVICE:-rehab-station-qwen.service}"
NPU_SERVICE="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
MODE_OWNER="${REHAB_RUN_USER:-${SUDO_USER:-${USER:-elf}}}"
WAIT_SECONDS="${REHAB_SWITCH_WAIT_SECONDS:-180}"
CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
CAMERA_RELEASE_SECONDS="${REHAB_CAMERA_RELEASE_SECONDS:-15}"

normalize_mode() {
  case "${1:-}" in
    cpu|cpu_8082|8082) printf '%s\n' cpu_8082 ;;
    npu|npu_8085|8085) printf '%s\n' npu_8085 ;;
    *) return 1 ;;
  esac
}

target_mode="$(normalize_mode "${1:-}")" || {
  echo "Usage: $0 cpu_8082|npu_8085" >&2
  exit 2
}

read_mode() {
  local value
  value="$(tr -d ' \t\r\n' < "$MODE_FILE" 2>/dev/null || true)"
  normalize_mode "$value" 2>/dev/null || printf '%s\n' npu_8085
}

service_for_mode() {
  [[ "$1" == "cpu_8082" ]] && printf '%s\n' "$CPU_SERVICE" || printf '%s\n' "$NPU_SERVICE"
}

url_for_mode() {
  [[ "$1" == "cpu_8082" ]] && printf '%s\n' 'http://127.0.0.1:8082/status' || printf '%s\n' 'http://127.0.0.1:8085/status'
}

run_systemctl() {
  if [[ "$(id -u)" == "0" ]]; then
    systemctl "$@"
  else
    sudo systemctl "$@"
  fi
}

unit_exists() {
  systemctl list-unit-files "$1" --no-legend 2>/dev/null | grep -q "^${1}[[:space:]]"
}

validate_status() {
  local mode="$1" url="$2"
  python3 - "$mode" "$url" <<'PY'
import json
import sys
from urllib.request import urlopen

mode, url = sys.argv[1:]
with urlopen(url, timeout=3.0) as response:
    status = json.load(response)

pose_backend = status.get("pose_backend")
pose_backend_actual = pose_backend.get("actual_backend") if isinstance(pose_backend, dict) else ""
actual = str(status.get("actual_backend") or pose_backend_actual or "")
camera_ok = status.get("camera_live_ok")
if mode == "cpu_8082":
    ok = actual == "mediapipe" and camera_ok is not False
else:
    pipeline = str(status.get("rknn_pipeline") or "")
    ok = (
        str(status.get("service_mode") or "") == "npu_rehab"
        and actual == "rknn"
        and pipeline == "yolov5n_rtmpose"
        and camera_ok is not False
    )
raise SystemExit(0 if ok else 1)
PY
}

wait_ready() {
  local mode="$1" url="$2" i
  for ((i = 1; i <= WAIT_SECONDS; i++)); do
    if validate_status "$mode" "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

persist_mode() {
  local mode="$1" tmp
  mkdir -p "$(dirname "$MODE_FILE")"
  tmp="${MODE_FILE}.tmp.$$"
  printf '%s\n' "$mode" > "$tmp"
  mv -f "$tmp" "$MODE_FILE"
  if [[ "$(id -u)" == "0" ]] && id "$MODE_OWNER" >/dev/null 2>&1; then
    chown "$MODE_OWNER":"$MODE_OWNER" "$MODE_FILE" 2>/dev/null || true
  fi
}

request_display_refresh() {
  [[ "${REHAB_SWITCH_SKIP_DISPLAY:-0}" == "1" ]] && return
  mkdir -p "$(dirname "$DISPLAY_REFRESH_FILE")"
  printf '%s-%s\n' "$(date +%s)" "$$" > "$DISPLAY_REFRESH_FILE"
  if [[ "$(id -u)" == "0" ]] && id "$MODE_OWNER" >/dev/null 2>&1; then
    chown "$MODE_OWNER":"$MODE_OWNER" "$DISPLAY_REFRESH_FILE" 2>/dev/null || true
  fi
}

resolved_camera_device() {
  readlink -f "$CAMERA_DEVICE" 2>/dev/null || printf '%s\n' "$CAMERA_DEVICE"
}

camera_owner_pids() {
  local device
  device="$(resolved_camera_device)"
  command -v fuser >/dev/null 2>&1 || return 0
  fuser "$device" 2>/dev/null || true
}

wait_camera_release() {
  local i owners
  command -v fuser >/dev/null 2>&1 || { sleep 1; return 0; }
  for ((i = 1; i <= CAMERA_RELEASE_SECONDS; i++)); do
    owners="$(camera_owner_pids)"
    [[ -z "${owners//[[:space:]]/}" ]] && return 0
    sleep 1
  done
  owners="$(camera_owner_pids)"
  echo "[FAIL] Camera is still owned by PID(s): ${owners:-unknown}" >&2
  echo "[CHECK] fuser -v $(resolved_camera_device)" >&2
  return 1
}

previous_mode="$(read_mode)"
target_service="$(service_for_mode "$target_mode")"
other_mode="cpu_8082"
[[ "$target_mode" == "cpu_8082" ]] && other_mode="npu_8085"
other_service="$(service_for_mode "$other_mode")"
target_url="$(url_for_mode "$target_mode")"

if ! unit_exists "$target_service" || ! unit_exists "$other_service"; then
  echo "[FAIL] Dual-mode systemd services are not installed." >&2
  echo "Run: ./scripts/install_rehab_station_autostart.sh" >&2
  exit 3
fi

mkdir -p "$(dirname "$MODE_SWITCH_FILE")"
printf '%s %s %s\n' "$target_mode" "$$" "$(date +%s)" > "$MODE_SWITCH_FILE"
cleanup_mode_switch_file() {
  rm -f "$MODE_SWITCH_FILE"
}
trap cleanup_mode_switch_file EXIT

echo "[SWITCH] ${previous_mode} -> ${target_mode}"
run_systemctl stop "$other_service" || true
run_systemctl stop "$target_service" || true
if ! wait_camera_release; then
  echo "[FAIL] Refusing to start ${target_mode} while another process owns the camera." >&2
  exit 4
fi
run_systemctl reset-failed "$target_service" || true

if ! run_systemctl start "$target_service" || ! wait_ready "$target_mode" "$target_url"; then
  echo "[FAIL] ${target_mode} did not become healthy; restoring ${previous_mode}." >&2
  run_systemctl stop "$target_service" || true
  previous_service="$(service_for_mode "$previous_mode")"
  run_systemctl reset-failed "$previous_service" || true
  run_systemctl start "$previous_service" || true
  echo "[LOG] sudo journalctl -u ${target_service} -n 120 --no-pager" >&2
  exit 5
fi

persist_mode "$target_mode"
request_display_refresh

echo "[OK] Active mode: ${target_mode}"
if [[ "$target_mode" == "cpu_8082" ]]; then
  echo "[TRAIN] http://127.0.0.1:8082/train?display=1"
  echo "[DOCTOR] http://127.0.0.1:8082/doctor"
else
  echo "[TRAIN] http://127.0.0.1:8085/train?display=1"
  echo "[DOCTOR] http://127.0.0.1:8085/doctor"
  echo "[DEBUG] http://127.0.0.1:8085/npu-debug"
fi
