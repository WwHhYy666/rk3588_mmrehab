#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1

MODE_FILE="${REHAB_MODE_FILE:-runtime/selected_rehab_mode}"
REFRESH_FILE="${REHAB_DISPLAY_REFRESH_FILE:-runtime/display_refresh}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}/rehab-station-display-${UID}"
PROFILE_DIR="${REHAB_BROWSER_PROFILE_DIR:-${RUNTIME_ROOT}/profile}"
LOG_DIR="${HOME}/.cache/rehab-station-display"
LOG_FILE="${LOG_DIR}/manager.log"
POLL_SECONDS="${REHAB_DISPLAY_POLL_SECONDS:-1}"
HEALTH_POLL_SECONDS="${REHAB_DISPLAY_HEALTH_POLL_SECONDS:-3}"
BROWSER_GRACE_SECONDS="${REHAB_DISPLAY_BROWSER_GRACE_SECONDS:-20}"
BROWSER_FAIL_LIMIT="${REHAB_DISPLAY_BROWSER_FAIL_LIMIT:-3}"
DEBUG_PORT="${REHAB_BROWSER_DEBUG_PORT:-9225}"
KEEP_AWAKE_SCRIPT="scripts/rehab_display_keepawake.sh"

mkdir -p "$LOG_DIR" "$PROFILE_DIR"
exec >>"$LOG_FILE" 2>&1

if [[ -f "$KEEP_AWAKE_SCRIPT" ]] && ! pgrep -f 'scripts/rehab_display_keepawake.sh' >/dev/null 2>&1; then
  nohup bash "$KEEP_AWAKE_SCRIPT" >/dev/null 2>&1 &
fi

read_mode() {
  local value
  value="$(tr -d ' \t\r\n' < "$MODE_FILE" 2>/dev/null || true)"
  case "$value" in
    cpu_8082|npu_8085) printf '%s\n' "$value" ;;
    *) printf '%s\n' npu_8085 ;;
  esac
}

read_refresh() {
  cat "$REFRESH_FILE" 2>/dev/null || true
}

stop_managed_browser() {
  local pid
  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && kill "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "--user-data-dir=${PROFILE_DIR}" 2>/dev/null || true)

  local attempt
  for attempt in {1..30}; do
    pgrep -f -- "--user-data-dir=${PROFILE_DIR}" >/dev/null 2>&1 || return 0
    sleep 0.1
  done

  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && kill -KILL "$pid" 2>/dev/null || true
  done < <(pgrep -f -- "--user-data-dir=${PROFILE_DIR}" 2>/dev/null || true)
}

browser_running() {
  pgrep -f -- "--user-data-dir=${PROFILE_DIR}" >/dev/null 2>&1
}

backend_reachable() {
  curl --fail --silent --connect-timeout 1 --max-time 2 "$1" >/dev/null 2>&1
}

# Return 0 for a healthy same-origin page, 1 for an explicit Chromium/network
# error page, 2 when all pages were closed, and 3 when debugging is temporarily
# unavailable or the visible page cannot be classified safely.
browser_page_state() {
  local expected_url="$1"
  local page_json
  if ! page_json="$(curl --fail --silent --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:${DEBUG_PORT}/json/list")"; then
    return 3
  fi
  printf '%s' "$page_json" | python3 -c '
import json
import sys
from urllib.parse import urlsplit

expected = sys.argv[1]
expected_parts = urlsplit(expected)
expected_origin = (expected_parts.scheme.lower(), expected_parts.hostname or "", expected_parts.port)
try:
    pages = [item for item in json.load(sys.stdin) if item.get("type") == "page"]
except Exception:
    raise SystemExit(3)
if not pages:
    raise SystemExit(2)
bad_markers = (
    "chrome-error://",
    "err_connection_refused",
    "this site can\u0027t be reached",
    "this site cannot be reached",
    "cant be reached",
    "cannot be reached",
    "aw, snap",
)
explicit_error = False
for page in pages:
    title = str(page.get("title") or "").strip().lower()
    url = str(page.get("url") or "").strip().lower()
    if any(marker in title or marker in url for marker in bad_markers):
        explicit_error = True
        continue
    parts = urlsplit(url)
    page_origin = (parts.scheme.lower(), parts.hostname or "", parts.port)
    if page_origin == expected_origin:
        raise SystemExit(0)
raise SystemExit(1 if explicit_error else 3)
' "$expected_url"
}

open_mode() {
  local mode="$1" url
  if [[ "$mode" == "cpu_8082" ]]; then
    url="http://127.0.0.1:8082/train?display=1"
  else
    url="http://127.0.0.1:8085/train?display=1"
  fi
  stop_managed_browser
  echo "[$(date '+%F %T')] opening ${mode}: ${url}"
  REHAB_STATION_URL="$url" \
  REHAB_BROWSER_WAIT_SECONDS=0 \
  REHAB_BROWSER_DEBUG_PORT="$DEBUG_PORT" \
  REHAB_BROWSER_PROFILE_DIR="$PROFILE_DIR" \
    bash scripts/open_rehab_station_kiosk.sh &
  browser_opened_at="$(date +%s)"
  browser_failures=0
}

last_state=""
browser_opened_at=0
browser_failures=0
next_health_check=0
while true; do
  mode="$(read_mode)"
  state="${mode}|$(read_refresh)"
  if [[ "$state" != "$last_state" ]]; then
    open_mode "$mode"
    last_state="$state"
  fi

  now="$(date +%s)"
  if (( now >= next_health_check )); then
    if [[ "$mode" == "cpu_8082" ]]; then
      display_url="http://127.0.0.1:8082/train?display=1"
    else
      display_url="http://127.0.0.1:8085/train?display=1"
    fi

    if browser_running && (( now - browser_opened_at >= BROWSER_GRACE_SECONDS )) && backend_reachable "$display_url"; then
      browser_page_state "$display_url"
      page_state=$?
      if [[ "$page_state" == "1" ]]; then
        browser_failures=$((browser_failures + 1))
        echo "[$(date '+%F %T')] Chromium display health failure ${browser_failures}/${BROWSER_FAIL_LIMIT}"
        if (( browser_failures >= BROWSER_FAIL_LIMIT )); then
          echo "[$(date '+%F %T')] backend is healthy but Chromium is not; restarting only the dedicated browser"
          open_mode "$mode"
        fi
      else
        browser_failures=0
        if [[ "$page_state" == "3" ]]; then
          echo "[$(date '+%F %T')] Chromium debugging endpoint/page state is temporarily unavailable; keeping the current browser"
        fi
      fi
    elif ! browser_running && (( now - browser_opened_at >= BROWSER_GRACE_SECONDS )) && backend_reachable "$display_url"; then
      echo "[$(date '+%F %T')] dedicated Chromium exited; opening a clean browser session"
      open_mode "$mode"
    else
      browser_failures=0
    fi
    next_health_check=$((now + HEALTH_POLL_SECONDS))
  fi
  sleep "$POLL_SECONDS"
done
