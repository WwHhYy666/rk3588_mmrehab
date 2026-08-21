#!/usr/bin/env bash
set -euo pipefail

URL="${REHAB_STATION_URL:-http://127.0.0.1:8082/train?display=1}"
WAIT_SECONDS="${REHAB_BROWSER_WAIT_SECONDS:-15}"
READY_TIMEOUT_SECONDS="${REHAB_BROWSER_READY_TIMEOUT_SECONDS:-90}"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}/rehab-station-display-${UID}"
DEFAULT_PROFILE_DIR="${RUNTIME_ROOT}/profile"
DEFAULT_CACHE_DIR="${RUNTIME_ROOT}/cache"
PROFILE_DIR="${REHAB_BROWSER_PROFILE_DIR:-${DEFAULT_PROFILE_DIR}}"
CACHE_DIR="${REHAB_BROWSER_CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
CLEAN_PROFILE="${REHAB_BROWSER_CLEAN_PROFILE:-1}"
DEBUG_PORT="${REHAB_BROWSER_DEBUG_PORT:-9225}"
# Hardware compositing is important on RK3588: forcing software rendering can
# make the whole desktop, mouse and camera preview sluggish. Enable this only
# as a targeted workaround for a reproducible GPU white-screen failure.
DISABLE_GPU="${REHAB_BROWSER_DISABLE_GPU:-0}"

CHROMIUM_COMPAT_FLAGS=(
  --disable-dev-shm-usage
  --disable-gpu-shader-disk-cache
  --disable-background-timer-throttling
  --disable-renderer-backgrounding
  --disable-backgrounding-occluded-windows
)
if [[ "$DISABLE_GPU" == "1" ]]; then
  CHROMIUM_COMPAT_FLAGS+=(--disable-gpu)
fi

sleep "$WAIT_SECONDS"

if [[ -n "${DISPLAY:-}" ]]; then
  if command -v xset >/dev/null 2>&1; then
    xset s off >/dev/null 2>&1 || true
    xset s noblank >/dev/null 2>&1 || true
    xset -dpms >/dev/null 2>&1 || true
  fi
  if command -v gsettings >/dev/null 2>&1; then
    gsettings set org.gnome.desktop.screensaver lock-enabled false >/dev/null 2>&1 || true
    gsettings set org.gnome.desktop.session idle-delay uint32 0 >/dev/null 2>&1 || true
  fi
fi

if command -v curl >/dev/null 2>&1; then
  ready=0
  for ((i = 1; i <= READY_TIMEOUT_SECONDS; i++)); do
    if curl -fsS "$URL" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "Rehab page did not become ready before browser launch: $URL" >&2
  fi
fi

if [[ "$CLEAN_PROFILE" == "1" && "$PROFILE_DIR" == "$DEFAULT_PROFILE_DIR" && "$CACHE_DIR" == "$DEFAULT_CACHE_DIR" ]]; then
  case "$RUNTIME_ROOT" in
    "${XDG_RUNTIME_DIR:-/tmp}"/rehab-station-display-*) rm -rf -- "$RUNTIME_ROOT" ;;
    *) echo "Refusing to clean unexpected browser runtime path: $RUNTIME_ROOT" >&2; exit 3 ;;
  esac
fi
mkdir -p "$PROFILE_DIR" "$CACHE_DIR"

# A Chromium crash can leave these profile locks behind. They are safe to
# remove before launch because the display manager has already stopped the
# browser process that owns this dedicated profile.
rm -f \
  "$PROFILE_DIR/SingletonCookie" \
  "$PROFILE_DIR/SingletonLock" \
  "$PROFILE_DIR/SingletonSocket" 2>/dev/null || true

for browser in chromium-browser chromium google-chrome-stable google-chrome; do
  if command -v "$browser" >/dev/null 2>&1; then
    exec env \
      GNOME_KEYRING_CONTROL= \
      GNOME_KEYRING_PID= \
      SSH_AUTH_SOCK= \
      "$browser" \
      --kiosk \
      --start-maximized \
      --window-position=0,0 \
      --no-proxy-server \
      --remote-debugging-address=127.0.0.1 \
      --remote-debugging-port="$DEBUG_PORT" \
      --disk-cache-dir="$CACHE_DIR" \
      --disk-cache-size=16777216 \
      --media-cache-size=16777216 \
      --no-first-run \
      --no-default-browser-check \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --disable-background-mode \
      --disable-component-update \
      --disable-extensions \
      --disable-features=TranslateUI \
      --disable-pinch \
      --overscroll-history-navigation=0 \
      --autoplay-policy=no-user-gesture-required \
      --password-store=basic \
      --check-for-update-interval=31536000 \
      --user-data-dir="$PROFILE_DIR" \
      "${CHROMIUM_COMPAT_FLAGS[@]}" \
      "$URL"
  fi
done

if command -v xdg-open >/dev/null 2>&1; then
  exec xdg-open "$URL"
fi

echo "No Chromium or xdg-open command found; open this URL manually: $URL" >&2
exit 1
