#!/usr/bin/env bash
set -euo pipefail

URL="${REHAB_STATION_URL:-http://127.0.0.1:8085/train?display=1}"
WAIT_SECONDS="${REHAB_BROWSER_WAIT_SECONDS:-5}"
READY_TIMEOUT_SECONDS="${REHAB_BROWSER_READY_TIMEOUT_SECONDS:-90}"
PROFILE_DIR="${REHAB_BROWSER_PROFILE_DIR:-${HOME}/.cache/npu-rehab-8085-browser}"

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
    echo "NPU 8085 page did not become ready before browser launch: $URL" >&2
  fi
fi

mkdir -p "$PROFILE_DIR"

for browser in chromium-browser chromium google-chrome-stable google-chrome; do
  if command -v "$browser" >/dev/null 2>&1; then
    exec env \
      GNOME_KEYRING_CONTROL= \
      GNOME_KEYRING_PID= \
      SSH_AUTH_SOCK= \
      "$browser" \
      --kiosk \
      --app="$URL" \
      --start-maximized \
      --window-position=0,0 \
      --no-first-run \
      --no-default-browser-check \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --disable-features=TranslateUI \
      --disable-pinch \
      --overscroll-history-navigation=0 \
      --autoplay-policy=no-user-gesture-required \
      --password-store=basic \
      --check-for-update-interval=31536000 \
      --user-data-dir="$PROFILE_DIR"
  fi
done

if command -v xdg-open >/dev/null 2>&1; then
  exec xdg-open "$URL"
fi

echo "No Chromium or xdg-open command found; open this URL manually: $URL" >&2
exit 1
