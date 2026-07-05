#!/usr/bin/env bash
set -euo pipefail

URL="${REHAB_STATION_URL:-http://127.0.0.1:8082/train?display=1}"
WAIT_SECONDS="${REHAB_BROWSER_WAIT_SECONDS:-15}"

sleep "$WAIT_SECONDS"

for browser in chromium-browser chromium google-chrome-stable google-chrome; do
  if command -v "$browser" >/dev/null 2>&1; then
    exec "$browser" \
      --app="$URL" \
      --start-fullscreen \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --check-for-update-interval=31536000
  fi
done

if command -v xdg-open >/dev/null 2>&1; then
  exec xdg-open "$URL"
fi

echo "No Chromium or xdg-open command found; open this URL manually: $URL" >&2
exit 1