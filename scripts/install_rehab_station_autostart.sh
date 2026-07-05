#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
if [[ "$RUN_USER" == "root" ]]; then
  RUN_USER="elf"
fi
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
if [[ -z "$RUN_HOME" ]]; then
  echo "[FAIL] Cannot find home directory for user: ${RUN_USER}" >&2
  exit 2
fi

SERVICE_NAME="${SERVICE_NAME:-rehab-station-qwen.service}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
AUTOSTART_DIR="${RUN_HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/rehab-station-browser.desktop"
START_SCRIPT="${PROJECT_DIR}/scripts/start_rehab_station_qwen.sh"
STOP_SCRIPT="${PROJECT_DIR}/scripts/stop_rehab_station_qwen.sh"
BROWSER_SCRIPT="${PROJECT_DIR}/scripts/open_rehab_station_kiosk.sh"
URL="${REHAB_STATION_URL:-http://127.0.0.1:8082/train?display=1}"

if [[ ! -f "$START_SCRIPT" ]]; then
  echo "[FAIL] Start script not found: $START_SCRIPT" >&2
  echo "Run this installer from the uploaded project, or set PROJECT_DIR=/actual/project/path." >&2
  exit 2
fi
if [[ ! -f "$STOP_SCRIPT" ]]; then
  echo "[FAIL] Stop script not found: $STOP_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$BROWSER_SCRIPT" ]]; then
  echo "[FAIL] Browser kiosk script not found: $BROWSER_SCRIPT" >&2
  exit 2
fi

chmod +x "$START_SCRIPT" "$STOP_SCRIPT" "$BROWSER_SCRIPT"

SERVICE_TMP="$(mktemp)"
cat >"$SERVICE_TMP" <<EOF
[Unit]
Description=RK3588 Rehab Station with local Qwen RKLLM
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/env bash ${START_SCRIPT}
ExecStop=/usr/bin/env bash ${STOP_SCRIPT}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "[INSTALL] systemd service ${SERVICE_NAME} -> ${SERVICE_PATH}"
sudo cp "$SERVICE_TMP" "$SERVICE_PATH"
rm -f "$SERVICE_TMP"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

mkdir -p "$AUTOSTART_DIR"
cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Rehab Station Browser
Exec=env REHAB_STATION_URL=${URL} ${BROWSER_SCRIPT}
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
chown "${RUN_USER}:${RUN_USER}" "$DESKTOP_FILE" 2>/dev/null || true

sudo loginctl enable-linger "$RUN_USER" >/dev/null 2>&1 || true

echo "[OK] Backend autostart installed: ${SERVICE_NAME}"
echo "[OK] Browser kiosk autostart installed: ${DESKTOP_FILE}"
echo "[URL] ${URL}"
echo "[NEXT] Start now with: sudo systemctl start ${SERVICE_NAME}"
echo "[CHECK] sudo systemctl status ${SERVICE_NAME} --no-pager"