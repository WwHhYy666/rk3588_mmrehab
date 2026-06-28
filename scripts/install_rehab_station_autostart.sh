#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/elf/project}"
SERVICE_NAME="${SERVICE_NAME:-rehab-station-qwen.service}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
AUTOSTART_DIR="${HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/rehab-station-browser.desktop"
START_SCRIPT="${PROJECT_DIR}/scripts/start_rehab_station_qwen.sh"
URL="${REHAB_STATION_URL:-http://127.0.0.1:8082/train}"

if [[ ! -f "$START_SCRIPT" ]]; then
  echo "[FAIL] Start script not found: $START_SCRIPT" >&2
  echo "Upload the project to /home/elf/project first, or set PROJECT_DIR." >&2
  exit 2
fi

chmod +x "$START_SCRIPT"

cat >/tmp/rehab-station-qwen.service <<EOF
[Unit]
Description=RK3588 Rehab Station with local Qwen RKLLM
After=network-online.target graphical.target
Wants=network-online.target

[Service]
Type=simple
User=elf
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${START_SCRIPT}
ExecStop=${PROJECT_DIR}/scripts/stop_rehab_station_qwen.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "[INSTALL] systemd service ${SERVICE_NAME}"
sudo cp /tmp/rehab-station-qwen.service "$SERVICE_PATH"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

mkdir -p "$AUTOSTART_DIR"
cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Rehab Station Browser
Exec=sh -lc 'sleep 12; if command -v chromium-browser >/dev/null 2>&1; then chromium-browser --kiosk ${URL}; elif command -v chromium >/dev/null 2>&1; then chromium --kiosk ${URL}; else xdg-open ${URL}; fi'
X-GNOME-Autostart-enabled=true
EOF

echo "[OK] Backend autostart installed: ${SERVICE_NAME}"
echo "[OK] Browser autostart installed: ${DESKTOP_FILE}"
echo "[NEXT] Start now with: sudo systemctl start ${SERVICE_NAME}"
echo "[CHECK] sudo systemctl status ${SERVICE_NAME} --no-pager"
