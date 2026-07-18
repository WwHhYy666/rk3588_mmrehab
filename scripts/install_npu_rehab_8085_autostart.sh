#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
SERVICE_NAME="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
ENABLE_SERVICE="${REHAB_ENABLE_SERVICE:-1}"
INSTALL_KIOSK="${REHAB_INSTALL_KIOSK:-1}"
SYSTEMD_DIR="/etc/systemd/system"

if [[ "$RUN_USER" == "root" ]]; then
  RUN_USER="elf"
fi

RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
if [[ -z "$RUN_HOME" ]]; then
  echo "[FAIL] Cannot find home directory for user: ${RUN_USER}" >&2
  exit 2
fi

START_SCRIPT="${PROJECT_DIR}/scripts/start_npu_rehab_8085.sh"
STOP_SCRIPT="${PROJECT_DIR}/scripts/stop_npu_rehab_8085.sh"
KIOSK_SCRIPT="${PROJECT_DIR}/scripts/open_npu_rehab_8085_kiosk.sh"

for path in "$START_SCRIPT" "$STOP_SCRIPT" "$KIOSK_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "[FAIL] Required script not found: $path" >&2
    exit 2
  fi
done

case "$ENABLE_SERVICE" in
  0|1) ;;
  *) echo "[FAIL] REHAB_ENABLE_SERVICE must be 0 or 1." >&2; exit 2 ;;
esac
case "$INSTALL_KIOSK" in
  0|1) ;;
  *) echo "[FAIL] REHAB_INSTALL_KIOSK must be 0 or 1." >&2; exit 2 ;;
esac

chmod +x "$START_SCRIPT" "$STOP_SCRIPT" "$KIOSK_SCRIPT"

unit_tmp="$(mktemp)"
cat >"$unit_tmp" <<EOF
[Unit]
Description=RK3588 NPU Rehabilitation Station on port 8085
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

sudo install -m 0644 "$unit_tmp" "${SYSTEMD_DIR}/${SERVICE_NAME}"
rm -f "$unit_tmp"
sudo systemctl daemon-reload

if [[ "$ENABLE_SERVICE" == "1" ]]; then
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl restart "$SERVICE_NAME"
  echo "[OK] Enabled and started ${SERVICE_NAME}"
else
  sudo systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 || true
  echo "[OK] Installed ${SERVICE_NAME} without enabling it"
fi

if [[ "$INSTALL_KIOSK" == "1" ]]; then
  autostart_dir="${RUN_HOME}/.config/autostart"
  desktop_file="${autostart_dir}/npu-rehab-8085-kiosk.desktop"
  mkdir -p "$autostart_dir"
  cat >"$desktop_file" <<EOF
[Desktop Entry]
Type=Application
Name=RK3588 NPU 康复训练
Comment=打开 8085 患者训练页面
Exec=${KIOSK_SCRIPT}
Icon=video-display
Terminal=false
X-GNOME-Autostart-enabled=true
EOF
  chmod +x "$desktop_file"
  chown -R "${RUN_USER}:${RUN_USER}" "$autostart_dir" 2>/dev/null || true
  echo "[OK] Installed desktop kiosk autostart for ${RUN_USER}"
fi

echo "[CHECK] systemctl status ${SERVICE_NAME}"
echo "[PAGE] http://127.0.0.1:8085/train"
