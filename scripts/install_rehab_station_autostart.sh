#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_USER="${RUN_USER:-${SUDO_USER:-$(id -un)}}"
if [[ "$RUN_USER" == "root" ]]; then
  RUN_USER="elf"
fi
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
if [[ -z "$RUN_HOME" ]]; then
  echo "[FAIL] Cannot find home directory for user: ${RUN_USER}" >&2
  exit 2
fi

CPU_SERVICE="${REHAB_CPU_SERVICE:-rehab-station-qwen.service}"
NPU_SERVICE="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
MODE_SERVICE="${REHAB_MODE_SERVICE:-rehab-station-mode.service}"
SYSTEMD_DIR="/etc/systemd/system"
AUTOSTART_DIR="${RUN_HOME}/.config/autostart"
DESKTOP_DIR="${RUN_HOME}/Desktop"
MODE_FILE="${PROJECT_DIR}/runtime/selected_rehab_mode"
REFRESH_FILE="${PROJECT_DIR}/runtime/display_refresh"
DEFAULT_MODE="${REHAB_DEFAULT_MODE:-npu_8085}"
AUTO_START_MODE="0"
ENABLE_OS_AUTOLOGIN="${REHAB_ENABLE_OS_AUTOLOGIN:-1}"
SYSTEMCTL_BIN="$(command -v systemctl)"

CPU_START="${PROJECT_DIR}/scripts/start_rehab_station_qwen.sh"
CPU_STOP="${PROJECT_DIR}/scripts/stop_rehab_station_qwen.sh"
NPU_START="${PROJECT_DIR}/scripts/start_npu_rehab_8085.sh"
NPU_STOP="${PROJECT_DIR}/scripts/stop_npu_rehab_8085.sh"
MODE_START="${PROJECT_DIR}/scripts/start_selected_rehab_mode.sh"
DISPLAY_MANAGER="${PROJECT_DIR}/scripts/rehab_display_manager.sh"
DISPLAY_KEEP_AWAKE="${PROJECT_DIR}/scripts/rehab_display_keepawake.sh"
DESKTOP_SWITCH="${PROJECT_DIR}/scripts/switch_rehab_mode_desktop.sh"
SWITCH_CPU="${PROJECT_DIR}/scripts/switch_to_cpu_8082.sh"
SWITCH_NPU="${PROJECT_DIR}/scripts/switch_to_npu_8085.sh"
KIOSK_SCRIPT="${PROJECT_DIR}/scripts/open_rehab_station_kiosk.sh"

case "$DEFAULT_MODE" in
  cpu_8082|npu_8085) ;;
  *) echo "[FAIL] REHAB_DEFAULT_MODE must be cpu_8082 or npu_8085." >&2; exit 2 ;;
esac
for path in "$CPU_START" "$CPU_STOP" "$NPU_START" "$NPU_STOP" "$MODE_START" "$DISPLAY_MANAGER" "$DISPLAY_KEEP_AWAKE" "$DESKTOP_SWITCH" "$SWITCH_CPU" "$SWITCH_NPU" "$KIOSK_SCRIPT"; do
  if [[ ! -f "$path" ]]; then
    echo "[FAIL] Required script not found: $path" >&2
    exit 2
  fi
done

chmod +x "$CPU_START" "$CPU_STOP" "$NPU_START" "$NPU_STOP" "$MODE_START" \
  "$DISPLAY_MANAGER" "$DISPLAY_KEEP_AWAKE" "$SWITCH_CPU" "$SWITCH_NPU" "$KIOSK_SCRIPT" \
  "$DESKTOP_SWITCH" \
  "${PROJECT_DIR}/scripts/switch_rehab_mode.sh"

install_unit() {
  local service_name="$1" content="$2" tmp
  tmp="$(mktemp)"
  printf '%s\n' "$content" > "$tmp"
  sudo cp "$tmp" "${SYSTEMD_DIR}/${service_name}"
  rm -f "$tmp"
  echo "[OK] Installed ${service_name}"
}

cpu_unit="[Unit]
Description=RK3588 Rehab Station CPU 8082 with local Qwen
After=network-online.target
Wants=network-online.target
Conflicts=${NPU_SERVICE}

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=REHAB_PORT=8082
Environment=POSE_BACKEND=mediapipe
Environment=RK_CAMERA_ENABLED=1
ExecStart=/usr/bin/env bash ${CPU_START}
ExecStop=/usr/bin/env bash ${CPU_STOP}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

npu_unit="[Unit]
Description=RK3588 Rehab Station NPU 8085 YOLOv5n RTMPose
After=network-online.target
Wants=network-online.target
Conflicts=${CPU_SERVICE}

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/env bash ${NPU_START}
ExecStop=/usr/bin/env bash ${NPU_STOP}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

mode_unit="[Unit]
Description=RK3588 Rehab Station persistent CPU/NPU mode selector
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=${PROJECT_DIR}
Environment=REHAB_RUN_USER=${RUN_USER}
ExecStart=/usr/bin/env bash ${MODE_START}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target"

install_unit "$CPU_SERVICE" "$cpu_unit"
install_unit "$NPU_SERVICE" "$npu_unit"
install_unit "$MODE_SERVICE" "$mode_unit"

mkdir -p "${PROJECT_DIR}/runtime"
if [[ ! -f "$MODE_FILE" ]]; then
  printf '%s\n' "$DEFAULT_MODE" > "$MODE_FILE"
  echo "[OK] Initial boot mode: ${DEFAULT_MODE}"
else
  echo "[KEEP] Existing boot mode: $(tr -d ' \t\r\n' < "$MODE_FILE")"
fi
touch "$REFRESH_FILE"
chown "${RUN_USER}:${RUN_USER}" "$MODE_FILE" "$REFRESH_FILE" 2>/dev/null || true

sudoers_tmp="$(mktemp)"
cat > "$sudoers_tmp" <<EOF
Cmnd_Alias REHAB_STATION_MODE = ${SYSTEMCTL_BIN} start ${CPU_SERVICE}, ${SYSTEMCTL_BIN} stop ${CPU_SERVICE}, ${SYSTEMCTL_BIN} restart ${CPU_SERVICE}, ${SYSTEMCTL_BIN} reset-failed ${CPU_SERVICE}, ${SYSTEMCTL_BIN} start ${NPU_SERVICE}, ${SYSTEMCTL_BIN} stop ${NPU_SERVICE}, ${SYSTEMCTL_BIN} restart ${NPU_SERVICE}, ${SYSTEMCTL_BIN} reset-failed ${NPU_SERVICE}
${RUN_USER} ALL=(root) NOPASSWD: REHAB_STATION_MODE
EOF
sudo visudo -cf "$sudoers_tmp" >/dev/null
sudo install -m 0440 "$sudoers_tmp" /etc/sudoers.d/rehab-station-mode
rm -f "$sudoers_tmp"
echo "[OK] Installed limited service-switch permission"

first_xsession() {
  local session
  session="$(find /usr/share/xsessions -maxdepth 1 -type f -name '*.desktop' 2>/dev/null | sort | head -n 1 || true)"
  [[ -n "$session" ]] && basename "$session"
}

configure_lightdm_autologin() {
  local tmp="$(mktemp)"
  cat > "$tmp" <<EOF
[Seat:*]
autologin-user=${RUN_USER}
autologin-user-timeout=0
EOF
  sudo mkdir -p /etc/lightdm/lightdm.conf.d
  sudo cp "$tmp" /etc/lightdm/lightdm.conf.d/50-rehab-station-autologin.conf
  rm -f "$tmp"
  echo "[OK] LightDM autologin configured"
}

configure_sddm_autologin() {
  local tmp="$(mktemp)" session
  session="$(first_xsession || true)"
  cat > "$tmp" <<EOF
[Autologin]
User=${RUN_USER}
EOF
  [[ -n "$session" ]] && printf 'Session=%s\n' "$session" >> "$tmp"
  sudo mkdir -p /etc/sddm.conf.d
  sudo cp "$tmp" /etc/sddm.conf.d/50-rehab-station-autologin.conf
  rm -f "$tmp"
  echo "[OK] SDDM autologin configured"
}

configure_gdm_autologin() {
  local path="$1" tmp="$(mktemp)"
  if [[ -f "$path" ]]; then
    awk -v user="$RUN_USER" '
      BEGIN { in_daemon=0; seen_daemon=0; seen_enable=0; seen_login=0 }
      function missing() {
        if (in_daemon) {
          if (!seen_enable) print "AutomaticLoginEnable=True"
          if (!seen_login) print "AutomaticLogin=" user
        }
      }
      /^\[.*\]/ {
        missing(); in_daemon=($0 == "[daemon]")
        if (in_daemon) { seen_daemon=1; seen_enable=0; seen_login=0 }
        print; next
      }
      in_daemon && /^#?AutomaticLoginEnable=/ { print "AutomaticLoginEnable=True"; seen_enable=1; next }
      in_daemon && /^#?AutomaticLogin=/ { print "AutomaticLogin=" user; seen_login=1; next }
      { print }
      END {
        missing()
        if (!seen_daemon) {
          print ""; print "[daemon]"; print "AutomaticLoginEnable=True"; print "AutomaticLogin=" user
        }
      }
    ' "$path" > "$tmp"
  else
    cat > "$tmp" <<EOF
[daemon]
AutomaticLoginEnable=True
AutomaticLogin=${RUN_USER}
EOF
  fi
  sudo mkdir -p "$(dirname "$path")"
  sudo cp "$tmp" "$path"
  rm -f "$tmp"
  echo "[OK] GDM autologin configured"
}

configure_os_autologin() {
  local dm
  if [[ "$ENABLE_OS_AUTOLOGIN" != "1" ]]; then
    echo "[SKIP] OS autologin disabled"
    return
  fi
  dm="$(basename "$(cat /etc/X11/default-display-manager 2>/dev/null || true)")"
  case "$dm" in
    lightdm) configure_lightdm_autologin ;;
    sddm) configure_sddm_autologin ;;
    gdm|gdm3)
      [[ -d /etc/gdm3 || "$dm" == "gdm3" ]] && configure_gdm_autologin /etc/gdm3/custom.conf || configure_gdm_autologin /etc/gdm/custom.conf
      ;;
    *) echo "[WARN] Display manager not detected; browser will start after manual desktop login." >&2 ;;
  esac
}

configure_os_autologin

mkdir -p "$AUTOSTART_DIR" "$DESKTOP_DIR"
rm -f "${AUTOSTART_DIR}/rehab-station-display-manager.desktop"
for entry in "${AUTOSTART_DIR}"/*.desktop; do
  [[ -f "$entry" ]] || continue
  if grep -Eq 'rehab_display_manager\.sh|open_rehab_station_kiosk\.sh|switch_rehab_mode_desktop\.sh' "$entry"; then
    rm -f -- "$entry"
  fi
done
cat > "${AUTOSTART_DIR}/rehab-display-keepawake.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Rehab Display Keep Awake
Comment=Prevent display blanking without starting rehabilitation services
Exec=/usr/bin/env bash ${DISPLAY_KEEP_AWAKE}
Path=${PROJECT_DIR}
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=2
EOF

cat > "${DESKTOP_DIR}/rehab-cpu-8082.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=康复训练 CPU 8082
Comment=切换到 MediaPipe CPU 康复训练
Exec=${DESKTOP_SWITCH} cpu_8082
Icon=video-display
Terminal=false
EOF

cat > "${DESKTOP_DIR}/rehab-npu-8085.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=康复训练 NPU 8085
Comment=切换到 YOLOv5n 和 RTMPose NPU 康复训练
Exec=${DESKTOP_SWITCH} npu_8085
Icon=applications-engineering
Terminal=false
EOF

rm -f "${DESKTOP_DIR}/rehab-recover-npu-8085.desktop"
chmod +x "${DESKTOP_DIR}/rehab-cpu-8082.desktop" "${DESKTOP_DIR}/rehab-npu-8085.desktop" "${AUTOSTART_DIR}/rehab-display-keepawake.desktop"
chown -R "${RUN_USER}:${RUN_USER}" "$AUTOSTART_DIR" "$DESKTOP_DIR" 2>/dev/null || true
sudo loginctl enable-linger "$RUN_USER" >/dev/null 2>&1 || true

sudo systemctl daemon-reload
sudo systemctl disable --now rehab-station-backend-watchdog.service >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/rehab-station-backend-watchdog.service
pkill -f 'scripts/rehab_backend_watchdog.sh' >/dev/null 2>&1 || true
sudo systemctl daemon-reload
sudo systemctl disable "$CPU_SERVICE" "$NPU_SERVICE" "$MODE_SERVICE" >/dev/null 2>&1 || true
systemctl --user disable --now rehab-station-display-manager.service rehab-station-browser.service >/dev/null 2>&1 || true
sudo systemctl stop "$MODE_SERVICE" "$CPU_SERVICE" "$NPU_SERVICE" >/dev/null 2>&1 || true
pkill -f 'scripts/rehab_display_manager.sh' >/dev/null 2>&1 || true
echo "[OK] Manual desktop switching installed; no rehab service/browser will start at boot"
echo "[MODE] $(tr -d ' \t\r\n' < "$MODE_FILE")"
echo "[CPU] ./scripts/switch_to_cpu_8082.sh"
echo "[NPU] ./scripts/switch_to_npu_8085.sh"
echo "[DESKTOP] 双击 康复训练 CPU 8082 或 康复训练 NPU 8085"
echo "[CHECK] systemctl is-active ${CPU_SERVICE} ${NPU_SERVICE}"
echo "[REBOOT] sudo reboot"
