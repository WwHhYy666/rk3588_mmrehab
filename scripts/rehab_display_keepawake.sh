#!/usr/bin/env bash
set -u

LOG_DIR="${HOME}/.cache/rehab-station-display"
LOG_FILE="${LOG_DIR}/keepawake.log"
INTERVAL_SECONDS="${REHAB_DISPLAY_KEEP_AWAKE_INTERVAL:-30}"
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi
if [[ -z "${XAUTHORITY:-}" && -f "${HOME}/.Xauthority" ]]; then
  export XAUTHORITY="${HOME}/.Xauthority"
fi
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "/run/user/${UID}/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${UID}/bus"
fi
mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1

if [[ "${REHAB_DISPLAY_INHIBITED:-0}" != "1" ]] && command -v systemd-inhibit >/dev/null 2>&1; then
  export REHAB_DISPLAY_INHIBITED=1
  systemd-inhibit \
    --what=idle:sleep \
    --who=rehab-display \
    --why="Keep the rehabilitation display available" \
    --mode=block \
    bash "$0"
  echo "[$(date '+%F %T')] systemd-inhibit ended; continuing with desktop keep-awake fallback"
fi

set_gsettings() {
  command -v gsettings >/dev/null 2>&1 || return 0
  gsettings set "$1" "$2" "$3" >/dev/null 2>&1 || true
}

set_xfconf() {
  command -v xfconf-query >/dev/null 2>&1 || return 0
  xfconf-query -c "$1" -p "$2" -s "$3" >/dev/null 2>&1 || true
}

apply_keepawake() {
  if [[ -n "${DISPLAY:-}" ]] && command -v xset >/dev/null 2>&1; then
    xset dpms force on >/dev/null 2>&1 || true
    xset s off >/dev/null 2>&1 || true
    xset s noblank >/dev/null 2>&1 || true
    xset -dpms >/dev/null 2>&1 || true
  fi
  command -v xscreensaver-command >/dev/null 2>&1 && xscreensaver-command -deactivate >/dev/null 2>&1 || true
  command -v gnome-screensaver-command >/dev/null 2>&1 && gnome-screensaver-command -d >/dev/null 2>&1 || true

  # GNOME / Ubuntu desktop. These settings persist across login and reboot.
  set_gsettings org.gnome.desktop.screensaver lock-enabled false
  set_gsettings org.gnome.desktop.screensaver idle-activation-enabled false
  set_gsettings org.gnome.desktop.session idle-delay "uint32 0"
  set_gsettings org.gnome.settings-daemon.plugins.power idle-dim false
  set_gsettings org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
  set_gsettings org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout 0

  # MATE variants used by some RK3588 Ubuntu images.
  set_gsettings org.mate.screensaver lock-enabled false
  set_gsettings org.mate.screensaver idle-activation-enabled false
  set_gsettings org.mate.session idle-delay 0
  set_gsettings org.mate.power-manager idle-dim-ac false
  set_gsettings org.mate.power-manager sleep-display-ac 0
  set_gsettings org.mate.power-manager sleep-display-battery 0

  # XFCE variants. Existing keys are updated without changing their types.
  set_xfconf xfce4-power-manager /xfce4-power-manager/dpms-enabled false
  set_xfconf xfce4-power-manager /xfce4-power-manager/blank-on-ac 0
  set_xfconf xfce4-power-manager /xfce4-power-manager/blank-on-battery 0
  set_xfconf xfce4-power-manager /xfce4-power-manager/dpms-on-ac-sleep 0
  set_xfconf xfce4-power-manager /xfce4-power-manager/dpms-on-ac-off 0
  set_xfconf xfce4-power-manager /xfce4-power-manager/dpms-on-battery-sleep 0
  set_xfconf xfce4-power-manager /xfce4-power-manager/dpms-on-battery-off 0
  set_xfconf xfce4-screensaver /saver/enabled false
  set_xfconf xfce4-screensaver /lock/enabled false
}

echo "===== $(date '+%F %T') display keep-awake started ====="
while true; do
  apply_keepawake
  sleep "$INTERVAL_SECONDS"
done
