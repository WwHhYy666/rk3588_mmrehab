#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOSTART_DIR="${HOME}/.config/autostart"
KEEP_AWAKE="${PROJECT_DIR}/scripts/rehab_display_keepawake.sh"
KEEP_AWAKE_DESKTOP="${AUTOSTART_DIR}/rehab-display-keepawake.desktop"
PERSISTENT_PROFILE="${HOME}/.cache/rehab-station-browser"
PERSISTENT_CACHE="${HOME}/.cache/rehab-station-browser-cache"
RUNTIME_ROOT="${XDG_RUNTIME_DIR:-/tmp}/rehab-station-display-${UID}"

safe_remove_dedicated_dir() {
  local target="$1"
  if [[ "$target" == "${HOME}/.cache/rehab-station-browser" \
    || "$target" == "${HOME}/.cache/rehab-station-browser-cache" \
    || "$target" == "${XDG_RUNTIME_DIR:-/tmp}/rehab-station-display-${UID}" ]]; then
    rm -rf -- "$target"
    return
  fi
  echo "[FAIL] Refusing to remove unexpected path: $target" >&2
  exit 3
}

echo "[1/6] Stop only the dedicated display manager and Chromium"
pkill -f 'scripts/rehab_display_manager.sh' >/dev/null 2>&1 || true
pkill -f -- "--user-data-dir=${PERSISTENT_PROFILE}" >/dev/null 2>&1 || true
pkill -f -- "--user-data-dir=${RUNTIME_ROOT}/profile" >/dev/null 2>&1 || true
sleep 1

echo "[2/6] Remove every legacy browser/display-manager autostart entry"
mkdir -p "$AUTOSTART_DIR"
rm -f -- \
  "${AUTOSTART_DIR}/rehab-station-display-manager.desktop" \
  "${AUTOSTART_DIR}/rehab-station-browser.desktop" \
  "${AUTOSTART_DIR}/rehab-npu-8085.desktop" 2>/dev/null || true
for entry in "${AUTOSTART_DIR}"/*.desktop; do
  [[ -f "$entry" ]] || continue
  if grep -Eq 'rehab_display_manager\.sh|open_rehab_station_kiosk\.sh|switch_rehab_mode_desktop\.sh' "$entry"; then
    rm -f -- "$entry"
  fi
done

echo "[3/6] Remove only the dedicated persistent Chromium profile/cache"
safe_remove_dedicated_dir "$PERSISTENT_PROFILE"
safe_remove_dedicated_dir "$PERSISTENT_CACHE"
safe_remove_dedicated_dir "$RUNTIME_ROOT"

echo "[4/6] Install display-only keep-awake autostart (does not start 8085)"
cat >"$KEEP_AWAKE_DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Rehab Display Keep Awake
Comment=Prevent RK3588 display blanking; does not start camera, NPU, backend or browser
Exec=/usr/bin/env bash ${KEEP_AWAKE}
Path=${PROJECT_DIR}
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=2
EOF
chmod +x "$KEEP_AWAKE" "$KEEP_AWAKE_DESKTOP"

echo "[5/6] Keep rehabilitation services manual-start only"
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl disable rehab-station-mode.service rehab-station-qwen.service rehab-station-npu-8085.service >/dev/null 2>&1 || true
  sudo systemctl stop rehab-station-mode.service rehab-station-qwen.service rehab-station-npu-8085.service >/dev/null 2>&1 || true
  systemctl --user disable --now rehab-station-display-manager.service rehab-station-browser.service >/dev/null 2>&1 || true
fi

echo "[6/6] Apply persistent desktop power settings now"
pkill -f 'scripts/rehab_display_keepawake.sh' >/dev/null 2>&1 || true
nohup bash "$KEEP_AWAKE" >/dev/null 2>&1 &

echo "[OK] Display stability repair installed."
echo "[OK] 8085, camera, NPU and Chromium still start only after double-clicking the desktop icon."
echo "[NEXT] Double-click the 8085 icon. A reboot is not required."
