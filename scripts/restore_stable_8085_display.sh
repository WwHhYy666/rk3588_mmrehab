#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

expected_manager="5dcd037c9bfa8c0333c105ca6173aa9220d72fc524eff94463211369cb8296e9"
expected_kiosk="ba53813da3b8d5bd382ee267f350f31827bbf3db448868a4626e756fe779a75f"
expected_train="f3c741ccc972ad1ef3ab5ab3471a137681359ae584ade12c3e3f1025f9bcd86b"

check_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "[FAIL] Missing stable file: $path" >&2; exit 2; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[FAIL] $path is not the confirmed stable version" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 3
  fi
  echo "[OK] $path $actual"
}

echo "[1/5] Verify the confirmed non-white-screen/non-lag display baseline"
check_hash "scripts/rehab_display_manager.sh" "$expected_manager"
check_hash "scripts/open_rehab_station_kiosk.sh" "$expected_kiosk"
check_hash "prescription/banzi/static/train.js" "$expected_train"

echo "[2/5] Keep optional extensions disabled"
if [[ -f scripts/set_rehab_extensions.sh ]]; then
  bash scripts/set_rehab_extensions.sh off
else
  mkdir -p runtime
  printf '%s\n' 0 > runtime/rehab_extensions.flag
fi

echo "[3/5] Remove browser/display-manager autostart; keep only display keep-awake"
autostart_file="${HOME}/.config/autostart/rehab-station-display-manager.desktop"
rm -f -- "$autostart_file"

echo "[4/5] Stop only the dedicated display manager and its Chromium"
pkill -f 'scripts/rehab_display_manager.sh' >/dev/null 2>&1 || true
sleep 1
legacy_profile_dir="${HOME}/.cache/rehab-station-browser"
runtime_root="${XDG_RUNTIME_DIR:-/tmp}/rehab-station-display-${UID}"
profile_dir="${REHAB_BROWSER_PROFILE_DIR:-${runtime_root}/profile}"
pkill -f -- "--user-data-dir=${legacy_profile_dir}" >/dev/null 2>&1 || true
pkill -f -- "--user-data-dir=${profile_dir}" >/dev/null 2>&1 || true

echo "[5/5] Remove only stale Chromium singleton locks"
rm -f -- \
  "${profile_dir}/SingletonCookie" \
  "${profile_dir}/SingletonLock" \
  "${profile_dir}/SingletonSocket" 2>/dev/null || true
rm -rf -- "${legacy_profile_dir}" "${HOME}/.cache/rehab-station-browser-cache" "${runtime_root}"

echo "[OK] Stable display files are restored and the white browser is closed."
echo "[NEXT] The screen should now be on the Ubuntu desktop. Double-click the 8085 icon to start; do not reboot."
