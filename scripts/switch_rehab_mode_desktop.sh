#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1

target="${1:-}"
log_dir="${HOME}/.cache/rehab-station-display"
log_file="${log_dir}/desktop-switch.log"
mkdir -p "$log_dir"
exec >>"$log_file" 2>&1

echo ""
echo "===== $(date '+%F %T') switch ${target} ====="
if ! bash scripts/switch_rehab_mode.sh "$target"; then
  command -v notify-send >/dev/null 2>&1 && notify-send "康复训练切换失败" "请查看 ${log_file}" >/dev/null 2>&1 || true
  exit 1
fi

if ! pgrep -f 'scripts/rehab_display_manager.sh' >/dev/null 2>&1; then
  nohup bash scripts/rehab_display_manager.sh >>"${log_dir}/manager-start.log" 2>&1 &
fi

command -v notify-send >/dev/null 2>&1 && notify-send "康复训练" "已切换到 ${target}" >/dev/null 2>&1 || true
