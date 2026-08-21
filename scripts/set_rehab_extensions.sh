#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "Usage: bash scripts/set_rehab_extensions.sh on|off [--restart]" >&2
  exit 2
}

case "${1:-}" in
  on|1|true) value=1 ;;
  off|0|false) value=0 ;;
  *) usage ;;
esac

flag_file="runtime/rehab_extensions.flag"
mkdir -p "$(dirname "$flag_file")"
temporary="${flag_file}.tmp.$$"
printf '%s\n' "$value" > "$temporary"
mv -f "$temporary" "$flag_file"
echo "REHAB_EXTENDED_GROUPS=$value"

if [[ "${2:-}" == "--restart" ]]; then
  bash scripts/switch_rehab_mode.sh npu_8085
elif [[ -n "${2:-}" ]]; then
  usage
fi
