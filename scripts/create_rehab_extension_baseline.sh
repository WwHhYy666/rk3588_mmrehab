#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

baseline_dir="${REHAB_EXTENSION_BASELINE_DIR:-runtime/stable_baselines/rehab_extensions_predeploy}"
project_root="$(pwd -P)"
case "$(realpath -m "$baseline_dir")" in
  "$project_root"/runtime/stable_baselines/*) ;;
  *) echo "[FAIL] Baseline directory must stay under runtime/stable_baselines" >&2; exit 2 ;;
esac
if [[ -e "$baseline_dir" && "${1:-}" != "--force" ]]; then
  echo "[FAIL] Baseline already exists: $baseline_dir" >&2
  echo "Use --force only after confirming the existing baseline is no longer needed." >&2
  exit 2
fi

files=(
  "prescription/banzi/static/home.js"
  "prescription/banzi/record_prescription_http.py"
  "prescription/banzi/npu_rehab_8085.py"
  "scripts/rehab_display_manager.sh"
  "scripts/open_rehab_station_kiosk.sh"
  "prescription/banzi/static/train.js"
)

expected_display_manager="5dcd037c9bfa8c0333c105ca6173aa9220d72fc524eff94463211369cb8296e9"
expected_kiosk="ba53813da3b8d5bd382ee267f350f31827bbf3db448868a4626e756fe779a75f"
expected_train_js="f3c741ccc972ad1ef3ab5ab3471a137681359ae584ade12c3e3f1025f9bcd86b"

check_locked_hash() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "[FAIL] Stable display contract mismatch: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 3
  fi
}

check_locked_hash "scripts/rehab_display_manager.sh" "$expected_display_manager"
check_locked_hash "scripts/open_rehab_station_kiosk.sh" "$expected_kiosk"
check_locked_hash "prescription/banzi/static/train.js" "$expected_train_js"

temporary="${baseline_dir}.tmp.$$"
[[ "$(realpath -m "$temporary")" == "$project_root"/runtime/stable_baselines/* ]] || exit 5
rm -rf -- "$temporary"
mkdir -p "$temporary/files"
for path in "${files[@]}"; do
  [[ -f "$path" ]] || { echo "[FAIL] Missing baseline file: $path" >&2; exit 4; }
  mkdir -p "$temporary/files/$(dirname "$path")"
  cp -a -- "$path" "$temporary/files/$path"
done
(
  cd "$temporary/files"
  sha256sum "${files[@]}" > ../SHA256SUMS
)
printf '%s\n' "$(date -Iseconds)" > "$temporary/CREATED_AT"
printf '%s\n' "REHAB_EXTENDED_GROUPS=0" > "$temporary/FEATURE_STATE"
[[ "$(realpath -m "$baseline_dir")" == "$project_root"/runtime/stable_baselines/* ]] || exit 5
rm -rf -- "$baseline_dir"
mkdir -p "$(dirname "$baseline_dir")"
mv "$temporary" "$baseline_dir"
echo "[OK] Stable baseline created: $baseline_dir"
cat "$baseline_dir/SHA256SUMS"
