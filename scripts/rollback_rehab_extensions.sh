#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

baseline_dir="${REHAB_EXTENSION_BASELINE_DIR:-runtime/stable_baselines/rehab_extensions_predeploy}"
service_name="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
project_root="$(pwd -P)"
case "$(realpath -m "$baseline_dir")" in
  "$project_root"/runtime/stable_baselines/*) ;;
  *) echo "[FAIL] Baseline directory must stay under runtime/stable_baselines" >&2; exit 2 ;;
esac
[[ -d "$baseline_dir/files" && -f "$baseline_dir/SHA256SUMS" ]] || {
  echo "[FAIL] Missing stable baseline: $baseline_dir" >&2
  exit 2
}

echo "[1/5] Disable extension flag and stop extension state"
bash scripts/set_rehab_extensions.sh off
curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8085/api/extension/stop >/dev/null 2>&1 || true

echo "[2/5] Verify baseline manifest"
(
  cd "$baseline_dir/files"
  sha256sum -c ../SHA256SUMS
)

echo "[3/5] Restore stable shared files"
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  mkdir -p "$(dirname "$path")"
  cp -a -- "$baseline_dir/files/$path" "$path"
done < <(awk '{print $2}' "$baseline_dir/SHA256SUMS")

echo "[4/5] Restart only 8085"
if systemctl list-unit-files "$service_name" --no-legend 2>/dev/null | grep -q "^${service_name}[[:space:]]"; then
  if [[ "$(id -u)" == "0" ]]; then
    systemctl restart "$service_name"
  else
    sudo systemctl restart "$service_name"
  fi
else
  bash scripts/stop_npu_rehab_8085.sh || true
  bash scripts/start_npu_rehab_8085.sh
fi

echo "[5/5] Verify /status, /train and stable three-action configuration"
python3 - <<'PY'
import json
import time
from pathlib import Path
from urllib.request import urlopen

deadline = time.time() + 120
status = None
while time.time() < deadline:
    try:
        with urlopen("http://127.0.0.1:8085/status", timeout=3) as response:
            status = json.load(response)
        if status.get("service_mode") == "npu_rehab":
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("[FAIL] 8085 /status did not recover")

if status.get("extension_enabled"):
    raise SystemExit("[FAIL] extension flag is still enabled")
with urlopen("http://127.0.0.1:8085/train", timeout=3) as response:
    if response.status != 200:
        raise SystemExit("[FAIL] /train is unavailable")
required = [
    Path("evaluate/configs/npu/sit_to_stand.yaml"),
    Path("evaluate/configs/npu/standing_hamstring_curl.yaml"),
    Path("evaluate/configs/npu/seated_knee_raise.yaml"),
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("[FAIL] missing stable action configs: " + ", ".join(missing))
print("[OK] 8085 healthy, extension disabled, /train available, stable three-action configs present")
PY

echo "[OK] Rollback complete. 8082 was not touched."
