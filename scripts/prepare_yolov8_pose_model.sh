#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-/home/elf/models/yolov8n-pose.rknn}"

if [[ -f "$DEST" ]]; then
  echo "model already exists: $DEST"
  ls -lh "$DEST"
  exit 0
fi

mapfile -t CANDIDATES < <(
  find /home/elf -name "*yolov8*pose*.rknn" 2>/dev/null \
    | grep -vi "_fp" \
    | sort -u
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "No yolov8 pose RKNN model found under /home/elf." >&2
  echo "Expected one of these paths, for example:" >&2
  echo "  /home/elf/rknn_yolov8_pose_demo/model/yolov8n-pose.rknn" >&2
  echo "  /home/elf/yolov8_pose/model/yolov8n-pose.rknn" >&2
  echo "Copy or convert an INT8 yolov8n-pose.rknn first, then rerun this script." >&2
  exit 2
fi

SOURCE="${CANDIDATES[0]}"
echo "using source model: $SOURCE"
sudo mkdir -p "$(dirname "$DEST")"
sudo cp -a "$SOURCE" "$DEST"
sudo chown "$(id -u)":"$(id -g)" "$DEST" 2>/dev/null || true
ls -lh "$DEST"
