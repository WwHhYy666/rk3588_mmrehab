#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RKNN_DET_MODEL="${RKNN_DET_MODEL:-rknn/rtmdet_nano_or_tiny.rknn}"
export RKNN_RTMPOSE_MODEL="${RKNN_RTMPOSE_MODEL:-rknn/rtmpose_fp16.rknn}"
export RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
export RKNN_MAX_POSE_PERSONS="${RKNN_MAX_POSE_PERSONS:-1}"
export RKNN_DET_SCORE_THRES="${RKNN_DET_SCORE_THRES:-0.50}"
export RKNN_POSE_KEYPOINT_THRES="${RKNN_POSE_KEYPOINT_THRES:-0.20}"
export RKNN_DET_NMS_PRE="${RKNN_DET_NMS_PRE:-100}"
export RKNN_DET_INTERVAL="${RKNN_DET_INTERVAL:-1}"
export RKNN_POSE_INTERVAL="${RKNN_POSE_INTERVAL:-1}"

if [[ ! -f "$RKNN_DET_MODEL" ]]; then
  echo "RKNN_DET_MODEL not found: $RKNN_DET_MODEL" >&2
  echo "Set RKNN_DET_MODEL to a converted RTMDet nano/tiny .rknn before running this check." >&2
  exit 2
fi

if [[ ! -f "$RKNN_RTMPOSE_MODEL" ]]; then
  echo "RKNN_RTMPOSE_MODEL not found: $RKNN_RTMPOSE_MODEL" >&2
  exit 2
fi

python3 vision/rknn_pose/smoke_test_rknn_pose.py \
  --pipeline rtmdet_rtmpose \
  --det-model "$RKNN_DET_MODEL" \
  --pose-model "$RKNN_RTMPOSE_MODEL" \
  --camera "$RK_CAMERA_DEVICE" \
  --out outputs/rtmdet_light_smoke.jpg \
  --fail-on-postprocess-error \
  --require-person \
  --max-det-ms 60 \
  --max-total-ms 100
