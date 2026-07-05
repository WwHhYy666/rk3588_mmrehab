#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export POSE_BACKEND="${POSE_BACKEND:-rknn}"
export RKNN_POSE_PIPELINE="${RKNN_POSE_PIPELINE:-yolov8_pose}"
export RKNN_POSE_MODEL="${RKNN_POSE_MODEL:-/home/elf/models/yolov8n-pose.rknn}"
export RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
export RK_CAMERA_OPEN_MODE="${RK_CAMERA_OPEN_MODE:-opencv}"
export RK_CAMERA_WIDTH="${RK_CAMERA_WIDTH:-640}"
export RK_CAMERA_HEIGHT="${RK_CAMERA_HEIGHT:-480}"
export RKNN_STREAM_WIDTH="${RKNN_STREAM_WIDTH:-640}"
export RKNN_STREAM_HEIGHT="${RKNN_STREAM_HEIGHT:-360}"
export RKNN_POSE_INPUT_SIZE="${RKNN_POSE_INPUT_SIZE:-640}"
export RKNN_POSE_CONF_THRES="${RKNN_POSE_CONF_THRES:-0.35}"
export RKNN_POSE_NMS_THRES="${RKNN_POSE_NMS_THRES:-0.45}"
export RKNN_POSE_KEYPOINT_THRES="${RKNN_POSE_KEYPOINT_THRES:-0.12}"
export RKNN_POSE_TOPK="${RKNN_POSE_TOPK:-50}"
export RKNN_POSE_MAX_DET="${RKNN_POSE_MAX_DET:-1}"
export RKNN_KEYPOINT_DECODE_MODE="${RKNN_KEYPOINT_DECODE_MODE:-auto}"
export RKNN_KEYPOINT_ANCHOR_ORDER="${RKNN_KEYPOINT_ANCHOR_ORDER:-auto}"
export RKNN_MIN_PERSON_BOX_HEIGHT_RATIO="${RKNN_MIN_PERSON_BOX_HEIGHT_RATIO:-0.42}"
export RKNN_MIN_PERSON_BOX_AREA_RATIO="${RKNN_MIN_PERSON_BOX_AREA_RATIO:-0.08}"

if [[ ! -f "$RKNN_POSE_MODEL" ]]; then
  echo "RKNN_POSE_MODEL not found: $RKNN_POSE_MODEL" >&2
  echo "Run scripts/prepare_yolov8_pose_model.sh first, or set RKNN_POSE_MODEL to an existing yolov8n-pose.rknn." >&2
  exit 2
fi

echo "Using YOLOv8-Pose RKNN model: $RKNN_POSE_MODEL"
echo "Camera: $RK_CAMERA_DEVICE ${RK_CAMERA_WIDTH}x${RK_CAMERA_HEIGHT}; stream ${RKNN_STREAM_WIDTH}x${RKNN_STREAM_HEIGHT}"
python3 prescription/banzi/record_prescription_http.py
