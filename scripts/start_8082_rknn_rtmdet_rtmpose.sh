#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export POSE_BACKEND="${POSE_BACKEND:-rknn}"
export RKNN_POSE_PIPELINE="${RKNN_POSE_PIPELINE:-rtmdet_rtmpose}"
export RKNN_DET_MODEL="${RKNN_DET_MODEL:-rknn/rtmdet_fp16.rknn}"
export RKNN_RTMPOSE_MODEL="${RKNN_RTMPOSE_MODEL:-rknn/rtmpose_fp16.rknn}"
export RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"
export RK_CAMERA_OPEN_MODE="${RK_CAMERA_OPEN_MODE:-opencv}"
export RK_CAMERA_WIDTH="${RK_CAMERA_WIDTH:-640}"
export RK_CAMERA_HEIGHT="${RK_CAMERA_HEIGHT:-480}"
export RKNN_STREAM_WIDTH="${RKNN_STREAM_WIDTH:-640}"
export RKNN_STREAM_HEIGHT="${RKNN_STREAM_HEIGHT:-360}"
export RKNN_MAX_POSE_PERSONS="${RKNN_MAX_POSE_PERSONS:-1}"
export RKNN_DET_SCORE_THRES="${RKNN_DET_SCORE_THRES:-0.50}"
export RKNN_POSE_KEYPOINT_THRES="${RKNN_POSE_KEYPOINT_THRES:-0.20}"
export RKNN_DET_INTERVAL="${RKNN_DET_INTERVAL:-8}"
export RKNN_DET_CACHE_SECONDS="${RKNN_DET_CACHE_SECONDS:-1.5}"
export RKNN_DET_NMS_PRE="${RKNN_DET_NMS_PRE:-100}"
export RKNN_POSE_INTERVAL="${RKNN_POSE_INTERVAL:-2}"
export RKNN_POSE_CACHE_SECONDS="${RKNN_POSE_CACHE_SECONDS:-1.0}"
export RKNN_MIN_PERSON_BOX_HEIGHT_RATIO="${RKNN_MIN_PERSON_BOX_HEIGHT_RATIO:-0.42}"
export RKNN_MIN_PERSON_BOX_AREA_RATIO="${RKNN_MIN_PERSON_BOX_AREA_RATIO:-0.08}"
export RKNN_PERSON_SELECT="${RKNN_PERSON_SELECT:-largest_center}"

echo "Using system librknnrt.so. Run scripts/install_rknnrt_system.sh first if the RKNN runtime is still old."
echo "For RTMDet nano/tiny candidates, run scripts/check_rtmdet_light_model.sh before starting 8082."
python3 prescription/banzi/record_prescription_http.py
