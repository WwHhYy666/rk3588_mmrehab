from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from vision.rknn_pose.pose_frame_adapter import StablePersonSelector, adapt_rknn_pose_frame
from vision.rknn_pose.rknn_backend import RKNNPoseBackend


ACTION_RULES = {
    "seated_knee_extension": {"angle_kind": "flexion", "point_names": ["hip", "knee", "ankle"]},
    "standing_hamstring_curl": {"angle_kind": "flexion", "point_names": ["hip", "knee", "ankle"]},
    "seated_knee_raise": {"angle_kind": "flexion", "metric_kind": "knee_raise_height_ratio", "point_names": ["shoulder", "hip", "knee"]},
    "sit_to_stand": {"angle_kind": "included", "point_names": ["hip", "knee", "ankle"]},
    "knee_flexion": {"angle_kind": "flexion", "point_names": ["hip", "knee", "ankle"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test RKNN pose without starting 8082.")
    parser.add_argument("--pipeline", default="yolov8_pose", choices=["yolov8_pose", "rtmdet_rtmpose", "rtmpose_fixed"])
    parser.add_argument("--model", help="Path to yolov8n-pose.rknn for the legacy YOLOv8-Pose pipeline")
    parser.add_argument("--det-model", help="Path to rtmdet_fp16.rknn for the RTMDet + RTMPose pipeline")
    parser.add_argument("--pose-model", help="Path to rtmpose_fp16.rknn for the RTMDet + RTMPose pipelines")
    parser.add_argument("--fixed-bbox", help="Fixed RTMPose ROI as x1,y1,x2,y2 for the rtmpose_fixed pipeline")
    parser.add_argument("--image", help="Static image path")
    parser.add_argument("--camera", help="Camera device, e.g. auto, 0, or /dev/video11")
    parser.add_argument("--action", default="seated_knee_extension")
    parser.add_argument("--side-mode", default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--out", default="outputs/rknn_pose_smoke.jpg")
    parser.add_argument("--require-person", action="store_true", help="Fail if no valid person is selected.")
    parser.add_argument("--fail-on-postprocess-error", action="store_true", help="Fail if RKNN postprocess reports an error.")
    parser.add_argument("--max-det-ms", type=float, help="Fail if RTMDet inference time is above this value.")
    parser.add_argument("--max-total-ms", type=float, help="Fail if total pose time is above this value.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_backend_env(args)
    frame = read_frame(args)
    backend = RKNNPoseBackend(model_path=args.model)
    try:
        result = backend.infer(frame)
    finally:
        backend.release()
    detections = list(result.meta.get("detections") or [])
    height, width = frame.shape[:2]
    adapted = adapt_rknn_pose_frame(
        detections,
        frame_width=width,
        frame_height=height,
        action_config=ACTION_RULES.get(args.action, ACTION_RULES["seated_knee_extension"]),
        side_mode=args.side_mode,
        selector=StablePersonSelector(),
        visibility_threshold=backend.keypoint_thres,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if result.annotated_frame is not None:
        cv2.imwrite(str(out_path), result.annotated_frame)
    selected = adapted["selected_result"]
    payload = {
        "backend": "rknn",
        "actual_backend": "rknn",
        "pipeline": result.meta.get("rknn_pipeline"),
        "fps": result.fps,
        "person_count": selected.get("person_count"),
        "selected_person_reason": selected.get("selected_person_reason"),
        "selected_side": selected.get("side"),
        "selected_angle": selected.get("selected_target_angle"),
        "quality_ok": selected.get("quality_ok"),
        "quality_message": selected.get("quality_message"),
        "missing_keypoints": selected.get("missing_keypoints"),
        "multi_person_warning": selected.get("multi_person_warning"),
        "rknn_decoder": result.meta.get("rknn_decoder"),
        "output_shapes": result.meta.get("output_shapes"),
        "det_output_shapes": result.meta.get("det_output_shapes"),
        "pose_output_shapes": result.meta.get("pose_output_shapes"),
        "fixed_bbox": result.meta.get("fixed_bbox"),
        "fixed_bbox_requested": result.meta.get("fixed_bbox_requested"),
        "fixed_bbox_mode": result.meta.get("fixed_bbox_mode"),
        "rtmdet_pairs": result.meta.get("rtmdet_pairs"),
        "rtmdet_compatible": rtmdet_compatible(result.meta),
        "performance_ms": result.meta.get("performance_ms"),
        "keypoint_conf_range": result.meta.get("keypoint_conf_range"),
        "keypoint_xy_range": result.meta.get("keypoint_xy_range"),
        "keypoint_decode_mode": result.meta.get("keypoint_decode_mode"),
        "keypoint_anchor_order": result.meta.get("keypoint_anchor_order"),
        "keypoint_anchor_order_setting": result.meta.get("keypoint_anchor_order_setting"),
        "keypoint_geometry_score_range": result.meta.get("keypoint_geometry_score_range"),
        "keypoint_global_index_range": result.meta.get("keypoint_global_index_range"),
        "keypoint_candidate_count_range": result.meta.get("keypoint_candidate_count_range"),
        "keypoint_branch_diagnostics": result.meta.get("keypoint_branch_diagnostics"),
        "keypoint_raw_shape": result.meta.get("keypoint_raw_shape"),
        "keypoint_raw_xy_range": result.meta.get("keypoint_raw_xy_range"),
        "keypoint_restored_xy_range": result.meta.get("keypoint_restored_xy_range"),
        "postprocess_error": result.meta.get("postprocess_error"),
        "rehab_keypoints": adapted.get("rehab_keypoints"),
        "annotated_image": str(out_path),
    }
    failures = validate_smoke_result(args, payload)
    payload["ok"] = not failures
    payload["failures"] = failures
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 3


def open_capture(source):
    normalized = normalize_camera_source(source)
    cap = cv2.VideoCapture(normalized, cv2.CAP_ANY)
    return cap


def configure_backend_env(args: argparse.Namespace) -> None:
    os.environ["RKNN_POSE_PIPELINE"] = args.pipeline
    if args.pipeline == "rtmdet_rtmpose":
        if args.det_model:
            os.environ["RKNN_DET_MODEL"] = args.det_model
        if args.pose_model:
            os.environ["RKNN_RTMPOSE_MODEL"] = args.pose_model
    elif args.pipeline == "rtmpose_fixed":
        if args.pose_model:
            os.environ["RKNN_RTMPOSE_MODEL"] = args.pose_model
        if args.fixed_bbox:
            os.environ["RKNN_RTMPOSE_FIXED_BBOX"] = args.fixed_bbox
    elif args.model:
        os.environ["RKNN_POSE_MODEL"] = args.model


def rtmdet_compatible(meta: dict[str, Any]) -> bool | None:
    if meta.get("rknn_pipeline") != "rtmdet_rtmpose":
        return None
    if meta.get("postprocess_error"):
        return False
    pairs = meta.get("rtmdet_pairs")
    if not isinstance(pairs, list) or not pairs:
        return False
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        if not {"stride", "cls_out", "bbox_out"}.issubset(pair):
            return False
    return True


def validate_smoke_result(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    postprocess_error = payload.get("postprocess_error")
    if args.fail_on_postprocess_error and postprocess_error:
        failures.append(f"postprocess_error: {postprocess_error}")
    if args.require_person:
        person_count = payload.get("person_count")
        if not isinstance(person_count, int) or person_count < 1:
            failures.append("person_count < 1")
        if not payload.get("quality_ok"):
            failures.append(f"pose quality failed: {payload.get('quality_message')}")
    if args.pipeline == "rtmdet_rtmpose" and payload.get("rtmdet_compatible") is False:
        failures.append("RTMDet output heads are not compatible with the current cls/bbox decoder.")

    performance = payload.get("performance_ms") or {}
    if args.max_det_ms is not None:
        det_ms = performance.get("det_inference_ms")
        if not isinstance(det_ms, (int, float)) or float(det_ms) > args.max_det_ms:
            failures.append(f"det_inference_ms {det_ms} > {args.max_det_ms}")
    if args.max_total_ms is not None:
        total_ms = performance.get("total_pose_ms")
        if not isinstance(total_ms, (int, float)) or float(total_ms) > args.max_total_ms:
            failures.append(f"total_pose_ms {total_ms} > {args.max_total_ms}")
    return failures


def read_frame(args: argparse.Namespace):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"failed to read image: {args.image}")
        return frame
    if args.camera:
        for source in camera_candidates(args.camera):
            cap = open_capture(source)
            success, frame = cap.read()
            cap.release()
            if success:
                return frame
        raise SystemExit(f"failed to read camera frame: {args.camera}")
    raise SystemExit("please provide --image or --camera")


def camera_candidates(value: str):
    text = str(value or "").strip()
    if text.lower() != "auto":
        if text.isdigit():
            return [int(text), text]
        return [text]
    candidates = []
    dev_dir = Path("/dev")
    if dev_dir.exists():
        candidates.extend(str(path) for path in sorted(dev_dir.glob("video*")))
    candidates.extend(range(0, 16))
    return candidates


def normalize_camera_source(value):
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"/dev/video(\d+)", text)
    if match:
        return int(match.group(1))
    return text


if __name__ == "__main__":
    raise SystemExit(main())
