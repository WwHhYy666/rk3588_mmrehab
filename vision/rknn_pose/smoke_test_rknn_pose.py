from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Smoke test RKNN YOLOv8-Pose without starting 8082.")
    parser.add_argument("--model", required=True, help="Path to yolov8n-pose.rknn")
    parser.add_argument("--image", help="Static image path")
    parser.add_argument("--camera", help="Camera device, e.g. /dev/video11")
    parser.add_argument("--action", default="seated_knee_extension")
    parser.add_argument("--side-mode", default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--out", default="outputs/rknn_pose_smoke.jpg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def read_frame(args: argparse.Namespace):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"failed to read image: {args.image}")
        return frame
    if args.camera:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise SystemExit(f"failed to read camera frame: {args.camera}")
        return frame
    raise SystemExit("please provide --image or --camera")


if __name__ == "__main__":
    raise SystemExit(main())
