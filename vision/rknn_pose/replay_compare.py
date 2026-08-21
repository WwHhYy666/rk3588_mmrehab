from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    parser = argparse.ArgumentParser(description="Compare MediaPipe and RKNN pose on the same replay video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--rknn-model", required=True)
    parser.add_argument("--action", default="seated_knee_extension")
    parser.add_argument("--side-mode", default="auto", choices=["auto", "left", "right"])
    parser.add_argument("--out", default="outputs/replay_compare")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_compare(args)
    csv_path = out_dir / "angle_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()) if rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try_write_plot(rows, out_dir / "angle_curves.png")
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path), **summary}, ensure_ascii=False, indent=2))
    return 0


def run_compare(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    import mediapipe as mp

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {args.video}")
    backend = RKNNPoseBackend(model_path=args.rknn_model)
    selector = StablePersonSelector()
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
    action_config = ACTION_RULES.get(args.action, ACTION_RULES["seated_knee_extension"])
    rows: list[dict[str, object]] = []
    rknn_frames = 0
    mediapipe_frames = 0
    missing_frames = 0
    start = time.perf_counter()
    try:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if args.max_frames and frame_index >= args.max_frames:
                break
            frame_index += 1
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
            mp_result = pose.process(rgb)
            mp_angle = None
            if mp_result.pose_landmarks:
                mp_angle = _mediapipe_angle(mp_result, action_config, args.side_mode)
                mediapipe_frames += 1 if mp_angle is not None else 0

            rknn_result = backend.infer(frame)
            detections = list(rknn_result.meta.get("detections") or [])
            adapted = adapt_rknn_pose_frame(
                detections,
                frame_width=width,
                frame_height=height,
                action_config=action_config,
                side_mode=args.side_mode,
                selector=selector,
                visibility_threshold=backend.keypoint_thres,
            )
            selected = adapted["selected_result"]
            rknn_angle = selected.get("selected_target_angle")
            rknn_frames += 1 if rknn_angle is not None else 0
            missing_frames += 0 if selected.get("quality_ok") else 1
            rows.append(
                {
                    "frame_index": frame_index,
                    "mediapipe_angle": mp_angle,
                    "rknn_angle": rknn_angle,
                    "rknn_quality_ok": selected.get("quality_ok"),
                    "missing_keypoints": ",".join(selected.get("missing_keypoints") or []),
                    "person_count": selected.get("person_count"),
                    "multi_person_warning": selected.get("multi_person_warning"),
                    "keypoint_decode_mode": rknn_result.meta.get("keypoint_decode_mode"),
                    "keypoint_anchor_order": rknn_result.meta.get("keypoint_anchor_order"),
                    "keypoint_geometry_score_range": json.dumps(rknn_result.meta.get("keypoint_geometry_score_range")),
                }
            )
    finally:
        backend.release()
        pose.close()
        cap.release()
    elapsed = max(time.perf_counter() - start, 1e-6)
    return rows, {
        "frames": len(rows),
        "fps_overall": round(len(rows) / elapsed, 2),
        "mediapipe_usable_frames": mediapipe_frames,
        "rknn_usable_frames": rknn_frames,
        "rknn_quality_missing_frames": missing_frames,
    }


def try_write_plot(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    xs = [int(row["frame_index"]) for row in rows]
    mp_values = [row.get("mediapipe_angle") for row in rows]
    rknn_values = [row.get("rknn_angle") for row in rows]
    plt.figure(figsize=(10, 4))
    plt.plot(xs, mp_values, label="MediaPipe")
    plt.plot(xs, rknn_values, label="RKNN")
    plt.xlabel("frame")
    plt.ylabel("angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _mediapipe_angle(result: object, action_config: dict[str, object], side_mode: str) -> float | None:
    landmarks = result.pose_landmarks.landmark
    side = "right" if side_mode == "right" else "left"
    if side_mode == "auto":
        left_vis = _side_visibility(landmarks, "left")
        right_vis = _side_visibility(landmarks, "right")
        side = "right" if right_vis > left_vis else "left"
    indices = {
        "left": {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27},
        "right": {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28},
    }[side]
    points = []
    for name in action_config.get("point_names", ["hip", "knee", "ankle"]):
        landmark = landmarks[indices[str(name)]]
        if landmark.visibility < 0.30:
            return None
        points.append((landmark.x, landmark.y))
    from vision.rknn_pose.pose_frame_adapter import calculate_angle, calculate_knee_raise_height_ratio, target_angle_from_included

    if str(action_config.get("metric_kind", "angle")) == "knee_raise_height_ratio":
        return calculate_knee_raise_height_ratio(points, [str(name) for name in action_config.get("point_names", ["shoulder", "hip", "knee"])])
    return target_angle_from_included(calculate_angle(points), str(action_config.get("angle_kind", "included")))


def _side_visibility(landmarks: object, side: str) -> float:
    indices = {"left": [11, 23, 25, 27], "right": [12, 24, 26, 28]}[side]
    values = [landmarks[index].visibility for index in indices]
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
