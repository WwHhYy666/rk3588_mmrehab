import cv2
import mediapipe as mp
import json
import os
import time
import math
from collections import deque
from datetime import datetime
from pathlib import Path


# =========================
# 可调参数区
# =========================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe 置信度阈值。太低会抖，太高可能丢帧。
VISIBILITY_THRESHOLD = 0.55

# 平滑窗口。数值越大越稳，但延迟越明显；5 比较适合演示。
SMOOTH_WINDOW_SIZE = 5

# 默认优先使用 MediaPipe 的 3D world landmarks 计算膝关节角度。
# 如果 3D 数据不可用，会自动退回 2D。
PREFER_3D_WORLD_ANGLE = True

# Pose 模型复杂度：0 最快，1 均衡，2 更准但更吃性能。电脑端建议 2，板端可改回 1。
MODEL_COMPLEXITY = 2

# 保存目录：无论从哪个目录运行，都保存到项目根目录下的 prescription_data
PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "prescription" else Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / "prescription_data"


LEFT_KNEE_RULE = {
    "side": "left",
    "target_joint": "left_knee",
    "hip_index": 23,
    "knee_index": 25,
    "ankle_index": 27
}

RIGHT_KNEE_RULE = {
    "side": "right",
    "target_joint": "right_knee",
    "hip_index": 24,
    "knee_index": 26,
    "ankle_index": 28
}


def clamp(value, low, high):
    return max(low, min(high, value))


def calculate_angle(points):
    """
    计算三点夹角，points = [a, b, c]，角度点在 b。
    支持 2D: (x, y)
    支持 3D: (x, y, z)

    返回的是“关节包含角”：
    - 腿伸直时接近 180°
    - 膝盖弯曲时变小，比如 120°、90°
    """
    if len(points) != 3:
        return None

    a, b, c = points

    if len(a) != len(b) or len(b) != len(c):
        return None

    ba = [a[i] - b[i] for i in range(len(a))]
    bc = [c[i] - b[i] for i in range(len(c))]

    dot_product = sum(ba[i] * bc[i] for i in range(len(ba)))
    ba_length = math.sqrt(sum(v * v for v in ba))
    bc_length = math.sqrt(sum(v * v for v in bc))

    if ba_length < 1e-8 or bc_length < 1e-8:
        return None

    cos_value = dot_product / (ba_length * bc_length)
    cos_value = clamp(cos_value, -1.0, 1.0)

    return math.degrees(math.acos(cos_value))


def knee_flexion_from_included_angle(included_angle):
    """
    把“关节包含角”转换成更符合康复表达的“膝关节屈曲角”：
    - 腿伸直时：包含角约 180°，屈曲角约 0°
    - 膝盖弯曲时：包含角约 90°，屈曲角约 90°

    注意：这个转换不会伪造动作幅度，只是让显示更直观。
    """
    if included_angle is None:
        return None

    flexion = 180.0 - included_angle
    return clamp(flexion, 0.0, 180.0)


def get_landmark_tuple(landmarks, index, use_3d=False):
    lm = landmarks[index]
    if use_3d:
        return (lm.x, lm.y, lm.z)
    return (lm.x, lm.y)


def get_visibility(landmarks, indices):
    values = [landmarks[i].visibility for i in indices]
    return min(values), sum(values) / len(values)


def compute_knee_angle(result, rule):
    """
    同时计算 2D 角度和 3D world 角度。

    返回:
    {
        "valid": bool,
        "side": "left/right",
        "visibility_min": float,
        "visibility_avg": float,
        "included_angle_2d": float | None,
        "flexion_angle_2d": float | None,
        "included_angle_3d": float | None,
        "flexion_angle_3d": float | None,
        "selected_included_angle": float | None,
        "selected_flexion_angle": float | None,
        "selected_source": "3d_world" | "2d_image" | None
    }
    """
    if not result.pose_landmarks:
        return {"valid": False}

    image_landmarks = result.pose_landmarks.landmark
    indices = [rule["hip_index"], rule["knee_index"], rule["ankle_index"]]

    visibility_min, visibility_avg = get_visibility(image_landmarks, indices)

    included_2d = None
    flexion_2d = None
    included_3d = None
    flexion_3d = None

    if visibility_min >= VISIBILITY_THRESHOLD:
        points_2d = [
            get_landmark_tuple(image_landmarks, rule["hip_index"], use_3d=False),
            get_landmark_tuple(image_landmarks, rule["knee_index"], use_3d=False),
            get_landmark_tuple(image_landmarks, rule["ankle_index"], use_3d=False),
        ]
        included_2d = calculate_angle(points_2d)
        flexion_2d = knee_flexion_from_included_angle(included_2d)

        # world landmarks 更适合做 3D 关节角，但单目估计仍不是医学级测量。
        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks.landmark
            points_3d = [
                get_landmark_tuple(world_landmarks, rule["hip_index"], use_3d=True),
                get_landmark_tuple(world_landmarks, rule["knee_index"], use_3d=True),
                get_landmark_tuple(world_landmarks, rule["ankle_index"], use_3d=True),
            ]
            included_3d = calculate_angle(points_3d)
            flexion_3d = knee_flexion_from_included_angle(included_3d)

    selected_source = None
    selected_included = None
    selected_flexion = None

    if PREFER_3D_WORLD_ANGLE and included_3d is not None:
        selected_source = "3d_world"
        selected_included = included_3d
        selected_flexion = flexion_3d
    elif included_2d is not None:
        selected_source = "2d_image"
        selected_included = included_2d
        selected_flexion = flexion_2d

    return {
        "valid": selected_flexion is not None,
        "side": rule["side"],
        "visibility_min": visibility_min,
        "visibility_avg": visibility_avg,
        "included_angle_2d": included_2d,
        "flexion_angle_2d": flexion_2d,
        "included_angle_3d": included_3d,
        "flexion_angle_3d": flexion_3d,
        "selected_included_angle": selected_included,
        "selected_flexion_angle": selected_flexion,
        "selected_source": selected_source
    }


class MovingAverage:
    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def update(self, value):
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()


def choose_knee_rule(mode, left_result, right_result):
    """
    mode:
    - left: 只用左膝
    - right: 只用右膝
    - auto: 自动选择更可信的一侧

    自动逻辑：
    优先选择有效角度；两侧都有效时，选 visibility_avg 更高的一侧。
    """
    if mode == "left":
        return LEFT_KNEE_RULE, left_result
    if mode == "right":
        return RIGHT_KNEE_RULE, right_result

    left_valid = left_result.get("valid", False)
    right_valid = right_result.get("valid", False)

    if left_valid and not right_valid:
        return LEFT_KNEE_RULE, left_result
    if right_valid and not left_valid:
        return RIGHT_KNEE_RULE, right_result
    if left_valid and right_valid:
        if right_result.get("visibility_avg", 0) > left_result.get("visibility_avg", 0):
            return RIGHT_KNEE_RULE, right_result
        return LEFT_KNEE_RULE, left_result

    # 都无效时，默认左侧
    return LEFT_KNEE_RULE, left_result


def save_prescription(patient_id, action_name, frames, knee_rule, meta):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_flexion_angles = [
        frame["selected_flexion_angle_smoothed"]
        for frame in frames
        if frame.get("selected_flexion_angle_smoothed") is not None
    ]

    selected_included_angles = [
        frame["selected_included_angle"]
        for frame in frames
        if frame.get("selected_included_angle") is not None
    ]

    if len(frames) >= 2:
        duration_seconds = frames[-1]["relative_time"] - frames[0]["relative_time"]
    else:
        duration_seconds = 0

    prescription = {
        "patient_id": patient_id,
        "action_name": action_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "description": "This is a personalized rehabilitation prescription recorded from the patient's safe standard movement.",
        "camera_instruction": "For knee flexion training, stand sideways to the camera. Keep hip, knee and ankle visible. Use the visible leg side if auto mode is unstable.",

        "algorithm_note": {
            "included_angle_meaning": "Straight leg is close to 180 degrees; bent knee becomes smaller.",
            "flexion_angle_meaning": "Flexion angle = 180 - included angle. Straight leg is close to 0 degrees; bent knee becomes larger.",
            "angle_source_priority": "Prefer 3D MediaPipe world landmarks; fallback to 2D image landmarks.",
            "smoothing": f"Moving average window = {SMOOTH_WINDOW_SIZE} frames.",
            "warning": "Single-camera MediaPipe angle is suitable for demo and trend feedback, not clinical-grade measurement."
        },

        "runtime_meta": meta,

        "keypoint_rule": knee_rule,

        "clinical_baseline": {
            "frame_count": len(frames),
            "duration_seconds": duration_seconds,

            "min_selected_included_angle": min(selected_included_angles) if selected_included_angles else None,
            "max_selected_included_angle": max(selected_included_angles) if selected_included_angles else None,

            "min_knee_flexion_angle": min(selected_flexion_angles) if selected_flexion_angles else None,
            "max_knee_flexion_angle": max(selected_flexion_angles) if selected_flexion_angles else None,
            "rom_flexion": max(selected_flexion_angles) - min(selected_flexion_angles) if selected_flexion_angles else None
        },

        "template_frames": frames
    }

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{patient_id}_{action_name}_{time_str}.json"
    output_path = OUTPUT_DIR / file_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prescription, f, ensure_ascii=False, indent=4)

    return str(output_path), prescription


def put_text(frame, text, y, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def main():
    patient_id = input("请输入患者编号，例如 patient_001：").strip()
    if patient_id == "":
        patient_id = "patient_001"

    action_name = input("请输入动作名称，例如 knee_flexion：").strip()
    if action_name == "":
        action_name = "knee_flexion"

    side_mode = input("请选择膝盖侧别 left / right / auto，默认 auto：").strip().lower()
    if side_mode not in ["left", "right", "auto"]:
        side_mode = "auto"

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("摄像头打开失败，请把 CAMERA_INDEX 从 0 改成 1 再试。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("摄像头打开成功")
    print("操作说明：")
    print("按 R 开始录制标准动作")
    print("按 S 停止录制并保存 JSON 处方")
    print("按 C 清空当前平滑缓存")
    print("按 Q 或 ESC 退出程序")
    print()
    print("站位建议：膝关节屈伸请尽量侧对摄像头，让髋、膝、踝三个点都能清楚被看到。")
    print("显示说明：Flexion 是膝关节屈曲角，腿伸直约 0°，弯曲越大数值越大。")

    is_recording = False
    start_time = None
    frames = []
    frame_index = 0
    invalid_frame_count = 0
    selected_rule_at_recording = None

    smoother = MovingAverage(SMOOTH_WINDOW_SIZE)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        while True:
            success, frame = cap.read()

            if not success:
                print("读取摄像头画面失败")
                break

            # 镜像翻转方便人站位操作。注意：如果你发现左右腿总是反，可以把这一行注释掉。
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(rgb_frame)

            selected_result = {"valid": False}
            selected_rule = LEFT_KNEE_RULE

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                left_result = compute_knee_angle(result, LEFT_KNEE_RULE)
                right_result = compute_knee_angle(result, RIGHT_KNEE_RULE)

                selected_rule, selected_result = choose_knee_rule(side_mode, left_result, right_result)

            raw_flexion = selected_result.get("selected_flexion_angle")
            smoothed_flexion = smoother.update(raw_flexion)

            if is_recording and selected_rule_at_recording is None:
                selected_rule_at_recording = selected_rule

            current_rom = None
            if frames:
                recorded_angles = [
                    f["selected_flexion_angle_smoothed"]
                    for f in frames
                    if f.get("selected_flexion_angle_smoothed") is not None
                ]
                if recorded_angles:
                    current_rom = max(recorded_angles) - min(recorded_angles)

            # 录制数据
            if is_recording:
                if selected_result.get("valid", False):
                    now = time.time()
                    landmarks = result.pose_landmarks.landmark

                    keypoints = {}
                    for i, lm in enumerate(landmarks):
                        keypoints[str(i)] = {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility
                        }

                    frame_data = {
                        "frame_index": frame_index,
                        "relative_time": now - start_time,
                        "selected_side": selected_rule["side"],
                        "selected_source": selected_result.get("selected_source"),

                        "visibility_min": selected_result.get("visibility_min"),
                        "visibility_avg": selected_result.get("visibility_avg"),

                        "selected_included_angle": selected_result.get("selected_included_angle"),
                        "selected_flexion_angle_raw": raw_flexion,
                        "selected_flexion_angle_smoothed": smoothed_flexion,

                        "included_angle_2d": selected_result.get("included_angle_2d"),
                        "flexion_angle_2d": selected_result.get("flexion_angle_2d"),
                        "included_angle_3d": selected_result.get("included_angle_3d"),
                        "flexion_angle_3d": selected_result.get("flexion_angle_3d"),

                        # 兼容你原来后续可能读取 left_knee_angle 的代码。
                        # 这里保留字段名，但它代表“当前选中侧的包含角”，不一定永远是左膝。
                        "left_knee_angle": selected_result.get("selected_included_angle"),

                        "keypoints": keypoints
                    }

                    frames.append(frame_data)
                    frame_index += 1
                else:
                    invalid_frame_count += 1

            # UI 显示
            status_text = f"Recording... valid frames: {len(frames)} invalid: {invalid_frame_count}" if is_recording else "Ready. Press R to record."
            put_text(frame, status_text, 35, (0, 255, 0) if is_recording else (255, 255, 255), scale=0.8)

            if selected_result.get("valid", False):
                side = selected_result.get("side", "?")
                source = selected_result.get("selected_source", "?")
                included = selected_result.get("selected_included_angle")
                vis = selected_result.get("visibility_min", 0)

                put_text(frame, f"Side: {side}   Source: {source}   Visibility: {vis:.2f}", 70, (255, 255, 255))
                put_text(frame, f"Included angle: {included:.1f} deg", 105, (255, 255, 0))
                put_text(frame, f"Knee flexion: raw {raw_flexion:.1f} deg   smooth {smoothed_flexion:.1f} deg", 140, (0, 255, 255))

                if current_rom is not None:
                    put_text(frame, f"Current ROM during recording: {current_rom:.1f} deg", 175, (0, 255, 0))
            else:
                put_text(frame, "No reliable knee landmarks. Show hip-knee-ankle clearly.", 80, (0, 0, 255))

            put_text(frame, "R: start   S: save   C: clear smooth   Q/ESC: quit", 460, (255, 255, 255), scale=0.65)
            put_text(frame, "Tip: stand sideways to camera; try left/right mode if auto is unstable.", 430, (200, 200, 200), scale=0.58)

            cv2.imshow("Improved Prescription Recorder", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                frames = []
                frame_index = 0
                invalid_frame_count = 0
                selected_rule_at_recording = None
                smoother.clear()
                start_time = time.time()
                is_recording = True
                print("开始录制标准动作...")

            elif key == ord("c"):
                smoother.clear()
                print("已清空平滑缓存。")

            elif key == ord("s"):
                if is_recording:
                    is_recording = False

                    if len(frames) == 0:
                        print("没有录到有效骨架数据，请重新录制。")
                    else:
                        meta = {
                            "camera_index": CAMERA_INDEX,
                            "frame_width": FRAME_WIDTH,
                            "frame_height": FRAME_HEIGHT,
                            "side_mode": side_mode,
                            "prefer_3d_world_angle": PREFER_3D_WORLD_ANGLE,
                            "model_complexity": MODEL_COMPLEXITY,
                            "visibility_threshold": VISIBILITY_THRESHOLD,
                            "smooth_window_size": SMOOTH_WINDOW_SIZE,
                            "invalid_frame_count": invalid_frame_count
                        }

                        output_path, prescription = save_prescription(
                            patient_id,
                            action_name,
                            frames,
                            selected_rule_at_recording or selected_rule,
                            meta
                        )

                        baseline = prescription["clinical_baseline"]

                        print("JSON 数字处方保存成功！")
                        print("保存路径：", output_path)
                        print("患者编号：", prescription["patient_id"])
                        print("动作名称：", prescription["action_name"])
                        print("帧数：", baseline["frame_count"])
                        print("动作时长：", baseline["duration_seconds"])
                        print("最小膝关节屈曲角：", baseline["min_knee_flexion_angle"])
                        print("最大膝关节屈曲角：", baseline["max_knee_flexion_angle"])
                        print("ROM 屈曲范围：", baseline["rom_flexion"])
                        print("无效帧数：", invalid_frame_count)

                        if baseline["rom_flexion"] is not None and baseline["rom_flexion"] < 20:
                            print("提醒：本次 ROM 仍然偏小。请确认是否侧对摄像头、是否录到明显屈膝动作、是否选择了正确腿侧。")
                else:
                    print("当前还没有开始录制，请先按 R。")

            elif key == ord("q") or key == 27:
                print("程序退出")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
