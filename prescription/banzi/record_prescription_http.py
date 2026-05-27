from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from feedback.feedback_engine import build_feedback_from_files
from prescription.common.result_storage import save_prescription_artifacts
from realtime.system_monitor import get_system_status
from realtime.training_session import RealtimeTrainingSession


CAMERA_DEVICE = os.environ.get("RK_CAMERA_DEVICE", "/dev/video21")
CAMERA_BACKEND = cv2.CAP_V4L2
FOURCC = "MJPG"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
INFER_WIDTH = 640
INFER_HEIGHT = 360
JPEG_QUALITY = 70
PORT = 8082

RUNTIME_DIR = PROJECT_ROOT / "runtime"
ACTIVE_TEMPLATES_PATH = RUNTIME_DIR / "active_templates.json"
EVALUATE_REPORTS_DIR = PROJECT_ROOT / "evaluate" / "reports"
PYTHON_EXE = Path("D:/anaconda/python.exe") if Path("D:/anaconda/python.exe").exists() else Path(sys.executable)
DEFAULT_CONFIG_BY_ACTION = {
    "knee_flexion": "evaluate/configs/knee_flexion.yaml",
    "seated_knee_extension": "evaluate/configs/seated_knee_extension.yaml",
    "seated_knee_raise": "evaluate/configs/seated_knee_raise.yaml",
    "standing_hamstring_curl": "evaluate/configs/standing_hamstring_curl.yaml",
    "sit_to_stand": "evaluate/configs/sit_to_stand.yaml",
}
FEEDBACK_RULE_BY_ACTION = {
    "knee_flexion": "feedback/rules/knee_flexion_feedback.yaml",
    "seated_knee_extension": "feedback/rules/seated_knee_extension_feedback.yaml",
    "seated_knee_raise": "feedback/rules/seated_knee_raise_feedback.yaml",
    "standing_hamstring_curl": "feedback/rules/standing_hamstring_curl_feedback.yaml",
    "sit_to_stand": "feedback/rules/sit_to_stand_feedback.yaml",
}
RECORD_ROLE_LABELS = {
    "doctor_template": "标准动作",
    "patient_attempt": "患者动作",
}

VISIBILITY_THRESHOLD = 0.55
SMOOTH_WINDOW_SIZE = 5
PREFER_3D_WORLD_ANGLE = True
MODEL_COMPLEXITY = 1

SIDE_MODE_LABELS = {
    "auto": "自动",
    "left": "左腿",
    "right": "右腿",
}

ANGLE_SOURCE_LABELS = {
    "3d_world": "3D 世界坐标",
    "2d_image": "2D 图像坐标",
}

LEFT_KNEE_RULE = {
    "side": "left",
    "target_joint": "left_knee",
    "hip_index": 23,
    "knee_index": 25,
    "ankle_index": 27,
}

RIGHT_KNEE_RULE = {
    "side": "right",
    "target_joint": "right_knee",
    "hip_index": 24,
    "knee_index": 26,
    "ankle_index": 28,
}

REHAB_KEYPOINT_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

ACTION_RULES = {
    "knee_flexion": {
        "display_name": "膝关节屈曲",
        "angle_kind": "flexion",
        "point_names": ["hip", "knee", "ankle"],
        "rules": {
            "left": {**LEFT_KNEE_RULE, "point_indices": {"hip": 23, "knee": 25, "ankle": 27}},
            "right": {**RIGHT_KNEE_RULE, "point_indices": {"hip": 24, "knee": 26, "ankle": 28}},
        },
    },
    "seated_knee_extension": {
        "display_name": "坐姿伸膝",
        "angle_kind": "flexion",
        "point_names": ["hip", "knee", "ankle"],
        "rules": {
            "left": {**LEFT_KNEE_RULE, "point_indices": {"hip": 23, "knee": 25, "ankle": 27}},
            "right": {**RIGHT_KNEE_RULE, "point_indices": {"hip": 24, "knee": 26, "ankle": 28}},
        },
    },
    "seated_knee_raise": {
        "display_name": "坐姿抬膝",
        "angle_kind": "flexion",
        "point_names": ["shoulder", "hip", "knee"],
        "rules": {
            "left": {
                "side": "left",
                "target_joint": "left_hip",
                "point_indices": {"shoulder": 11, "hip": 23, "knee": 25},
            },
            "right": {
                "side": "right",
                "target_joint": "right_hip",
                "point_indices": {"shoulder": 12, "hip": 24, "knee": 26},
            },
        },
    },
    "standing_hamstring_curl": {
        "display_name": "站姿屈膝后勾腿",
        "angle_kind": "flexion",
        "point_names": ["hip", "knee", "ankle"],
        "rules": {
            "left": {**LEFT_KNEE_RULE, "point_indices": {"hip": 23, "knee": 25, "ankle": 27}},
            "right": {**RIGHT_KNEE_RULE, "point_indices": {"hip": 24, "knee": 26, "ankle": 28}},
        },
    },
    "sit_to_stand": {
        "display_name": "坐站训练",
        "angle_kind": "included",
        "point_names": ["hip", "knee", "ankle"],
        "rules": {
            "left": {**LEFT_KNEE_RULE, "point_indices": {"hip": 23, "knee": 25, "ankle": 27}},
            "right": {**RIGHT_KNEE_RULE, "point_indices": {"hip": 24, "knee": 26, "ankle": 28}},
        },
    },
}

ACTION_ALIASES = {
    "屈膝": "knee_flexion",
    "膝关节屈曲": "knee_flexion",
    "knee_flexion": "knee_flexion",
    "坐姿伸膝": "seated_knee_extension",
    "坐姿平举腿": "seated_knee_extension",
    "seated_knee_extension": "seated_knee_extension",
    "坐姿抬膝": "seated_knee_raise",
    "seated_knee_raise": "seated_knee_raise",
    "站姿屈膝后勾腿": "standing_hamstring_curl",
    "站姿腘绳肌弯举": "standing_hamstring_curl",
    "腘绳肌弯举": "standing_hamstring_curl",
    "后勾腿": "standing_hamstring_curl",
    "hamstring_curl": "standing_hamstring_curl",
    "standing_hamstring_curl": "standing_hamstring_curl",
    "坐站训练": "sit_to_stand",
    "坐站": "sit_to_stand",
    "sit_to_stand": "sit_to_stand",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calculate_angle(points: list[tuple[float, ...]]) -> float | None:
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

    cos_value = clamp(dot_product / (ba_length * bc_length), -1.0, 1.0)
    return math.degrees(math.acos(cos_value))


def target_angle_from_included_angle(included_angle: float | None, angle_kind: str) -> float | None:
    if included_angle is None:
        return None
    if angle_kind == "flexion":
        return clamp(180.0 - included_angle, 0.0, 180.0)
    return clamp(included_angle, 0.0, 180.0)


def get_landmark_tuple(landmarks, index: int, use_3d: bool = False) -> tuple[float, ...]:
    landmark = landmarks[index]
    if use_3d:
        return (landmark.x, landmark.y, landmark.z)
    return (landmark.x, landmark.y)


def get_visibility(landmarks, indices: list[int]) -> tuple[float, float]:
    values = [landmarks[i].visibility for i in indices]
    return min(values), sum(values) / len(values)


def normalize_action_id(value: object) -> str:
    text = str(value or "").strip()
    return ACTION_ALIASES.get(text, ACTION_ALIASES.get(text.lower(), "knee_flexion"))


def get_action_config(action_id: str) -> dict[str, object]:
    return ACTION_RULES.get(action_id, ACTION_RULES["knee_flexion"])


def get_rule_indices(rule: dict[str, object]) -> dict[str, int]:
    if isinstance(rule.get("point_indices"), dict):
        return {str(name): int(index) for name, index in dict(rule["point_indices"]).items()}
    return {
        "hip": int(rule["hip_index"]),
        "knee": int(rule["knee_index"]),
        "ankle": int(rule["ankle_index"]),
    }


def compute_action_angle(result, rule: dict[str, object], action_config: dict[str, object]) -> dict[str, object]:
    if not result.pose_landmarks:
        return {"valid": False}

    image_landmarks = result.pose_landmarks.landmark
    point_indices = get_rule_indices(rule)
    indices = list(point_indices.values())
    visibility_min, visibility_avg = get_visibility(image_landmarks, indices)
    angle_kind = str(action_config.get("angle_kind", "included"))

    included_2d = None
    target_angle_2d = None
    included_3d = None
    target_angle_3d = None

    if visibility_min >= VISIBILITY_THRESHOLD:
        points_2d = [
            get_landmark_tuple(image_landmarks, index, use_3d=False)
            for index in indices
        ]
        included_2d = calculate_angle(points_2d)
        target_angle_2d = target_angle_from_included_angle(included_2d, angle_kind)

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks.landmark
            points_3d = [
                get_landmark_tuple(world_landmarks, index, use_3d=True)
                for index in indices
            ]
            included_3d = calculate_angle(points_3d)
            target_angle_3d = target_angle_from_included_angle(included_3d, angle_kind)

    selected_source = None
    selected_included = None
    selected_target_angle = None

    if PREFER_3D_WORLD_ANGLE and included_3d is not None:
        selected_source = "3d_world"
        selected_included = included_3d
        selected_target_angle = target_angle_3d
    elif included_2d is not None:
        selected_source = "2d_image"
        selected_included = included_2d
        selected_target_angle = target_angle_2d

    return {
        "valid": selected_target_angle is not None,
        "side": rule["side"],
        "target_joint": rule.get("target_joint"),
        "visibility_min": visibility_min,
        "visibility_avg": visibility_avg,
        "included_angle_2d": included_2d,
        "target_angle_2d": target_angle_2d,
        "flexion_angle_2d": target_angle_2d,
        "included_angle_3d": included_3d,
        "target_angle_3d": target_angle_3d,
        "flexion_angle_3d": target_angle_3d,
        "selected_included_angle": selected_included,
        "selected_target_angle": selected_target_angle,
        "selected_flexion_angle": selected_target_angle,
        "selected_source": selected_source,
    }


def compute_knee_angle(result, rule: dict[str, object]) -> dict[str, object]:
    return compute_action_angle(result, rule, ACTION_RULES["knee_flexion"])


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.values = deque(maxlen=window_size)

    def update(self, value: float | None) -> float | None:
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self) -> None:
        self.values.clear()


def choose_action_rule(
    mode: str,
    action_config: dict[str, object],
    left_result: dict[str, object],
    right_result: dict[str, object],
):
    rules = action_config["rules"]
    left_rule = rules["left"]
    right_rule = rules["right"]
    if mode == "left":
        return left_rule, left_result
    if mode == "right":
        return right_rule, right_result

    left_valid = bool(left_result.get("valid", False))
    right_valid = bool(right_result.get("valid", False))

    if left_valid and not right_valid:
        return left_rule, left_result
    if right_valid and not left_valid:
        return right_rule, right_result
    if left_valid and right_valid:
        if right_result.get("visibility_avg", 0) > left_result.get("visibility_avg", 0):
            return right_rule, right_result
        return left_rule, left_result

    return left_rule, left_result


def choose_knee_rule(mode: str, left_result: dict[str, object], right_result: dict[str, object]):
    return choose_action_rule(mode, ACTION_RULES["knee_flexion"], left_result, right_result)


def build_compact_keypoints(landmarks, selected_rule: dict[str, object]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, index in get_rule_indices(selected_rule).items():
        landmark = landmarks[index]
        compact[name] = {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }
    return compact


def build_rehab_keypoints(landmarks) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, index in REHAB_KEYPOINT_INDICES.items():
        landmark = landmarks[index]
        compact[name] = {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }
    return compact


def split_host_port(host_header: str) -> tuple[str, str]:
    text = host_header.strip()
    if not text:
        return "unknown", str(PORT)

    if text.startswith("[") and "]:" in text:
        host, port = text.rsplit(":", 1)
        return host.strip("[]"), port

    if text.count(":") == 1:
        host, port = text.rsplit(":", 1)
        return host, port

    return text.strip("[]"), str(PORT)


def project_relative(path: str | Path) -> str:
    value = Path(path)
    absolute = value if value.is_absolute() else PROJECT_ROOT / value
    try:
        return absolute.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def reject_report_input(path: str | Path) -> None:
    relative = project_relative(path).replace("\\", "/")
    if relative.startswith("evaluate/reports/"):
        raise ValueError("evaluate/reports/*.json 是评估输出，不能作为模板或患者动作输入。")


def load_active_templates() -> dict[str, dict[str, object]]:
    if not ACTIVE_TEMPLATES_PATH.exists():
        return {}
    try:
        payload = json.loads(ACTIVE_TEMPLATES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def save_active_templates(payload: dict[str, dict[str, object]]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_TEMPLATES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_active_template(action_id: str) -> dict[str, object] | None:
    entry = load_active_templates().get(action_id)
    return entry if isinstance(entry, dict) else None


def set_active_template(action_id: str, template_file: str | Path) -> dict[str, object]:
    templates = load_active_templates()
    entry = {
        "action_id": action_id,
        "template_file": project_relative(template_file),
        "config_file": DEFAULT_CONFIG_BY_ACTION.get(action_id, f"evaluate/configs/{action_id}.yaml"),
        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
    }
    templates[action_id] = entry
    save_active_templates(templates)
    return entry


def normalize_record_role(value: object) -> str:
    text = str(value or "").strip()
    return text if text in RECORD_ROLE_LABELS else "doctor_template"


def report_file_for(action_id: str) -> Path:
    EVALUATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EVALUATE_REPORTS_DIR / f"report_{action_id}_{timestamp}.json"


def evaluate_attempt(action_id: str, attempt_file: str | Path) -> dict[str, object]:
    active_template = get_active_template(action_id)
    if not active_template:
        return {"ok": False, "error": "请先录入标准动作"}

    template_file = str(active_template.get("template_file", ""))
    config_file = str(active_template.get("config_file", DEFAULT_CONFIG_BY_ACTION.get(action_id, "")))
    if not template_file:
        return {"ok": False, "error": "active template 缺少 template_file。"}
    if not config_file:
        return {"ok": False, "error": f"动作 {action_id} 缺少评估配置。"}

    try:
        reject_report_input(template_file)
        reject_report_input(attempt_file)
    except ValueError as error:
        return {"ok": False, "error": str(error)}

    template_path = resolve_project_path(template_file)
    attempt_path = resolve_project_path(attempt_file)
    config_path = resolve_project_path(config_file)
    report_path = report_file_for(action_id)

    if not template_path.exists():
        return {"ok": False, "error": f"active template 文件不存在：{template_file}"}
    if not attempt_path.exists():
        return {"ok": False, "error": f"patient attempt 文件不存在：{project_relative(attempt_path)}"}
    if not config_path.exists():
        return {"ok": False, "error": f"评估配置不存在：{config_file}"}

    command = [
        str(PYTHON_EXE),
        "evaluate/banzi/run_evaluate.py",
        "--template",
        project_relative(template_path),
        "--attempt",
        project_relative(attempt_path),
        "--config",
        project_relative(config_path),
        "--out",
        project_relative(report_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            shell=False,
            timeout=30,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "ok": False,
            "error": "评估超时",
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "returncode": None,
        }

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": "评估失败",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rule_path = resolve_project_path(FEEDBACK_RULE_BY_ACTION.get(action_id, "feedback/rules/knee_flexion_feedback.yaml"))
        feedback = build_feedback_from_files(report_path, rule_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "ok": False,
            "error": f"评估报告或反馈生成失败：{error}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    print(f"[feedback mock] TTS: {feedback['tts']['text']}")
    print(f"[feedback mock] Motor: {feedback['motor']['pattern']}")
    return {
        "ok": True,
        "report_file": project_relative(report_path),
        "report": report,
        "feedback": feedback,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_prescription(
    patient_id: str,
    action_name: str,
    frames: list[dict[str, object]],
    knee_rule: dict[str, object],
    meta: dict[str, object],
) -> dict[str, object]:
    action_id = normalize_action_id(action_name)
    target_angles = [
        frame["target_angle_smoothed"]
        for frame in frames
        if frame.get("target_angle_smoothed") is not None
    ]
    if not target_angles:
        target_angles = [
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
        duration_seconds = 0.0

    return {
        "patient_id": patient_id,
        "action_id": action_id,
        "action_name": action_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "这是一次用于个性化标准动作录入与实时比对的动作模板结果。",
        "camera_instruction": "录制时请让目标关节相关的三个关键点持续清晰可见；如果自动选侧不稳定，请改用左侧或右侧固定模式。",
        "algorithm_note": {
            "included_angle_meaning": "三点夹角由目标关节相邻两个肢段计算得到。",
            "target_angle_meaning": "target_angle 是用于后续 template 与 attempt replay 对比的统一角度序列。",
            "angle_source_priority": "优先使用 MediaPipe 的 3D world landmarks，不可用时回退到 2D 图像 landmarks。",
            "smoothing": f"平滑窗口 = {SMOOTH_WINDOW_SIZE} 帧。",
            "warning": "单目 MediaPipe 角度适合演示和趋势反馈，不属于临床级测量。",
        },
        "runtime_meta": {**meta, "action_id": action_id},
        "keypoint_rule": knee_rule,
        "clinical_baseline": {
            "frame_count": len(frames),
            "duration_seconds": duration_seconds,
            "min_selected_included_angle": min(selected_included_angles) if selected_included_angles else None,
            "max_selected_included_angle": max(selected_included_angles) if selected_included_angles else None,
            "min_target_angle": min(target_angles) if target_angles else None,
            "max_target_angle": max(target_angles) if target_angles else None,
            "rom_target_angle": max(target_angles) - min(target_angles) if target_angles else None,
            "min_knee_flexion_angle": min(target_angles) if target_angles else None,
            "max_knee_flexion_angle": max(target_angles) if target_angles else None,
            "rom_flexion": max(target_angles) - min(target_angles) if target_angles else None,
        },
        "template_frames": frames,
    }


def sanitize_text(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def open_camera() -> cv2.VideoCapture:
    candidates: list[object] = [CAMERA_DEVICE]
    if os.name == "nt":
        candidates.extend([0, 1, 2])
    else:
        candidates.extend([21, 0, 1, 2])

    seen: set[str] = set()
    tried: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if isinstance(candidate, int):
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(candidate, backend)
        else:
            cap = cv2.VideoCapture(str(candidate), CAMERA_BACKEND if str(candidate).startswith("/dev/") else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if cap.isOpened():
            print(f"摄像头已打开: {candidate}")
            return cap
        cap.release()

    raise RuntimeError(f"无法打开摄像头，已尝试: {', '.join(tried)}。可设置环境变量 RK_CAMERA_DEVICE 指定设备。")


class RecorderState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.running = True
        self.frame_id = 0
        self.jpg_bytes: bytes | None = None
        self.last_status = "等待首帧画面"
        self.frame_timestamps = deque(maxlen=60)
        self.pose_fps: float | None = None

        self.is_recording = False
        self.patient_id = "patient_001"
        self.action_name = "knee_flexion"
        self.current_record_role = "doctor_template"
        self.side_mode = "auto"
        self.start_time: float | None = None
        self.frames: list[dict[str, object]] = []
        self.frame_index = 0
        self.invalid_frame_count = 0
        self.selected_rule_at_recording: dict[str, object] | None = None
        self.smoother = MovingAverage(SMOOTH_WINDOW_SIZE)

        self.selected_result: dict[str, object] = {"valid": False}
        self.selected_rule: dict[str, object] = LEFT_KNEE_RULE
        self.current_rom: float | None = None

        self.last_export_payload: dict[str, object] | None = None
        self.last_export_summary: dict[str, object] | None = None
        self.last_export_board_result: dict[str, object] | None = None
        self.last_export_error: str | None = None
        self.awaiting_ack = False
        self.last_patient_attempt_path: str | None = None
        self.last_patient_attempt_summary: dict[str, object] | None = None
        self.last_evaluation_report_path: str | None = None
        self.last_feedback: dict[str, object] | None = None

    def reset_recording(self) -> None:
        self.frames = []
        self.frame_index = 0
        self.invalid_frame_count = 0
        self.selected_rule_at_recording = None
        self.current_rom = None
        self.start_time = None
        self.is_recording = False
        self.smoother.clear()

    def clear_export(self) -> None:
        self.last_export_payload = None
        self.last_export_summary = None
        self.last_export_board_result = None
        self.last_export_error = None
        self.awaiting_ack = False

    def clear_all(self) -> None:
        self.reset_recording()
        self.clear_export()
        self.last_patient_attempt_path = None
        self.last_patient_attempt_summary = None
        self.last_evaluation_report_path = None
        self.last_feedback = None

    def update_frame(self, jpg_bytes: bytes, status: str) -> None:
        with self.condition:
            self.frame_id += 1
            now = time.time()
            self.frame_timestamps.append(now)
            if len(self.frame_timestamps) >= 2:
                elapsed = self.frame_timestamps[-1] - self.frame_timestamps[0]
                if elapsed > 1e-6:
                    self.pose_fps = (len(self.frame_timestamps) - 1) / elapsed
            self.jpg_bytes = jpg_bytes
            self.last_status = status
            self.condition.notify_all()

    def snapshot_status(self) -> dict[str, object]:
        smoothed = self.smoother.values[-1] if self.smoother.values else None
        action_id = normalize_action_id(self.action_name)
        active_template = get_active_template(action_id)
        return {
            "recording": self.is_recording,
            "patient_id": self.patient_id,
            "action_name": self.action_name,
            "action_id": action_id,
            "current_record_role": self.current_record_role,
            "current_record_role_label": RECORD_ROLE_LABELS.get(self.current_record_role, self.current_record_role),
            "side_mode": self.side_mode,
            "side_mode_label": SIDE_MODE_LABELS.get(self.side_mode, self.side_mode),
            "valid_frames": len(self.frames),
            "invalid_frames": self.invalid_frame_count,
            "selected_side": self.selected_rule.get("side"),
            "selected_side_label": SIDE_MODE_LABELS.get(str(self.selected_rule.get("side")), self.selected_rule.get("side")),
            "selected_source": self.selected_result.get("selected_source"),
            "selected_source_label": ANGLE_SOURCE_LABELS.get(str(self.selected_result.get("selected_source")), self.selected_result.get("selected_source")),
            "selected_flexion_angle": self.selected_result.get("selected_flexion_angle"),
            "smoothed_flexion_angle": smoothed,
            "visibility_min": self.selected_result.get("visibility_min"),
            "visibility_avg": self.selected_result.get("visibility_avg"),
            "current_rom": self.current_rom,
            "pending_export": self.last_export_payload is not None,
            "awaiting_ack": self.awaiting_ack,
            "last_export_error": self.last_export_error,
            "active_template": active_template,
            "patient_attempt_file": self.last_patient_attempt_path,
            "patient_attempt_summary": self.last_patient_attempt_summary,
            "evaluation_report_file": self.last_evaluation_report_path,
            "feedback": self.last_feedback,
            "training": realtime_session.snapshot(),
            "pose_fps": round(self.pose_fps, 2) if self.pose_fps is not None else None,
            "status": self.last_status,
        }


state = RecorderState()
realtime_session = RealtimeTrainingSession()
cap = open_camera()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def make_json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status_code: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def build_home_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RK3588 统一训练台</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#ffffff; --ink:#17202a; --muted:#5f6b7a; --line:#d9e0e7; --accent:#0f766e; --blue:#2457a6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink); background:var(--bg); }
    .wrap { max-width:1100px; margin:0 auto; padding:28px; display:grid; gap:18px; }
    .top { display:flex; justify-content:space-between; gap:16px; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:16px; }
    h1 { margin:0; font-size:30px; }
    p { margin:6px 0 0; color:var(--muted); line-height:1.6; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .entry { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; display:grid; gap:12px; min-height:230px; }
    .entry h2 { margin:0; font-size:20px; }
    .entry a { justify-self:start; text-decoration:none; color:white; background:var(--accent); padding:11px 14px; border-radius:6px; }
    .entry a.train { background:var(--blue); }
    .status { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .stat { background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px; min-height:64px; }
    .stat b { display:block; font-size:12px; color:var(--muted); margin-bottom:5px; }
    .mono { font-family:Consolas,"Courier New",monospace; word-break:break-all; }
    @media (max-width:850px) { .grid,.status { grid-template-columns:1fr; } .top { display:block; } }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="top">
      <div>
        <h1>RK3588 统一训练台</h1>
        <p>一个入口完成医生标准动作录制和患者实时屈膝训练。</p>
      </div>
      <div class="mono">http://板子IP:8082</div>
    </section>
    <section class="status" id="status-grid"></section>
    <section class="grid">
      <article class="entry">
        <h2>医生录制标准动作</h2>
        <p>采集医生或标准示范动作，保存后自动写入 active template，供患者实时训练使用。</p>
        <a href="/doctor">进入医生录制</a>
      </article>
      <article class="entry">
        <h2>患者实时训练</h2>
        <p>读取当前 active template，连续识别屈膝 reps，实时提示并在整组完成后生成完整评估报告。</p>
        <a class="train" href="/train">进入患者训练</a>
      </article>
    </section>
  </main>
  <script>
    const grid = document.getElementById("status-grid");
    function text(value) { return value == null || value === "" ? "-" : value; }
    async function refresh() {
      const res = await fetch("/status");
      const status = await res.json();
      const training = status.training || {};
      const rows = [
        ["active template", status.active_template?.template_file || "未设置"],
        ["摄像头", status.status || "-"],
        ["训练状态", training.status || "idle"],
        ["最近报告", training.report_file || status.evaluation_report_file || "-"],
      ];
      grid.innerHTML = rows.map(([k,v]) => `<div class="stat"><b>${k}</b><span class="mono">${text(v)}</span></div>`).join("");
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def build_train_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>患者实时屈膝训练</title>
  <style>
    :root { --bg:#f6f7f9; --panel:#fff; --ink:#17202a; --muted:#5f6b7a; --line:#d9e0e7; --accent:#2457a6; --ok:#147a4b; --warn:#b85c38; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink); background:var(--bg); }
    .wrap { max-width:1240px; margin:0 auto; padding:22px; display:grid; gap:16px; }
    header { display:flex; justify-content:space-between; align-items:center; gap:12px; border-bottom:1px solid var(--line); padding-bottom:12px; }
    h1 { margin:0; font-size:24px; }
    a { color:var(--accent); text-decoration:none; }
    .grid { display:grid; grid-template-columns:1.15fr .85fr; gap:16px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    img.stream { width:100%; display:block; background:#000; border:1px solid var(--line); border-radius:6px; }
    .controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    label { display:grid; gap:5px; color:var(--muted); font-size:13px; }
    input, select, button { font:inherit; }
    input, select { padding:9px 10px; border:1px solid var(--line); border-radius:6px; background:#fff; }
    button { border:0; border-radius:6px; padding:10px 12px; cursor:pointer; color:#fff; background:var(--accent); }
    button.ok { background:var(--ok); }
    button.warn { background:var(--warn); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .stats { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:12px; }
    .stat { border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; min-height:62px; }
    .stat b { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    .system-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }
    .prompt { font-size:24px; font-weight:700; padding:14px; border:1px solid var(--line); border-radius:8px; background:#eef5ff; }
    .rep { border:1px solid var(--line); border-radius:6px; padding:10px; background:#fff; margin-top:8px; }
    .mono { font-family:Consolas,"Courier New",monospace; word-break:break-all; }
    @media (max-width:960px) { .grid,.controls,.stats,.system-grid { grid-template-columns:1fr; } header { display:block; } }
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <div>
        <h1>患者实时屈膝训练</h1>
        <div><a href="/">训练台首页</a> / <a href="/doctor">医生录制</a></div>
      </div>
      <div class="mono" id="active-template">active template: -</div>
    </header>
    <section class="grid">
      <div class="panel">
        <img class="stream" src="/stream.mjpg" alt="实时预览">
        <div class="stats" id="live-stats"></div>
      </div>
      <div class="panel">
        <div class="controls">
          <label>患者编号<input id="patient-id" value="patient_001"></label>
          <label>目标次数<input id="target-reps" type="number" min="1" max="50" value="10"></label>
          <label>侧别模式
            <select id="side-mode">
              <option value="auto">auto（自动）</option>
              <option value="left">left（左腿）</option>
              <option value="right">right（右腿）</option>
            </select>
          </label>
          <label>动作
            <input id="action-id" value="seated_knee_extension" list="train-action-options">
            <datalist id="train-action-options">
              <option value="seated_knee_extension">坐姿伸膝</option>
              <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
              <option value="sit_to_stand">坐站训练</option>
              <option value="knee_flexion">屈膝旧流程</option>
            </datalist>
          </label>
        </div>
        <div class="controls" style="margin-top:12px">
          <button id="playlist-btn" class="ok">开始完整训练</button>
          <button id="start-btn" class="ok">开始训练</button>
          <button id="pause-btn">暂停 / 继续</button>
          <button id="stop-btn" class="warn">结束训练</button>
        </div>
        <div class="prompt" id="prompt" style="margin-top:14px">等待开始训练</div>
        <div class="stats" id="feedback-stats"></div>
        <h2>每遍小评估</h2>
        <div id="rep-list"></div>
        <h2>整组报告</h2>
        <div id="report-box" class="mono">尚未生成</div>
      </div>
    </section>
    <section class="panel">
      <h2>板端资源监控</h2>
      <div class="system-grid" id="system-stats"></div>
    </section>
  </main>
  <script>
    const startBtn = document.getElementById("start-btn");
    const playlistBtn = document.getElementById("playlist-btn");
    const pauseBtn = document.getElementById("pause-btn");
    const stopBtn = document.getElementById("stop-btn");
    const promptBox = document.getElementById("prompt");
    const liveStats = document.getElementById("live-stats");
    const feedbackStats = document.getElementById("feedback-stats");
    const repList = document.getElementById("rep-list");
    const reportBox = document.getElementById("report-box");
    const activeTemplate = document.getElementById("active-template");
    const systemStats = document.getElementById("system-stats");

    function fmt(value, digits = 1) {
      if (value == null || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }
    function row(label, value) {
      return `<div class="stat"><b>${label}</b><span>${value == null || value === "" ? "-" : value}</span></div>`;
    }
    async function post(url, body) {
      const res = await fetch(url, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {}) });
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || "请求失败");
      return data;
    }
    function render(status) {
      const training = status.training || {};
      activeTemplate.textContent = `active template: ${status.active_template?.template_file || "未设置"}`;
      promptBox.textContent = training.prompt || "等待开始训练";
      liveStats.innerHTML = [
        row("训练状态", training.status || "idle"),
        row("完整训练", training.playlist_mode ? `${(training.playlist_index || 0) + 1} / ${training.playlist_total || 0}` : "未开启"),
        row("当前动作", training.current_action_name || training.action_id || "-"),
        row("镜头提示", training.current_camera_prompt || "-"),
        row("休息倒计时", training.rest_remaining_seconds == null ? "-" : `${training.rest_remaining_seconds} 秒`),
        row("状态机", training.current_state || "-"),
        row("当前指标", `${fmt(training.current_metric ?? training.current_angle)} ${training.metric?.metric_unit || "度"}`),
        row("baseline", `${fmt(training.baseline_angle)} 度`),
        row("rep", `${training.completed_reps || 0} / ${training.target_reps || 10}`),
        row("未计数尝试", training.invalid_attempts || 0),
        row("指标类型", training.metric?.metric_name || "-"),
        row("目标区间", Array.isArray(training.target_range) ? `${fmt(training.target_range[0])} - ${fmt(training.target_range[1])} ${training.metric?.metric_unit || "度"}` : "-"),
      ].join("");
      feedbackStats.innerHTML = [
        row("TTS", training.tts_text || "-"),
        row("TTS backend", training.tts?.backend || "-"),
        row("TTS 队列", training.tts ? `${training.tts.queued || 0}` : "-"),
        row("最近未计数", training.last_invalid_attempt?.screen_prompt || "-"),
        row("Motor mock", training.motor_mock_pattern || "-"),
        row("attempt", training.saved_attempt_file || "-"),
        row("report", training.report_file || "-"),
      ].join("");
      const reps = training.rep_results || [];
      repList.innerHTML = reps.length ? reps.map(rep => `
        <div class="rep">
          <b>第 ${rep.rep_index} 遍：${rep.primary_error}</b><br>
          ROM ${fmt(rep.rom)} / TUT ${fmt(rep.tut_seconds)} 秒 / Speed ${fmt(rep.peak_speed)} 度每秒<br>
          ${rep.screen_prompt || ""}
        </div>
      `).join("") : (training.last_invalid_attempt ? `<div class="rep"><b>未计数：</b>${training.last_invalid_attempt.screen_prompt || "动作不到位"}</div>` : "暂无");
      const playlistReports = training.playlist_reports || [];
      if (playlistReports.length) {
        const lines = playlistReports.map(item => `${item.action_name || item.action_id}: ${item.report_file || "-"}`);
        if (training.report_file && !lines.some(line => line.includes(training.report_file))) {
          lines.push(`当前报告: ${training.report_file}`);
        }
        reportBox.textContent = lines.join("\\n");
      } else if (training.report_file) {
        const errors = training.report?.errors || {};
        const metrics = training.report?.metrics || {};
        reportBox.textContent = `报告：${training.report_file}\nprimary_error：${errors.primary_error || "-"}\nROM：${fmt(metrics.rom?.actual)} / ${fmt(metrics.rom?.target)}\nTUT：${fmt(metrics.tut?.actual)} / ${fmt(metrics.tut?.target)}`;
      }
      const running = training.status === "running";
      const active = Boolean(status.active_template?.template_file);
      startBtn.disabled = running || !active;
      playlistBtn.disabled = running || training.status === "resting";
      pauseBtn.disabled = !["running", "paused"].includes(training.status);
      stopBtn.disabled = !["running", "paused", "resting"].includes(training.status);
    }
    async function refresh() {
      const res = await fetch("/status");
      render(await res.json());
    }
    function renderSystem(status) {
      const cpu = status.cpu || {};
      const memory = status.memory || {};
      const temperature = status.temperature || {};
      const npu = status.npu || {};
      const pose = status.pose_fps || {};
      systemStats.innerHTML = [
        row("CPU", cpu.available ? `${fmt(cpu.percent)}%` : cpu.note),
        row("Memory", memory.available ? `${fmt(memory.percent)}% (${fmt(memory.used_mb, 0)} / ${fmt(memory.total_mb, 0)} MB)` : memory.note),
        row("Temperature", temperature.available ? `${fmt(temperature.max_celsius)} °C` : temperature.note),
        row("NPU load", npu.available ? (npu.percent == null ? npu.raw : `${fmt(npu.percent)}%`) : npu.note),
        row("Pose FPS", pose.available ? `${fmt(pose.fps, 2)} FPS` : pose.note),
      ].join("");
    }
    async function refreshSystem() {
      try {
        const res = await fetch("/api/system/status");
        renderSystem(await res.json());
      } catch (error) {
        systemStats.innerHTML = row("监控状态", "系统资源读取失败");
      }
    }
    startBtn.addEventListener("click", async () => {
      try {
        await post("/api/realtime/start", {
          patient_id: document.getElementById("patient-id").value,
          action_id: document.getElementById("action-id").value,
          side_mode: document.getElementById("side-mode").value,
          target_reps: Number(document.getElementById("target-reps").value || 10),
        });
      } catch (error) { promptBox.textContent = error.message || String(error); }
      refresh();
    });
    playlistBtn.addEventListener("click", async () => {
      try {
        await post("/api/realtime/start_playlist", {
          patient_id: document.getElementById("patient-id").value,
          side_mode: document.getElementById("side-mode").value,
          target_reps: Number(document.getElementById("target-reps").value || 3),
        });
      } catch (error) { promptBox.textContent = error.message || String(error); }
      refresh();
    });
    pauseBtn.addEventListener("click", async () => { await post("/api/realtime/pause", {}); refresh(); });
    stopBtn.addEventListener("click", async () => { await post("/api/realtime/stop", {}); refresh(); });
    refresh();
    refreshSystem();
    setInterval(refresh, 800);
    setInterval(refreshSystem, 1000);
  </script>
</body>
</html>
"""


def build_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>医生标准动作录制</title>
  <style>
    :root {
      --bg: #f2efe7;
      --panel: #fffdf8;
      --ink: #1b1b1b;
      --muted: #5a5a5a;
      --line: #d4cfc3;
      --accent: #146356;
      --warn: #b85c38;
      --ok: #2b6f3e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: linear-gradient(135deg, #ede8dc 0%, #f8f5ee 40%, #e7efe8 100%);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 20px;
    }
    .hero, .panel {
      background: rgba(255, 253, 248, 0.95);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 40px rgba(35, 35, 35, 0.08);
    }
    .hero { padding: 20px 24px; }
    .hero h1 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
    }
    .panel { padding: 18px; }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    img.stream {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
      display: block;
    }
    form {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 14px;
      color: var(--muted);
    }
    input, select, button {
      font: inherit;
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 11px 14px;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }
    button.alt { background: #5c6b73; }
    button.warn { background: var(--warn); }
    button.ok { background: var(--ok); }
    button:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
    }
    .stat b {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.04em;
    }
    .message {
      min-height: 72px;
      border-radius: 12px;
      padding: 12px 14px;
      background: #f7f3ea;
      border: 1px solid var(--line);
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .mono {
      font-family: Consolas, "Courier New", monospace;
      word-break: break-all;
    }
    .hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .workflow {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .subpanel {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #fff;
    }
    .subpanel h3 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .result-box {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .feedback-card {
      border-radius: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      background: #f7f3ea;
      line-height: 1.6;
    }
    .feedback-card.red { border-color: #d94b3d; background: #fff0ee; }
    .feedback-card.orange { border-color: #d8902f; background: #fff6e8; }
    .feedback-card.yellow { border-color: #d2b743; background: #fffbe8; }
    .feedback-card.green { border-color: #3d9b63; background: #edf8ef; }
    @media (max-width: 920px) {
      .grid { grid-template-columns: 1fr; }
      .buttons, .status-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>医生标准动作录制</h1>
      <p><a href="/">训练台首页</a> / <a href="/train">患者实时训练</a>。流程：录入标准动作 → 保存为 active template → 患者实时训练使用该模板。</p>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>实时预览</h2>
        <img class="stream" src="/stream.mjpg" alt="实时预览">
        <div class="hint">
          建议尽量侧身面对摄像头，保证髋、膝、踝三个关键点持续可见。<br>
          如果自动选腿不稳定，请改成固定的 <span class="mono">left</span> 或 <span class="mono">right</span> 模式。
        </div>
      </section>
      <section class="panel">
        <h2>录制控制</h2>
        <form id="record-form">
          <label>患者编号
            <input id="patient_id" name="patient_id" value="patient_001">
          </label>
          <label>动作名称
            <input id="action_name" name="action_name" value="seated_knee_extension" list="action-options">
            <datalist id="action-options">
              <option value="seated_knee_extension">坐姿伸膝</option>
              <option value="standing_hamstring_curl">站姿屈膝后勾腿</option>
              <option value="sit_to_stand">坐站训练</option>
              <option value="knee_flexion">屈膝旧流程</option>
            </datalist>
          </label>
          <label>侧别模式
            <select id="side_mode" name="side_mode">
              <option value="auto">auto（自动）</option>
              <option value="left">left（左腿）</option>
              <option value="right">right（右腿）</option>
            </select>
          </label>
        </form>
        <div class="workflow">
          <div class="subpanel">
            <h3>当前动作</h3>
            <div>action_id：<span class="mono" id="action-id-label">knee_flexion</span></div>
            <div>action_name：<span class="mono" id="action-name-label">knee_flexion</span></div>
          </div>
          <div class="subpanel">
            <h3>标准动作区域</h3>
            <div class="buttons">
              <button id="start-template-btn">录入标准动作</button>
              <button id="save-template-btn" class="ok">保存为 active template</button>
            </div>
            <div class="hint">当前 active template：<span class="mono" id="active-template-label">未设置</span></div>
          </div>
          <div class="subpanel">
            <h3>患者训练区域</h3>
            <div class="buttons">
              <button id="start-attempt-btn">录入患者动作</button>
              <button id="save-attempt-btn" class="ok">保存 patient attempt</button>
              <button id="evaluate-btn" class="warn">结束并评估</button>
              <button id="cancel-btn" class="alt">取消本轮录制</button>
            </div>
            <div class="hint">当前 patient attempt：<span class="mono" id="patient-attempt-label">未设置</span></div>
          </div>
          <div class="buttons">
            <button id="clear-btn" class="alt">清空缓存</button>
          </div>
        </div>
        <div style="margin-top:14px">
          <div class="message" id="message">等待操作。</div>
        </div>
        <div class="status-grid" id="status-grid"></div>
        <div class="result-box" id="result-box"></div>
      </section>
    </div>
  </div>
  <script>
    const statusGrid = document.getElementById("status-grid");
    const messageBox = document.getElementById("message");
    const form = document.getElementById("record-form");
    const startTemplateBtn = document.getElementById("start-template-btn");
    const saveTemplateBtn = document.getElementById("save-template-btn");
    const startAttemptBtn = document.getElementById("start-attempt-btn");
    const saveAttemptBtn = document.getElementById("save-attempt-btn");
    const evaluateBtn = document.getElementById("evaluate-btn");
    const clearBtn = document.getElementById("clear-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    const actionIdLabel = document.getElementById("action-id-label");
    const actionNameLabel = document.getElementById("action-name-label");
    const activeTemplateLabel = document.getElementById("active-template-label");
    const patientAttemptLabel = document.getElementById("patient-attempt-label");
    const resultBox = document.getElementById("result-box");
    let latestStatus = {};
    let isEvaluating = false;

    function formatNumber(value, unit = "") {
      if (value == null || Number.isNaN(Number(value))) {
        return "-";
      }
      return `${Number(value).toFixed(1)}${unit}`;
    }

    function setMessage(text) {
      messageBox.textContent = text;
    }

    function updateButtons(status) {
      const recording = Boolean(status.recording);
      const hasActiveTemplate = Boolean(status.active_template && status.active_template.template_file);
      const hasAttempt = Boolean(status.patient_attempt_file);
      startTemplateBtn.disabled = recording || isEvaluating;
      startAttemptBtn.disabled = recording || isEvaluating;
      saveTemplateBtn.disabled = !recording || status.current_record_role !== "doctor_template" || isEvaluating;
      saveAttemptBtn.disabled = !recording || status.current_record_role !== "patient_attempt" || isEvaluating;
      evaluateBtn.disabled = recording || isEvaluating || !hasActiveTemplate || !hasAttempt;
      clearBtn.disabled = isEvaluating;
      cancelBtn.disabled = isEvaluating;
    }

    function displayStatus(status) {
      if (status.recording) return "录制中";
      if (status.awaiting_ack) return "等待板端确认";
      if (status.pending_export) return "等待本机保存";
      return status.status || "等待操作";
    }

    function displaySaveState(status) {
      if (status.awaiting_ack) return "本机已保存，等待板端确认";
      if (status.pending_export) return "已保存到板端";
      return "无待处理导出";
    }

    function renderStatus(status) {
      latestStatus = status || {};
      actionIdLabel.textContent = status.action_id || "knee_flexion";
      actionNameLabel.textContent = status.action_name || "-";
      activeTemplateLabel.textContent = status.active_template?.template_file || "未设置";
      patientAttemptLabel.textContent = status.patient_attempt_file || "未设置";
      const rows = [
        ["当前状态", displayStatus(status)],
        ["保存状态", displaySaveState(status)],
        ["录制角色", status.current_record_role_label || status.current_record_role || "-"],
        ["action_id", status.action_id || "-"],
        ["患者编号", status.patient_id || "-"],
        ["动作名称", status.action_name || "-"],
        ["侧别模式", status.side_mode_label || status.side_mode || "-"],
        ["已录有效帧", String(status.valid_frames ?? "-")],
        ["无效帧", String(status.invalid_frames ?? "-")],
        ["当前选腿", status.selected_side_label || status.selected_side || "-"],
        ["角度来源", status.selected_source_label || status.selected_source || "-"],
        ["最低可见度", formatNumber(status.visibility_min)],
        ["平均可见度", formatNumber(status.visibility_avg)],
        ["当前屈曲角", formatNumber(status.selected_flexion_angle, " 度")],
        ["平滑屈曲角", formatNumber(status.smoothed_flexion_angle, " 度")],
        ["当前 ROM", formatNumber(status.current_rom, " 度")],
        ["待重试导出", status.pending_export ? "有" : "无"],
        ["active template", status.active_template?.template_file || "-"],
        ["patient attempt", status.patient_attempt_file || "-"],
        ["评估报告", status.evaluation_report_file || "-"],
        ["最近错误", status.last_export_error || "-"],
      ];
      statusGrid.innerHTML = rows.map(([label, value]) => `
        <div class="stat">
          <b>${label}</b>
          <span>${value}</span>
        </div>
      `).join("");
      updateButtons(status);
    }

    async function getStatus() {
      try {
        const response = await fetch("/status");
        const status = await response.json();
        renderStatus(status);
      } catch (error) {
        renderStatus({});
        setMessage("状态拉取失败，请确认板端服务仍在运行。");
      }
    }

    function collectPayload(recordRole) {
      const data = new FormData(form);
      return {
        patient_id: String(data.get("patient_id") || "").trim(),
        action_name: String(data.get("action_name") || "").trim(),
        side_mode: String(data.get("side_mode") || "auto").trim(),
        record_role: recordRole,
      };
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `请求失败: ${response.status}`);
      }
      return data;
    }

    function renderEvaluation(result) {
      const report = result.report || {};
      const metrics = report.metrics || {};
      const errors = report.errors || {};
      const structured = report.structured_feedback || {};
      const feedback = result.feedback || {};
      const screen = feedback.screen || {};
      const tts = feedback.tts || {};
      const motor = feedback.motor || {};
      resultBox.innerHTML = `
        <div class="subpanel">
          <h3>评估结果</h3>
          <div class="status-grid">
            <div class="stat"><b>ROM target</b><span>${formatNumber(metrics.rom?.target, " 度")}</span></div>
            <div class="stat"><b>ROM actual</b><span>${formatNumber(metrics.rom?.actual, " 度")}</span></div>
            <div class="stat"><b>ROM diff</b><span>${formatNumber(metrics.rom?.diff, " 度")}</span></div>
            <div class="stat"><b>TUT target</b><span>${formatNumber(metrics.tut?.target, " 秒")}</span></div>
            <div class="stat"><b>TUT actual</b><span>${formatNumber(metrics.tut?.actual, " 秒")}</span></div>
            <div class="stat"><b>TUT ratio</b><span>${formatNumber(metrics.tut?.ratio)}</span></div>
            <div class="stat"><b>DTW normalized</b><span>${formatNumber(metrics.dtw?.normalized_distance)}</span></div>
            <div class="stat"><b>Speed ratio</b><span>${formatNumber(metrics.speed?.ratio)}</span></div>
            <div class="stat"><b>primary_error</b><span>${errors.primary_error || "-"}</span></div>
            <div class="stat"><b>report</b><span class="mono">${result.report_file || "-"}</span></div>
          </div>
          <div class="hint">structured_feedback：<span class="mono">${JSON.stringify(structured)}</span></div>
        </div>
        <div class="feedback-card ${screen.color || ""}">
          <b>${screen.title || "反馈"}</b><br>
          ${screen.message || "-"}<br>
          TTS mock：<span class="mono">${tts.text || "-"}</span><br>
          Motor mock：<span class="mono">${motor.pattern || "-"}</span>
        </div>
      `;
    }

    async function startRecording(recordRole) {
      try {
        const result = await postJson("/api/start", collectPayload(recordRole));
        setMessage(result.message || "已开始录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        getStatus();
      }
    }

    async function saveRecording(recordRole) {
      try {
        const result = await postJson("/api/save", { record_role: recordRole });
        if (recordRole === "doctor_template") {
          setMessage(`标准动作已保存，并设为 active template。\n${result.active_template?.template_file || result.board_saved_relative_path || "-"}`);
        } else {
          setMessage(`患者动作已保存。\n${result.patient_attempt_file || result.board_saved_relative_path || "-"}`);
        }
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        getStatus();
      }
    }

    async function runEvaluate() {
      isEvaluating = true;
      updateButtons(latestStatus);
      try {
        const result = await postJson("/api/evaluate", {
          action_id: latestStatus.action_id || "knee_flexion",
          attempt_file: latestStatus.patient_attempt_file || undefined,
        });
        renderEvaluation(result);
        setMessage(`评估完成。\n报告：${result.report_file || "-"}`);
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        isEvaluating = false;
        getStatus();
      }
    }

    startTemplateBtn.addEventListener("click", () => startRecording("doctor_template"));
    startAttemptBtn.addEventListener("click", () => startRecording("patient_attempt"));
    saveTemplateBtn.addEventListener("click", () => saveRecording("doctor_template"));
    saveAttemptBtn.addEventListener("click", () => saveRecording("patient_attempt"));
    evaluateBtn.addEventListener("click", runEvaluate);

    clearBtn.addEventListener("click", async () => {
      try {
        const result = await postJson("/api/clear", { clear_export: true });
        resultBox.innerHTML = "";
        setMessage(result.message || "已清空缓存。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        getStatus();
      }
    });

    cancelBtn.addEventListener("click", async () => {
      try {
        const result = await postJson("/api/cancel", {});
        setMessage(result.message || "已取消本轮录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        getStatus();
      }
    });

    getStatus();
    setInterval(getStatus, 1000);
  </script>
</body>
</html>
"""


def pose_worker() -> None:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while state.running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            infer_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            output_frame = frame.copy()
            selected_result: dict[str, object] = {"valid": False}
            selected_rule = LEFT_KNEE_RULE

            if result.pose_landmarks:
                output_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in result.pose_landmarks.landmark:
                    output_landmarks.landmark.add(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility,
                    )
                mp_drawing.draw_landmarks(output_frame, output_landmarks, mp_pose.POSE_CONNECTIONS)

                training_snapshot = realtime_session.snapshot()
                if training_snapshot.get("status") in {"running", "resting"} and training_snapshot.get("action_id"):
                    action_id = normalize_action_id(training_snapshot.get("action_id"))
                else:
                    action_id = normalize_action_id(state.action_name)
                action_config = get_action_config(action_id)
                action_rules = action_config.get("rules", {})
                left_rule = dict(action_rules.get("left", LEFT_KNEE_RULE))
                right_rule = dict(action_rules.get("right", RIGHT_KNEE_RULE))
                left_result = compute_action_angle(result, left_rule, action_config)
                right_result = compute_action_angle(result, right_rule, action_config)
                selected_rule, selected_result = choose_action_rule(state.side_mode, action_config, left_result, right_result)

            raw_flexion = selected_result.get("selected_flexion_angle")
            smoothed_flexion = state.smoother.update(raw_flexion)
            current_frame_data: dict[str, object] | None = None

            if state.is_recording and state.selected_rule_at_recording is None:
                state.selected_rule_at_recording = selected_rule

            if selected_result.get("valid", False) and result.pose_landmarks:
                now = time.time()
                landmarks = result.pose_landmarks.landmark
                current_frame_data = {
                    "frame_index": state.frame_index,
                    "relative_time": (now - state.start_time) if state.start_time is not None else 0,
                    "selected_side": selected_rule["side"],
                    "selected_source": selected_result.get("selected_source"),
                    "visibility_min": selected_result.get("visibility_min"),
                    "visibility_avg": selected_result.get("visibility_avg"),
                    "selected_included_angle": selected_result.get("selected_included_angle"),
                    "target_angle_raw": raw_flexion,
                    "target_angle_smoothed": smoothed_flexion,
                    "selected_flexion_angle_raw": raw_flexion,
                    "selected_flexion_angle_smoothed": smoothed_flexion,
                    "included_angle_2d": selected_result.get("included_angle_2d"),
                    "target_angle_2d": selected_result.get("target_angle_2d"),
                    "flexion_angle_2d": selected_result.get("flexion_angle_2d"),
                    "included_angle_3d": selected_result.get("included_angle_3d"),
                    "target_angle_3d": selected_result.get("target_angle_3d"),
                    "flexion_angle_3d": selected_result.get("flexion_angle_3d"),
                    "left_knee_angle": selected_result.get("selected_included_angle"),
                    "visibility": selected_result.get("visibility_min"),
                    "visibility_threshold": VISIBILITY_THRESHOLD,
                    "keypoints": build_compact_keypoints(landmarks, selected_rule),
                    "rehab_keypoints": build_rehab_keypoints(landmarks),
                }
                if state.is_recording:
                    state.frames.append(current_frame_data)
                    state.frame_index += 1
            elif state.is_recording:
                state.invalid_frame_count += 1

            realtime_session.process_frame(current_frame_data, selected_rule)

            recorded_angles = [
                frame_item["target_angle_smoothed"]
                for frame_item in state.frames
                if frame_item.get("target_angle_smoothed") is not None
            ]
            state.current_rom = max(recorded_angles) - min(recorded_angles) if recorded_angles else None

            state.selected_result = selected_result
            state.selected_rule = selected_rule

            if selected_result.get("valid", False):
                side = str(selected_result.get("side", ""))
                status = f"已检测到{SIDE_MODE_LABELS.get(side, side)}"
            elif state.awaiting_ack and state.last_export_payload is not None:
                status = "等待保存确认"
            elif state.is_recording:
                status = "录制中"
            else:
                status = "未检测到可靠关键点"

            ok, jpg = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                state.update_frame(jpg.tobytes(), status)
    finally:
        pose.close()


class PrescriptionHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            body = build_home_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/doctor":
            body = build_page_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/train":
            body = build_train_page_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/status":
            make_json_response(self, state.snapshot_status())
            return

        if parsed.path == "/api/realtime/status":
            make_json_response(self, {"ok": True, "training": realtime_session.snapshot()})
            return

        if parsed.path == "/api/system/status":
            make_json_response(self, get_system_status(state.pose_fps))
            return

        if parsed.path == "/api/active_template":
            action_id = "knee_flexion"
            for item in (parsed.query or "").split("&"):
                if item.startswith("action_id="):
                    action_id = normalize_action_id(item.split("=", 1)[1])
                    break
            active_template = get_active_template(action_id)
            make_json_response(
                self,
                {
                    "ok": True,
                    "action_id": action_id,
                    "active_template": active_template,
                    "active_template_file": active_template.get("template_file") if active_template else None,
                },
            )
            return

        if parsed.path == "/api/export_last":
            if state.last_export_payload is None:
                make_json_response(self, {"ok": False, "error": "没有待重试导出的结果。"}, status_code=404)
                return
            response_payload = {
                "ok": True,
                "prescription": state.last_export_payload,
                "summary": state.last_export_summary,
            }
            if state.last_export_board_result is not None:
                response_payload["board_saved_path"] = state.last_export_board_result.get("saved_path")
                response_payload["board_summary_path"] = state.last_export_board_result.get("summary_path")
                response_payload["board_summary"] = state.last_export_board_result.get("summary")
            make_json_response(self, response_payload)
            return

        if parsed.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_sent_frame_id = -1
            while state.running:
                with state.condition:
                    state.condition.wait_for(lambda: state.frame_id != last_sent_frame_id or not state.running)
                    if not state.running:
                        break
                    last_sent_frame_id = state.frame_id
                    data = state.jpg_bytes

                if data is None:
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            payload = read_json_body(self)
        except json.JSONDecodeError:
            make_json_response(self, {"ok": False, "error": "请求体不是有效 JSON。"}, status_code=400)
            return

        if self.path == "/api/realtime/start":
            if state.is_recording:
                make_json_response(self, {"ok": False, "error": "当前正在医生录制中，请先保存或取消。"}, status_code=400)
                return
            action_id = normalize_action_id(payload.get("action_id") or state.action_name)
            patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            target_reps = payload.get("target_reps")
            try:
                target_reps_value = int(target_reps) if target_reps is not None else None
            except (TypeError, ValueError):
                target_reps_value = None
            result = realtime_session.start(
                patient_id=patient_id,
                action_id=action_id,
                side_mode=side_mode,
                target_reps=target_reps_value,
            )
            if not result.get("ok"):
                make_json_response(self, result, status_code=400)
                return
            make_json_response(self, result)
            return

        if self.path == "/api/realtime/start_playlist":
            if state.is_recording:
                make_json_response(self, {"ok": False, "error": "当前正在医生录制中，请先保存或取消。"}, status_code=400)
                return
            patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            target_reps = payload.get("target_reps")
            try:
                target_reps_value = int(target_reps) if target_reps is not None else None
            except (TypeError, ValueError):
                target_reps_value = None
            result = realtime_session.start_playlist(
                patient_id=patient_id,
                side_mode=side_mode,
                target_reps=target_reps_value,
            )
            if not result.get("ok"):
                make_json_response(self, result, status_code=400)
                return
            make_json_response(self, result)
            return

        if self.path == "/api/realtime/pause":
            make_json_response(self, realtime_session.pause())
            return

        if self.path == "/api/realtime/stop":
            make_json_response(self, realtime_session.stop())
            return

        if self.path == "/api/start":
            if state.is_recording:
                make_json_response(self, {"ok": False, "error": "当前正在录制中，请先保存或取消本轮录制。"}, status_code=400)
                return
            if realtime_session.snapshot().get("status") == "running":
                make_json_response(self, {"ok": False, "error": "当前正在患者实时训练中，请先结束训练。"}, status_code=400)
                return
            state.patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            state.action_name = sanitize_text(payload.get("action_name"), "knee_flexion")
            state.current_record_role = normalize_record_role(payload.get("record_role"))
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            state.side_mode = side_mode if side_mode in {"left", "right", "auto"} else "auto"
            state.reset_recording()
            state.start_time = time.time()
            state.is_recording = True
            state.last_export_error = None
            role_label = RECORD_ROLE_LABELS.get(state.current_record_role, state.current_record_role)
            make_json_response(self, {"ok": True, "record_role": state.current_record_role, "message": f"已开始录制{role_label}。"})
            return

        if self.path == "/api/clear":
            clear_export = bool(payload.get("clear_export", False))
            state.reset_recording()
            if clear_export:
                state.clear_export()
            state.last_export_error = None
            make_json_response(self, {"ok": True, "message": "已清空录制缓存。"})
            return

        if self.path == "/api/cancel":
            state.clear_all()
            make_json_response(self, {"ok": True, "message": "已取消当前录制，并清空板端缓存。"})
            return

        if self.path == "/api/save":
            if not state.frames:
                state.last_export_error = "没有录到有效骨架数据，请重新录制。"
                make_json_response(self, {"ok": False, "error": state.last_export_error}, status_code=400)
                return

            state.is_recording = False
            record_role = normalize_record_role(payload.get("record_role") or state.current_record_role)
            meta = {
                "camera_device": CAMERA_DEVICE,
                "camera_backend": "cv2.CAP_V4L2",
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "infer_width": INFER_WIDTH,
                "infer_height": INFER_HEIGHT,
                "side_mode": state.side_mode,
                "prefer_3d_world_angle": PREFER_3D_WORLD_ANGLE,
                "model_complexity": MODEL_COMPLEXITY,
                "visibility_threshold": VISIBILITY_THRESHOLD,
                "smooth_window_size": SMOOTH_WINDOW_SIZE,
                "invalid_frame_count": state.invalid_frame_count,
                "result_format": "compact_v1",
                "record_role": record_role,
            }
            prescription = build_prescription(
                state.patient_id,
                state.action_name,
                list(state.frames),
                state.selected_rule_at_recording or state.selected_rule,
                meta,
            )
            board_ip, board_port = split_host_port(self.headers.get("Host", ""))
            try:
                board_save_result = save_prescription_artifacts(
                    prescription,
                    board_ip=board_ip,
                    board_port=board_port,
                    source="record_prescription_http_board",
                )
            except OSError as error:
                state.last_export_error = f"板端保存失败：{error}"
                make_json_response(self, {"ok": False, "error": state.last_export_error}, status_code=500)
                return

            state.last_export_payload = prescription
            baseline = prescription["clinical_baseline"]
            state.last_export_summary = {
                "patient_id": prescription["patient_id"],
                "action_name": prescription["action_name"],
                "frame_count": baseline["frame_count"],
                "duration_seconds": baseline["duration_seconds"],
                "rom_flexion": baseline["rom_flexion"],
            }
            state.last_export_board_result = board_save_result
            state.last_export_error = None
            state.awaiting_ack = False
            active_template = None
            saved_relative_path = project_relative(board_save_result["saved_path"])
            if record_role == "doctor_template":
                active_template = set_active_template(normalize_action_id(prescription["action_name"]), board_save_result["saved_path"])
            elif record_role == "patient_attempt":
                state.last_patient_attempt_path = saved_relative_path
                state.last_patient_attempt_summary = state.last_export_summary
                state.last_evaluation_report_path = None
                state.last_feedback = None
            make_json_response(
                self,
                {
                    "ok": True,
                    "record_role": record_role,
                    "active_template": active_template,
                    "patient_attempt_file": state.last_patient_attempt_path,
                    "prescription": prescription,
                    "summary": state.last_export_summary,
                    "board_saved_path": board_save_result["saved_path"],
                    "board_saved_relative_path": saved_relative_path,
                    "board_summary_path": board_save_result["summary_path"],
                    "board_summary": board_save_result["summary"],
                    "message": "已保存到板端 docs/results/，第三阶段离线闭环不需要同步 Windows。",
                },
            )
            return

        if self.path == "/api/evaluate":
            action_id = normalize_action_id(payload.get("action_id") or state.action_name)
            attempt_file = payload.get("attempt_file") or state.last_patient_attempt_path
            if not attempt_file:
                make_json_response(self, {"ok": False, "error": "请先录入患者动作"}, status_code=400)
                return
            result = evaluate_attempt(action_id, str(attempt_file))
            if not result.get("ok"):
                make_json_response(self, result, status_code=400)
                return
            state.last_evaluation_report_path = str(result.get("report_file"))
            state.last_feedback = result.get("feedback") if isinstance(result.get("feedback"), dict) else None
            make_json_response(self, result)
            return

        if self.path == "/api/ack_saved":
            if state.last_export_payload is None or not state.awaiting_ack:
                make_json_response(self, {"ok": False, "error": "当前没有待确认清理的导出结果。"}, status_code=400)
                return

            state.reset_recording()
            state.clear_export()
            make_json_response(self, {"ok": True, "message": "板端已确认本机保存成功，并清空本次录制缓存。"})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> None:
    worker = threading.Thread(target=pose_worker, daemon=True)
    worker.start()

    server = ThreadedHTTPServer(("0.0.0.0", PORT), PrescriptionHTTPHandler)

    print(f"8082 统一训练台已启动: http://板子IP:{PORT}")
    print(f"摄像头设备: {CAMERA_DEVICE}")
    print(f"采集分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"推理分辨率: {INFER_WIDTH}x{INFER_HEIGHT}")
    print("说明: /doctor 用于医生标准动作录制，/train 用于患者实时屈膝训练。")

    try:
        server.serve_forever()
    finally:
        state.running = False
        realtime_session.stop()
        with state.condition:
            state.condition.notify_all()
        cap.release()


if __name__ == "__main__":
    main()
