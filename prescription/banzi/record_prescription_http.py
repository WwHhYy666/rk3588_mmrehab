from __future__ import annotations

import json
import math
import mimetypes
import os
import queue
import base64
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import cv2
except Exception as exc:  # pragma: no cover - environment dependent
    cv2 = None
    CV2_IMPORT_ERROR = str(exc)
else:
    CV2_IMPORT_ERROR = None

try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
except Exception as exc:  # pragma: no cover - environment dependent
    mp = None
    landmark_pb2 = None
    MEDIAPIPE_IMPORT_ERROR = str(exc)
else:
    MEDIAPIPE_IMPORT_ERROR = None

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is expected on the board
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from feedback.feedback_engine import build_feedback_from_files
from prescription.common.active_templates import (
    get_active_template as registry_get_active_template,
    load_active_templates as registry_load_active_templates,
    normalize_pose_backend,
    save_active_templates as registry_save_active_templates,
    set_active_template as registry_set_active_template,
)
from prescription.common.llm_assistant import answer_question, get_llm_status, summarize_report
from prescription.common.report_visuals import (
    attach_keyframe_urls,
    build_keyframe_notes,
    build_metric_cards,
    render_report_images,
    resolve_keyframe_path,
)
from prescription.common.result_storage import save_prescription_artifacts
from realtime.system_monitor import get_system_status
from realtime.training_session import RealtimeTrainingSession
from realtime.tts_worker import TTSWorker
from vision.pose_backend_selector import PoseBackendSelection, resolve_pose_backend

try:
    from vision.rknn_pose.pose_frame_adapter import RknnPoseStabilizer, StablePersonSelector, adapt_rknn_pose_frame
    from vision.rknn_pose.rknn_backend import RKNNPoseBackend
except Exception as exc:  # pragma: no cover - RKNN branch is board dependent
    RknnPoseStabilizer = None
    StablePersonSelector = None
    RKNNPoseBackend = None
    RKNN_IMPORT_ERROR = str(exc)
else:
    RKNN_IMPORT_ERROR = None


DEFAULT_USB_CAMERA_BY_ID = "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"
DEFAULT_CAMERA_DEVICE = DEFAULT_USB_CAMERA_BY_ID if os.name != "nt" and Path(DEFAULT_USB_CAMERA_BY_ID).exists() else "auto"
CAMERA_DEVICE = os.environ.get("RK_CAMERA_DEVICE", DEFAULT_CAMERA_DEVICE)
CAMERA_BACKEND = cv2.CAP_V4L2 if cv2 is not None else 0
CAMERA_OPEN_MODE = os.environ.get("RK_CAMERA_OPEN_MODE", "opencv").strip().lower() or "opencv"
CAMERA_FPS = int(os.environ.get("RK_CAMERA_FPS", "30"))
CAMERA_GST_FORMAT = os.environ.get("RK_CAMERA_GST_FORMAT", "MJPG").strip().upper() or "MJPG"
FOURCC = "MJPG"
FRAME_WIDTH = int(os.environ.get("RK_CAMERA_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("RK_CAMERA_HEIGHT", "360"))
INFER_WIDTH = 640
INFER_HEIGHT = 360
JPEG_QUALITY = 70
KEYFRAME_MAX_WIDTH = 640
KEYFRAME_JPEG_QUALITY = 70
try:
    REHAB_KEYFRAME_EVERY_N = max(1, int(os.environ.get("REHAB_KEYFRAME_EVERY_N", "3")))
except ValueError:
    REHAB_KEYFRAME_EVERY_N = 3
RKNN_STREAM_WIDTH = int(os.environ.get("RKNN_STREAM_WIDTH", "960"))
RKNN_STREAM_HEIGHT = int(os.environ.get("RKNN_STREAM_HEIGHT", "540"))
RKNN_PREVIEW_FPS = float(os.environ.get("RKNN_PREVIEW_FPS", "6"))
PORT = 8082

RUNTIME_DIR = PROJECT_ROOT / "runtime"
ACTIVE_TEMPLATES_PATH = RUNTIME_DIR / "active_templates.json"
EVALUATE_REPORTS_DIR = PROJECT_ROOT / "evaluate" / "reports"
STATIC_DIR = Path(__file__).resolve().parent / "static"
POSE_BACKEND_CALIBRATION_PATH = PROJECT_ROOT / "realtime" / "configs" / "pose_backend_calibration.yaml"
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
ACTION_METRIC_VISIBILITY_THRESHOLD = float(os.environ.get("ACTION_METRIC_VISIBILITY_THRESHOLD", "0.35"))
SMOOTH_WINDOW_SIZE = 5
PREFER_3D_WORLD_ANGLE = True
MODEL_COMPLEXITY = 1
ALLOW_POSE_BACKEND_MISMATCH = os.environ.get("ALLOW_POSE_BACKEND_MISMATCH", "0").strip() == "1"
SIDE_VIEW_RATIO_MAX = float(os.environ.get("SIDE_VIEW_RATIO_MAX", "0.32"))
RKNN_DEBUG_KEYPOINT_DRAW_THRESHOLD = float(os.environ.get("RKNN_DEBUG_KEYPOINT_DRAW_THRESHOLD", "0.05"))
RKNN_DRAW_RAW_DEBUG_SKELETON = os.environ.get("RKNN_DRAW_RAW_DEBUG_SKELETON", "0").strip() == "1"
RKNN_RELAXED_VISIBILITY_THRESHOLD = float(os.environ.get("RKNN_RELAXED_VISIBILITY_THRESHOLD", "0.12"))
RKNN_FAST_PREVIEW = os.environ.get("RKNN_FAST_PREVIEW", "0").strip() == "1"
RKNN_FAST_FRAME_DATA = os.environ.get("RKNN_FAST_FRAME_DATA", "0").strip() == "1"
RKNN_FIXED_STRICT_LEG_VISIBILITY = os.environ.get("RKNN_FIXED_STRICT_LEG_VISIBILITY", "1").strip() == "1"
RKNN_FIXED_LEG_VISIBILITY_THRESHOLD = float(os.environ.get("RKNN_FIXED_LEG_VISIBILITY_THRESHOLD", "0.30"))
RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD = float(os.environ.get("RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD", "0.35"))
RKNN_DRAW_FIXED_BBOX = os.environ.get("RKNN_DRAW_FIXED_BBOX", "0").strip() == "1"
try:
    RK_CAMERA_FAIL_DISPLAY_THRESHOLD = max(1, int(os.environ.get("RK_CAMERA_FAIL_DISPLAY_THRESHOLD", "30")))
except ValueError:
    RK_CAMERA_FAIL_DISPLAY_THRESHOLD = 30
ACTIVE_REALTIME_STATUSES = {
    "running",
    "paused",
    "resting",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
}
POSE_BACKEND_VERSION = {
    "mediapipe": "mediapipe_pose",
    "rknn": "rknn_pose",
}
POSE_KEYPOINT_SCHEMA = {
    "mediapipe": "mediapipe33_to_rehab_v1",
    "rknn": "coco17_to_rehab_v1",
}


def load_pose_backend_calibration() -> dict[str, object]:
    defaults: dict[str, object] = {
        "mediapipe": {
            "visibility_min": VISIBILITY_THRESHOLD,
            "smoothing_window": SMOOTH_WINDOW_SIZE,
            "angle_offset": {},
        },
        "rknn": {
            "visibility_min": 0.12,
            "smoothing_window": SMOOTH_WINDOW_SIZE,
            "angle_offset": {},
            "stabilizer": {
                "alpha": 0.35,
                "low_conf_alpha": 0.18,
                "jump_scale": 0.35,
                "max_hold_frames": 8,
                "lock_confirm_frames": 4,
            },
        },
    }
    if yaml is None or not POSE_BACKEND_CALIBRATION_PATH.exists():
        return defaults
    try:
        payload = yaml.safe_load(POSE_BACKEND_CALIBRATION_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    if not isinstance(payload, dict):
        return defaults
    merged = dict(defaults)
    for backend in ("mediapipe", "rknn"):
        current = dict(merged.get(backend) or {})
        incoming = payload.get(backend)
        if isinstance(incoming, dict):
            for key, value in incoming.items():
                if isinstance(value, dict) and isinstance(current.get(key), dict):
                    nested = dict(current[key])
                    nested.update(value)
                    current[key] = nested
                else:
                    current[key] = value
        merged[backend] = current
    return merged


POSE_BACKEND_CALIBRATION = load_pose_backend_calibration()


def backend_calibration(backend: str | None = None) -> dict[str, object]:
    key = normalize_pose_backend(backend or getattr(globals().get("state", object()), "pose_backend_actual", "mediapipe"))
    value = POSE_BACKEND_CALIBRATION.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def backend_visibility_threshold(backend: str | None = None) -> float:
    key = normalize_pose_backend(backend or getattr(globals().get("state", object()), "pose_backend_actual", "mediapipe"))
    if key == "mediapipe":
        return VISIBILITY_THRESHOLD
    if key == "rknn" and "RKNN_POSE_KEYPOINT_THRES" in os.environ and globals().get("rknn_backend") is not None:
        return float(globals()["rknn_backend"].keypoint_thres)
    calibration = backend_calibration(key)
    try:
        return float(calibration.get("visibility_min", VISIBILITY_THRESHOLD))
    except (TypeError, ValueError):
        return VISIBILITY_THRESHOLD


def backend_smoothing_window(backend: str | None = None) -> int:
    calibration = backend_calibration(backend)
    try:
        return max(1, int(calibration.get("smoothing_window", SMOOTH_WINDOW_SIZE)))
    except (TypeError, ValueError):
        return SMOOTH_WINDOW_SIZE


def backend_angle_offset(action_id: str, backend: str | None = None) -> float:
    calibration = backend_calibration(backend)
    offsets = calibration.get("angle_offset", {})
    if not isinstance(offsets, dict):
        return 0.0
    try:
        return float(offsets.get(action_id, 0.0))
    except (TypeError, ValueError):
        return 0.0
RKNN_COCO17_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

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
        "metric_kind": "knee_raise_height_ratio",
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

PREFERRED_SIDE_BY_ACTION = {
    "seated_knee_extension": "left",
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


def calculate_knee_raise_height_ratio(points: list[tuple[float, ...]], point_names: list[str]) -> float | None:
    named = {name: point for name, point in zip(point_names, points)}
    shoulder = named.get("shoulder")
    hip = named.get("hip")
    knee = named.get("knee")
    if shoulder is None or hip is None or knee is None:
        return None
    if len(shoulder) < 2 or len(hip) < 2 or len(knee) < 2:
        return None
    torso_scale = math.hypot(float(shoulder[0]) - float(hip[0]), float(shoulder[1]) - float(hip[1]))
    if torso_scale <= 1e-8:
        return None
    return (float(hip[1]) - float(knee[1])) / torso_scale


def get_landmark_tuple(landmarks, index: int, use_3d: bool = False) -> tuple[float, ...]:
    landmark = landmarks[index]
    if use_3d:
        return (landmark.x, landmark.y, landmark.z)
    return (landmark.x, landmark.y)


def get_visibility(landmarks, indices: list[int]) -> tuple[float, float]:
    values = [landmarks[i].visibility for i in indices]
    return min(values), sum(values) / len(values)


def _point_number(point: dict[str, object] | None, field: str) -> float | None:
    if not isinstance(point, dict):
        return None
    value = point.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def target_leg_visibility(
    rehab_keypoints: dict[str, object],
    selected_rule: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    side = str(selected_rule.get("side") or "left")
    values: dict[str, float] = {}
    missing: list[str] = []
    for joint in ("hip", "knee", "ankle"):
        keypoint_name = f"{side}_{joint}"
        point = rehab_keypoints.get(keypoint_name)
        visibility = _point_number(point if isinstance(point, dict) else None, "visibility")
        value = float(visibility or 0.0)
        values[joint] = round(value, 3)
        if value < threshold:
            missing.append(keypoint_name)
    visibility_values = list(values.values())
    visibility_min = min(visibility_values) if visibility_values else 0.0
    visibility_avg = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
    return {
        "target_side": side,
        "target_leg_visibility": values,
        "target_side_keypoint_visibility": values,
        "target_leg_visibility_min": round(visibility_min, 3),
        "target_leg_visibility_avg": round(visibility_avg, 3),
        "target_leg_visibility_threshold": round(float(threshold), 3),
        "target_leg_visibility_ok": len(missing) == 0,
        "target_leg_missing_keypoints": missing,
    }


def apply_rtmpose_fixed_visibility_guard(
    selected_result: dict[str, object],
    selected_rule: dict[str, object],
    rehab_keypoints: dict[str, object],
) -> None:
    diagnostics = target_leg_visibility(rehab_keypoints, selected_rule, RKNN_FIXED_LEG_VISIBILITY_THRESHOLD)
    selected_result.update(diagnostics)
    if not RKNN_FIXED_STRICT_LEG_VISIBILITY or diagnostics["target_leg_visibility_ok"]:
        return
    missing = sorted(set(list(selected_result.get("missing_keypoints") or []) + list(diagnostics["target_leg_missing_keypoints"])))
    selected_result.update(
        {
            "valid": False,
            "quality_ok": False,
            "quality_message": "请让髋、膝、踝完整入镜",
            "missing_keypoints": missing,
        }
    )
    for key in (
        "included_angle_2d",
        "target_angle_2d",
        "flexion_angle_2d",
        "selected_included_angle",
        "selected_target_angle",
        "selected_flexion_angle",
    ):
        selected_result[key] = None


def compute_side_view_metrics(
    rehab_keypoints: dict[str, object],
    selected_result: dict[str, object],
    visibility_threshold: float,
    selected_rule: dict[str, object] | None = None,
    backend: str | None = None,
) -> dict[str, object]:
    backend_name = str(backend or selected_result.get("pose_backend") or "")
    if backend_name == "rknn" or selected_result.get("selected_source") == "rknn_2d_image":
        return compute_rknn_side_view_metrics(rehab_keypoints, selected_result, visibility_threshold, selected_rule)

    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    points = {name: rehab_keypoints.get(name) for name in required}
    visibilities = [
        _point_number(point if isinstance(point, dict) else None, "visibility") or 0.0
        for point in points.values()
    ]
    has_any_torso = any(value > 0.0 for value in visibilities)
    pose_detected = bool(selected_result.get("valid")) or int(selected_result.get("person_count") or 0) > 0 or has_any_torso
    if not pose_detected:
        return {
            "pose_detected": False,
            "orientation_ok": False,
            "orientation_ratio": None,
            "orientation_visibility": 0.0,
            "orientation_message": "未检测到训练者",
        }

    if any(value < visibility_threshold for value in visibilities):
        return {
            "pose_detected": True,
            "orientation_ok": False,
            "orientation_ratio": None,
            "orientation_visibility": min(visibilities) if visibilities else 0.0,
            "orientation_message": "请保持肩部和髋部关键点可见",
        }

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]
    ls_x = _point_number(left_shoulder if isinstance(left_shoulder, dict) else None, "x")
    rs_x = _point_number(right_shoulder if isinstance(right_shoulder, dict) else None, "x")
    lh_x = _point_number(left_hip if isinstance(left_hip, dict) else None, "x")
    rh_x = _point_number(right_hip if isinstance(right_hip, dict) else None, "x")
    ls_y = _point_number(left_shoulder if isinstance(left_shoulder, dict) else None, "y")
    rs_y = _point_number(right_shoulder if isinstance(right_shoulder, dict) else None, "y")
    lh_y = _point_number(left_hip if isinstance(left_hip, dict) else None, "y")
    rh_y = _point_number(right_hip if isinstance(right_hip, dict) else None, "y")
    values = [ls_x, rs_x, lh_x, rh_x, ls_y, rs_y, lh_y, rh_y]
    if any(value is None for value in values):
        return {
            "pose_detected": True,
            "orientation_ok": False,
            "orientation_ratio": None,
            "orientation_visibility": min(visibilities),
            "orientation_message": "请保持身体侧面完整入镜",
        }

    shoulder_width = abs(float(ls_x) - float(rs_x))
    hip_width = abs(float(lh_x) - float(rh_x))
    torso_height = max((abs(float(ls_y) - float(lh_y)) + abs(float(rs_y) - float(rh_y))) / 2.0, 1e-6)
    orientation_ratio = max(shoulder_width, hip_width) / torso_height
    orientation_ok = orientation_ratio <= SIDE_VIEW_RATIO_MAX
    return {
        "pose_detected": True,
        "orientation_ok": orientation_ok,
        "orientation_ratio": orientation_ratio,
        "orientation_visibility": min(visibilities),
        "orientation_message": "侧身角度合适" if orientation_ok else "请调整角度，将身体侧面对准摄像头",
    }


def compute_rknn_side_view_metrics(
    rehab_keypoints: dict[str, object],
    selected_result: dict[str, object],
    visibility_threshold: float,
    selected_rule: dict[str, object] | None = None,
) -> dict[str, object]:
    visibilities = [
        _point_number(point if isinstance(point, dict) else None, "visibility") or 0.0
        for point in rehab_keypoints.values()
    ]
    has_any_keypoint = any(value > 0.01 for value in visibilities)
    pose_detected = bool(selected_result.get("valid")) or int(selected_result.get("person_count") or 0) > 0 or has_any_keypoint
    if not pose_detected:
        return {
            "pose_detected": False,
            "orientation_ok": False,
            "orientation_ratio": None,
            "orientation_visibility": 0.0,
            "orientation_message": "未检测到训练者",
            "rknn_orientation_relaxed": True,
        }

    relaxed_threshold = min(float(visibility_threshold), RKNN_RELAXED_VISIBILITY_THRESHOLD)
    selected_side = str(
        selected_result.get("side")
        or (selected_rule or {}).get("side")
        or "left"
    )
    sides = [selected_side]
    other_side = "right" if selected_side == "left" else "left"
    if other_side not in sides:
        sides.append(other_side)

    checked_chains: list[str] = []
    best_visibility = 0.0
    best_count = 0
    for side in sides:
        for chain in (
            [f"{side}_hip", f"{side}_knee", f"{side}_ankle"],
            [f"{side}_shoulder", f"{side}_hip", f"{side}_knee"],
        ):
            values = [
                _point_number(rehab_keypoints.get(name) if isinstance(rehab_keypoints.get(name), dict) else None, "visibility") or 0.0
                for name in chain
            ]
            visible_count = sum(1 for value in values if value >= relaxed_threshold)
            checked_chains.append(f"{'-'.join(chain)}:{visible_count}/3")
            if visible_count > best_count:
                best_count = visible_count
                best_visibility = min(values) if values else 0.0
            if visible_count == len(chain):
                return {
                    "pose_detected": True,
                    "orientation_ok": True,
                    "orientation_ratio": None,
                    "orientation_visibility": min(values),
                    "orientation_message": "RKNN relaxed side view ok",
                    "rknn_orientation_relaxed": True,
                    "rknn_orientation_chains": checked_chains,
                }

    if selected_result.get("valid"):
        return {
            "pose_detected": True,
            "orientation_ok": True,
            "orientation_ratio": None,
            "orientation_visibility": selected_result.get("visibility_min"),
            "orientation_message": "RKNN selected action keypoints ok",
            "rknn_orientation_relaxed": True,
            "rknn_orientation_chains": checked_chains,
        }

    return {
        "pose_detected": True,
        "orientation_ok": False,
        "orientation_ratio": None,
        "orientation_visibility": best_visibility,
        "orientation_message": "RKNN detected person, waiting for near-side hip/knee/ankle",
        "rknn_orientation_relaxed": True,
        "rknn_orientation_chains": checked_chains,
    }


def draw_rehab_skeleton_overlay(
    frame,
    rehab_keypoints: dict[str, object],
    selected_rule: dict[str, object],
    action_config: dict[str, object],
    visibility_threshold: float,
    draw_visibility_threshold: float | None = None,
):
    if cv2 is None or frame is None or not rehab_keypoints:
        return frame
    height, width = frame.shape[:2]
    if draw_visibility_threshold is None:
        draw_threshold = max(0.05, min(float(visibility_threshold), 0.30))
    else:
        draw_threshold = max(0.05, float(draw_visibility_threshold))
    base_edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    for start, end in base_edges:
        _draw_rehab_edge(frame, rehab_keypoints, start, end, width, height, draw_threshold, (40, 230, 120), 3)

    selected_side = str(selected_rule.get("side") or "left")
    point_names = [str(name) for name in action_config.get("point_names", ["hip", "knee", "ankle"])]
    highlight_names = [f"{selected_side}_{name}" for name in point_names]
    for start, end in zip(highlight_names, highlight_names[1:]):
        _draw_rehab_edge(frame, rehab_keypoints, start, end, width, height, draw_threshold, (255, 210, 60), 5)
    for name in sorted(rehab_keypoints):
        color = (70, 220, 255) if name in highlight_names else (60, 180, 255)
        radius = 6 if name in highlight_names else 4
        _draw_rehab_point(frame, rehab_keypoints, name, width, height, draw_threshold, color, radius)
    return frame


def draw_fixed_bbox_overlay(frame, fixed_bbox: object) -> None:
    if cv2 is None or frame is None or not isinstance(fixed_bbox, (list, tuple)) or len(fixed_bbox) != 4:
        return
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in fixed_bbox]
    except (TypeError, ValueError):
        return
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 2)
    cv2.putText(frame, "fixed bbox", (x1 + 6, max(24, y1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)


def _rehab_pixel(
    rehab_keypoints: dict[str, object],
    name: str,
    width: int,
    height: int,
    visibility_threshold: float,
) -> tuple[int, int] | None:
    point = rehab_keypoints.get(name)
    if not isinstance(point, dict):
        return None
    visibility = _point_number(point, "visibility") or 0.0
    x_value = _point_number(point, "x")
    y_value = _point_number(point, "y")
    if x_value is None or y_value is None or visibility < visibility_threshold:
        return None
    x_pixel = int(round(max(0.0, min(1.0, x_value)) * width))
    y_pixel = int(round(max(0.0, min(1.0, y_value)) * height))
    return x_pixel, y_pixel


def _draw_rehab_edge(
    frame,
    rehab_keypoints: dict[str, object],
    start: str,
    end: str,
    width: int,
    height: int,
    visibility_threshold: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    start_pixel = _rehab_pixel(rehab_keypoints, start, width, height, visibility_threshold)
    end_pixel = _rehab_pixel(rehab_keypoints, end, width, height, visibility_threshold)
    if start_pixel is not None and end_pixel is not None:
        cv2.line(frame, start_pixel, end_pixel, color, thickness, cv2.LINE_AA)


def _draw_rehab_point(
    frame,
    rehab_keypoints: dict[str, object],
    name: str,
    width: int,
    height: int,
    visibility_threshold: float,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    pixel = _rehab_pixel(rehab_keypoints, name, width, height, visibility_threshold)
    if pixel is not None:
        cv2.circle(frame, pixel, radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pixel, radius + 1, (10, 20, 30), 1, cv2.LINE_AA)


def draw_rknn_debug_overlay(
    frame,
    selected_detection: dict[str, object] | None,
    selected_result: dict[str, object],
    frame_data: dict[str, object],
    *,
    detection_count: int,
    keypoint_threshold: float,
    postprocess_error: object | None,
):
    if cv2 is None or frame is None:
        return frame
    keypoints = []
    if isinstance(selected_detection, dict):
        keypoints = list(selected_detection.get("keypoints") or [])
    draw_threshold = max(0.0, RKNN_DEBUG_KEYPOINT_DRAW_THRESHOLD)
    if RKNN_DRAW_RAW_DEBUG_SKELETON:
        for start, end in RKNN_COCO17_EDGES:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_point = keypoints[start]
            end_point = keypoints[end]
            if len(start_point) < 3 or len(end_point) < 3:
                continue
            if float(start_point[2]) < draw_threshold or float(end_point[2]) < draw_threshold:
                continue
            cv2.line(
                frame,
                (int(start_point[0]), int(start_point[1])),
                (int(end_point[0]), int(end_point[1])),
                (255, 80, 220),
                2,
                cv2.LINE_AA,
            )
        for index, point in enumerate(keypoints):
            if len(point) < 3 or float(point[2]) < draw_threshold:
                continue
            pixel = (int(point[0]), int(point[1]))
            cv2.circle(frame, pixel, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pixel, 6, (20, 20, 20), 1, cv2.LINE_AA)
            if index in {5, 6, 11, 12, 13, 14, 15, 16}:
                cv2.putText(frame, str(index), (pixel[0] + 6, pixel[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    missing = selected_result.get("missing_keypoints") or []
    if isinstance(missing, list):
        missing_text = ",".join(str(item) for item in missing[:4]) or "-"
    else:
        missing_text = str(missing)
    debug_lines = [
        f"RKNN det={detection_count} decoder={selected_result.get('rknn_decoder') or '-'}",
        f"kpt_mode={selected_result.get('keypoint_decode_mode') or '-'} order={selected_result.get('keypoint_anchor_order') or '-'} geom={selected_result.get('keypoint_geometry_score_range') or '-'}",
        f"override={selected_result.get('keypoint_decode_override') or '-'} conf={selected_result.get('keypoint_conf_range')}",
        f"kp_thr={keypoint_threshold:.2f} draw={draw_threshold:.2f} raw={selected_result.get('keypoint_raw_xy_range')}",
        f"locked={selected_result.get('locked_side') or '-'} held={len(selected_result.get('held_keypoints') or [])} jump={len(selected_result.get('jump_rejected') or [])} swap={bool(selected_result.get('side_switch_blocked'))}",
        f"pose={bool(frame_data.get('pose_detected'))} orient={bool(frame_data.get('orientation_ok'))} vis={frame_data.get('orientation_visibility')}",
        f"missing={missing_text}",
    ]
    if postprocess_error:
        debug_lines.append(f"postprocess={str(postprocess_error)[:52]}")
    y = 28
    for line in debug_lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return frame


def normalize_action_id(value: object) -> str:
    text = str(value or "").strip()
    return ACTION_ALIASES.get(text, ACTION_ALIASES.get(text.lower(), "knee_flexion"))


def get_action_config(action_id: str) -> dict[str, object]:
    return ACTION_RULES.get(action_id, ACTION_RULES["knee_flexion"])


def preferred_side_for_action(action_id: str) -> str | None:
    side = PREFERRED_SIDE_BY_ACTION.get(action_id)
    return side if side in {"left", "right"} else None


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
    metric_kind = str(action_config.get("metric_kind", "angle"))

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
        if metric_kind == "knee_raise_height_ratio":
            target_angle_2d = calculate_knee_raise_height_ratio(points_2d, list(point_indices.keys()))
        else:
            target_angle_2d = target_angle_from_included_angle(included_2d, angle_kind)

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks.landmark
            points_3d = [
                get_landmark_tuple(world_landmarks, index, use_3d=True)
                for index in indices
            ]
            included_3d = calculate_angle(points_3d)
            target_angle_3d = None if metric_kind == "knee_raise_height_ratio" else target_angle_from_included_angle(included_3d, angle_kind)

    selected_source = None
    selected_included = None
    selected_target_angle = None

    if metric_kind == "knee_raise_height_ratio" and target_angle_2d is not None:
        selected_source = "2d_image_ratio"
        selected_included = included_2d
        selected_target_angle = target_angle_2d
    elif PREFER_3D_WORLD_ANGLE and included_3d is not None:
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
        self.window_size = max(1, int(window_size))
        self.values = deque(maxlen=self.window_size)

    def update(self, value: float | None) -> float | None:
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self) -> None:
        self.values.clear()

    def set_window_size(self, window_size: int) -> None:
        normalized = max(1, int(window_size))
        if normalized == self.window_size:
            return
        recent = list(self.values)[-normalized:]
        self.window_size = normalized
        self.values = deque(recent, maxlen=self.window_size)


def choose_action_rule(
    mode: str,
    action_config: dict[str, object],
    left_result: dict[str, object],
    right_result: dict[str, object],
    locked_side: str | None = None,
    preferred_side: str | None = None,
):
    rules = action_config["rules"]
    left_rule = rules["left"]
    right_rule = rules["right"]
    if mode == "left":
        return left_rule, left_result
    if mode == "right":
        return right_rule, right_result
    if locked_side == "left":
        return left_rule, {**left_result, "locked_side": "left", "side_lock_reason": "mediapipe_locked"}
    if locked_side == "right":
        return right_rule, {**right_result, "locked_side": "right", "side_lock_reason": "mediapipe_locked"}
    if preferred_side == "left":
        return left_rule, {**left_result, "locked_side": "left", "side_lock_reason": "action_preferred"}
    if preferred_side == "right":
        return right_rule, {**right_result, "locked_side": "right", "side_lock_reason": "action_preferred"}

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


def capabilities_snapshot() -> dict[str, dict[str, object]]:
    evaluate_available = (PROJECT_ROOT / "evaluate" / "run_evaluate.py").exists()
    llm_status = get_llm_status()
    llm_note = f"{llm_status.get('provider')}/{llm_status.get('model')}"
    if llm_status.get("provider") == "glm4v_api" and not llm_status.get("api_key_configured"):
        llm_note = "GLM API 未配置 API Key"
    vision_ready = cv2 is not None and cap is not None and state.pose_backend_actual in {"mediapipe", "rknn"}
    return {
        "cv2": {"available": cv2 is not None, "note": CV2_IMPORT_ERROR or "ok"},
        "mediapipe": {"available": mp is not None, "note": MEDIAPIPE_IMPORT_ERROR or "ok"},
        "pose_backend": {"available": vision_ready, "note": state.pose_backend_message},
        "vision": {"available": vision_ready, "note": vision_boot_error or "ok"},
        "evaluate": {"available": evaluate_available, "note": "ok" if evaluate_available else "missing evaluate pipeline"},
        "llm": {"available": bool(llm_status.get("enabled")), "note": llm_note},
        "tts": {"available": True, "note": "runtime fallback enabled"},
    }


def load_report_context(report_file: str | Path | None) -> dict[str, object] | None:
    if not report_file:
        return None
    path = resolve_project_path(report_file)
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "report_file": project_relative(path),
        "report": report,
        "report_card": report.get("report_card") if isinstance(report.get("report_card"), dict) else None,
        "summary_bundle": report.get("summary_bundle") if isinstance(report.get("summary_bundle"), dict) else None,
    }


def load_recent_reports(limit: int = 3) -> list[dict[str, object]]:
    if not EVALUATE_REPORTS_DIR.exists():
        return []
    reports: list[dict[str, object]] = []
    paths = sorted(EVALUATE_REPORTS_DIR.glob("report_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in paths[:limit]:
        context = load_report_context(path)
        if context is not None:
            reports.append(context)
    return reports


def resolve_llm_report(report_id: object) -> tuple[Path | None, dict[str, object] | None, str | None]:
    value = str(report_id or "latest").strip() or "latest"
    if value == "latest":
        recent = load_recent_reports(limit=1)
        if not recent:
            return None, None, "暂无训练报告，请先完成一次训练。"
        report_file = recent[0].get("report_file")
        path = resolve_project_path(str(report_file))
        return path, recent[0], None

    if "/" in value or "\\" in value or value in {".", ".."}:
        return None, None, "report_id 只能是 evaluate/reports/ 下的安全文件名。"
    if not value.startswith("report_") or not value.endswith(".json"):
        return None, None, "report_id 必须是 report_*.json。"

    path = (EVALUATE_REPORTS_DIR / value).resolve()
    try:
        path.relative_to(EVALUATE_REPORTS_DIR.resolve())
    except ValueError:
        return None, None, "report_id 路径不安全，已拒绝读取。"
    context = load_report_context(path)
    if not context:
        return None, None, "训练报告不存在或无法读取。"
    return path, context, None


def llm_report_source_payload(path: Path | None, context: dict[str, object] | None) -> dict[str, object]:
    report = context.get("report") if isinstance(context, dict) else None
    source: dict[str, object] = {
        "source_report_file": context.get("report_file") if isinstance(context, dict) else None,
        "source_report_mtime": None,
        "source_evaluated_at": None,
    }
    if path and path.exists():
        source["source_report_mtime"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(report, dict):
        meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
        source["source_evaluated_at"] = report.get("evaluated_at") or meta.get("evaluated_at")
    return source


def frame_b64_from_payload(payload: dict[str, object]) -> str | None:
    raw_frame = str(payload.get("frame_b64") or "").strip()
    if raw_frame:
        return raw_frame
    if not payload.get("include_current_frame"):
        return None
    with state.condition:
        jpg_bytes = state.jpg_bytes
    if not jpg_bytes:
        return None
    return base64.b64encode(jpg_bytes).decode("ascii")


def first_keyframe_b64(report: dict[str, object]) -> str | None:
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    for item in keyframes:
        if not isinstance(item, dict):
            continue
        path = resolve_keyframe_path(PROJECT_ROOT, item.get("image_path"))
        if path is not None and path.exists() and path.is_file():
            try:
                return base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError:
                return None
    return None


def speak_llm_text(text: object, event_type: str = "llm_summary") -> dict[str, object]:
    global llm_tts_started
    text_value = " ".join(str(text or "").split()).strip()
    if not text_value:
        return {"ok": False, "error": "没有可朗读的 AI 文本。"}
    if len(text_value) > 120:
        text_value = text_value[:119] + "…"
    event = event_type if event_type in {"llm_summary", "llm_qa"} else "llm_summary"
    training_status = str(realtime_session.snapshot().get("status") or "")
    if training_status in ACTIVE_REALTIME_STATUSES:
        return {"ok": False, "error": "当前正在实时训练中，暂不朗读 AI 内容，避免抢占计数和纠错语音。"}
    with llm_tts_lock:
        if not llm_tts_started:
            llm_tts_worker.start()
            llm_tts_started = True
        queued = llm_tts_worker.speak(text_value, priority="low", event_type=event)
    return {
        "ok": True,
        "queued": bool(queued),
        "spoken_text": text_value,
        "tts": llm_tts_worker.snapshot(),
    }


def serve_static_asset(handler: BaseHTTPRequestHandler, raw_path: str) -> bool:
    requested = unquote(raw_path.removeprefix("/assets/")).strip("/")
    asset_path = (STATIC_DIR / requested).resolve()
    if not str(asset_path).startswith(str(STATIC_DIR.resolve())) or not asset_path.exists() or not asset_path.is_file():
        return False
    body = asset_path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(asset_path))
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def serve_report_image(handler: BaseHTTPRequestHandler, raw_path: str) -> bool:
    requested = unquote(raw_path.removeprefix("/report-images/")).strip("/")
    image_path = resolve_keyframe_path(PROJECT_ROOT, requested)
    if image_path is None or not image_path.exists() or not image_path.is_file():
        return False
    body = image_path.read_bytes()
    content_type, _ = mimetypes.guess_type(str(image_path))
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or "image/jpeg")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def build_app_shell(page_id: str, title: str, description: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{description}">
  <link rel="stylesheet" href="/assets/app.css">
</head>
<body data-page="{page_id}">
  <div id="app"></div>
  <script>window.__REHAB_APP__ = {{"page": "{page_id}"}};</script>
  <script src="/assets/common.js"></script>
  <script src="/assets/{page_id}.js"></script>
</body>
</html>
"""


def reject_report_input(path: str | Path) -> None:
    relative = project_relative(path).replace("\\", "/")
    if relative.startswith("evaluate/reports/"):
        raise ValueError("evaluate/reports/*.json 是评估输出，不能作为模板或患者动作输入。")


def load_active_templates() -> dict[str, object]:
    return registry_load_active_templates(ACTIVE_TEMPLATES_PATH)


def save_active_templates(payload: dict[str, object]) -> None:
    registry_save_active_templates(payload, ACTIVE_TEMPLATES_PATH)


def get_active_template(action_id: str, backend: str | None = None) -> dict[str, object] | None:
    backend_name = normalize_pose_backend(backend or getattr(state, "pose_backend_actual", "mediapipe"))
    return registry_get_active_template(action_id, backend_name, ACTIVE_TEMPLATES_PATH)


def set_active_template(action_id: str, template_file: str | Path, pose_meta: dict[str, object] | None = None) -> dict[str, object]:
    backend_name = normalize_pose_backend((pose_meta or {}).get("actual_backend") or getattr(state, "pose_backend_actual", "mediapipe"))
    return registry_set_active_template(
        action_id,
        template_file,
        config_file=DEFAULT_CONFIG_BY_ACTION.get(action_id, f"evaluate/configs/{action_id}.yaml"),
        pose_backend=backend_name,
        pose_meta=pose_meta,
        path=ACTIVE_TEMPLATES_PATH,
        project_root=PROJECT_ROOT,
    )


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
    summary_bundle = report.get("summary_bundle") if isinstance(report.get("summary_bundle"), dict) else None
    report_card = report.get("report_card") if isinstance(report.get("report_card"), dict) else None
    return {
        "ok": True,
        "report_file": project_relative(report_path),
        "report": report,
        "report_card": report_card,
        "summary_bundle": summary_bundle,
        "feedback": feedback,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def current_pose_meta() -> dict[str, object]:
    return {
        "requested_backend": state.pose_backend_requested,
        "actual_backend": state.pose_backend_actual,
        "pose_backend": state.pose_backend_actual,
        "pose_backend_version": POSE_BACKEND_VERSION.get(state.pose_backend_actual),
        "pose_keypoint_schema": POSE_KEYPOINT_SCHEMA.get(state.pose_backend_actual),
        "rknn_model_path": state.rknn_model_path,
        "rknn_pipeline": state.rknn_pipeline,
        "rknn_z_valid": False if state.pose_backend_actual == "rknn" else None,
        "angle_mode": "2d_image" if state.pose_backend_actual == "rknn" else "mediapipe_3d_world_then_2d",
        "camera_pose_requirement": "RKNN/NPU 要求侧身固定机位、单人入镜优先。" if state.pose_backend_actual == "rknn" else None,
    }


def validate_template_backend(action_ids: list[str]) -> dict[str, object]:
    backend_name = normalize_pose_backend(state.pose_backend_actual)
    missing = []
    mismatches = []
    for action_id in action_ids:
        active_template = get_active_template(action_id, backend_name)
        if not active_template:
            missing.append(action_id)
            continue
        template_backend = str(active_template.get("actual_backend") or active_template.get("pose_backend") or "mediapipe")
        if normalize_pose_backend(template_backend) != backend_name:
            mismatches.append({"action_id": action_id, "template_backend": template_backend, "actual_backend": backend_name})
    if missing:
        return {
            "ok": False,
            "error": f"当前 {backend_name} 后端缺少 active template，请先用该后端录入：{'、'.join(missing)}",
            "missing": missing,
            "pose_backend": backend_name,
        }
    if mismatches and not ALLOW_POSE_BACKEND_MISMATCH:
        names = "、".join(item["action_id"] for item in mismatches)
        return {
            "ok": False,
            "error": f"active template 后端与当前后端不一致，请用 {backend_name} 重新录模板：{names}",
            "backend_mismatches": mismatches,
        }
    return {"ok": True, "backend_mismatches": mismatches}


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
            "angle_source_priority": "RKNN 第一版只使用 2D 图像角度。" if meta.get("actual_backend") == "rknn" else "优先使用 MediaPipe 的 3D world landmarks，不可用时回退到 2D 图像 landmarks。",
            "smoothing": f"平滑窗口 = {SMOOTH_WINDOW_SIZE} 帧。",
            "warning": "RKNN 第一版要求侧身固定机位、单人入镜优先，角度适合演示和趋势反馈，不属于临床级测量。" if meta.get("actual_backend") == "rknn" else "单目 MediaPipe 角度适合演示和趋势反馈，不属于临床级测量。",
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


def build_camera_candidates(requested_device: str) -> list[object]:
    candidates: list[object] = []
    requested = str(requested_device or "auto").strip()
    if requested and requested.lower() != "auto":
        candidates.append(int(requested) if requested.isdigit() else requested)
    if os.name == "nt":
        candidates.extend([0, 1, 2])
    else:
        for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
            if directory.exists():
                candidates.extend(str(path) for path in sorted(directory.glob("*")))
        candidates.extend(str(path) for path in sorted(Path("/dev").glob("video*")))
    return candidates


def normalize_camera_source(candidate: object) -> object:
    if isinstance(candidate, int):
        return candidate
    text = str(candidate or "").strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"/dev/video(\d+)", text)
    if match:
        return int(match.group(1))
    return text


def camera_source_device_path(source: object) -> str:
    if isinstance(source, int):
        return str(source) if os.name == "nt" else f"/dev/video{source}"
    return str(source)


def build_gstreamer_pipeline(source: object) -> str:
    device = camera_source_device_path(source)
    if CAMERA_GST_FORMAT in {"MJPG", "MJPEG", "JPEG"}:
        caps = f"image/jpeg,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={CAMERA_FPS}/1"
        return (
            f"v4l2src device={device} io-mode=2 ! {caps} ! "
            "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
    caps = f"video/x-raw,format=YUY2,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={CAMERA_FPS}/1"
    return (
        f"v4l2src device={device} io-mode=2 ! {caps} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera_opencv(source: object) -> Any:
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
    else:
        cap = cv2.VideoCapture(str(source), cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_camera_gstreamer(source: object) -> Any:
    if os.name == "nt":
        raise RuntimeError("GStreamer camera mode is Linux-only")
    return cv2.VideoCapture(build_gstreamer_pipeline(source), cv2.CAP_GSTREAMER)


def camera_open_modes() -> list[str]:
    if CAMERA_OPEN_MODE == "auto":
        return ["gstreamer", "opencv"] if os.name != "nt" else ["opencv"]
    if CAMERA_OPEN_MODE == "gstreamer":
        return ["gstreamer"]
    return ["opencv"]


def resize_capture_frame(frame: Any) -> Any:
    if cv2 is None or frame is None:
        return frame
    height, width = frame.shape[:2]
    if width == FRAME_WIDTH and height == FRAME_HEIGHT:
        return frame
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def open_camera() -> Any:
    global active_camera_device, active_camera_frame_shape, active_camera_open_mode
    global active_camera_actual_width, active_camera_actual_height, active_camera_actual_fps
    global camera_open_attempts
    if cv2 is None:
        raise RuntimeError(f"cv2 unavailable: {CV2_IMPORT_ERROR}")
    candidates = build_camera_candidates(CAMERA_DEVICE)

    seen: set[str] = set()
    tried: list[str] = []
    failures: list[str] = []
    active_camera_device = None
    active_camera_frame_shape = None
    active_camera_open_mode = None
    active_camera_actual_width = None
    active_camera_actual_height = None
    active_camera_actual_fps = None
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        tried.append(f"{key}({CAMERA_OPEN_MODE})")
        source = normalize_camera_source(candidate)
        if CAMERA_OPEN_MODE == "gstreamer":
            cap = open_camera_gstreamer(source)
            active_camera_open_mode = "gstreamer"
        elif isinstance(source, int):
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend)
            active_camera_open_mode = "opencv"
        else:
            cap = cv2.VideoCapture(str(source), cv2.CAP_ANY)
            active_camera_open_mode = "opencv"
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            failures.append(f"{key}: open failed")
            cap.release()
            continue
        read_ok = False
        frame_shape = None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                frame = resize_capture_frame(frame)
                read_ok = True
                frame_shape = tuple(frame.shape)
                break
            time.sleep(0.05)
        if read_ok:
            active_camera_device = key
            active_camera_frame_shape = frame_shape
            try:
                active_camera_actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (frame_shape[1] if frame_shape else None)
                active_camera_actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (frame_shape[0] if frame_shape else None)
                active_camera_actual_fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 2)
            except Exception:
                active_camera_actual_width = frame_shape[1] if frame_shape else None
                active_camera_actual_height = frame_shape[0] if frame_shape else None
                active_camera_actual_fps = None
            camera_open_attempts = list(tried)
            print(f"摄像头已打开并读到帧: {candidate} shape={frame_shape}")
            return cap
        failures.append(f"{key}: opened but read_frame failed")
        cap.release()

    camera_open_attempts = list(tried)
    raise RuntimeError(
        f"无法打开并读取摄像头，已尝试: {', '.join(tried)}。"
        f"失败原因: {'; '.join(failures)}。"
        f"当前 RK_CAMERA_DEVICE={CAMERA_DEVICE}，可设置 RK_CAMERA_DEVICE=auto 或 /dev/v4l/by-id/... 指定稳定设备。"
    )


def reopen_camera_after_read_failures() -> bool:
    global cap, vision_boot_error
    if cv2 is None:
        return False
    old_cap = cap
    cap = None
    if old_cap is not None:
        old_cap.release()
    try:
        cap = open_camera()
    except Exception as exc:  # pragma: no cover - board dependent
        vision_boot_error = str(exc)
        state.update_status(f"摄像头重新打开失败：{exc}")
        return False
    vision_boot_error = None
    state.note_camera_reopen()
    state.update_status(f"摄像头已重新打开：{active_camera_device or CAMERA_DEVICE}")
    return True


class RecorderState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.running = True
        self.frame_id = 0
        self.jpg_bytes: bytes | None = None
        self.last_status = "等待首帧画面"
        self.frame_timestamps = deque(maxlen=60)
        self.pose_fps: float | None = None
        self.pose_backend_requested = "mediapipe"
        self.pose_backend_actual = "mediapipe"
        self.pose_backend_fallback_used = False
        self.pose_backend_error_message: str | None = None
        self.rknn_model_path: str | None = None
        self.rknn_pipeline: str | None = None
        self.pose_backend_message = "Using MediaPipe CPU pose backend."
        self.pose_perf: dict[str, object] = {}
        self.pose_worker_error: str | None = None
        self.pose_worker_error_count = 0
        self.last_pose_worker_error_at: str | None = None
        self.last_pose_worker_frame_at: float | None = None
        self.last_pose_worker_frame_time: str | None = None
        self.camera_read_failure_count = 0
        self.camera_consecutive_read_failures = 0
        self.camera_reopen_count = 0
        self.camera_read_ms: float | None = None
        self.camera_capture_fps: float | None = None
        self.camera_read_timestamps = deque(maxlen=60)
        self.frame_queue_drops = 0
        self.pose_loop_index = 0
        self.last_camera_read_success_at: str | None = None
        self.last_camera_read_failure_at: str | None = None
        self.pose_quality: dict[str, object] = {
            "quality_ok": False,
            "quality_message": "等待首帧画面",
            "missing_keypoints": [],
            "person_count": 0,
            "multi_person_warning": False,
            "selected_person_reason": None,
        }
        self.mediapipe_locked_side: str | None = None
        self.mediapipe_locked_action_id: str | None = None

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
        self.rom_min_angle: float | None = None
        self.rom_max_angle: float | None = None

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
        self.rom_min_angle = None
        self.rom_max_angle = None
        self.start_time = None
        self.is_recording = False
        self.smoother.clear()

    def reset_mediapipe_side_lock(self) -> None:
        self.mediapipe_locked_side = None
        self.mediapipe_locked_action_id = None

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
            self.last_pose_worker_frame_at = now
            self.last_pose_worker_frame_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.jpg_bytes = jpg_bytes
            self.last_status = status
            self.condition.notify_all()

    def update_preview_frame(self, jpg_bytes: bytes) -> None:
        with self.condition:
            self.frame_id += 1
            self.jpg_bytes = jpg_bytes
            self.condition.notify_all()

    def update_status(self, status: str) -> None:
        with self.condition:
            self.last_status = status
            self.condition.notify_all()

    def note_camera_read_success(self, read_ms: float | None = None) -> None:
        with self.condition:
            self.camera_consecutive_read_failures = 0
            if read_ms is not None:
                self.camera_read_ms = read_ms
            now = time.time()
            self.camera_read_timestamps.append(now)
            if len(self.camera_read_timestamps) >= 2:
                elapsed = self.camera_read_timestamps[-1] - self.camera_read_timestamps[0]
                if elapsed > 1e-6:
                    self.camera_capture_fps = (len(self.camera_read_timestamps) - 1) / elapsed
            self.last_camera_read_success_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def note_frame_queue_drop(self) -> None:
        with self.condition:
            self.frame_queue_drops += 1

    def note_camera_read_failure(self, consecutive_read_failures: int) -> None:
        with self.condition:
            self.camera_read_failure_count += 1
            self.camera_consecutive_read_failures = consecutive_read_failures
            self.last_camera_read_failure_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def note_camera_reopen(self) -> None:
        with self.condition:
            self.camera_reopen_count += 1

    def note_pose_worker_error(self, exc: Exception) -> None:
        with self.condition:
            self.pose_worker_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.pose_worker_error_count += 1
            self.last_pose_worker_error_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_status = f"识别线程单帧异常，正在恢复：{exc}"
            self.condition.notify_all()

    def clear_pose_worker_error(self) -> None:
        with self.condition:
            self.pose_worker_error = None

    def snapshot_status(self) -> dict[str, object]:
        smoothed = self.smoother.values[-1] if self.smoother.values else None
        action_id = normalize_action_id(self.action_name)
        active_templates = load_active_templates()
        active_template = get_active_template(action_id, self.pose_backend_actual)
        recent_reports = load_recent_reports()
        latest_report = load_report_context(self.last_evaluation_report_path) or (recent_reports[0] if recent_reports else None)
        camera_display_failures = self.camera_consecutive_read_failures if self.camera_consecutive_read_failures >= RK_CAMERA_FAIL_DISPLAY_THRESHOLD else 0
        pose_worker_idle_ms = None
        if self.last_pose_worker_frame_at is not None:
            pose_worker_idle_ms = max(0.0, (time.time() - self.last_pose_worker_frame_at) * 1000.0)
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
            "fixed_bbox": self.selected_result.get("fixed_bbox"),
            "fixed_bbox_requested": self.selected_result.get("fixed_bbox_requested"),
            "person_box_height_ratio": self.selected_result.get("person_box_height_ratio"),
            "person_box_area_ratio": self.selected_result.get("person_box_area_ratio"),
            "target_leg_visibility": self.selected_result.get("target_leg_visibility"),
            "target_side_keypoint_visibility": self.selected_result.get("target_side_keypoint_visibility"),
            "target_leg_visibility_min": self.selected_result.get("target_leg_visibility_min"),
            "target_leg_visibility_avg": self.selected_result.get("target_leg_visibility_avg"),
            "target_leg_visibility_ok": self.selected_result.get("target_leg_visibility_ok"),
            "current_rom": self.current_rom,
            "pending_export": self.last_export_payload is not None,
            "awaiting_ack": self.awaiting_ack,
            "last_export_error": self.last_export_error,
            "active_template": active_template,
            "patient_attempt_file": self.last_patient_attempt_path,
            "patient_attempt_summary": self.last_patient_attempt_summary,
            "evaluation_report_file": self.last_evaluation_report_path,
            "feedback": self.last_feedback,
            "latest_report": latest_report,
            "recent_reports": recent_reports,
            "active_templates": active_templates,
            "active_templates_by_backend": active_templates.get("by_backend", {}) if isinstance(active_templates, dict) else {},
            "active_template_backend": self.pose_backend_actual,
            "llm": get_llm_status(),
            "capabilities": capabilities_snapshot(),
            "stream_available": cv2 is not None and cap is not None and self.pose_backend_actual in {"mediapipe", "rknn"},
            "stream_ready": self.jpg_bytes is not None and self.frame_id > 0,
            "frame_id": self.frame_id,
            "vision_boot_error": vision_boot_error,
            "camera_device": CAMERA_DEVICE,
            "camera_device_requested": CAMERA_DEVICE,
            "camera_device_active": active_camera_device,
            "camera_open_mode_requested": CAMERA_OPEN_MODE,
            "camera_open_mode_active": active_camera_open_mode,
            "camera_width_requested": FRAME_WIDTH,
            "camera_height_requested": FRAME_HEIGHT,
            "camera_frame_shape": active_camera_frame_shape,
            "camera_actual_width": active_camera_actual_width,
            "camera_actual_height": active_camera_actual_height,
            "camera_actual_fps": active_camera_actual_fps,
            "camera_open_attempts": camera_open_attempts,
            "camera_read_ms": round(self.camera_read_ms, 2) if self.camera_read_ms is not None else None,
            "camera_capture_fps": round(self.camera_capture_fps, 2) if self.camera_capture_fps is not None else None,
            "frame_queue_drops": self.frame_queue_drops,
            "camera_read_failure_count": self.camera_read_failure_count,
            "camera_consecutive_read_failures": self.camera_consecutive_read_failures,
            "camera_display_failures": camera_display_failures,
            "camera_fail_display_threshold": RK_CAMERA_FAIL_DISPLAY_THRESHOLD,
            "camera_status": "fail" if camera_display_failures > 0 else "ok",
            "camera_reopen_count": self.camera_reopen_count,
            "last_camera_read_success_at": self.last_camera_read_success_at,
            "last_camera_read_failure_at": self.last_camera_read_failure_at,
            "training": realtime_session.snapshot(),
            "pose_fps": round(self.pose_fps, 2) if self.pose_fps is not None else None,
            "requested_backend": self.pose_backend_requested,
            "actual_backend": self.pose_backend_actual,
            "pose_backend": self.pose_backend_actual,
            "fallback_used": self.pose_backend_fallback_used,
            "backend_error_message": self.pose_backend_error_message,
            "pose_backend_message": self.pose_backend_message,
            "pose_backend_version": POSE_BACKEND_VERSION.get(self.pose_backend_actual),
            "pose_keypoint_schema": POSE_KEYPOINT_SCHEMA.get(self.pose_backend_actual),
            "rknn_model_path": self.rknn_model_path,
            "rknn_pipeline": self.rknn_pipeline,
            "allow_pose_backend_mismatch": ALLOW_POSE_BACKEND_MISMATCH,
            "pose_performance": self.pose_perf,
            "pose_worker_error": self.pose_worker_error,
            "pose_worker_error_count": self.pose_worker_error_count,
            "last_pose_worker_error_at": self.last_pose_worker_error_at,
            "last_pose_worker_frame_at": self.last_pose_worker_frame_time,
            "pose_worker_idle_ms": round(pose_worker_idle_ms, 2) if pose_worker_idle_ms is not None else None,
            "pose_quality": self.pose_quality,
            "quality_ok": self.pose_quality.get("quality_ok"),
            "quality_message": self.pose_quality.get("quality_message"),
            "missing_keypoints": self.pose_quality.get("missing_keypoints"),
            "person_count": self.pose_quality.get("person_count"),
            "multi_person_warning": self.pose_quality.get("multi_person_warning"),
            "selected_person_reason": self.pose_quality.get("selected_person_reason"),
            "status": self.last_status,
        }


state = RecorderState()
realtime_session = RealtimeTrainingSession()
llm_tts_worker = TTSWorker(global_cooldown=1.0, same_text_cooldown=5.0, use_real_tts=True)
llm_tts_started = False
llm_tts_lock = threading.Lock()
cap = None
mp_pose = None
mp_drawing = None
rknn_backend = None
rknn_person_selector = StablePersonSelector() if StablePersonSelector is not None else None


def make_rknn_pose_stabilizer():
    if RknnPoseStabilizer is None:
        return None
    config = backend_calibration("rknn").get("stabilizer", {})
    if not isinstance(config, dict):
        config = {}

    def _float(name: str, default: float) -> float:
        env_name = f"RKNN_STABILIZER_{name.upper()}"
        try:
            return float(os.environ.get(env_name, config.get(name, default)))
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int) -> int:
        env_name = f"RKNN_STABILIZER_{name.upper()}"
        try:
            return int(os.environ.get(env_name, config.get(name, default)))
        except (TypeError, ValueError):
            return default

    return RknnPoseStabilizer(
        alpha=_float("alpha", 0.35),
        low_conf_alpha=_float("low_conf_alpha", 0.18),
        jump_scale=_float("jump_scale", 0.35),
        max_hold_frames=_int("max_hold_frames", 8),
        lock_confirm_frames=_int("lock_confirm_frames", 4),
    )


rknn_pose_stabilizer = make_rknn_pose_stabilizer()
frame_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
pose_backend_selection: PoseBackendSelection | None = None
vision_boot_error: str | None = None
active_camera_device: str | None = None
active_camera_frame_shape: tuple[int, ...] | None = None
active_camera_open_mode: str | None = None
active_camera_actual_width: int | None = None
active_camera_actual_height: int | None = None
active_camera_actual_fps: float | None = None
camera_open_attempts: list[str] = []


def probe_rknn_backend() -> None:
    global rknn_backend
    if RKNNPoseBackend is None:
        raise RuntimeError(f"RKNN backend import failed: {RKNN_IMPORT_ERROR}")
    if rknn_backend is None:
        keypoint_thres = None if "RKNN_POSE_KEYPOINT_THRES" in os.environ else backend_visibility_threshold("rknn")
        backend = RKNNPoseBackend(keypoint_thres=keypoint_thres)
        for model_path_text in str(backend.model_path).split(";"):
            model_path = Path(model_path_text)
            if not model_path.exists():
                raise FileNotFoundError(f"RKNN model not found: {model_path}")
        try:
            from rknnlite.api import RKNNLite  # noqa: F401
        except Exception as exc:  # pragma: no cover - board only
            raise RuntimeError(f"RKNNLite import failed: {exc}") from exc
        rknn_backend = backend


try:
    pose_backend_selection = resolve_pose_backend(probe_rknn_backend)
except Exception as exc:
    raise RuntimeError(f"RKNN pose backend initialization failed: {exc}") from exc

state.pose_backend_requested = pose_backend_selection.requested_backend
state.pose_backend_actual = pose_backend_selection.actual_backend
state.pose_backend_fallback_used = pose_backend_selection.fallback_used
state.pose_backend_error_message = pose_backend_selection.backend_error_message
state.pose_backend_message = pose_backend_selection.message
if rknn_backend is not None:
    state.rknn_model_path = rknn_backend.model_path
    state.rknn_pipeline = getattr(rknn_backend, "pipeline", None)

vision_requires_mediapipe = pose_backend_selection.actual_backend == "mediapipe"
if cv2 is not None and (not vision_requires_mediapipe or mp is not None):
    try:
        cap = open_camera()
        if vision_requires_mediapipe:
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
    except Exception as exc:  # pragma: no cover - environment dependent
        vision_boot_error = str(exc)
else:
    vision_boot_error = CV2_IMPORT_ERROR or (MEDIAPIPE_IMPORT_ERROR if vision_requires_mediapipe else None)


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
              <option value="seated_knee_raise">坐姿抬膝</option>
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
              <option value="seated_knee_raise">坐姿抬膝</option>
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


def build_home_html() -> str:
    return build_app_shell("home", "RK3588 康复训练台", "RK3588 orthopedic rehab station")


def build_train_page_html() -> str:
    return build_app_shell("train", "患者训练驾驶舱", "Realtime rehab training cockpit")


def build_ai_page_html() -> str:
    return build_app_shell("ai", "AI 康复复盘", "Post-training AI rehab review")


def build_page_html() -> str:
    return build_app_shell("doctor", "医生动作录制台", "Doctor recording and evaluation workspace")


def camera_capture_worker() -> None:
    if cv2 is None or cap is None:
        state.last_status = "视觉链路未就绪"
        return
    consecutive_read_failures = 0
    last_rknn_preview_at = 0.0
    while state.running:
        current_cap = cap
        if current_cap is None:
            time.sleep(0.2)
            continue
        read_start = time.perf_counter()
        success, frame = current_cap.read()
        read_ms = (time.perf_counter() - read_start) * 1000.0
        if not success:
            consecutive_read_failures += 1
            state.note_camera_read_failure(consecutive_read_failures)
            if consecutive_read_failures == 1:
                state.update_status("摄像头已打开，等待首帧画面")
            elif consecutive_read_failures % 30 == 0:
                state.update_status(f"摄像头已打开但读取帧失败，连续失败 {consecutive_read_failures} 次")
            if consecutive_read_failures >= 60 and consecutive_read_failures % 60 == 0:
                reopen_camera_after_read_failures()
            time.sleep(0.1)
            continue
        consecutive_read_failures = 0
        state.note_camera_read_success(read_ms)
        frame = resize_capture_frame(frame)
        frame = cv2.flip(frame, 1)
        if state.pose_backend_actual == "rknn" and RKNN_PREVIEW_FPS > 0:
            now = time.time()
            last_pose_at = state.frame_timestamps[-1] if state.frame_timestamps else 0.0
            if now - last_rknn_preview_at >= 1.0 / RKNN_PREVIEW_FPS and now - last_pose_at > 0.25:
                preview = resize_rknn_stream_frame(frame)
                ok, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    state.update_preview_frame(jpg.tobytes())
                    last_rknn_preview_at = now
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            state.note_frame_queue_drop()
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put_nowait(frame)


def pose_worker() -> None:
    if cv2 is None or cap is None:
        state.last_status = "视觉链路未就绪"
        while state.running:
            time.sleep(0.5)
        return
    if state.pose_backend_actual == "mediapipe" and (mp_pose is None or mp_drawing is None or landmark_pb2 is None):
        state.last_status = "MediaPipe 视觉链路未就绪"
        while state.running:
            time.sleep(0.5)
        return
    if state.pose_backend_actual == "rknn" and (rknn_backend is None or rknn_person_selector is None):
        state.last_status = "RKNN 视觉链路未就绪"
        while state.running:
            time.sleep(0.5)
        return

    pose = None
    if state.pose_backend_actual == "mediapipe":
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
            queue_wait_start = time.perf_counter()
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            queue_wait_ms = (time.perf_counter() - queue_wait_start) * 1000.0
            pose_loop_start = time.perf_counter()
            state.pose_loop_index += 1
            pose_loop_index = state.pose_loop_index
            pose_process_start = time.perf_counter()
            if state.pose_backend_actual == "rknn":
                try:
                    output_frame, selected_rule, selected_result, current_frame_data, realtime_frame_data = process_rknn_frame(frame)
                except Exception as exc:
                    if state.pose_backend_requested == "auto" and mp_pose is not None and mp_drawing is not None and landmark_pb2 is not None:
                        state.pose_backend_actual = "mediapipe"
                        state.pose_backend_fallback_used = True
                        state.pose_backend_error_message = str(exc)
                        state.pose_backend_message = f"RKNN lazy load/inference failed; fallback to MediaPipe: {exc}"
                        pose = mp_pose.Pose(
                            static_image_mode=False,
                            model_complexity=MODEL_COMPLEXITY,
                            smooth_landmarks=True,
                            enable_segmentation=False,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6,
                        )
                        output_frame, selected_rule, selected_result, current_frame_data, realtime_frame_data = process_mediapipe_frame(frame, pose)
                    else:
                        state.pose_backend_error_message = str(exc)
                        state.pose_backend_message = f"RKNN lazy load/inference failed: {exc}"
                        state.update_status(f"RKNN 初始化或推理失败：{exc}")
                        time.sleep(0.5)
                        continue
            else:
                output_frame, selected_rule, selected_result, current_frame_data, realtime_frame_data = process_mediapipe_frame(frame, pose)
            pose_process_ms = (time.perf_counter() - pose_process_start) * 1000.0

            if state.is_recording and state.selected_rule_at_recording is None:
                state.selected_rule_at_recording = selected_rule
            if current_frame_data is not None and state.is_recording:
                state.frames.append(current_frame_data)
                state.frame_index += 1
                frame_angle = current_frame_data.get("target_angle_smoothed")
                if frame_angle is not None:
                    try:
                        frame_angle_value = float(frame_angle)
                        state.rom_min_angle = frame_angle_value if state.rom_min_angle is None else min(state.rom_min_angle, frame_angle_value)
                        state.rom_max_angle = frame_angle_value if state.rom_max_angle is None else max(state.rom_max_angle, frame_angle_value)
                    except (TypeError, ValueError):
                        pass
            elif current_frame_data is None and state.is_recording:
                state.invalid_frame_count += 1

            keyframe_jpeg = None
            keyframe_encode_ms = 0.0
            keyframe_skipped = False
            training_status = str(realtime_session.snapshot().get("status") or "")
            if training_status == "running" and realtime_frame_data.get("target_angle_smoothed") is not None:
                if pose_loop_index % REHAB_KEYFRAME_EVERY_N == 0:
                    keyframe_start = time.perf_counter()
                    keyframe_jpeg = encode_keyframe_candidate(output_frame)
                    keyframe_encode_ms = (time.perf_counter() - keyframe_start) * 1000.0
                else:
                    keyframe_skipped = True
            realtime_process_start = time.perf_counter()
            try:
                realtime_session.process_frame(realtime_frame_data, selected_rule, keyframe_jpeg=keyframe_jpeg)
                state.clear_pose_worker_error()
            except Exception as exc:
                state.note_pose_worker_error(exc)
                print(f"[pose_worker] realtime process_frame failed: {exc}")
                continue
            realtime_process_ms = (time.perf_counter() - realtime_process_start) * 1000.0
            state_update_start = time.perf_counter()
            if state.rom_min_angle is not None and state.rom_max_angle is not None:
                state.current_rom = state.rom_max_angle - state.rom_min_angle
            else:
                state.current_rom = None
            state.selected_result = selected_result
            state.selected_rule = selected_rule
            state.pose_quality = build_pose_quality(selected_result)
            pose_perf = dict(selected_result.get("performance_ms") or {})
            pose_perf["camera_read_ms"] = round(state.camera_read_ms, 2) if state.camera_read_ms is not None else None
            pose_perf["camera_capture_fps"] = round(state.camera_capture_fps, 2) if state.camera_capture_fps is not None else None
            pose_perf["frame_queue_drops"] = state.frame_queue_drops
            pose_perf["queue_wait_ms"] = round(queue_wait_ms, 2)
            pose_perf["pose_process_ms"] = round(pose_process_ms, 2)
            pose_perf["keyframe_encode_ms"] = round(keyframe_encode_ms, 2)
            pose_perf["keyframe_every_n"] = REHAB_KEYFRAME_EVERY_N
            pose_perf["keyframe_skipped"] = keyframe_skipped
            pose_perf["realtime_process_ms"] = round(realtime_process_ms, 2)

            if selected_result.get("valid", False):
                side = str(selected_result.get("side", ""))
                status = f"已检测到{SIDE_MODE_LABELS.get(side, side)}"
            elif selected_result.get("quality_message"):
                status = str(selected_result.get("quality_message"))
            elif state.awaiting_ack and state.last_export_payload is not None:
                status = "等待保存确认"
            elif state.is_recording:
                status = "录制中"
            else:
                status = "未检测到可靠关键点"
            pose_perf["state_update_ms"] = round((time.perf_counter() - state_update_start) * 1000.0, 2)

            if state.pose_backend_actual == "rknn":
                resize_start = time.perf_counter()
                output_frame = resize_rknn_stream_frame(output_frame)
                pose_perf["stream_resize_ms"] = round((time.perf_counter() - resize_start) * 1000.0, 2)
            else:
                pose_perf["stream_resize_ms"] = 0.0
            jpeg_start = time.perf_counter()
            ok, jpg = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            pose_perf["jpeg_encode_ms"] = round((time.perf_counter() - jpeg_start) * 1000.0, 2)
            pose_perf["pose_loop_ms"] = round((time.perf_counter() - pose_loop_start) * 1000.0, 2)
            state.pose_perf = pose_perf
            if ok:
                state.update_frame(jpg.tobytes(), status)
    finally:
        if pose is not None:
            pose.close()


def resize_rknn_stream_frame(frame):
    if RKNN_STREAM_WIDTH <= 0 or RKNN_STREAM_HEIGHT <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= RKNN_STREAM_WIDTH and height <= RKNN_STREAM_HEIGHT:
        return frame
    return cv2.resize(frame, (RKNN_STREAM_WIDTH, RKNN_STREAM_HEIGHT), interpolation=cv2.INTER_AREA)


def encode_keyframe_candidate(frame) -> bytes | None:
    if cv2 is None or frame is None:
        return None
    try:
        image = frame
        height, width = image.shape[:2]
        if width > KEYFRAME_MAX_WIDTH:
            scale = KEYFRAME_MAX_WIDTH / float(width)
            image = cv2.resize(image, (KEYFRAME_MAX_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, KEYFRAME_JPEG_QUALITY])
        return jpg.tobytes() if ok else None
    except Exception:
        return None


def current_action_context() -> tuple[str, dict[str, object]]:
    training_snapshot = realtime_session.snapshot()
    if training_snapshot.get("status") in ACTIVE_REALTIME_STATUSES and training_snapshot.get("action_id"):
        action_id = normalize_action_id(training_snapshot.get("action_id"))
    else:
        action_id = normalize_action_id(state.action_name)
    return action_id, get_action_config(action_id)


def current_action_id_fast() -> str:
    status = getattr(realtime_session, "status", None)
    action_id = getattr(realtime_session, "action_id", None)
    if status in ACTIVE_REALTIME_STATUSES and action_id:
        return normalize_action_id(action_id)
    return normalize_action_id(state.action_name)


def process_mediapipe_frame(frame, pose) -> tuple[object, dict[str, object], dict[str, object], dict[str, object] | None, dict[str, object]]:
    infer_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
    rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    output_frame = frame.copy()
    selected_result: dict[str, object] = {"valid": False, "quality_ok": False, "quality_message": "未检测到可靠关键点"}
    selected_rule = LEFT_KNEE_RULE
    keypoints: dict[str, object] = {}
    rehab_keypoints: dict[str, object] = {}

    if result.pose_landmarks:
        output_landmarks = landmark_pb2.NormalizedLandmarkList()
        for landmark in result.pose_landmarks.landmark:
            output_landmarks.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility)
        mp_drawing.draw_landmarks(output_frame, output_landmarks, mp_pose.POSE_CONNECTIONS)
        action_id, action_config = current_action_context()
        action_rules = action_config.get("rules", {})
        left_rule = dict(action_rules.get("left", LEFT_KNEE_RULE))
        right_rule = dict(action_rules.get("right", RIGHT_KNEE_RULE))
        left_result = compute_action_angle(result, left_rule, action_config)
        right_result = compute_action_angle(result, right_rule, action_config)
        if state.mediapipe_locked_action_id != action_id:
            state.reset_mediapipe_side_lock()
        locked_side = state.mediapipe_locked_side if state.mediapipe_locked_action_id == action_id else None
        preferred_side = preferred_side_for_action(action_id) if state.side_mode == "auto" else None
        selected_rule, selected_result = choose_action_rule(
            state.side_mode,
            action_config,
            left_result,
            right_result,
            locked_side=locked_side,
            preferred_side=preferred_side,
        )
        selected_side = str(selected_rule.get("side") or "")
        if state.side_mode in {"left", "right"}:
            state.mediapipe_locked_side = state.side_mode
            state.mediapipe_locked_action_id = action_id
        elif selected_side in {"left", "right"} and (preferred_side or selected_result.get("valid")):
            state.mediapipe_locked_side = selected_side
            state.mediapipe_locked_action_id = action_id
            selected_result = {**selected_result, "locked_side": selected_side, "side_lock_reason": selected_result.get("side_lock_reason") or "first_stable_side"}
        selected_result = {**selected_result, "quality_ok": bool(selected_result.get("valid")), "quality_message": "关键点质量正常" if selected_result.get("valid") else "关键点置信度不足"}
        keypoints = build_compact_keypoints(result.pose_landmarks.landmark, selected_rule)
        rehab_keypoints = build_rehab_keypoints(result.pose_landmarks.landmark)
        output_frame = draw_rehab_skeleton_overlay(
            output_frame,
            rehab_keypoints,
            selected_rule,
            action_config,
            current_visibility_threshold(),
        )

    current_frame_data, realtime_frame_data = build_frame_data_pair(
        selected_result,
        selected_rule,
        keypoints,
        rehab_keypoints,
    )
    return output_frame, selected_rule, selected_result, current_frame_data, realtime_frame_data


def process_rknn_frame(frame) -> tuple[object, dict[str, object], dict[str, object], dict[str, object] | None, dict[str, object]]:
    assert rknn_backend is not None and rknn_person_selector is not None
    infer_call_start = time.perf_counter()
    result = rknn_backend.infer(frame)
    infer_call_ms = (time.perf_counter() - infer_call_start) * 1000.0
    detections = list(result.meta.get("detections") or [])
    action_context_start = time.perf_counter()
    action_id = current_action_id_fast()
    action_config = get_action_config(action_id)
    action_context_ms = (time.perf_counter() - action_context_start) * 1000.0
    height, width = frame.shape[:2]
    adapt_start = time.perf_counter()
    adapted = adapt_rknn_pose_frame(
        detections,
        frame_width=width,
        frame_height=height,
        action_config=action_config,
        side_mode=state.side_mode,
        selector=rknn_person_selector,
        visibility_threshold=rknn_backend.keypoint_thres,
        stabilizer=rknn_pose_stabilizer,
    )
    adapt_ms = (time.perf_counter() - adapt_start) * 1000.0
    selected_result = dict(adapted["selected_result"])
    selected_result["performance_ms"] = dict(result.meta.get("performance_ms") or {})
    selected_result["performance_ms"]["rknn_infer_call_ms"] = round(infer_call_ms, 2)
    selected_result["performance_ms"]["rknn_action_context_ms"] = round(action_context_ms, 2)
    selected_result["performance_ms"]["rknn_adapt_ms"] = round(adapt_ms, 2)
    rknn_pipeline = result.meta.get("rknn_pipeline")
    fast_frame_data = bool(RKNN_FAST_FRAME_DATA and rknn_pipeline == "rtmpose_fixed")
    selected_result["pose_backend"] = "rknn"
    selected_result["rknn_detection_count"] = len(detections)
    selected_result["rknn_pipeline"] = rknn_pipeline
    selected_result["rknn_keypoint_threshold"] = float(rknn_backend.keypoint_thres)
    selected_result["rknn_decoder"] = result.meta.get("rknn_decoder")
    selected_result["rknn_fast_frame_data"] = fast_frame_data
    selected_result["fixed_bbox"] = result.meta.get("fixed_bbox")
    selected_result["fixed_bbox_requested"] = result.meta.get("fixed_bbox_requested")
    selected_result["fixed_bbox_mode"] = result.meta.get("fixed_bbox_mode")
    selected_result["rtmpose_draw_enabled"] = result.meta.get("rtmpose_draw_enabled")
    selected_result["rknn_fixed_strict_leg_visibility"] = RKNN_FIXED_STRICT_LEG_VISIBILITY if rknn_pipeline == "rtmpose_fixed" else None
    selected_result["rknn_fixed_leg_visibility_threshold"] = RKNN_FIXED_LEG_VISIBILITY_THRESHOLD if rknn_pipeline == "rtmpose_fixed" else None
    selected_result["rknn_fixed_draw_visibility_threshold"] = RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD if rknn_pipeline == "rtmpose_fixed" else None
    if not fast_frame_data:
        selected_result["rknn_output_shapes"] = result.meta.get("output_shapes")
        selected_result["rknn_det_output_shapes"] = result.meta.get("det_output_shapes")
        selected_result["rknn_pose_output_shapes"] = result.meta.get("pose_output_shapes")
        selected_result["rknn_det_cache_hit"] = result.meta.get("det_cache_hit")
        selected_result["rknn_det_interval"] = result.meta.get("det_interval")
        selected_result["rknn_pose_cache_hit"] = result.meta.get("pose_cache_hit")
        selected_result["rknn_pose_reused"] = result.meta.get("pose_reused")
        selected_result["rknn_pose_interval"] = result.meta.get("pose_interval")
        selected_result["rknn_person_select"] = result.meta.get("person_select")
        selected_result["keypoint_conf_range"] = result.meta.get("keypoint_conf_range")
        selected_result["keypoint_xy_range"] = result.meta.get("keypoint_xy_range")
        selected_result["keypoint_decode_mode"] = result.meta.get("keypoint_decode_mode")
        selected_result["keypoint_decode_override"] = result.meta.get("keypoint_decode_override")
        selected_result["keypoint_anchor_order"] = result.meta.get("keypoint_anchor_order")
        selected_result["keypoint_anchor_order_setting"] = result.meta.get("keypoint_anchor_order_setting")
        selected_result["keypoint_geometry_score_range"] = result.meta.get("keypoint_geometry_score_range")
        selected_result["keypoint_global_index_range"] = result.meta.get("keypoint_global_index_range")
        selected_result["keypoint_candidate_count_range"] = result.meta.get("keypoint_candidate_count_range")
        selected_result["keypoint_branch_diagnostics"] = result.meta.get("keypoint_branch_diagnostics")
        selected_result["keypoint_raw_shape"] = result.meta.get("keypoint_raw_shape")
        selected_result["keypoint_raw_xy_range"] = result.meta.get("keypoint_raw_xy_range")
        selected_result["keypoint_restored_xy_range"] = result.meta.get("keypoint_restored_xy_range")
    selected_result.setdefault("locked_side", adapted.get("selected_result", {}).get("locked_side"))
    selected_result.setdefault("pose_stabilized", adapted.get("selected_result", {}).get("pose_stabilized"))
    selected_result.setdefault("held_keypoints", adapted.get("selected_result", {}).get("held_keypoints") or [])
    selected_result.setdefault("jump_rejected", adapted.get("selected_result", {}).get("jump_rejected") or [])
    selected_result.setdefault("side_switch_blocked", adapted.get("selected_result", {}).get("side_switch_blocked"))
    selected_result.setdefault("side_lock_reason", adapted.get("selected_result", {}).get("side_lock_reason"))
    postprocess_error = result.meta.get("postprocess_error")
    selected_result["postprocess_error"] = postprocess_error
    if postprocess_error:
        selected_result.update(
            {
                "valid": False,
                "quality_ok": False,
                "quality_message": f"RKNN 后处理失败：{postprocess_error}",
                "missing_keypoints": [],
            }
        )
    selected_rule = dict(adapted["selected_rule"])
    keypoint_copy_start = time.perf_counter()
    if fast_frame_data:
        keypoints = adapted.get("keypoints") or {}
        rehab_keypoints = adapted.get("rehab_keypoints") or {}
    else:
        keypoints = dict(adapted.get("keypoints") or {})
        rehab_keypoints = dict(adapted.get("rehab_keypoints") or {})
    selected_result["performance_ms"]["rknn_keypoint_copy_ms"] = round((time.perf_counter() - keypoint_copy_start) * 1000.0, 2)
    if rknn_pipeline == "rtmpose_fixed":
        visibility_guard_start = time.perf_counter()
        if postprocess_error:
            selected_result.update(target_leg_visibility(rehab_keypoints, selected_rule, RKNN_FIXED_LEG_VISIBILITY_THRESHOLD))
        else:
            apply_rtmpose_fixed_visibility_guard(selected_result, selected_rule, rehab_keypoints)
        selected_result["performance_ms"]["rknn_fixed_visibility_guard_ms"] = round((time.perf_counter() - visibility_guard_start) * 1000.0, 2)
    frame_data_start = time.perf_counter()
    if fast_frame_data:
        current_frame_data, realtime_frame_data, frame_data_timings = build_fast_rknn_frame_data_pair(
            selected_result,
            selected_rule,
            keypoints,
            rehab_keypoints,
            action_id=action_id,
        )
        selected_result["performance_ms"].update(frame_data_timings)
    else:
        current_frame_data, realtime_frame_data = build_frame_data_pair(
            selected_result,
            selected_rule,
            keypoints,
            rehab_keypoints,
            fast_rknn_diagnostics=fast_frame_data,
        )
    selected_result["performance_ms"]["rknn_frame_data_ms"] = round((time.perf_counter() - frame_data_start) * 1000.0, 2)
    if selected_result.get("rknn_pose_reused"):
        current_frame_data = None
        realtime_frame_data["target_angle_smoothed"] = None
        realtime_frame_data["selected_flexion_angle_smoothed"] = None
        realtime_frame_data["action_keypoints_valid"] = False
        realtime_frame_data["rknn_pose_reused"] = True
        realtime_frame_data["quality_message"] = "RKNN reused pose preview frame"
    draw_start = time.perf_counter()
    output_frame = frame.copy()
    if rknn_pipeline == "rtmpose_fixed" and RKNN_DRAW_FIXED_BBOX:
        draw_fixed_bbox_overlay(output_frame, selected_result.get("fixed_bbox"))
    rknn_draw_visibility_threshold = None
    if rknn_pipeline == "rtmpose_fixed":
        rknn_draw_visibility_threshold = max(float(current_visibility_threshold()), RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD)
    output_frame = draw_rehab_skeleton_overlay(
        output_frame,
        rehab_keypoints,
        selected_rule,
        action_config,
        current_visibility_threshold(),
        draw_visibility_threshold=rknn_draw_visibility_threshold,
    )
    if not RKNN_FAST_PREVIEW:
        output_frame = draw_rknn_debug_overlay(
            output_frame,
            adapted.get("selected_detection") if isinstance(adapted.get("selected_detection"), dict) else None,
            selected_result,
            realtime_frame_data,
            detection_count=len(detections),
            keypoint_threshold=float(rknn_backend.keypoint_thres),
            postprocess_error=postprocess_error,
        )
    if postprocess_error:
        cv2.putText(output_frame, "RKNN postprocess failed", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    selected_result["performance_ms"]["rknn_draw_ms"] = round((time.perf_counter() - draw_start) * 1000.0, 2)
    return output_frame, selected_rule, selected_result, current_frame_data, realtime_frame_data


def rknn_diagnostic_fields(selected_result: dict[str, object]) -> dict[str, object]:
    return {
        "rknn_output_shapes": selected_result.get("rknn_output_shapes"),
        "rknn_det_output_shapes": selected_result.get("rknn_det_output_shapes"),
        "rknn_pose_output_shapes": selected_result.get("rknn_pose_output_shapes"),
        "rknn_det_cache_hit": selected_result.get("rknn_det_cache_hit"),
        "rknn_det_interval": selected_result.get("rknn_det_interval"),
        "rknn_pose_cache_hit": selected_result.get("rknn_pose_cache_hit"),
        "rknn_pose_reused": selected_result.get("rknn_pose_reused"),
        "rknn_pose_interval": selected_result.get("rknn_pose_interval"),
        "rknn_person_select": selected_result.get("rknn_person_select"),
        "keypoint_conf_range": selected_result.get("keypoint_conf_range"),
        "keypoint_xy_range": selected_result.get("keypoint_xy_range"),
        "keypoint_decode_mode": selected_result.get("keypoint_decode_mode"),
        "keypoint_decode_override": selected_result.get("keypoint_decode_override"),
        "keypoint_anchor_order": selected_result.get("keypoint_anchor_order"),
        "keypoint_anchor_order_setting": selected_result.get("keypoint_anchor_order_setting"),
        "keypoint_geometry_score_range": selected_result.get("keypoint_geometry_score_range"),
        "keypoint_global_index_range": selected_result.get("keypoint_global_index_range"),
        "keypoint_candidate_count_range": selected_result.get("keypoint_candidate_count_range"),
        "keypoint_branch_diagnostics": selected_result.get("keypoint_branch_diagnostics"),
        "keypoint_raw_shape": selected_result.get("keypoint_raw_shape"),
        "keypoint_raw_xy_range": selected_result.get("keypoint_raw_xy_range"),
        "keypoint_restored_xy_range": selected_result.get("keypoint_restored_xy_range"),
    }


def build_fast_rknn_frame_data_pair(
    selected_result: dict[str, object],
    selected_rule: dict[str, object],
    keypoints: dict[str, object],
    rehab_keypoints: dict[str, object],
    *,
    action_id: str,
) -> tuple[dict[str, object] | None, dict[str, object], dict[str, object]]:
    prelude_start = time.perf_counter()
    timings: dict[str, object] = {}
    angle_smooth_start = time.perf_counter()
    raw_flexion = selected_result.get("selected_flexion_angle")
    if raw_flexion is not None:
        try:
            raw_flexion = float(raw_flexion) + backend_angle_offset(action_id, state.pose_backend_actual)
        except (TypeError, ValueError):
            raw_flexion = selected_result.get("selected_flexion_angle")
    state.smoother.set_window_size(backend_smoothing_window(state.pose_backend_actual))
    smoothed_flexion = state.smoother.update(raw_flexion)
    timings["rknn_angle_smooth_ms"] = round((time.perf_counter() - angle_smooth_start) * 1000.0, 2)
    now = time.time()
    relative_time = (now - state.start_time) if state.start_time is not None else 0
    threshold_start = time.perf_counter()
    metric_visibility = current_action_metric_visibility_threshold()
    quality_visibility = current_visibility_threshold()
    timings["rknn_threshold_ms"] = round((time.perf_counter() - threshold_start) * 1000.0, 2)
    valid = bool(selected_result.get("valid", False))
    timings["rknn_frame_prelude_ms"] = round((time.perf_counter() - prelude_start) * 1000.0, 2)

    realtime_start = time.perf_counter()
    realtime_frame_data: dict[str, object] = {
        "frame_index": state.frame_index,
        "relative_time": relative_time,
        "selected_side": selected_rule.get("side"),
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
        "visibility_threshold": metric_visibility,
        "quality_visibility_threshold": quality_visibility,
        "quality_ok": selected_result.get("quality_ok"),
        "quality_message": selected_result.get("quality_message"),
        "missing_keypoints": selected_result.get("missing_keypoints") or [],
        "person_count": selected_result.get("person_count"),
        "multi_person_warning": selected_result.get("multi_person_warning"),
        "selected_person_reason": selected_result.get("selected_person_reason"),
        "person_box_quality_ok": selected_result.get("person_box_quality_ok"),
        "person_box_height_ratio": selected_result.get("person_box_height_ratio"),
        "person_box_area_ratio": selected_result.get("person_box_area_ratio"),
        "person_box_quality_message": selected_result.get("person_box_quality_message"),
        "fixed_bbox": selected_result.get("fixed_bbox"),
        "fixed_bbox_requested": selected_result.get("fixed_bbox_requested"),
        "fixed_bbox_mode": selected_result.get("fixed_bbox_mode"),
        "target_leg_visibility": selected_result.get("target_leg_visibility"),
        "target_side_keypoint_visibility": selected_result.get("target_side_keypoint_visibility"),
        "target_leg_visibility_min": selected_result.get("target_leg_visibility_min"),
        "target_leg_visibility_avg": selected_result.get("target_leg_visibility_avg"),
        "target_leg_visibility_threshold": selected_result.get("target_leg_visibility_threshold"),
        "target_leg_visibility_ok": selected_result.get("target_leg_visibility_ok"),
        "target_leg_missing_keypoints": selected_result.get("target_leg_missing_keypoints") or [],
        "locked_side": selected_result.get("locked_side"),
        "side_lock_reason": selected_result.get("side_lock_reason"),
        "pose_stabilized": selected_result.get("pose_stabilized"),
        "held_keypoints": selected_result.get("held_keypoints") or [],
        "jump_rejected": selected_result.get("jump_rejected") or [],
        "side_switch_blocked": selected_result.get("side_switch_blocked"),
        "rknn_detection_count": selected_result.get("rknn_detection_count"),
        "rknn_pipeline": selected_result.get("rknn_pipeline"),
        "rknn_keypoint_threshold": selected_result.get("rknn_keypoint_threshold"),
        "rknn_decoder": selected_result.get("rknn_decoder"),
        "postprocess_error": selected_result.get("postprocess_error"),
        "requested_backend": state.pose_backend_requested,
        "actual_backend": state.pose_backend_actual,
        "pose_backend": state.pose_backend_actual,
        "pose_backend_version": POSE_BACKEND_VERSION.get(state.pose_backend_actual),
        "pose_keypoint_schema": POSE_KEYPOINT_SCHEMA.get(state.pose_backend_actual),
        "rknn_model_path": state.rknn_model_path,
        "keypoints": keypoints,
        "rehab_keypoints": rehab_keypoints,
    }
    timings["rknn_realtime_frame_data_ms"] = round((time.perf_counter() - realtime_start) * 1000.0, 2)

    side_start = time.perf_counter()
    side_view = compute_side_view_metrics(
        rehab_keypoints,
        selected_result,
        quality_visibility,
        selected_rule,
        state.pose_backend_actual,
    )
    timings["rknn_side_view_ms"] = round((time.perf_counter() - side_start) * 1000.0, 2)
    person_visible = bool(side_view.get("pose_detected"))
    realtime_frame_data.update(side_view)
    realtime_frame_data["person_visible"] = person_visible
    realtime_frame_data["pose_detected"] = person_visible
    realtime_frame_data["action_keypoints_valid"] = valid

    current_start = time.perf_counter()
    current_frame_data: dict[str, object] | None = None
    if state.is_recording and valid:
        current_frame_data = dict(realtime_frame_data)
        current_frame_data["person_visible"] = True
        current_frame_data["pose_detected"] = True
        current_frame_data["action_keypoints_valid"] = True
    timings["rknn_current_frame_data_ms"] = round((time.perf_counter() - current_start) * 1000.0, 2)
    return current_frame_data, realtime_frame_data, timings


def build_frame_data_pair(
    selected_result: dict[str, object],
    selected_rule: dict[str, object],
    keypoints: dict[str, object],
    rehab_keypoints: dict[str, object],
    *,
    fast_rknn_diagnostics: bool = False,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    current_frame_data = build_current_frame_data(
        selected_result,
        selected_rule,
        keypoints,
        rehab_keypoints,
        fast_rknn_diagnostics=fast_rknn_diagnostics,
    )
    realtime_frame_data: dict[str, object] = dict(current_frame_data or {})
    if current_frame_data is None:
        now = time.time()
        realtime_frame_data.update(
            {
                "frame_index": state.frame_index,
                "relative_time": (now - state.start_time) if state.start_time is not None else 0,
                "selected_side": selected_rule.get("side"),
                "selected_source": selected_result.get("selected_source"),
                "visibility_min": selected_result.get("visibility_min"),
                "visibility_avg": selected_result.get("visibility_avg"),
                "visibility": selected_result.get("visibility_min"),
                "visibility_threshold": current_action_metric_visibility_threshold(),
                "quality_visibility_threshold": current_visibility_threshold(),
                "quality_ok": selected_result.get("quality_ok"),
                "quality_message": selected_result.get("quality_message"),
                "missing_keypoints": selected_result.get("missing_keypoints") or [],
                "person_count": selected_result.get("person_count"),
                "multi_person_warning": selected_result.get("multi_person_warning"),
                "selected_person_reason": selected_result.get("selected_person_reason"),
                "person_box_quality_ok": selected_result.get("person_box_quality_ok"),
                "person_box_height_ratio": selected_result.get("person_box_height_ratio"),
                "person_box_area_ratio": selected_result.get("person_box_area_ratio"),
                "person_box_quality_message": selected_result.get("person_box_quality_message"),
                "fixed_bbox": selected_result.get("fixed_bbox"),
                "fixed_bbox_requested": selected_result.get("fixed_bbox_requested"),
                "fixed_bbox_mode": selected_result.get("fixed_bbox_mode"),
                "target_leg_visibility": selected_result.get("target_leg_visibility"),
                "target_side_keypoint_visibility": selected_result.get("target_side_keypoint_visibility"),
                "target_leg_visibility_min": selected_result.get("target_leg_visibility_min"),
                "target_leg_visibility_avg": selected_result.get("target_leg_visibility_avg"),
                "target_leg_visibility_threshold": selected_result.get("target_leg_visibility_threshold"),
                "target_leg_visibility_ok": selected_result.get("target_leg_visibility_ok"),
                "target_leg_missing_keypoints": selected_result.get("target_leg_missing_keypoints") or [],
                "locked_side": selected_result.get("locked_side"),
                "side_lock_reason": selected_result.get("side_lock_reason"),
                "pose_stabilized": selected_result.get("pose_stabilized"),
                "held_keypoints": selected_result.get("held_keypoints") or [],
                "jump_rejected": selected_result.get("jump_rejected") or [],
                "side_switch_blocked": selected_result.get("side_switch_blocked"),
                "rknn_detection_count": selected_result.get("rknn_detection_count"),
                "rknn_pipeline": selected_result.get("rknn_pipeline"),
                "rknn_keypoint_threshold": selected_result.get("rknn_keypoint_threshold"),
                "rknn_decoder": selected_result.get("rknn_decoder"),
                "postprocess_error": selected_result.get("postprocess_error"),
                "requested_backend": state.pose_backend_requested,
                "actual_backend": state.pose_backend_actual,
                "pose_backend": state.pose_backend_actual,
                "pose_backend_version": POSE_BACKEND_VERSION.get(state.pose_backend_actual),
                "pose_keypoint_schema": POSE_KEYPOINT_SCHEMA.get(state.pose_backend_actual),
                "rknn_model_path": state.rknn_model_path,
                "rknn_pipeline": state.rknn_pipeline,
                "keypoints": keypoints,
                "rehab_keypoints": rehab_keypoints,
            }
        )
        if not fast_rknn_diagnostics:
            realtime_frame_data.update(rknn_diagnostic_fields(selected_result))
    side_view = compute_side_view_metrics(
        rehab_keypoints,
        selected_result,
        current_visibility_threshold(),
        selected_rule,
        state.pose_backend_actual,
    )
    person_visible = bool(side_view.get("pose_detected"))
    action_keypoints_valid = bool(selected_result.get("valid", False))
    realtime_frame_data.update(side_view)
    realtime_frame_data["person_visible"] = person_visible
    realtime_frame_data["pose_detected"] = person_visible
    realtime_frame_data["action_keypoints_valid"] = action_keypoints_valid
    return current_frame_data, realtime_frame_data


def build_current_frame_data(
    selected_result: dict[str, object],
    selected_rule: dict[str, object],
    keypoints: dict[str, object],
    rehab_keypoints: dict[str, object],
    *,
    fast_rknn_diagnostics: bool = False,
) -> dict[str, object] | None:
    action_id, _ = current_action_context()
    raw_flexion = selected_result.get("selected_flexion_angle")
    if raw_flexion is not None:
        try:
            raw_flexion = float(raw_flexion) + backend_angle_offset(action_id, state.pose_backend_actual)
        except (TypeError, ValueError):
            raw_flexion = selected_result.get("selected_flexion_angle")
    state.smoother.set_window_size(backend_smoothing_window(state.pose_backend_actual))
    smoothed_flexion = state.smoother.update(raw_flexion)
    if not selected_result.get("valid", False):
        return None
    now = time.time()
    frame_data = {
        "frame_index": state.frame_index,
        "relative_time": (now - state.start_time) if state.start_time is not None else 0,
        "selected_side": selected_rule.get("side"),
        "selected_source": selected_result.get("selected_source"),
        "person_visible": True,
        "action_keypoints_valid": True,
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
        "visibility_threshold": current_action_metric_visibility_threshold(),
        "quality_visibility_threshold": current_visibility_threshold(),
        "quality_ok": selected_result.get("quality_ok"),
        "quality_message": selected_result.get("quality_message"),
        "missing_keypoints": selected_result.get("missing_keypoints") or [],
        "person_count": selected_result.get("person_count"),
        "multi_person_warning": selected_result.get("multi_person_warning"),
        "selected_person_reason": selected_result.get("selected_person_reason"),
        "person_box_quality_ok": selected_result.get("person_box_quality_ok"),
        "person_box_height_ratio": selected_result.get("person_box_height_ratio"),
        "person_box_area_ratio": selected_result.get("person_box_area_ratio"),
        "person_box_quality_message": selected_result.get("person_box_quality_message"),
        "fixed_bbox": selected_result.get("fixed_bbox"),
        "fixed_bbox_requested": selected_result.get("fixed_bbox_requested"),
        "fixed_bbox_mode": selected_result.get("fixed_bbox_mode"),
        "target_leg_visibility": selected_result.get("target_leg_visibility"),
        "target_side_keypoint_visibility": selected_result.get("target_side_keypoint_visibility"),
        "target_leg_visibility_min": selected_result.get("target_leg_visibility_min"),
        "target_leg_visibility_avg": selected_result.get("target_leg_visibility_avg"),
        "target_leg_visibility_threshold": selected_result.get("target_leg_visibility_threshold"),
        "target_leg_visibility_ok": selected_result.get("target_leg_visibility_ok"),
        "target_leg_missing_keypoints": selected_result.get("target_leg_missing_keypoints") or [],
        "locked_side": selected_result.get("locked_side"),
        "side_lock_reason": selected_result.get("side_lock_reason"),
        "pose_stabilized": selected_result.get("pose_stabilized"),
        "held_keypoints": selected_result.get("held_keypoints") or [],
        "jump_rejected": selected_result.get("jump_rejected") or [],
        "side_switch_blocked": selected_result.get("side_switch_blocked"),
        "rknn_detection_count": selected_result.get("rknn_detection_count"),
        "rknn_pipeline": selected_result.get("rknn_pipeline"),
        "rknn_keypoint_threshold": selected_result.get("rknn_keypoint_threshold"),
        "rknn_decoder": selected_result.get("rknn_decoder"),
        "postprocess_error": selected_result.get("postprocess_error"),
        "requested_backend": state.pose_backend_requested,
        "actual_backend": state.pose_backend_actual,
        "pose_backend": state.pose_backend_actual,
        "pose_backend_version": POSE_BACKEND_VERSION.get(state.pose_backend_actual),
        "pose_keypoint_schema": POSE_KEYPOINT_SCHEMA.get(state.pose_backend_actual),
        "rknn_model_path": state.rknn_model_path,
        "rknn_pipeline": state.rknn_pipeline,
        "keypoints": keypoints,
        "rehab_keypoints": rehab_keypoints,
    }
    if not fast_rknn_diagnostics:
        frame_data.update(rknn_diagnostic_fields(selected_result))
    return frame_data


def build_pose_quality(selected_result: dict[str, object]) -> dict[str, object]:
    return {
        "quality_ok": bool(selected_result.get("quality_ok")),
        "quality_message": selected_result.get("quality_message"),
        "missing_keypoints": selected_result.get("missing_keypoints") or [],
        "person_count": selected_result.get("person_count", 1 if selected_result.get("valid") else 0),
        "multi_person_warning": bool(selected_result.get("multi_person_warning")),
        "selected_person_reason": selected_result.get("selected_person_reason"),
        "selected_person_score": selected_result.get("selected_person_score"),
        "person_box_quality_ok": selected_result.get("person_box_quality_ok"),
        "person_box_height_ratio": selected_result.get("person_box_height_ratio"),
        "person_box_area_ratio": selected_result.get("person_box_area_ratio"),
        "person_box_quality_message": selected_result.get("person_box_quality_message"),
        "fixed_bbox": selected_result.get("fixed_bbox"),
        "fixed_bbox_requested": selected_result.get("fixed_bbox_requested"),
        "fixed_bbox_mode": selected_result.get("fixed_bbox_mode"),
        "target_leg_visibility": selected_result.get("target_leg_visibility"),
        "target_side_keypoint_visibility": selected_result.get("target_side_keypoint_visibility"),
        "target_leg_visibility_min": selected_result.get("target_leg_visibility_min"),
        "target_leg_visibility_avg": selected_result.get("target_leg_visibility_avg"),
        "target_leg_visibility_threshold": selected_result.get("target_leg_visibility_threshold"),
        "target_leg_visibility_ok": selected_result.get("target_leg_visibility_ok"),
        "target_leg_missing_keypoints": selected_result.get("target_leg_missing_keypoints") or [],
        "locked_side": selected_result.get("locked_side"),
        "side_lock_reason": selected_result.get("side_lock_reason"),
        "pose_stabilized": selected_result.get("pose_stabilized"),
        "held_keypoints": selected_result.get("held_keypoints") or [],
        "jump_rejected": selected_result.get("jump_rejected") or [],
        "side_switch_blocked": bool(selected_result.get("side_switch_blocked")),
        "rknn_detection_count": selected_result.get("rknn_detection_count"),
        "rknn_pipeline": selected_result.get("rknn_pipeline"),
        "rknn_keypoint_threshold": selected_result.get("rknn_keypoint_threshold"),
        "rknn_output_shapes": selected_result.get("rknn_output_shapes"),
        "rknn_det_output_shapes": selected_result.get("rknn_det_output_shapes"),
        "rknn_pose_output_shapes": selected_result.get("rknn_pose_output_shapes"),
        "rknn_det_cache_hit": selected_result.get("rknn_det_cache_hit"),
        "rknn_det_interval": selected_result.get("rknn_det_interval"),
        "rknn_person_select": selected_result.get("rknn_person_select"),
        "rknn_decoder": selected_result.get("rknn_decoder"),
        "keypoint_conf_range": selected_result.get("keypoint_conf_range"),
        "keypoint_xy_range": selected_result.get("keypoint_xy_range"),
        "keypoint_decode_mode": selected_result.get("keypoint_decode_mode"),
        "keypoint_decode_override": selected_result.get("keypoint_decode_override"),
        "keypoint_anchor_order": selected_result.get("keypoint_anchor_order"),
        "keypoint_anchor_order_setting": selected_result.get("keypoint_anchor_order_setting"),
        "keypoint_geometry_score_range": selected_result.get("keypoint_geometry_score_range"),
        "keypoint_global_index_range": selected_result.get("keypoint_global_index_range"),
        "keypoint_candidate_count_range": selected_result.get("keypoint_candidate_count_range"),
        "keypoint_branch_diagnostics": selected_result.get("keypoint_branch_diagnostics"),
        "keypoint_raw_shape": selected_result.get("keypoint_raw_shape"),
        "keypoint_raw_xy_range": selected_result.get("keypoint_raw_xy_range"),
        "keypoint_restored_xy_range": selected_result.get("keypoint_restored_xy_range"),
        "postprocess_error": selected_result.get("postprocess_error"),
    }


def current_visibility_threshold() -> float:
    if state.pose_backend_actual == "rknn" and rknn_backend is not None:
        return float(rknn_backend.keypoint_thres)
    return backend_visibility_threshold(state.pose_backend_actual)


def current_action_metric_visibility_threshold() -> float:
    return min(float(current_visibility_threshold()), ACTION_METRIC_VISIBILITY_THRESHOLD)


class PrescriptionHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path.startswith("/report-images/"):
            if serve_report_image(self, parsed.path):
                return
            self.send_response(404)
            self.end_headers()
            return

        if parsed.path.startswith("/assets/"):
            if serve_static_asset(self, parsed.path):
                return
            self.send_response(404)
            self.end_headers()
            return

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

        if parsed.path == "/ai":
            body = build_ai_page_html().encode("utf-8")
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
            payload = get_system_status(state.pose_fps)
            payload["pose_backend"] = {
                "requested_backend": state.pose_backend_requested,
                "actual_backend": state.pose_backend_actual,
                "fallback_used": state.pose_backend_fallback_used,
                "backend_error_message": state.pose_backend_error_message,
                "rknn_model_path": state.rknn_model_path,
                "rknn_pipeline": state.rknn_pipeline,
                "quality": state.pose_quality,
                "performance": state.pose_perf,
            }
            make_json_response(self, payload)
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
            if cv2 is None or cap is None:
                body = b"vision stream unavailable"
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
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

        if self.path == "/api/llm/report_summary":
            report_path, context, error = resolve_llm_report(payload.get("report_id") or "latest")
            if error:
                make_json_response(self, {"ok": False, "error": error, "message": error}, status_code=404)
                return
            report = context.get("report") if isinstance(context, dict) else None
            if not isinstance(report, dict):
                make_json_response(self, {"ok": False, "error": "训练报告格式无效。", "message": "训练报告格式无效。"}, status_code=400)
                return
            include_keyframes = bool(payload.get("include_keyframes"))
            render_metric_cards = bool(payload.get("render_metric_cards"))
            result = summarize_report(
                report,
                audience=str(payload.get("audience") or "both"),
                include_calorie=bool(payload.get("include_calorie", True)),
                include_keyframes=include_keyframes,
                keyframe_frame_b64=first_keyframe_b64(report) if include_keyframes else None,
            )
            metric_cards = build_metric_cards(report) if (include_keyframes or render_metric_cards) else []
            if include_keyframes:
                result["keyframes"] = attach_keyframe_urls(report)
                if not result.get("keyframe_notes"):
                    result["keyframe_notes"] = build_keyframe_notes(report)[:2]
            if metric_cards:
                result["metric_cards"] = metric_cards
            if render_metric_cards:
                result["rendered_images"] = render_report_images(PROJECT_ROOT, report, metric_cards)
            if isinstance(context, dict):
                result["report_file"] = context.get("report_file")
            result.update(llm_report_source_payload(report_path, context))
            status_code = 200 if result.get("ok") else 502
            make_json_response(self, result, status_code=status_code)
            return

        if self.path == "/api/llm/ask":
            question = str(payload.get("question") or "").strip()
            if not question:
                make_json_response(self, {"ok": False, "error": "请输入要咨询的问题。", "message": "请输入要咨询的问题。"}, status_code=400)
                return
            if len(question) > 300:
                make_json_response(self, {"ok": False, "error": "问题太长，请控制在 300 字以内。", "message": "问题太长，请控制在 300 字以内。"}, status_code=400)
                return
            report_path, context, error = resolve_llm_report(payload.get("report_id") or "latest")
            if error:
                make_json_response(self, {"ok": False, "error": error, "message": error}, status_code=404)
                return
            report = context.get("report") if isinstance(context, dict) else None
            if not isinstance(report, dict):
                make_json_response(self, {"ok": False, "error": "训练报告格式无效。", "message": "训练报告格式无效。"}, status_code=400)
                return
            result = answer_question(report, question, frame_b64=frame_b64_from_payload(payload))
            if isinstance(context, dict):
                result["report_file"] = context.get("report_file")
            result.update(llm_report_source_payload(report_path, context))
            if result.get("ok") and payload.get("speak"):
                result["tts"] = speak_llm_text(result.get("spoken_text"), "llm_qa")
            status_code = 200 if result.get("ok") else 502
            make_json_response(self, result, status_code=status_code)
            return

        if self.path == "/api/llm/speak":
            result = speak_llm_text(payload.get("text"), str(payload.get("event_type") or "llm_summary"))
            make_json_response(self, result, status_code=200 if result.get("ok") else 400)
            return

        if self.path == "/api/realtime/start":
            if state.is_recording:
                make_json_response(self, {"ok": False, "error": "当前正在医生录制中，请先保存或取消。"}, status_code=400)
                return
            action_id = normalize_action_id(payload.get("action_id") or state.action_name)
            patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            state.side_mode = side_mode if side_mode in {"left", "right", "auto"} else "auto"
            state.reset_mediapipe_side_lock()
            target_reps = payload.get("target_reps")
            try:
                target_reps_value = int(target_reps) if target_reps is not None else None
            except (TypeError, ValueError):
                target_reps_value = None
            backend_check = validate_template_backend([action_id])
            if not backend_check.get("ok"):
                make_json_response(self, backend_check, status_code=400)
                return
            result = realtime_session.start(
                patient_id=patient_id,
                action_id=action_id,
                side_mode=side_mode,
                target_reps=target_reps_value,
                pose_backend=state.pose_backend_actual,
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
            state.side_mode = side_mode if side_mode in {"left", "right", "auto"} else "auto"
            state.reset_mediapipe_side_lock()
            target_reps = payload.get("target_reps")
            try:
                target_reps_value = int(target_reps) if target_reps is not None else None
            except (TypeError, ValueError):
                target_reps_value = None
            backend_check = validate_template_backend(["seated_knee_extension", "standing_hamstring_curl", "seated_knee_raise"])
            if not backend_check.get("ok"):
                make_json_response(self, backend_check, status_code=400)
                return
            result = realtime_session.start_playlist(
                patient_id=patient_id,
                side_mode=side_mode,
                target_reps=target_reps_value,
                pose_backend=state.pose_backend_actual,
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

        if self.path == "/api/realtime/care_response":
            result = realtime_session.respond_to_care(bool(payload.get("needs_rest")))
            make_json_response(self, result, status_code=200 if result.get("ok") else 400)
            return

        if self.path == "/api/start":
            if state.is_recording:
                make_json_response(self, {"ok": False, "error": "当前正在录制中，请先保存或取消本轮录制。"}, status_code=400)
                return
            if realtime_session.snapshot().get("status") in ACTIVE_REALTIME_STATUSES:
                make_json_response(self, {"ok": False, "error": "当前正在患者实时训练中，请先结束训练。"}, status_code=400)
                return
            state.patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            state.action_name = sanitize_text(payload.get("action_name"), "knee_flexion")
            state.current_record_role = normalize_record_role(payload.get("record_role"))
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            state.side_mode = side_mode if side_mode in {"left", "right", "auto"} else "auto"
            state.reset_recording()
            state.reset_mediapipe_side_lock()
            if rknn_pose_stabilizer is not None:
                rknn_pose_stabilizer.reset()
            state.start_time = time.time()
            state.is_recording = True
            state.last_export_error = None
            role_label = RECORD_ROLE_LABELS.get(state.current_record_role, state.current_record_role)
            make_json_response(self, {"ok": True, "record_role": state.current_record_role, "message": f"已开始录制{role_label}。"})
            return

        if self.path == "/api/clear":
            clear_export = bool(payload.get("clear_export", False))
            state.reset_recording()
            state.reset_mediapipe_side_lock()
            if rknn_pose_stabilizer is not None:
                rknn_pose_stabilizer.reset()
            if clear_export:
                state.clear_export()
            state.last_export_error = None
            make_json_response(self, {"ok": True, "message": "已清空录制缓存。"})
            return

        if self.path == "/api/cancel":
            state.clear_all()
            if rknn_pose_stabilizer is not None:
                rknn_pose_stabilizer.reset()
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
                "camera_device_active": active_camera_device,
                "camera_backend": active_camera_open_mode or CAMERA_OPEN_MODE,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "camera_frame_shape": active_camera_frame_shape,
                "camera_actual_width": active_camera_actual_width,
                "camera_actual_height": active_camera_actual_height,
                "camera_actual_fps": active_camera_actual_fps,
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
                **current_pose_meta(),
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
                active_template = set_active_template(
                    normalize_action_id(prescription["action_name"]),
                    board_save_result["saved_path"],
                    current_pose_meta(),
                )
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
            if rknn_pose_stabilizer is not None:
                rknn_pose_stabilizer.reset()
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
    backend_ready = cap is not None and (
        state.pose_backend_actual == "rknn"
        or (state.pose_backend_actual == "mediapipe" and mp_pose is not None)
    )
    if cv2 is not None and backend_ready:
        capture_worker = threading.Thread(target=camera_capture_worker, daemon=True)
        inference_worker = threading.Thread(target=pose_worker, daemon=True)
        capture_worker.start()
        inference_worker.start()
    else:
        state.last_status = f"视觉链路未就绪: {vision_boot_error or 'unknown error'}"

    server = ThreadedHTTPServer(("0.0.0.0", PORT), PrescriptionHTTPHandler)

    print(f"8082 统一训练台已启动: http://板子IP:{PORT}")
    print(f"请求摄像头设备: {CAMERA_DEVICE}")
    print(f"实际摄像头设备: {active_camera_device or '未连接'}")
    print(f"请求采集分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"实际采集帧 shape: {active_camera_frame_shape or '未读到'}")
    print(f"推理分辨率: {INFER_WIDTH}x{INFER_HEIGHT}")
    print(f"姿态后端: requested={state.pose_backend_requested}, actual={state.pose_backend_actual}, fallback={state.pose_backend_fallback_used}")
    print("说明: /doctor 用于医生标准动作录制，/train 用于患者实时屈膝训练。")

    try:
        server.serve_forever()
    finally:
        state.running = False
        realtime_session.stop()
        llm_tts_worker.stop()
        with state.condition:
            state.condition.notify_all()
        if cap is not None:
            cap.release()
        if rknn_backend is not None:
            rknn_backend.release()


if __name__ == "__main__":
    main()
