from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CANONICAL_KEYPOINT_ORDER = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

LEFT_RIGHT_MIRROR_MAP = {
    "left_eye": "right_eye",
    "right_eye": "left_eye",
    "left_ear": "right_ear",
    "right_ear": "left_ear",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
}

REHAB_POINT_INDEX = {name: index for index, name in enumerate(CANONICAL_KEYPOINT_ORDER)}


@dataclass(frozen=True)
class FeatureMeta:
    valid_frames: int
    total_frames: int
    used_keypoint_names: tuple[str, ...]


def build_input_tensor(rep_payload: dict[str, Any], target_frames: int = 30) -> tuple[np.ndarray | None, FeatureMeta]:
    sequence = build_feature_sequence(rep_payload, target_frames=target_frames)
    if sequence is None:
        return None, FeatureMeta(valid_frames=0, total_frames=target_frames, used_keypoint_names=())
    tensor, valid_frames, used_names = sequence
    return tensor[np.newaxis, :, :].astype(np.float32), FeatureMeta(
        valid_frames=valid_frames,
        total_frames=target_frames,
        used_keypoint_names=tuple(used_names),
    )


def build_feature_sequence(rep_payload: dict[str, Any], target_frames: int = 30) -> tuple[np.ndarray, int, list[str]] | None:
    frames = extract_skeleton_frames(rep_payload)
    if not frames:
        return None
    normalized = [_frame_to_feature_vector(frame) for frame in frames]
    valid_count = sum(1 for vector, _ in normalized if vector is not None)
    if valid_count == 0:
        return None
    stacked = np.stack([vector if vector is not None else np.zeros(51, dtype=np.float32) for vector, _ in normalized], axis=0)
    resampled = _resample_time(stacked, target_frames)
    used_names: set[str] = set()
    for _, names in normalized:
        used_names.update(names)
    return resampled.T.astype(np.float32), valid_count, sorted(used_names)


def extract_skeleton_frames(rep_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(rep_payload, dict):
        return []
    sequence = rep_payload.get("skeleton_sequence")
    if isinstance(sequence, list):
        return [frame for frame in sequence if isinstance(frame, dict)]
    return []


def mirror_rep_payload(rep_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(rep_payload)
    mirrored_frames: list[dict[str, Any]] = []
    for frame in extract_skeleton_frames(rep_payload):
        mirrored = dict(frame)
        for field in ("keypoints", "rehab_keypoints"):
            source = frame.get(field)
            if not isinstance(source, dict):
                continue
            target: dict[str, Any] = {}
            for name, point in source.items():
                mapped = LEFT_RIGHT_MIRROR_MAP.get(str(name), str(name))
                if isinstance(point, dict):
                    target[mapped] = {
                        "x": _mirror_x(point.get("x")),
                        "y": _as_float(point.get("y")),
                        "z": _as_float(point.get("z")),
                        "visibility": _as_float(point.get("visibility")),
                    }
            mirrored[field] = target
        mirrored_frames.append(mirrored)
    payload["skeleton_sequence"] = mirrored_frames
    return payload


def _frame_to_feature_vector(frame: dict[str, Any]) -> tuple[np.ndarray | None, list[str]]:
    points = _extract_keypoint_map(frame)
    if not points:
        return None, []
    center_x, center_y = _body_center(points)
    scale = _body_scale(points)
    used_names: list[str] = []
    features: list[float] = []
    for name in CANONICAL_KEYPOINT_ORDER:
        point = points.get(name)
        if not isinstance(point, dict):
            features.extend([0.0, 0.0, 0.0])
            continue
        x = _as_float(point.get("x"))
        y = _as_float(point.get("y"))
        visibility = _clip_visibility(point.get("visibility"))
        if x is None or y is None or visibility <= 0.0:
            features.extend([0.0, 0.0, 0.0])
            continue
        used_names.append(name)
        nx = ((x - center_x) / scale) * visibility
        ny = ((y - center_y) / scale) * visibility
        features.extend([float(nx), float(ny), visibility])
    return np.asarray(features, dtype=np.float32), used_names


def _extract_keypoint_map(frame: dict[str, Any]) -> dict[str, dict[str, float]]:
    for field in ("keypoints", "rehab_keypoints"):
        source = frame.get(field)
        if not isinstance(source, dict):
            continue
        normalized: dict[str, dict[str, float]] = {}
        for name, point in source.items():
            if not isinstance(point, dict):
                continue
            x = _as_float(point.get("x"))
            y = _as_float(point.get("y"))
            if x is None or y is None:
                continue
            normalized[str(name)] = {
                "x": x,
                "y": y,
                "z": _as_float(point.get("z")) or 0.0,
                "visibility": _clip_visibility(point.get("visibility")),
            }
        if normalized:
            return normalized
    return {}


def _body_center(points: dict[str, dict[str, float]]) -> tuple[float, float]:
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")
    if left_hip and right_hip:
        return ((left_hip["x"] + right_hip["x"]) / 2.0, (left_hip["y"] + right_hip["y"]) / 2.0)
    xs = [point["x"] for point in points.values()]
    ys = [point["y"] for point in points.values()]
    if xs and ys:
        return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    return 0.0, 0.0


def _body_scale(points: dict[str, dict[str, float]]) -> float:
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    candidates: list[float] = []
    if left_hip and right_hip:
        candidates.append(abs(left_hip["x"] - right_hip["x"]))
        candidates.append(abs(left_hip["y"] - right_hip["y"]))
    if left_shoulder and right_shoulder:
        candidates.append(abs(left_shoulder["x"] - right_shoulder["x"]))
        candidates.append(abs(left_shoulder["y"] - right_shoulder["y"]))
    xs = [point["x"] for point in points.values()]
    ys = [point["y"] for point in points.values()]
    if xs and ys:
        candidates.append(max(xs) - min(xs))
        candidates.append(max(ys) - min(ys))
    scale = max([value for value in candidates if value > 1e-6], default=1.0)
    return float(scale)


def _resample_time(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    frame_count = int(sequence.shape[0])
    if frame_count == target_frames:
        return sequence
    if frame_count <= 0:
        return np.zeros((target_frames, int(sequence.shape[1]) if sequence.ndim == 2 else 51), dtype=np.float32)
    if frame_count == 1:
        return np.repeat(sequence, target_frames, axis=0)
    source_positions = np.linspace(0.0, 1.0, num=frame_count)
    target_positions = np.linspace(0.0, 1.0, num=target_frames)
    resampled = np.zeros((target_frames, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        resampled[:, feature_index] = np.interp(target_positions, source_positions, sequence[:, feature_index]).astype(np.float32)
    return resampled


def _mirror_x(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return 1.0 - number


def _clip_visibility(value: Any) -> float:
    number = _as_float(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, number))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None
