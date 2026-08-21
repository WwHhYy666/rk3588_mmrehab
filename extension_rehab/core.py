from __future__ import annotations

import math
from typing import Any

from .catalog import ACTIONS


COCO17_NAMES = (
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


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_coco17_points(raw: object) -> dict[str, dict[str, float | None]]:
    if isinstance(raw, dict):
        result: dict[str, dict[str, float | None]] = {}
        for name in COCO17_NAMES:
            point = raw.get(name)
            if not isinstance(point, dict):
                continue
            result[name] = {
                "x": _number(point.get("x")),
                "y": _number(point.get("y")),
                "visibility": _number(point.get("visibility")) or 0.0,
            }
        return result
    if isinstance(raw, (list, tuple)):
        result = {}
        for index, name in enumerate(COCO17_NAMES):
            point = raw[index] if index < len(raw) else None
            if isinstance(point, dict):
                x = _number(point.get("x"))
                y = _number(point.get("y"))
                visibility = _number(point.get("visibility")) or 0.0
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x = _number(point[0])
                y = _number(point[1])
                visibility = _number(point[2]) if len(point) >= 3 else 1.0
                visibility = visibility or 0.0
            else:
                continue
            result[name] = {"x": x, "y": y, "visibility": visibility}
        return result
    return {}


def _point(points: dict[str, dict[str, float | None]], name: str) -> tuple[float, float] | None:
    point = points.get(name)
    if not isinstance(point, dict):
        return None
    x = _number(point.get("x"))
    y = _number(point.get("y"))
    return (x, y) if x is not None and y is not None else None


def _visibility(points: dict[str, dict[str, float | None]], name: str) -> float:
    point = points.get(name)
    return float(_number(point.get("visibility")) or 0.0) if isinstance(point, dict) else 0.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    first = (a[0] - b[0], a[1] - b[1])
    second = (c[0] - b[0], c[1] - b[1])
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _lean_from_vertical(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    return math.degrees(math.atan2(abs(top[0] - bottom[0]), max(abs(top[1] - bottom[1]), 1e-8)))


def _shoulder_tilt(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.degrees(math.atan2(abs(left[1] - right[1]), max(abs(left[0] - right[0]), 1e-8)))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _best_visible_side(points: dict[str, dict[str, float | None]], joints: tuple[str, ...]) -> str:
    def score(side: str) -> tuple[int, float]:
        names = [f"{side}_{joint}" for joint in joints]
        present = sum(1 for name in names if _point(points, name) is not None)
        visibility = sum(_visibility(points, name) for name in names)
        return present, visibility

    return "right" if score("right") > score("left") else "left"


def compute_extension_frame(action_id: str, raw_points: object) -> dict[str, Any]:
    config = ACTIONS.get(str(action_id))
    if config is None:
        return {"valid": False, "metric": None, "components": {}, "compensation_errors": ["unknown_action"], "missing_keypoints": []}
    points = normalize_coco17_points(raw_points)
    threshold = float(config.get("visibility_threshold", 0.15))
    selected_side: str | None = None
    if action_id in {"seated_biceps_curl", "seated_shoulder_flexion"}:
        selected_side = _best_visible_side(points, ("shoulder", "elbow", "wrist", "hip"))
    elif action_id == "mini_squat":
        selected_side = _best_visible_side(points, ("shoulder", "hip", "knee", "ankle"))
    required = [str(name) for name in config.get("required_points") or []]
    if selected_side == "right":
        required = [name.replace("left_", "right_", 1) if name.startswith("left_") else name for name in required]
    missing = [name for name in required if _point(points, name) is None or _visibility(points, name) < threshold]
    errors: list[str] = []
    components: dict[str, float] = {}
    metric: float | None = None

    if action_id == "seated_biceps_curl":
        side = selected_side or "left"
        if _visibility(points, f"{side}_wrist") < threshold:
            errors.append("wrist_visibility")
        shoulder, elbow, wrist, hip = (_point(points, f"{side}_{joint}") for joint in ("shoulder", "elbow", "wrist", "hip"))
        if shoulder and elbow and wrist and hip:
            included = _angle(shoulder, elbow, wrist)
            metric = 180.0 - included if included is not None else None
            torso = max(_distance(shoulder, hip), 1e-6)
            components["upper_arm_swing_ratio"] = abs(shoulder[0] - elbow[0]) / torso
            components["trunk_lean_deg"] = _lean_from_vertical(shoulder, hip)
            if components["upper_arm_swing_ratio"] > float(config["compensation"]["max_upper_arm_swing_ratio"]):
                errors.append("upper_arm_swing")
            if components["trunk_lean_deg"] > float(config["compensation"]["max_trunk_lean_deg"]):
                errors.append("trunk_lean")

    elif action_id == "seated_shoulder_flexion":
        side = selected_side or "left"
        hip, shoulder, elbow, wrist = (_point(points, f"{side}_{joint}") for joint in ("hip", "shoulder", "elbow", "wrist"))
        if hip and shoulder and elbow and wrist:
            metric = _angle(hip, shoulder, elbow)
            elbow_angle = _angle(shoulder, elbow, wrist)
            components["elbow_angle_deg"] = elbow_angle if elbow_angle is not None else 0.0
            components["trunk_lean_deg"] = _lean_from_vertical(shoulder, hip)
            if components["elbow_angle_deg"] < float(config["compensation"]["min_elbow_angle_deg"]):
                errors.append("elbow_bent")
            if components["trunk_lean_deg"] > float(config["compensation"]["max_trunk_lean_deg"]):
                errors.append("trunk_lean")

    elif action_id == "standing_shoulder_abduction":
        left_hip, right_hip, left_shoulder, right_shoulder, left_elbow, right_elbow = (
            _point(points, name)
            for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
        )
        if left_hip and right_hip and left_shoulder and right_shoulder and left_elbow and right_elbow:
            left_angle = _angle(left_hip, left_shoulder, left_elbow)
            right_angle = _angle(right_hip, right_shoulder, right_elbow)
            if left_angle is not None and right_angle is not None:
                metric = (left_angle + right_angle) * 0.5
                components.update(left_angle_deg=left_angle, right_angle_deg=right_angle)
                if abs(left_angle - right_angle) > float(config["compensation"]["max_bilateral_difference_deg"]):
                    errors.append("bilateral_asymmetry")
            shoulder_mid = _midpoint(left_shoulder, right_shoulder)
            hip_mid = _midpoint(left_hip, right_hip)
            components["body_side_lean_deg"] = _lean_from_vertical(shoulder_mid, hip_mid)
            if components["body_side_lean_deg"] > float(config["compensation"]["max_body_side_lean_deg"]):
                errors.append("body_side_lean")

    elif action_id == "mini_squat":
        side = selected_side or "left"
        shoulder, hip, knee, ankle = (_point(points, f"{side}_{joint}") for joint in ("shoulder", "hip", "knee", "ankle"))
        if shoulder and hip and knee and ankle:
            included = _angle(hip, knee, ankle)
            metric = 180.0 - included if included is not None else None
            components["hip_y"] = hip[1]
            components["trunk_lean_deg"] = _lean_from_vertical(shoulder, hip)
            if components["trunk_lean_deg"] > float(config["compensation"]["max_trunk_lean_deg"]):
                errors.append("trunk_lean")

    elif action_id in {"lateral_step_touch", "low_impact_step_jack"}:
        left_shoulder, right_shoulder, left_wrist, right_wrist, left_ankle, right_ankle = (
            _point(points, name)
            for name in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
        )
        if _visibility(points, "left_ankle") < threshold or _visibility(points, "right_ankle") < threshold:
            errors.append("ankle_visibility")
        if left_shoulder and right_shoulder and left_ankle and right_ankle:
            shoulder_width = max(_distance(left_shoulder, right_shoulder), 1e-6)
            ankle_span_ratio = abs(right_ankle[0] - left_ankle[0]) / shoulder_width
            tilt = _shoulder_tilt(left_shoulder, right_shoulder)
            components.update(ankle_span_ratio=ankle_span_ratio, shoulder_tilt_deg=tilt)
            if tilt > float(config["compensation"]["max_shoulder_tilt_deg"]):
                errors.append("shoulder_tilt")
            if action_id == "lateral_step_touch":
                metric = ankle_span_ratio
            elif left_wrist and right_wrist:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) * 0.5
                wrist_y = (left_wrist[1] + right_wrist[1]) * 0.5
                arm_raise_ratio = max(0.0, (shoulder_y - wrist_y) / shoulder_width)
                leg_open_progress = max(0.0, ankle_span_ratio - 1.0)
                components.update(arm_raise_ratio=arm_raise_ratio, leg_open_progress=leg_open_progress)
                metric = min(arm_raise_ratio, leg_open_progress)
                if abs(arm_raise_ratio - leg_open_progress) > float(config["compensation"]["max_coordination_gap"]):
                    errors.append("coordination")

    valid = not missing and metric is not None and math.isfinite(metric)
    if missing:
        errors.append("low_visibility")
    unique_errors = list(dict.fromkeys(errors))
    quality_score = max(0.0, 100.0 - 18.0 * len(unique_errors) - 4.0 * len(missing))
    return {
        "action_id": action_id,
        "group": config["group"],
        "view": config["view"],
        "selected_side": selected_side,
        "valid": valid,
        "metric": round(float(metric), 6) if valid and metric is not None else None,
        "metric_unit": config["metric_unit"],
        "components": {name: round(float(value), 6) for name, value in components.items()},
        "compensation_errors": unique_errors,
        "missing_keypoints": missing,
        "quality_score": round(quality_score, 1),
    }
