"""Action-specific metric extraction for rehab evaluation and realtime reps."""

from __future__ import annotations

import math
from typing import Any


METRIC_VALUE_FIELD = "metric_value"

METRIC_DEFINITIONS: dict[str, dict[str, str]] = {
    "knee_extension_angle": {
        "unit": "degree",
        "description": "Knee extension included angle from hip-knee-ankle. A larger value means the knee is straighter.",
    },
    "knee_raise_height_ratio": {
        "unit": "body_ratio",
        "description": "Knee height above the hip normalized by shoulder-hip distance. A larger value means the knee is raised higher.",
    },
    "hip_rise_height_ratio": {
        "unit": "body_ratio",
        "description": "Hip rise from the initial seated baseline normalized by shoulder-hip distance. A larger value means the patient stood higher.",
    },
    "hamstring_curl_flexion_angle": {
        "unit": "degree",
        "description": "Knee flexion angle from hip-knee-ankle for standing hamstring curl. A larger value means the lower leg bends farther backward.",
    },
}


def extract_metric_sequence(frames: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metric_name = str(config.get("primary_metric") or "").strip()
    if not metric_name:
        raise ValueError("config missing primary_metric")
    return extract_named_metric_sequence(frames, metric_name, config)


def extract_named_metric_sequence(
    frames: list[dict[str, Any]],
    metric_name: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    definition = METRIC_DEFINITIONS.get(metric_name)
    if definition is None:
        raise ValueError(f"unsupported primary_metric: {metric_name}")

    raw_values: list[tuple[int, float, float]] = []
    baseline_hip_y: float | None = None
    baseline_torso_height: float | None = None

    for position, frame in enumerate(frames):
        time_value = _as_float(frame.get("relative_time"))
        if time_value is None:
            continue
        try:
            value, baseline_hip_y, baseline_torso_height = _extract_frame_value(
                frame,
                metric_name,
                config,
                baseline_hip_y,
                baseline_torso_height,
            )
        except ValueError:
            continue
        raw_values.append((position, time_value, value))

    if not raw_values:
        raise ValueError(
            f"no usable {metric_name} values found; please re-record this action template with rehab_keypoints"
        )

    values = [value for _, _, value in raw_values]
    times = [time_value for _, time_value, _ in raw_values]
    metric_frames = [
        {
            "frame_index": int(_as_float(frames[position].get("frame_index")) or position),
            "relative_time": time_value,
            METRIC_VALUE_FIELD: value,
        }
        for position, time_value, value in raw_values
    ]
    return {
        "metric_name": metric_name,
        "metric_unit": definition["unit"],
        "metric_description": definition["description"],
        "value_field": METRIC_VALUE_FIELD,
        "values": values,
        "frame_times": times,
        "frames": metric_frames,
        "sample_count": len(values),
    }


def extract_metric_from_frame(
    frame: dict[str, Any],
    config: dict[str, Any],
    *,
    baseline_hip_y: float | None = None,
    baseline_torso_height: float | None = None,
) -> dict[str, Any]:
    metric_name = str(config.get("primary_metric") or "").strip()
    if not metric_name:
        raise ValueError("config missing primary_metric")
    value, next_baseline_hip_y, next_baseline_torso_height = _extract_frame_value(
        frame,
        metric_name,
        config,
        baseline_hip_y,
        baseline_torso_height,
    )
    definition = METRIC_DEFINITIONS.get(metric_name) or {}
    return {
        "metric_name": metric_name,
        "metric_unit": definition.get("unit", ""),
        "metric_description": definition.get("description", ""),
        "value": value,
        "baseline_hip_y": next_baseline_hip_y,
        "baseline_torso_height": next_baseline_torso_height,
    }


def _extract_frame_value(
    frame: dict[str, Any],
    metric_name: str,
    config: dict[str, Any],
    baseline_hip_y: float | None,
    baseline_torso_height: float | None,
) -> tuple[float, float | None, float | None]:
    side = _select_side(frame, config)
    if metric_name == "knee_extension_angle":
        hip = _point(frame, side, "hip")
        knee = _point(frame, side, "knee")
        ankle = _point(frame, side, "ankle")
        return _angle_3_points(hip, knee, ankle), baseline_hip_y, baseline_torso_height

    if metric_name == "hamstring_curl_flexion_angle":
        hip = _point(frame, side, "hip")
        knee = _point(frame, side, "knee")
        ankle = _point(frame, side, "ankle")
        included_angle = _angle_3_points(hip, knee, ankle)
        return max(0.0, min(180.0, 180.0 - included_angle)), baseline_hip_y, baseline_torso_height

    if metric_name == "knee_raise_height_ratio":
        shoulder = _point(frame, side, "shoulder")
        hip = _point(frame, side, "hip")
        knee = _point(frame, side, "knee")
        torso_height = _torso_height(shoulder, hip)
        return (hip["y"] - knee["y"]) / torso_height, baseline_hip_y, baseline_torso_height

    if metric_name == "hip_rise_height_ratio":
        shoulder = _point(frame, side, "shoulder")
        hip = _point(frame, side, "hip")
        torso_height = baseline_torso_height or _torso_height(shoulder, hip)
        start_hip_y = baseline_hip_y if baseline_hip_y is not None else hip["y"]
        return (start_hip_y - hip["y"]) / torso_height, start_hip_y, torso_height

    raise ValueError(f"unsupported metric: {metric_name}")


def _select_side(frame: dict[str, Any], config: dict[str, Any]) -> str:
    side = str(frame.get("selected_side") or config.get("side") or "left").lower()
    return side if side in {"left", "right"} else "left"


def _point(frame: dict[str, Any], side: str, joint: str) -> dict[str, float]:
    keypoints = frame.get("rehab_keypoints")
    if not isinstance(keypoints, dict):
        raise ValueError("frame missing rehab_keypoints")
    point = keypoints.get(f"{side}_{joint}")
    if not isinstance(point, dict):
        raise ValueError(f"frame missing rehab_keypoints.{side}_{joint}")
    visibility = _as_float(point.get("visibility"))
    if visibility is not None and visibility < float(frame.get("visibility_threshold") or 0.0):
        raise ValueError(f"low visibility for {side}_{joint}")
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    z = _as_float(point.get("z")) or 0.0
    if x is None or y is None:
        raise ValueError(f"invalid point for {side}_{joint}")
    return {"x": x, "y": y, "z": z}


def _torso_height(shoulder: dict[str, float], hip: dict[str, float]) -> float:
    height = abs(float(hip["y"]) - float(shoulder["y"]))
    if height <= 1e-6:
        raise ValueError("shoulder-hip distance is too small")
    return height


def _angle_3_points(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> float:
    ba = [float(a[key]) - float(b[key]) for key in ("x", "y")]
    bc = [float(c[key]) - float(b[key]) for key in ("x", "y")]
    ba_len = math.sqrt(sum(value * value for value in ba))
    bc_len = math.sqrt(sum(value * value for value in bc))
    if ba_len <= 1e-12 or bc_len <= 1e-12:
        raise ValueError("angle is undefined for zero-length vectors")
    dot = sum(ba[index] * bc[index] for index in range(2))
    cos_value = max(-1.0, min(1.0, dot / (ba_len * bc_len)))
    return math.degrees(math.acos(cos_value))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


if __name__ == "__main__":
    def frame(shoulder_y: float, hip_y: float, knee_y: float, ankle_y: float, ankle_x: float = 0.65) -> dict[str, Any]:
        return {
            "relative_time": 0.0,
            "selected_side": "left",
            "rehab_keypoints": {
                "left_shoulder": {"x": 0.5, "y": shoulder_y, "z": 0.0, "visibility": 1.0},
                "left_hip": {"x": 0.5, "y": hip_y, "z": 0.0, "visibility": 1.0},
                "left_knee": {"x": 0.5, "y": knee_y, "z": 0.0, "visibility": 1.0},
                "left_ankle": {"x": ankle_x, "y": ankle_y, "z": 0.0, "visibility": 1.0},
            },
        }

    bent = frame(0.2, 0.5, 0.7, 0.7, ankle_x=0.7)
    straight = frame(0.2, 0.5, 0.65, 0.9, ankle_x=0.5)
    values = extract_named_metric_sequence([bent, straight], "knee_extension_angle")["values"]
    assert values[1] > values[0]
    low_knee = frame(0.2, 0.5, 0.7, 0.9)
    high_knee = frame(0.2, 0.5, 0.35, 0.9)
    assert extract_named_metric_sequence([low_knee, high_knee], "knee_raise_height_ratio")["values"][1] > 0.0
    seated = frame(0.2, 0.6, 0.75, 0.9)
    standing = frame(0.15, 0.35, 0.55, 0.9)
    assert extract_named_metric_sequence([seated, standing], "hip_rise_height_ratio")["values"][1] > 0.0
    straight_leg = frame(0.2, 0.5, 0.7, 0.9, ankle_x=0.5)
    curled_leg = frame(0.2, 0.5, 0.7, 0.62, ankle_x=0.35)
    curl_values = extract_named_metric_sequence([straight_leg, curled_leg], "hamstring_curl_flexion_angle")["values"]
    assert curl_values[1] > curl_values[0]
    print("action_metrics inline tests passed")
