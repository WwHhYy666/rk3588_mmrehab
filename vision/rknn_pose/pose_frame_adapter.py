"""Convert RKNN COCO17 detections into the 8082 training frame schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vision.rknn_pose.coco17_to_rehab import COCO17_INDEX, REHAB_REQUIRED_NAMES


REHAB_POINT_TO_RULE = {
    "shoulder": "shoulder",
    "hip": "hip",
    "knee": "knee",
    "ankle": "ankle",
}


@dataclass
class PersonSelectionState:
    previous_center: tuple[float, float] | None = None


@dataclass
class PoseStabilizerState:
    previous_rehab: dict[str, dict[str, Any]] | None = None
    missing_counts: dict[str, int] | None = None
    locked_side: str | None = None
    candidate_side: str | None = None
    candidate_count: int = 0
    last_side_mode: str = "auto"


class StablePersonSelector:
    def __init__(self) -> None:
        self.state = PersonSelectionState()

    def select(self, detections: list[dict[str, Any]], frame_width: int, frame_height: int) -> dict[str, Any]:
        candidates = [_score_detection(item, frame_width, frame_height, self.state.previous_center) for item in detections]
        candidates = [item for item in candidates if item is not None]
        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not candidates:
            return {
                "detection": None,
                "person_count": len(detections),
                "selected_person_score": None,
                "selected_person_reason": "no valid person detection",
                "multi_person_warning": False,
                "selection_stable": False,
            }

        selected = candidates[0]
        self.state.previous_center = selected["center"]
        return {
            "detection": selected["detection"],
            "person_count": len(detections),
            "selected_person_score": round(float(selected["score"]), 3),
            "selected_person_reason": "默认选择训练者",
            "multi_person_warning": False,
            "selection_stable": True,
        }


class RknnPoseStabilizer:
    def __init__(
        self,
        *,
        alpha: float = 0.35,
        low_conf_alpha: float = 0.20,
        jump_scale: float = 0.35,
        max_hold_frames: int = 5,
        lock_confirm_frames: int = 5,
    ) -> None:
        self.alpha = float(alpha)
        self.low_conf_alpha = float(low_conf_alpha)
        self.jump_scale = float(jump_scale)
        self.max_hold_frames = int(max_hold_frames)
        self.lock_confirm_frames = int(lock_confirm_frames)
        self.state = PoseStabilizerState(missing_counts={})

    def reset(self) -> None:
        self.state = PoseStabilizerState(missing_counts={})

    def stabilize(
        self,
        rehab_keypoints: dict[str, dict[str, Any]],
        *,
        side_mode: str,
        action_config: dict[str, Any],
        visibility_threshold: float,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if side_mode != self.state.last_side_mode:
            self.state.locked_side = side_mode if side_mode in {"left", "right"} else None
            self.state.candidate_side = None
            self.state.candidate_count = 0
            self.state.last_side_mode = side_mode

        working = {name: dict(point) for name, point in rehab_keypoints.items()}
        diagnostics = {
            "pose_stabilized": True,
            "held_keypoints": [],
            "jump_rejected": [],
            "side_switch_blocked": False,
            "locked_side": self.state.locked_side,
            "side_lock_reason": None,
        }

        preferred_side = _best_side_for_lock(working, action_config, visibility_threshold)
        if side_mode in {"left", "right"}:
            self.state.locked_side = side_mode
            diagnostics["side_lock_reason"] = "manual"
        elif self.state.locked_side is None and preferred_side is not None:
            if self.state.candidate_side == preferred_side:
                self.state.candidate_count += 1
            else:
                self.state.candidate_side = preferred_side
                self.state.candidate_count = 1
            if self.state.candidate_count >= self.lock_confirm_frames:
                self.state.locked_side = preferred_side
                diagnostics["side_lock_reason"] = "auto_confirmed"
        elif self.state.locked_side is not None:
            diagnostics["side_lock_reason"] = "locked"

        locked_side = self.state.locked_side
        previous = self.state.previous_rehab or {}
        if locked_side in {"left", "right"} and previous:
            swapped = _maybe_swap_sides_for_locked_leg(working, previous, locked_side, action_config)
            if swapped:
                working = swapped
                diagnostics["side_switch_blocked"] = True

        body_scale = _body_scale(working, previous)
        jump_threshold = max(0.06, body_scale * self.jump_scale)
        missing_counts = self.state.missing_counts or {}
        stabilized: dict[str, dict[str, Any]] = {}
        for name in REHAB_REQUIRED_NAMES:
            raw_point = dict(working.get(name, {}))
            prev_point = previous.get(name)
            x = _as_float(raw_point.get("x"))
            y = _as_float(raw_point.get("y"))
            visibility = _as_float(raw_point.get("visibility")) or 0.0
            missing = x is None or y is None or visibility <= 0.01

            if missing and isinstance(prev_point, dict) and missing_counts.get(name, 0) < self.max_hold_frames:
                point = dict(prev_point)
                point["visibility"] = max(visibility, (_as_float(prev_point.get("visibility")) or 0.0) * 0.82)
                stabilized[name] = point
                missing_counts[name] = missing_counts.get(name, 0) + 1
                diagnostics["held_keypoints"].append(name)
                continue

            if missing:
                stabilized[name] = raw_point
                missing_counts[name] = missing_counts.get(name, 0) + 1
                continue

            missing_counts[name] = 0
            if isinstance(prev_point, dict):
                prev_x = _as_float(prev_point.get("x"))
                prev_y = _as_float(prev_point.get("y"))
                if prev_x is not None and prev_y is not None:
                    distance = math.hypot(float(x) - prev_x, float(y) - prev_y)
                    if distance > jump_threshold:
                        point = dict(prev_point)
                        point["visibility"] = max(visibility, (_as_float(prev_point.get("visibility")) or 0.0) * 0.85)
                        stabilized[name] = point
                        diagnostics["jump_rejected"].append(name)
                        continue
                    alpha = self.alpha if visibility >= 0.50 else self.low_conf_alpha
                    raw_point["x"] = prev_x * (1.0 - alpha) + float(x) * alpha
                    raw_point["y"] = prev_y * (1.0 - alpha) + float(y) * alpha
            stabilized[name] = raw_point

        self.state.previous_rehab = {name: dict(point) for name, point in stabilized.items()}
        self.state.missing_counts = missing_counts
        diagnostics["locked_side"] = self.state.locked_side
        return stabilized, diagnostics


def adapt_rknn_pose_frame(
    detections: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    action_config: dict[str, Any],
    side_mode: str,
    selector: StablePersonSelector,
    visibility_threshold: float,
    stabilizer: RknnPoseStabilizer | None = None,
) -> dict[str, Any]:
    selection = selector.select(detections, frame_width, frame_height)
    detection = selection["detection"]
    if detection is None:
        return _invalid_payload(selection, "未检测到训练者")

    rehab_keypoints = coco17_detection_to_rehab(detection, frame_width, frame_height)
    stabilization: dict[str, Any] = {
        "pose_stabilized": False,
        "held_keypoints": [],
        "jump_rejected": [],
        "side_switch_blocked": False,
        "locked_side": side_mode if side_mode in {"left", "right"} else None,
        "side_lock_reason": "manual" if side_mode in {"left", "right"} else None,
    }
    effective_side_mode = side_mode
    if stabilizer is not None:
        rehab_keypoints, stabilization = stabilizer.stabilize(
            rehab_keypoints,
            side_mode=side_mode,
            action_config=action_config,
            visibility_threshold=visibility_threshold,
        )
        if stabilization.get("locked_side") in {"left", "right"}:
            effective_side_mode = str(stabilization["locked_side"])
    left_result = _compute_side_result(rehab_keypoints, "left", action_config, visibility_threshold)
    right_result = _compute_side_result(rehab_keypoints, "right", action_config, visibility_threshold)
    selected_side, selected_result = _choose_side(effective_side_mode, left_result, right_result)
    missing = selected_result.get("missing_keypoints", [])
    quality_ok = bool(selected_result.get("quality_ok"))
    quality_message = str(selected_result.get("quality_message") or "关键点质量正常")

    selected_result = {
        **selected_result,
        "valid": quality_ok and selected_result.get("selected_target_angle") is not None,
        "side": selected_side,
        "target_joint": f"{selected_side}_{_target_joint_name(action_config)}",
        "selected_source": "rknn_2d_image",
        "included_angle_3d": None,
        "target_angle_3d": None,
        "flexion_angle_3d": None,
        "quality_ok": quality_ok,
        "missing_keypoints": missing,
        "quality_message": quality_message,
        "person_count": selection["person_count"],
        "selected_person_score": selection["selected_person_score"],
        "selected_person_reason": selection["selected_person_reason"],
        "multi_person_warning": selection["multi_person_warning"],
        "selection_stable": selection["selection_stable"],
        "locked_side": stabilization.get("locked_side"),
        "side_lock_reason": stabilization.get("side_lock_reason"),
        "pose_stabilized": stabilization.get("pose_stabilized"),
        "held_keypoints": stabilization.get("held_keypoints") or [],
        "jump_rejected": stabilization.get("jump_rejected") or [],
        "side_switch_blocked": bool(stabilization.get("side_switch_blocked")),
    }
    return {
        "selected_rule": _rule_for_side(selected_side, action_config),
        "selected_result": selected_result,
        "rehab_keypoints": rehab_keypoints,
        "keypoints": _compact_keypoints(rehab_keypoints, selected_side, action_config),
        "selected_detection": detection,
    }


def coco17_detection_to_rehab(detection: dict[str, Any], frame_width: int, frame_height: int) -> dict[str, dict[str, Any]]:
    keypoints = detection.get("keypoints") or []
    rehab: dict[str, dict[str, Any]] = {}
    for name in REHAB_REQUIRED_NAMES:
        index = COCO17_INDEX[name]
        point = keypoints[index] if index < len(keypoints) else None
        if point is None:
            rehab[name] = {"x": None, "y": None, "z": None, "z_valid": False, "visibility": 0.0}
            continue
        x = _as_float(point[0])
        y = _as_float(point[1])
        score = _as_float(point[2]) or 0.0
        rehab[name] = {
            "x": _normalize(x, frame_width),
            "y": _normalize(y, frame_height),
            "z": None,
            "z_valid": False,
            "visibility": score,
        }
    return rehab


def _compute_side_result(
    rehab_keypoints: dict[str, dict[str, Any]],
    side: str,
    action_config: dict[str, Any],
    visibility_threshold: float,
) -> dict[str, Any]:
    point_names = [str(name) for name in action_config.get("point_names", ["hip", "knee", "ankle"])]
    metric_kind = str(action_config.get("metric_kind", "angle"))
    points = []
    missing: list[str] = []
    visibilities: list[float] = []
    for point_name in point_names:
        key = f"{side}_{point_name}"
        point = rehab_keypoints.get(key)
        visibility = _as_float(point.get("visibility") if isinstance(point, dict) else None) or 0.0
        x = _as_float(point.get("x") if isinstance(point, dict) else None)
        y = _as_float(point.get("y") if isinstance(point, dict) else None)
        if x is None or y is None or visibility < visibility_threshold:
            missing.append(key)
        points.append((x, y))
        visibilities.append(visibility)

    included = calculate_angle(points) if not missing else None
    if metric_kind == "knee_raise_height_ratio" and not missing:
        target = calculate_knee_raise_height_ratio(points, point_names)
    else:
        target = target_angle_from_included(included, str(action_config.get("angle_kind", "included")))
    visibility_min = min(visibilities) if visibilities else 0.0
    visibility_avg = sum(visibilities) / len(visibilities) if visibilities else 0.0
    quality_ok = included is not None and not missing
    return {
        "quality_ok": quality_ok,
        "quality_message": "关键点质量正常" if quality_ok else "关键点缺失或置信度过低",
        "missing_keypoints": missing,
        "visibility_min": visibility_min,
        "visibility_avg": visibility_avg,
        "included_angle_2d": included,
        "target_angle_2d": target,
        "flexion_angle_2d": target,
        "selected_included_angle": included,
        "selected_target_angle": target,
        "selected_flexion_angle": target,
    }


def calculate_angle(points: list[tuple[float | None, float | None]]) -> float | None:
    if len(points) != 3:
        return None
    if any(point[0] is None or point[1] is None for point in points):
        return None
    a, b, c = [(float(point[0]), float(point[1])) for point in points]
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ba_len = math.hypot(*ba)
    bc_len = math.hypot(*bc)
    if ba_len <= 1e-12 or bc_len <= 1e-12:
        return None
    cos_value = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_len * bc_len)))
    return math.degrees(math.acos(cos_value))


def target_angle_from_included(included_angle: float | None, angle_kind: str) -> float | None:
    if included_angle is None:
        return None
    if angle_kind == "flexion":
        return max(0.0, min(180.0, 180.0 - included_angle))
    return max(0.0, min(180.0, included_angle))


def calculate_knee_raise_height_ratio(points: list[tuple[float | None, float | None]], point_names: list[str]) -> float | None:
    named = {name: point for name, point in zip(point_names, points)}
    shoulder = named.get("shoulder")
    hip = named.get("hip")
    knee = named.get("knee")
    if shoulder is None or hip is None or knee is None:
        return None
    if any(point[0] is None or point[1] is None for point in (shoulder, hip, knee)):
        return None
    shoulder_x, shoulder_y = float(shoulder[0]), float(shoulder[1])
    hip_x, hip_y = float(hip[0]), float(hip[1])
    knee_y = float(knee[1])
    torso_scale = math.hypot(shoulder_x - hip_x, shoulder_y - hip_y)
    if torso_scale <= 1e-12:
        return None
    return (hip_y - knee_y) / torso_scale


def _choose_side(side_mode: str, left_result: dict[str, Any], right_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if side_mode == "left":
        return "left", left_result
    if side_mode == "right":
        return "right", right_result
    left_valid = bool(left_result.get("quality_ok"))
    right_valid = bool(right_result.get("quality_ok"))
    if left_valid and not right_valid:
        return "left", left_result
    if right_valid and not left_valid:
        return "right", right_result
    if right_result.get("visibility_avg", 0.0) > left_result.get("visibility_avg", 0.0):
        return "right", right_result
    return "left", left_result


def _best_side_for_lock(
    rehab_keypoints: dict[str, dict[str, Any]],
    action_config: dict[str, Any],
    visibility_threshold: float,
) -> str | None:
    left_result = _compute_side_result(rehab_keypoints, "left", action_config, visibility_threshold)
    right_result = _compute_side_result(rehab_keypoints, "right", action_config, visibility_threshold)
    left_ok = bool(left_result.get("quality_ok"))
    right_ok = bool(right_result.get("quality_ok"))
    if left_ok and not right_ok:
        return "left"
    if right_ok and not left_ok:
        return "right"
    if not left_ok and not right_ok:
        return None
    return "right" if right_result.get("visibility_avg", 0.0) > left_result.get("visibility_avg", 0.0) else "left"


def _maybe_swap_sides_for_locked_leg(
    current: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    locked_side: str,
    action_config: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    other_side = "right" if locked_side == "left" else "left"
    point_names = [str(name) for name in action_config.get("point_names", ["hip", "knee", "ankle"])]
    locked_distance = _side_distance_to_previous(current, previous, locked_side, locked_side, point_names)
    other_distance = _side_distance_to_previous(current, previous, other_side, locked_side, point_names)
    if locked_distance is None or other_distance is None:
        return None
    if other_distance + 0.03 >= locked_distance * 0.65:
        return None
    swapped = {name: dict(point) for name, point in current.items()}
    for base_name in ("shoulder", "hip", "knee", "ankle"):
        left_key = f"left_{base_name}"
        right_key = f"right_{base_name}"
        if left_key in swapped and right_key in swapped:
            swapped[left_key], swapped[right_key] = swapped[right_key], swapped[left_key]
    return swapped


def _side_distance_to_previous(
    current: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    current_side: str,
    previous_side: str,
    point_names: list[str],
) -> float | None:
    distances = []
    for point_name in point_names:
        current_point = current.get(f"{current_side}_{point_name}")
        previous_point = previous.get(f"{previous_side}_{point_name}")
        cx = _as_float(current_point.get("x") if isinstance(current_point, dict) else None)
        cy = _as_float(current_point.get("y") if isinstance(current_point, dict) else None)
        px = _as_float(previous_point.get("x") if isinstance(previous_point, dict) else None)
        py = _as_float(previous_point.get("y") if isinstance(previous_point, dict) else None)
        if cx is None or cy is None or px is None or py is None:
            continue
        distances.append(math.hypot(cx - px, cy - py))
    if not distances:
        return None
    return sum(distances) / len(distances)


def _body_scale(current: dict[str, dict[str, Any]], previous: dict[str, dict[str, Any]]) -> float:
    values = []
    for source in (current, previous):
        for side in ("left", "right"):
            for start, end in (("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
                start_point = source.get(f"{side}_{start}")
                end_point = source.get(f"{side}_{end}")
                sx = _as_float(start_point.get("x") if isinstance(start_point, dict) else None)
                sy = _as_float(start_point.get("y") if isinstance(start_point, dict) else None)
                ex = _as_float(end_point.get("x") if isinstance(end_point, dict) else None)
                ey = _as_float(end_point.get("y") if isinstance(end_point, dict) else None)
                if sx is not None and sy is not None and ex is not None and ey is not None:
                    distance = math.hypot(sx - ex, sy - ey)
                    if distance > 0:
                        values.append(distance)
    if not values:
        return 0.20
    return max(0.08, sorted(values)[len(values) // 2])


def _rule_for_side(side: str, action_config: dict[str, Any]) -> dict[str, Any]:
    rule = dict((action_config.get("rules") or {}).get(side, {}))
    rule["side"] = side
    rule["target_joint"] = f"{side}_{_target_joint_name(action_config)}"
    return rule


def _target_joint_name(action_config: dict[str, Any]) -> str:
    point_names = [str(name) for name in action_config.get("point_names", ["hip", "knee", "ankle"])]
    return point_names[1] if len(point_names) >= 2 else "knee"


def _compact_keypoints(rehab_keypoints: dict[str, dict[str, Any]], side: str, action_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    compact = {}
    for name in action_config.get("point_names", ["hip", "knee", "ankle"]):
        key = f"{side}_{name}"
        compact[str(name)] = dict(rehab_keypoints.get(key, {}))
    return compact


def _score_detection(
    detection: dict[str, Any],
    frame_width: int,
    frame_height: int,
    previous_center: tuple[float, float] | None,
) -> dict[str, Any] | None:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_width * frame_height))
    area_score = max(0.0, min(1.0, area / frame_area * 4.0))
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    dx = abs(center[0] - frame_width / 2.0) / max(frame_width / 2.0, 1.0)
    dy = abs(center[1] - frame_height / 2.0) / max(frame_height / 2.0, 1.0)
    center_score = max(0.0, 1.0 - (dx + dy) / 2.0)
    visibility_score = _visibility_score(detection.get("keypoints") or [])
    stability_score = 0.5
    if previous_center is not None:
        distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
        stability_score = max(0.0, 1.0 - distance / max(frame_width, frame_height, 1))
    score = 0.38 * area_score + 0.32 * visibility_score + 0.20 * center_score + 0.10 * stability_score
    return {"detection": detection, "score": score, "center": center}


def _visibility_score(keypoints: list[Any]) -> float:
    values = []
    for name in REHAB_REQUIRED_NAMES:
        index = COCO17_INDEX[name]
        if index < len(keypoints):
            values.append(_as_float(keypoints[index][2]) or 0.0)
    return sum(values) / len(values) if values else 0.0


def _invalid_payload(selection: dict[str, Any], message: str) -> dict[str, Any]:
    selected_result = {
        "valid": False,
        "quality_ok": False,
        "quality_message": message,
        "missing_keypoints": list(REHAB_REQUIRED_NAMES),
        "selected_source": "rknn_2d_image",
        "included_angle_3d": None,
        "target_angle_3d": None,
        "flexion_angle_3d": None,
        "person_count": selection.get("person_count", 0),
        "selected_person_reason": selection.get("selected_person_reason"),
        "selected_person_score": selection.get("selected_person_score"),
        "multi_person_warning": selection.get("multi_person_warning", False),
        "selection_stable": selection.get("selection_stable", False),
        "locked_side": None,
        "side_lock_reason": None,
        "pose_stabilized": False,
        "held_keypoints": [],
        "jump_rejected": [],
        "side_switch_blocked": False,
    }
    return {
        "selected_rule": {"side": "left"},
        "selected_result": selected_result,
        "rehab_keypoints": {},
        "keypoints": {},
        "selected_detection": selection.get("detection"),
    }


def _normalize(value: float | None, size: int) -> float | None:
    if value is None or size <= 0:
        return None
    return max(0.0, min(1.0, float(value) / float(size)))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None
