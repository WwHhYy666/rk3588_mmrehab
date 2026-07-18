"""Convert RKNN COCO-17 detections into the shared training-frame schema."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

from pose_estimation.rknn_pose.coco17_to_rehab import COCO17_INDEX, REHAB_REQUIRED_NAMES


REHAB_POINT_TO_RULE = {
    "shoulder": "shoulder",
    "hip": "hip",
    "knee": "knee",
    "ankle": "ankle",
}

COCO17_NAMES = tuple(COCO17_INDEX)
COCO17_DISPLAY_THRESHOLDS = {
    "nose": 0.10,
    "left_eye": 0.10,
    "right_eye": 0.10,
    "left_ear": 0.10,
    "right_ear": 0.10,
    "left_elbow": 0.12,
    "right_elbow": 0.12,
    "left_wrist": 0.10,
    "right_wrist": 0.10,
}


def compute_coco17_orientation_metrics(
    rehab_keypoints: dict[str, object],
    selected_result: dict[str, object],
    visibility_threshold: float,
    selected_rule: dict[str, object] | None = None,
    *,
    front_view_ratio_min: float = 0.55,
    side_view_ratio_max: float = 0.32,
    coordinate_visibility_threshold: float = 0.08,
    relaxed_visibility_threshold: float = 0.12,
) -> dict[str, object]:
    visibilities = [
        _as_float(point.get("visibility")) or 0.0
        for point in rehab_keypoints.values()
        if isinstance(point, dict)
    ]
    has_any_keypoint = any(value > 0.01 for value in visibilities)
    pose_detected = bool(selected_result.get("valid")) or int(selected_result.get("person_count") or 0) > 0 or has_any_keypoint
    if not pose_detected:
        return {
            "pose_detected": False,
            "orientation_ok": False,
            "side_view_ok": False,
            "front_view_ok": False,
            "orientation_ratio": None,
            "orientation_visibility": 0.0,
            "orientation_message": "pose not detected",
            "rknn_orientation_mode": "no_person",
            "rknn_orientation_relaxed": True,
            "rknn_orientation_chains": [],
        }

    torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    torso_points: dict[str, tuple[float, float, float]] = {}
    for name in torso_names:
        point = rehab_keypoints.get(name)
        if not isinstance(point, dict):
            continue
        x = _as_float(point.get("x"))
        y = _as_float(point.get("y"))
        visibility = _as_float(point.get("visibility")) or 0.0
        if x is None or y is None or visibility < float(coordinate_visibility_threshold):
            continue
        torso_points[name] = (float(x), float(y), float(visibility))

    if len(torso_points) == len(torso_names):
        left_shoulder = torso_points["left_shoulder"]
        right_shoulder = torso_points["right_shoulder"]
        left_hip = torso_points["left_hip"]
        right_hip = torso_points["right_hip"]
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        torso_height = max(
            (abs(left_shoulder[1] - left_hip[1]) + abs(right_shoulder[1] - right_hip[1])) / 2.0,
            1e-6,
        )
        orientation_ratio = max(shoulder_width, hip_width) / torso_height
        side_view_ok = orientation_ratio <= float(side_view_ratio_max)
        front_view_ok = orientation_ratio >= float(front_view_ratio_min)
        torso_visibility = min(point[2] for point in torso_points.values())
        return {
            "pose_detected": True,
            "orientation_ok": side_view_ok,
            "side_view_ok": side_view_ok,
            "front_view_ok": front_view_ok,
            "orientation_ratio": orientation_ratio,
            "front_view_ratio_min": float(front_view_ratio_min),
            "side_view_ratio_max": float(side_view_ratio_max),
            "orientation_visibility": torso_visibility,
            "orientation_message": "side view ok" if side_view_ok else ("front view ok" if front_view_ok else "adjust camera angle"),
            "rknn_orientation_mode": "torso_ratio",
            "rknn_orientation_relaxed": True,
            "rknn_orientation_chains": [],
        }

    relaxed_threshold = min(float(visibility_threshold), float(relaxed_visibility_threshold))
    selected_side = str(selected_result.get("side") or (selected_rule or {}).get("side") or "left")
    sides = [selected_side]
    other_side = "right" if selected_side == "left" else "left"
    if other_side not in sides:
        sides.append(other_side)

    checked_chains: list[str] = []
    best_visibility = 0.0
    for side in sides:
        for chain in (
            [f"{side}_hip", f"{side}_knee", f"{side}_ankle"],
            [f"{side}_shoulder", f"{side}_hip", f"{side}_knee"],
        ):
            values = [
                _as_float(point.get("visibility")) or 0.0
                for name in chain
                for point in [rehab_keypoints.get(name)]
                if isinstance(point, dict)
            ]
            while len(values) < len(chain):
                values.append(0.0)
            visible_count = sum(1 for value in values if value >= relaxed_threshold)
            checked_chains.append(f"{'-'.join(chain)}:{visible_count}/3")
            best_visibility = max(best_visibility, min(values) if values else 0.0)
            if visible_count == len(chain):
                return {
                    "pose_detected": True,
                    "orientation_ok": True,
                    "side_view_ok": True,
                    "front_view_ok": False,
                    "orientation_ratio": None,
                    "front_view_ratio_min": float(front_view_ratio_min),
                    "side_view_ratio_max": float(side_view_ratio_max),
                    "orientation_visibility": min(values),
                    "orientation_message": "RKNN side chain fallback ok",
                    "rknn_orientation_mode": "side_chain_fallback",
                    "rknn_orientation_relaxed": True,
                    "rknn_orientation_chains": checked_chains,
                }

    return {
        "pose_detected": True,
        "orientation_ok": False,
        "side_view_ok": False,
        "front_view_ok": False,
        "orientation_ratio": None,
        "front_view_ratio_min": float(front_view_ratio_min),
        "side_view_ratio_max": float(side_view_ratio_max),
        "orientation_visibility": best_visibility,
        "orientation_message": "RKNN torso points incomplete and no complete side chain",
        "rknn_orientation_mode": "insufficient_torso",
        "rknn_orientation_relaxed": True,
        "rknn_orientation_chains": checked_chains,
    }


@dataclass
class PersonSelectionState:
    previous_center: tuple[float, float] | None = None


@dataclass
class PoseStabilizerState:
    previous_rehab: dict[str, dict[str, Any]] | None = None
    missing_counts: dict[str, int] | None = None
    jump_candidates: dict[str, dict[str, Any]] | None = None
    jump_counts: dict[str, int] | None = None
    locked_side: str | None = None
    candidate_side: str | None = None
    candidate_count: int = 0
    last_side_mode: str = "auto"


@dataclass
class DisplayStabilizerState:
    previous_keypoints: dict[str, dict[str, Any]] | None = None
    missing_counts: dict[str, int] | None = None
    jump_candidates: dict[str, dict[str, Any]] | None = None
    jump_counts: dict[str, int] | None = None
    previous_bbox: list[float] | None = None
    bbox_missing_count: int = 0
    bbox_jump_candidate: list[float] | None = None
    bbox_jump_count: int = 0
    missing_since: dict[str, float] | None = None
    jump_started_at: dict[str, float] | None = None
    bbox_missing_since: float | None = None
    bbox_jump_started_at: float | None = None
    last_reliable_at: float | None = None


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
        self.state = PoseStabilizerState(missing_counts={}, jump_candidates={}, jump_counts={})

    def reset(self) -> None:
        self.state = PoseStabilizerState(missing_counts={}, jump_candidates={}, jump_counts={})

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
            "jump_pending": [],
            "jump_recovery_accepted": [],
            "jump_reject_counts": {},
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
        jump_consistency_threshold = max(0.04, body_scale * 0.30)
        missing_counts = self.state.missing_counts or {}
        jump_candidates = self.state.jump_candidates or {}
        jump_counts = self.state.jump_counts or {}
        stabilized: dict[str, dict[str, Any]] = {}
        for name in REHAB_REQUIRED_NAMES:
            raw_point = dict(working.get(name, {}))
            prev_point = previous.get(name)
            x = _as_float(raw_point.get("x"))
            y = _as_float(raw_point.get("y"))
            visibility = _as_float(raw_point.get("visibility")) or 0.0
            missing = x is None or y is None or visibility <= 0.01
            weak = not missing and visibility < visibility_threshold

            if (missing or weak) and isinstance(prev_point, dict) and missing_counts.get(name, 0) < self.max_hold_frames:
                jump_candidates.pop(name, None)
                jump_counts.pop(name, None)
                point = dict(prev_point) if missing else dict(raw_point)
                prev_x = _as_float(prev_point.get("x"))
                prev_y = _as_float(prev_point.get("y"))
                if not missing and prev_x is not None and prev_y is not None and x is not None and y is not None:
                    point["x"] = prev_x * (1.0 - self.low_conf_alpha) + float(x) * self.low_conf_alpha
                    point["y"] = prev_y * (1.0 - self.low_conf_alpha) + float(y) * self.low_conf_alpha
                point["visibility"] = max(visibility, (_as_float(prev_point.get("visibility")) or 0.0) * 0.82)
                stabilized[name] = point
                missing_counts[name] = missing_counts.get(name, 0) + 1
                diagnostics["held_keypoints"].append(name)
                continue

            if missing:
                jump_candidates.pop(name, None)
                jump_counts.pop(name, None)
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
                        candidate = jump_candidates.get(name)
                        candidate_x = _as_float(candidate.get("x")) if isinstance(candidate, dict) else None
                        candidate_y = _as_float(candidate.get("y")) if isinstance(candidate, dict) else None
                        consistent_jump = bool(
                            candidate_x is not None
                            and candidate_y is not None
                            and math.hypot(float(x) - candidate_x, float(y) - candidate_y) <= jump_consistency_threshold
                        )
                        jump_counts[name] = jump_counts.get(name, 0) + 1 if consistent_jump else 1
                        jump_candidates[name] = dict(raw_point)
                        diagnostics["jump_reject_counts"][name] = jump_counts[name]
                        if consistent_jump and jump_counts[name] >= 2:
                            jump_candidates.pop(name, None)
                            jump_counts.pop(name, None)
                            diagnostics["jump_recovery_accepted"].append(name)
                            stabilized[name] = raw_point
                            continue
                        point = dict(prev_point)
                        point["visibility"] = max(visibility, (_as_float(prev_point.get("visibility")) or 0.0) * 0.85)
                        stabilized[name] = point
                        diagnostics["jump_rejected"].append(name)
                        diagnostics["jump_pending"].append(name)
                        continue
                    jump_candidates.pop(name, None)
                    jump_counts.pop(name, None)
                    base_alpha = self.alpha if visibility >= 0.50 else self.low_conf_alpha
                    motion_ratio = min(1.0, distance / max(jump_threshold, 1e-6))
                    alpha = min(0.85, base_alpha + 0.35 * motion_ratio)
                    raw_point["x"] = prev_x * (1.0 - alpha) + float(x) * alpha
                    raw_point["y"] = prev_y * (1.0 - alpha) + float(y) * alpha
            stabilized[name] = raw_point

        self.state.previous_rehab = {name: dict(point) for name, point in stabilized.items()}
        self.state.missing_counts = missing_counts
        self.state.jump_candidates = jump_candidates
        self.state.jump_counts = jump_counts
        diagnostics["locked_side"] = self.state.locked_side
        return stabilized, diagnostics


class Coco17DisplayStabilizer:
    """Display-only smoothing for all COCO-17 points and the person box."""

    def __init__(
        self,
        *,
        alpha: float = 0.50,
        low_conf_alpha: float = 0.30,
        jump_scale: float = 0.35,
        max_hold_frames: int = 4,
        jump_confirm_frames: int = 2,
        bbox_alpha: float = 0.45,
        bbox_hold_frames: int = 6,
        hold_seconds: float | None = None,
        bbox_hold_seconds: float | None = None,
        jump_confirm_seconds: float | None = None,
        disappear_threshold_ratio: float = 0.65,
        bbox_iou_jump_threshold: float = 0.35,
        max_stale_seconds: float = 0.50,
    ) -> None:
        self.alpha = float(alpha)
        self.low_conf_alpha = float(low_conf_alpha)
        self.jump_scale = float(jump_scale)
        self.max_hold_frames = max(0, int(max_hold_frames))
        self.jump_confirm_frames = max(1, int(jump_confirm_frames))
        self.bbox_alpha = float(bbox_alpha)
        self.bbox_hold_frames = max(0, int(bbox_hold_frames))
        self.hold_seconds = max(0.0, float(hold_seconds)) if hold_seconds is not None else None
        self.bbox_hold_seconds = max(0.0, float(bbox_hold_seconds)) if bbox_hold_seconds is not None else None
        self.jump_confirm_seconds = max(0.0, float(jump_confirm_seconds)) if jump_confirm_seconds is not None else None
        self.disappear_threshold_ratio = min(1.0, max(0.1, float(disappear_threshold_ratio)))
        self.bbox_iou_jump_threshold = min(0.95, max(0.0, float(bbox_iou_jump_threshold)))
        self.max_stale_seconds = max(0.05, float(max_stale_seconds))
        self.state = DisplayStabilizerState(missing_counts={}, jump_candidates={}, jump_counts={}, missing_since={}, jump_started_at={})

    def reset(self) -> None:
        self.state = DisplayStabilizerState(missing_counts={}, jump_candidates={}, jump_counts={}, missing_since={}, jump_started_at={})

    def stabilize(
        self,
        detection: dict[str, Any] | None,
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[dict[str, dict[str, Any]], list[float] | None, dict[str, Any]]:
        raw = coco17_detection_to_display(detection, frame_width, frame_height) if detection else {}
        previous = self.state.previous_keypoints or {}
        missing_counts = self.state.missing_counts or {}
        jump_candidates = self.state.jump_candidates or {}
        jump_counts = self.state.jump_counts or {}
        missing_since = self.state.missing_since or {}
        jump_started_at = self.state.jump_started_at or {}
        now = time.monotonic()
        reliable_now = any(
            (_as_float(point.get("visibility")) or 0.0) >= _display_visibility_threshold(name)
            for name, point in raw.items()
        )
        if reliable_now:
            self.state.last_reliable_at = now
        stale_pose = self.state.last_reliable_at is not None and now - self.state.last_reliable_at > self.max_stale_seconds
        body_scale = _display_body_scale(raw, previous)
        jump_threshold = max(0.06, body_scale * self.jump_scale)
        consistency_threshold = max(0.035, jump_threshold * 0.45)
        stabilized: dict[str, dict[str, Any]] = {}
        held: list[str] = []
        jump_pending: list[str] = []
        jump_recovered: list[str] = []

        for name in COCO17_NAMES:
            appear_threshold = _display_visibility_threshold(name)
            raw_point = dict(raw.get(name, {}))
            prev_point = previous.get(name)
            previous_visible = isinstance(prev_point, dict) and (_as_float(prev_point.get("visibility")) or 0.0) >= 0.05
            threshold = appear_threshold * self.disappear_threshold_ratio if previous_visible else appear_threshold
            x = _as_float(raw_point.get("x"))
            y = _as_float(raw_point.get("y"))
            visibility = _as_float(raw_point.get("visibility")) or 0.0
            missing = x is None or y is None or visibility < threshold

            if stale_pose:
                jump_candidates.pop(name, None)
                jump_counts.pop(name, None)
                jump_started_at.pop(name, None)
                missing_since[name] = now
                missing_counts[name] = missing_counts.get(name, 0) + 1
                stabilized[name] = _empty_display_point()
                continue

            if missing:
                jump_candidates.pop(name, None)
                jump_counts.pop(name, None)
                jump_started_at.pop(name, None)
                missing_since.setdefault(name, now)
                within_time_hold = self.hold_seconds is not None and now - missing_since[name] <= self.hold_seconds
                within_frame_hold = self.hold_seconds is None and missing_counts.get(name, 0) < self.max_hold_frames
                if isinstance(prev_point, dict) and (within_time_hold or within_frame_hold):
                    point = dict(prev_point)
                    point["visibility"] = max(0.05, (_as_float(prev_point.get("visibility")) or threshold) * 0.86)
                    stabilized[name] = point
                    missing_counts[name] = missing_counts.get(name, 0) + 1
                    held.append(name)
                else:
                    stabilized[name] = _empty_display_point()
                    missing_counts[name] = missing_counts.get(name, 0) + 1
                continue

            missing_counts[name] = 0
            missing_since.pop(name, None)
            if isinstance(prev_point, dict):
                prev_x = _as_float(prev_point.get("x"))
                prev_y = _as_float(prev_point.get("y"))
                if prev_x is not None and prev_y is not None:
                    distance = math.hypot(float(x) - prev_x, float(y) - prev_y)
                    if distance > jump_threshold:
                        candidate = jump_candidates.get(name)
                        candidate_x = _as_float(candidate.get("x")) if isinstance(candidate, dict) else None
                        candidate_y = _as_float(candidate.get("y")) if isinstance(candidate, dict) else None
                        consistent = bool(
                            candidate_x is not None
                            and candidate_y is not None
                            and math.hypot(float(x) - candidate_x, float(y) - candidate_y) <= consistency_threshold
                        )
                        jump_counts[name] = jump_counts.get(name, 0) + 1 if consistent else 1
                        jump_candidates[name] = dict(raw_point)
                        jump_started_at.setdefault(name, now)
                        timed_out = self.jump_confirm_seconds is not None and now - jump_started_at[name] >= self.jump_confirm_seconds
                        if (consistent and jump_counts[name] >= self.jump_confirm_frames) or timed_out:
                            jump_candidates.pop(name, None)
                            jump_counts.pop(name, None)
                            jump_started_at.pop(name, None)
                            stabilized[name] = raw_point
                            jump_recovered.append(name)
                            continue
                        point = dict(prev_point)
                        point["visibility"] = max(visibility, (_as_float(prev_point.get("visibility")) or threshold) * 0.90)
                        stabilized[name] = point
                        jump_pending.append(name)
                        continue
                    jump_candidates.pop(name, None)
                    jump_counts.pop(name, None)
                    jump_started_at.pop(name, None)
                    base_alpha = self.alpha if visibility >= 0.50 else self.low_conf_alpha
                    motion_ratio = min(1.0, distance / max(jump_threshold, 1e-6))
                    alpha = min(0.82, base_alpha + 0.25 * motion_ratio)
                    raw_point["x"] = prev_x * (1.0 - alpha) + float(x) * alpha
                    raw_point["y"] = prev_y * (1.0 - alpha) + float(y) * alpha
            stabilized[name] = raw_point

        if stale_pose:
            self.state.previous_bbox = None
            self.state.bbox_missing_since = None
            self.state.bbox_missing_count = 0
            self.state.bbox_jump_candidate = None
            self.state.bbox_jump_count = 0
            self.state.bbox_jump_started_at = None
        bbox, bbox_diag = self._stabilize_bbox(
            None if stale_pose else detection.get("bbox") if isinstance(detection, dict) else None,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.state.previous_keypoints = {name: dict(point) for name, point in stabilized.items()}
        self.state.missing_counts = missing_counts
        self.state.jump_candidates = jump_candidates
        self.state.jump_counts = jump_counts
        self.state.missing_since = missing_since
        self.state.jump_started_at = jump_started_at
        visible_count = sum(
            1
            for point in stabilized.values()
            if _as_float(point.get("x")) is not None
            and _as_float(point.get("y")) is not None
            and (_as_float(point.get("visibility")) or 0.0) >= 0.05
        )
        return stabilized, bbox, {
            "display_keypoint_count": visible_count,
            "display_held_keypoints": held,
            "display_jump_pending": jump_pending,
            "display_jump_recovered": jump_recovered,
            "display_pose_stale": stale_pose,
            **bbox_diag,
        }

    def _stabilize_bbox(
        self,
        bbox_value: object,
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[list[float] | None, dict[str, Any]]:
        raw_bbox = _valid_bbox(bbox_value, frame_width, frame_height)
        previous = self.state.previous_bbox
        now = time.monotonic()
        if raw_bbox is None:
            self.state.bbox_jump_candidate = None
            self.state.bbox_jump_count = 0
            self.state.bbox_jump_started_at = None
            if self.state.bbox_missing_since is None:
                self.state.bbox_missing_since = now
            within_time_hold = self.bbox_hold_seconds is not None and now - self.state.bbox_missing_since <= self.bbox_hold_seconds
            within_frame_hold = self.bbox_hold_seconds is None and self.state.bbox_missing_count < self.bbox_hold_frames
            if previous is not None and (within_time_hold or within_frame_hold):
                self.state.bbox_missing_count += 1
                return list(previous), {"display_bbox_held": True, "display_bbox_jump_pending": False}
            self.state.bbox_missing_count += 1
            self.state.previous_bbox = None
            return None, {"display_bbox_held": False, "display_bbox_jump_pending": False}

        self.state.bbox_missing_count = 0
        self.state.bbox_missing_since = None
        if previous is None:
            self.state.bbox_jump_candidate = None
            self.state.bbox_jump_count = 0
            self.state.bbox_jump_started_at = None
            self.state.previous_bbox = list(raw_bbox)
            return list(raw_bbox), {"display_bbox_held": False, "display_bbox_jump_pending": False}

        previous_center = _bbox_center(previous)
        raw_center = _bbox_center(raw_bbox)
        center_distance = math.hypot(raw_center[0] - previous_center[0], raw_center[1] - previous_center[1])
        jump_threshold = max(24.0, math.hypot(frame_width, frame_height) * 0.12)
        bbox_iou = _bbox_iou(previous, raw_bbox)
        if center_distance > jump_threshold or bbox_iou < self.bbox_iou_jump_threshold:
            candidate = self.state.bbox_jump_candidate
            candidate_center = _bbox_center(candidate) if candidate is not None else None
            consistent = bool(
                candidate_center is not None
                and math.hypot(raw_center[0] - candidate_center[0], raw_center[1] - candidate_center[1]) <= jump_threshold * 0.45
            )
            self.state.bbox_jump_count = self.state.bbox_jump_count + 1 if consistent else 1
            self.state.bbox_jump_candidate = list(raw_bbox)
            if self.state.bbox_jump_started_at is None:
                self.state.bbox_jump_started_at = now
            timed_out = self.jump_confirm_seconds is not None and now - self.state.bbox_jump_started_at >= self.jump_confirm_seconds
            if not timed_out and (not consistent or self.state.bbox_jump_count < self.jump_confirm_frames):
                return list(previous), {"display_bbox_held": True, "display_bbox_jump_pending": True, "display_bbox_iou": bbox_iou}

        self.state.bbox_jump_candidate = None
        self.state.bbox_jump_count = 0
        self.state.bbox_jump_started_at = None
        smoothed = [old * (1.0 - self.bbox_alpha) + new * self.bbox_alpha for old, new in zip(previous, raw_bbox)]
        self.state.previous_bbox = smoothed
        return list(smoothed), {"display_bbox_held": False, "display_bbox_jump_pending": False, "display_bbox_iou": bbox_iou}


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
    display_stabilizer: Coco17DisplayStabilizer | None = None,
) -> dict[str, Any]:
    selection = selector.select(detections, frame_width, frame_height)
    detection = selection["detection"]
    if display_stabilizer is not None:
        display_keypoints, display_bbox, display_diagnostics = display_stabilizer.stabilize(
            detection if isinstance(detection, dict) else None,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    else:
        display_keypoints = coco17_detection_to_display(detection, frame_width, frame_height) if isinstance(detection, dict) else {}
        display_bbox = _valid_bbox(detection.get("bbox"), frame_width, frame_height) if isinstance(detection, dict) else None
        display_diagnostics = {
            "display_keypoint_count": sum(
                1 for point in display_keypoints.values() if (_as_float(point.get("visibility")) or 0.0) >= 0.05
            ),
            "display_held_keypoints": [],
            "display_jump_pending": [],
            "display_jump_recovered": [],
            "display_bbox_held": False,
            "display_bbox_jump_pending": False,
        }
    if detection is None:
        if stabilizer is not None:
            held_rehab, stabilization = stabilizer.stabilize(
                {},
                side_mode=side_mode,
                action_config=action_config,
                visibility_threshold=visibility_threshold,
            )
            effective_side_mode = side_mode
            if stabilization.get("locked_side") in {"left", "right"}:
                effective_side_mode = str(stabilization["locked_side"])
            left_result = _compute_side_result(held_rehab, "left", action_config, visibility_threshold)
            right_result = _compute_side_result(held_rehab, "right", action_config, visibility_threshold)
            selected_side, held_result = _choose_side(effective_side_mode, left_result, right_result)
            if held_result.get("quality_ok") and held_result.get("selected_target_angle") is not None:
                selected_result = {
                    **held_result,
                    "valid": True,
                    "side": selected_side,
                    "target_joint": f"{selected_side}_{_target_joint_name(action_config)}",
                    "selected_source": "rknn_2d_held",
                    "included_angle_3d": None,
                    "target_angle_3d": None,
                    "flexion_angle_3d": None,
                    "person_count": 0,
                    "selected_person_score": None,
                    "selected_person_reason": "短暂漏检，沿用稳定姿态",
                    "multi_person_warning": False,
                    "selection_stable": True,
                    "locked_side": stabilization.get("locked_side"),
                    "side_lock_reason": stabilization.get("side_lock_reason"),
                    "pose_stabilized": True,
                    "pose_dropout_held": True,
                    "held_keypoints": stabilization.get("held_keypoints") or [],
                    "jump_rejected": stabilization.get("jump_rejected") or [],
                    "jump_pending": stabilization.get("jump_pending") or [],
                    "jump_recovery_accepted": stabilization.get("jump_recovery_accepted") or [],
                    "jump_reject_counts": stabilization.get("jump_reject_counts") or {},
                    "side_switch_blocked": bool(stabilization.get("side_switch_blocked")),
                    "person_box_quality_ok": True,
                    "person_box_height_ratio": None,
                    "person_box_area_ratio": None,
                    "person_box_quality_message": "短暂漏检保持",
                    **display_diagnostics,
                }
                return {
                    "selected_rule": _rule_for_side(selected_side, action_config),
                    "selected_result": selected_result,
                    "orientation_rehab_keypoints": held_rehab,
                    "rehab_keypoints": held_rehab,
                    "keypoints": _compact_keypoints(held_rehab, selected_side, action_config),
                    "selected_detection": None,
                    "display_keypoints": display_keypoints,
                    "display_bbox": display_bbox,
                }
        payload = _invalid_payload(selection, "未检测到训练者")
        payload["selected_result"].update(display_diagnostics)
        payload["display_keypoints"] = display_keypoints
        payload["display_bbox"] = display_bbox
        return payload

    raw_rehab_keypoints = coco17_detection_to_rehab(detection, frame_width, frame_height)
    rehab_keypoints = {name: dict(point) for name, point in raw_rehab_keypoints.items()}
    bbox_quality = _bbox_quality(detection, frame_width, frame_height)
    stabilization: dict[str, Any] = {
        "pose_stabilized": False,
        "held_keypoints": [],
        "jump_rejected": [],
        "jump_pending": [],
        "jump_recovery_accepted": [],
        "jump_reject_counts": {},
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
        "jump_pending": stabilization.get("jump_pending") or [],
        "jump_recovery_accepted": stabilization.get("jump_recovery_accepted") or [],
        "jump_reject_counts": stabilization.get("jump_reject_counts") or {},
        "side_switch_blocked": bool(stabilization.get("side_switch_blocked")),
        "person_box_quality_ok": bbox_quality["ok"],
        "person_box_height_ratio": bbox_quality["height_ratio"],
        "person_box_area_ratio": bbox_quality["area_ratio"],
        "person_box_quality_message": bbox_quality["message"],
        **display_diagnostics,
    }
    if not bbox_quality["ok"]:
        selected_result["valid"] = False
        selected_result["quality_ok"] = False
        selected_result["quality_message"] = bbox_quality["message"]
        selected_result["missing_keypoints"] = sorted(set(list(missing) + ["person_box"]))
        for key in (
            "included_angle_2d",
            "target_angle_2d",
            "flexion_angle_2d",
            "selected_included_angle",
            "selected_target_angle",
            "selected_flexion_angle",
        ):
            selected_result[key] = None
    return {
        "selected_rule": _rule_for_side(selected_side, action_config),
        "selected_result": selected_result,
        "orientation_rehab_keypoints": raw_rehab_keypoints,
        "rehab_keypoints": rehab_keypoints,
        "keypoints": _compact_keypoints(rehab_keypoints, selected_side, action_config),
        "selected_detection": detection,
        "display_keypoints": display_keypoints,
        "display_bbox": display_bbox,
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


def coco17_detection_to_display(
    detection: dict[str, Any] | None,
    frame_width: int,
    frame_height: int,
) -> dict[str, dict[str, Any]]:
    keypoints = detection.get("keypoints") or [] if isinstance(detection, dict) else []
    display: dict[str, dict[str, Any]] = {}
    for name in COCO17_NAMES:
        index = COCO17_INDEX[name]
        point = keypoints[index] if index < len(keypoints) else None
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            display[name] = _empty_display_point()
            continue
        display[name] = {
            "x": _normalize(_as_float(point[0]), frame_width),
            "y": _normalize(_as_float(point[1]), frame_height),
            "z": None,
            "z_valid": False,
            "visibility": _as_float(point[2]) or 0.0 if len(point) > 2 else 0.0,
        }
    return display


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


def _bbox_quality(detection: dict[str, Any], frame_width: int, frame_height: int) -> dict[str, Any]:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return {"ok": False, "height_ratio": 0.0, "area_ratio": 0.0, "message": "RKNN invalid person box"}
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_width * frame_height))
    height_ratio = height / max(float(frame_height), 1.0)
    area_ratio = (width * height) / frame_area
    min_height = _env_float("RKNN_MIN_PERSON_BOX_HEIGHT_RATIO", 0.42)
    min_area = _env_float("RKNN_MIN_PERSON_BOX_AREA_RATIO", 0.08)
    if height_ratio < min_height:
        return {
            "ok": False,
            "height_ratio": round(height_ratio, 3),
            "area_ratio": round(area_ratio, 3),
            "message": "RKNN person box too short; keep full body in view",
        }
    if area_ratio < min_area:
        return {
            "ok": False,
            "height_ratio": round(height_ratio, 3),
            "area_ratio": round(area_ratio, 3),
            "message": "RKNN person box too small; move camera farther or center the body",
        }
    return {
        "ok": True,
        "height_ratio": round(height_ratio, 3),
        "area_ratio": round(area_ratio, 3),
        "message": "RKNN person box OK",
    }


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


def _display_visibility_threshold(name: str) -> float:
    return float(COCO17_DISPLAY_THRESHOLDS.get(name, 0.16))


def _empty_display_point() -> dict[str, Any]:
    return {"x": None, "y": None, "z": None, "z_valid": False, "visibility": 0.0}


def _display_body_scale(
    current: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]],
) -> float:
    source = current or previous
    distances: list[float] = []
    for start, end in (
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
    ):
        start_point = source.get(start)
        end_point = source.get(end)
        if not isinstance(start_point, dict) or not isinstance(end_point, dict):
            continue
        start_x = _as_float(start_point.get("x"))
        start_y = _as_float(start_point.get("y"))
        end_x = _as_float(end_point.get("x"))
        end_y = _as_float(end_point.get("y"))
        if None not in {start_x, start_y, end_x, end_y}:
            distances.append(math.hypot(float(end_x) - float(start_x), float(end_y) - float(start_y)))
    return max(0.20, sum(distances) / len(distances)) if distances else 0.35


def _valid_bbox(value: object, frame_width: int, frame_height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(frame_width - 1), x1))
    y1 = max(0.0, min(float(frame_height - 1), y1))
    x2 = max(0.0, min(float(frame_width - 1), x2))
    y2 = max(0.0, min(float(frame_height - 1), y2))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0)


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-6 else 0.0


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)
