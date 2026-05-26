from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_ACTIONS_PATH = PACKAGE_DIR / "actions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "prescription" / "docs" / "results"


@dataclass(frozen=True)
class TimeSeries:
    action_id: str
    action: dict[str, Any]
    times: list[float]
    angles: list[float]
    patient_id: str
    action_name: str
    source_path: Path | None = None


def load_actions(path: Path = DEFAULT_ACTIONS_PATH) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest_json(directory: Path = DEFAULT_RESULTS_DIR) -> Path:
    files = [path for path in directory.glob("*.json") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"没有找到处方 JSON：{directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_action_id(payload: dict[str, Any], actions: dict[str, dict[str, Any]]) -> str:
    candidates = [
        payload.get("action_id"),
        _safe_get(payload, "runtime_meta", "action_id"),
        payload.get("action_name"),
    ]
    normalized_aliases: dict[str, str] = {}
    for action_id, action in actions.items():
        normalized_aliases[_normalize(action_id)] = action_id
        for alias in action.get("aliases", []):
            normalized_aliases[_normalize(str(alias))] = action_id

    for candidate in candidates:
        key = _normalize(str(candidate or ""))
        if key in normalized_aliases:
            return normalized_aliases[key]

    raise ValueError(f"未知动作类型：{payload.get('action_name') or payload.get('action_id') or '空'}")


def extract_time_series(
    payload: dict[str, Any],
    *,
    actions: dict[str, dict[str, Any]] | None = None,
    source_path: Path | None = None,
) -> TimeSeries:
    action_map = actions or load_actions()
    action_id = resolve_action_id(payload, action_map)
    action = action_map[action_id]
    frames = payload.get("template_frames")
    if not isinstance(frames, list):
        frames = []

    times: list[float] = []
    angles: list[float] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        angle = _frame_angle(frame)
        if angle is None:
            angle = _angle_from_keypoints(frame, action)
        if angle is None:
            continue
        time_value = _as_float(frame.get("relative_time"))
        times.append(time_value if time_value is not None else float(index))
        angles.append(angle)

    if not angles:
        clinical = payload.get("clinical_baseline")
        if isinstance(clinical, dict):
            min_angle = _as_float(
                clinical.get("min_target_angle")
                if clinical.get("min_target_angle") is not None
                else clinical.get("min_knee_flexion_angle")
            )
            max_angle = _as_float(
                clinical.get("max_target_angle")
                if clinical.get("max_target_angle") is not None
                else clinical.get("max_knee_flexion_angle")
            )
            if min_angle is not None and max_angle is not None:
                times = [0.0, _as_float(clinical.get("duration_seconds")) or 1.0]
                angles = [min_angle, max_angle]

    return TimeSeries(
        action_id=action_id,
        action=action,
        times=times,
        angles=angles,
        patient_id=str(payload.get("patient_id") or "患者"),
        action_name=str(payload.get("action_name") or action.get("display_name") or action_id),
        source_path=source_path,
    )


def evaluate_single(series: TimeSeries) -> dict[str, Any]:
    if not series.angles:
        return {
            "ok": False,
            "quality_level": "invalid",
            "action_id": series.action_id,
            "action_name": series.action_name,
            "feedback_text": "没有可用角度序列，请检查关键点可见度后重新录制。",
            "motor_command": None,
        }

    min_angle = min(series.angles)
    max_angle = max(series.angles)
    rom = max_angle - min_angle
    duration = max(series.times) - min(series.times) if len(series.times) >= 2 else 0.0
    tut = calculate_tut(series.times, series.angles, series.action["target_angle_range"])
    level = quality_level(rom, tut, series.action)
    motor_command = motor_for_level(level)
    feedback_text = build_feedback_text(
        series.action,
        series.action_name,
        level,
        min_angle,
        max_angle,
        rom,
        tut,
    )
    return {
        "ok": True,
        "action_id": series.action_id,
        "action_name": series.action_name,
        "patient_id": series.patient_id,
        "frame_count": len(series.angles),
        "duration_seconds": duration,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "rom": rom,
        "target_angle_range": series.action["target_angle_range"],
        "tut_seconds": tut,
        "target_tut_seconds": series.action["target_tut_seconds"],
        "quality_level": level,
        "feedback_text": feedback_text,
        "motor_command": motor_command,
    }


def evaluate_pair(template: TimeSeries, attempt: TimeSeries) -> dict[str, Any]:
    if template.action_id != attempt.action_id:
        raise ValueError(f"模板动作 {template.action_id} 与本次动作 {attempt.action_id} 不一致。")

    result = evaluate_single(attempt)
    if not result.get("ok"):
        return result

    if not template.angles or not attempt.angles:
        result.update({"dtw_distance": None, "dtw_level": "invalid"})
        return result

    distance = dtw_distance(template.angles, attempt.angles)
    thresholds = attempt.action["dtw_thresholds"]
    if distance <= thresholds["good"]:
        dtw_level = "good"
    elif distance <= thresholds["warn"]:
        dtw_level = "warn"
    else:
        dtw_level = "bad"

    result["dtw_distance"] = distance
    result["dtw_level"] = dtw_level
    if dtw_level == "bad":
        result["quality_level"] = "warn" if result["quality_level"] == "good" else result["quality_level"]
        result["motor_command"] = motor_for_level(result["quality_level"])
        result["feedback_text"] += f" 与模板曲线差异较大，DTW 距离 {distance:.1f}。"
    else:
        result["feedback_text"] += f" 与模板曲线匹配度{_dtw_level_text(dtw_level)}，DTW 距离 {distance:.1f}。"
    return result


def replay_compare(template: TimeSeries, attempt: TimeSeries, *, tolerance_degrees: float = 8.0) -> dict[str, Any]:
    if template.action_id != attempt.action_id:
        raise ValueError(f"模板动作 {template.action_id} 与患者动作 {attempt.action_id} 不一致。")
    if not template.angles:
        raise ValueError("标准动作没有可用角度序列。")
    if not attempt.angles:
        raise ValueError("患者动作没有可用角度序列。")

    template_duration = _series_duration(template.times)
    attempt_duration = _series_duration(attempt.times)
    rows: list[dict[str, Any]] = []

    for index, (time_value, attempt_angle) in enumerate(zip(attempt.times, attempt.angles)):
        progress = _progress(time_value, attempt.times, attempt_duration)
        template_time = template.times[0] + progress * template_duration
        template_angle = interpolate_angle(template.times, template.angles, template_time)
        diff = attempt_angle - template_angle
        abs_diff = abs(diff)
        rows.append(
            {
                "frame_index": index,
                "attempt_time": time_value,
                "progress": progress,
                "template_time": template_time,
                "template_angle": template_angle,
                "attempt_angle": attempt_angle,
                "diff": diff,
                "abs_diff": abs_diff,
                "status": "match" if abs_diff <= tolerance_degrees else "off",
                "hint": frame_hint(diff, tolerance_degrees),
            }
        )

    aggregate = evaluate_pair(template, attempt)
    mean_abs_error = sum(row["abs_diff"] for row in rows) / len(rows)
    max_abs_error = max(row["abs_diff"] for row in rows)
    off_frame_count = sum(1 for row in rows if row["status"] == "off")
    aggregate.update(
        {
            "replay": rows,
            "replay_frame_count": len(rows),
            "replay_tolerance_degrees": tolerance_degrees,
            "mean_abs_error": mean_abs_error,
            "max_abs_error": max_abs_error,
            "off_frame_count": off_frame_count,
            "off_frame_ratio": off_frame_count / len(rows),
        }
    )
    aggregate["feedback_text"] += (
        f" 逐帧 replay 平均误差 {mean_abs_error:.1f} 度，"
        f"最大误差 {max_abs_error:.1f} 度，超出容差帧 {off_frame_count}/{len(rows)}。"
    )
    return aggregate


def calculate_tut(times: list[float], angles: list[float], target_range: list[float]) -> float:
    if len(times) < 2 or len(times) != len(angles):
        return 0.0
    low, high = float(target_range[0]), float(target_range[1])
    total = 0.0
    for index in range(1, len(times)):
        previous_angle = angles[index - 1]
        current_angle = angles[index]
        if low <= previous_angle <= high and low <= current_angle <= high:
            total += max(0.0, times[index] - times[index - 1])
    return total


def dtw_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return math.inf
    previous = [math.inf] * (len(right) + 1)
    previous[0] = 0.0
    for left_value in left:
        current = [math.inf] * (len(right) + 1)
        for j, right_value in enumerate(right, start=1):
            cost = abs(left_value - right_value)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1] / max(len(left), len(right))


def interpolate_angle(times: list[float], angles: list[float], target_time: float) -> float:
    if len(times) != len(angles) or not times:
        raise ValueError("时间序列和角度序列长度不一致。")
    if target_time <= times[0]:
        return angles[0]
    if target_time >= times[-1]:
        return angles[-1]
    for index in range(1, len(times)):
        left_time = times[index - 1]
        right_time = times[index]
        if left_time <= target_time <= right_time:
            span = right_time - left_time
            if span <= 1e-8:
                return angles[index]
            ratio = (target_time - left_time) / span
            return angles[index - 1] + ratio * (angles[index] - angles[index - 1])
    return angles[-1]


def frame_hint(diff: float, tolerance_degrees: float) -> str:
    if abs(diff) <= tolerance_degrees:
        return "动作角度接近标准。"
    if diff < 0:
        return "当前角度低于标准，请把动作幅度做大一些。"
    return "当前角度高于标准，请放慢并控制到模板幅度。"


def quality_level(rom: float, tut: float, action: dict[str, Any]) -> str:
    thresholds = action["rom_thresholds"]
    if rom >= thresholds["good"] and tut >= float(action["target_tut_seconds"]):
        return "good"
    if rom >= thresholds["warn"]:
        return "warn"
    return "bad"


def motor_for_level(level: str) -> str | None:
    if level == "bad":
        return "long"
    if level == "warn":
        return "short"
    return None


def build_feedback_text(
    action: dict[str, Any],
    action_name: str,
    level: str,
    min_angle: float,
    max_angle: float,
    rom: float,
    tut: float,
) -> str:
    target_low, target_high = action["target_angle_range"]
    target_tut = float(action["target_tut_seconds"])
    tut_gap = max(0.0, target_tut - tut)
    base = action.get("feedback", {}).get(level, "")
    return (
        f"{action_name}评估完成：最小角度 {min_angle:.1f} 度，最大角度 {max_angle:.1f} 度，"
        f"活动范围 {rom:.1f} 度；目标角度区间 {target_low:.0f}-{target_high:.0f} 度内"
        f"有效保持 {tut:.1f} 秒，目标 {target_tut:.1f} 秒，还差 {tut_gap:.1f} 秒。{base}"
    )


def _frame_angle(frame: dict[str, Any]) -> float | None:
    for key in (
        "target_angle_smoothed",
        "target_angle_raw",
        "selected_flexion_angle_smoothed",
        "selected_flexion_angle_raw",
        "selected_flexion_angle",
    ):
        value = _as_float(frame.get(key))
        if value is not None:
            return value
    return None


def _angle_from_keypoints(frame: dict[str, Any], action: dict[str, Any]) -> float | None:
    keypoints = frame.get("keypoints")
    if not isinstance(keypoints, dict):
        return None
    points = []
    for name in action["point_names"]:
        point = keypoints.get(name)
        if not isinstance(point, dict):
            return None
        x = _as_float(point.get("x"))
        y = _as_float(point.get("y"))
        z = _as_float(point.get("z"))
        if x is None or y is None:
            return None
        points.append((x, y) if z is None else (x, y, z))
    included = calculate_angle(points)
    if included is None:
        return None
    if action.get("angle_kind") == "flexion":
        return max(0.0, min(180.0, 180.0 - included))
    return included


def calculate_angle(points: list[tuple[float, ...]]) -> float | None:
    if len(points) != 3:
        return None
    a, b, c = points
    if len(a) != len(b) or len(b) != len(c):
        return None
    ba = [a[i] - b[i] for i in range(len(a))]
    bc = [c[i] - b[i] for i in range(len(c))]
    dot_product = sum(ba[i] * bc[i] for i in range(len(ba)))
    ba_length = math.sqrt(sum(value * value for value in ba))
    bc_length = math.sqrt(sum(value * value for value in bc))
    if ba_length < 1e-8 or bc_length < 1e-8:
        return None
    cos_value = max(-1.0, min(1.0, dot_product / (ba_length * bc_length)))
    return math.degrees(math.acos(cos_value))


def _safe_get(payload: dict[str, Any], object_key: str, value_key: str) -> Any:
    value = payload.get(object_key)
    if isinstance(value, dict):
        return value.get(value_key)
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _series_duration(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    return max(0.0, times[-1] - times[0])


def _progress(time_value: float, times: list[float], duration: float) -> float:
    if duration <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, (time_value - times[0]) / duration))


def _dtw_level_text(level: str) -> str:
    if level == "good":
        return "较好"
    if level == "warn":
        return "一般"
    return "较差"
