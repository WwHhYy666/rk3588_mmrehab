"""Health checks for RKNN doctor templates used by realtime training."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from evaluation.core.action_metrics import extract_metric_sequence


NPU_TEMPLATE_LIMITS: dict[str, dict[str, float]] = {
    "sit_to_stand": {
        "min_rom": 0.35,
        "max_rom": 1.50,
        "return_absolute_tolerance": 0.10,
        "start_geometry_min": 55.0,
        "start_geometry_max": 140.0,
    },
    "standing_hamstring_curl": {
        "min_rom": 45.0,
        "max_rom": 150.0,
        "return_absolute_tolerance": 10.0,
        "start_geometry_min": 145.0,
        "start_geometry_max": 180.0,
    },
    "seated_knee_raise": {
        "min_rom": 0.25,
        "max_rom": 1.20,
        "return_absolute_tolerance": 0.10,
        "start_geometry_min": 45.0,
        "start_geometry_max": 140.0,
    },
}


def validate_template_payload(
    payload: dict[str, Any],
    eval_config: dict[str, Any],
    *,
    pose_backend: str,
) -> dict[str, Any]:
    action_id = str(payload.get("action_id") or eval_config.get("action_id") or "").strip()
    if str(pose_backend or "").strip().lower() != "rknn" or action_id not in NPU_TEMPLATE_LIMITS:
        return {"ok": True, "required": False, "action_id": action_id, "reason": "not_required"}

    limits = dict(NPU_TEMPLATE_LIMITS[action_id])
    configured = eval_config.get("template_validation")
    if isinstance(configured, dict):
        for key in limits:
            if configured.get(key) is not None:
                limits[key] = float(configured[key])
    options = configured if isinstance(configured, dict) else {}
    min_frames = max(1, int(options.get("min_valid_frames", 20)))
    min_duration = float(options.get("min_duration_seconds", 2.0))
    peak_edge_fraction = float(options.get("peak_edge_fraction", 0.15))
    return_ratio = float(options.get("return_rom_tolerance_ratio", 0.20))

    frames = payload.get("template_frames")
    if not isinstance(frames, list):
        frames = []
    errors: list[dict[str, Any]] = []
    try:
        sequence = extract_metric_sequence(frames, eval_config)
    except ValueError as exc:
        return _result(
            action_id,
            limits,
            errors=[{"code": "metric_unavailable", "stage": "metric", "message": f"无法计算动作指标：{exc}"}],
            valid_frames=0,
        )

    values = [float(value) for value in sequence.get("values") or []]
    times = [float(value) for value in sequence.get("frame_times") or []]
    valid_frames = len(values)
    duration = max(times) - min(times) if len(times) >= 2 else 0.0
    if valid_frames < min_frames:
        errors.append({"code": "too_few_frames", "stage": "duration", "message": f"有效帧不足：{valid_frames}，至少需要 {min_frames} 帧"})
    if duration < min_duration:
        errors.append({"code": "duration_too_short", "stage": "duration", "message": f"录制时间过短：{duration:.2f} 秒，至少需要 {min_duration:.1f} 秒"})

    if not values:
        return _result(action_id, limits, errors=errors, valid_frames=valid_frames, duration_seconds=duration)

    window = max(3, min(len(values), int(round(len(values) * peak_edge_fraction))))
    start_value = float(median(values[:window]))
    end_value = float(median(values[-window:]))
    minimum = min(values)
    maximum = max(values)
    rom = maximum - minimum
    peak_index = values.index(maximum)
    peak_fraction = peak_index / max(1, len(values) - 1)
    return_tolerance = max(float(limits["return_absolute_tolerance"]), rom * return_ratio)
    return_error = abs(end_value - start_value)

    if rom < limits["min_rom"]:
        errors.append({"code": "rom_too_low", "stage": "peak", "message": f"动作幅度不足：当前 {rom:.3f}，至少需要 {limits['min_rom']:.3f}"})
    if rom > limits["max_rom"]:
        errors.append({"code": "rom_too_high", "stage": "peak", "message": f"动作幅度异常：当前 {rom:.3f}，允许上限 {limits['max_rom']:.3f}"})
    if peak_fraction < peak_edge_fraction or peak_fraction > 1.0 - peak_edge_fraction:
        errors.append({"code": "peak_at_edge", "stage": "cycle", "message": "峰值出现在录制开头或结尾，请完整录制起始、动作和返回"})
    if return_error > return_tolerance:
        errors.append({"code": "did_not_return", "stage": "return", "message": f"动作结束时未回到起始姿势：偏差 {return_error:.3f}，允许 {return_tolerance:.3f}"})

    start_geometry, end_geometry = _start_end_geometry(frames, peak_edge_fraction)
    geometry_min = limits["start_geometry_min"]
    geometry_max = limits["start_geometry_max"]
    if start_geometry is None or not geometry_min <= start_geometry <= geometry_max:
        errors.append({"code": "invalid_start_pose", "stage": "start", "message": "录制开始时不是该动作要求的起始姿势"})
    if end_geometry is None or not geometry_min <= end_geometry <= geometry_max:
        errors.append({"code": "invalid_end_pose", "stage": "return", "message": "录制结束时没有回到该动作要求的起始姿势"})

    return _result(
        action_id,
        limits,
        errors=errors,
        valid_frames=valid_frames,
        duration_seconds=duration,
        rom=rom,
        minimum=minimum,
        maximum=maximum,
        start_value=start_value,
        end_value=end_value,
        peak_index=peak_index,
        peak_fraction=peak_fraction,
        return_error=return_error,
        return_tolerance=return_tolerance,
        start_geometry=start_geometry,
        end_geometry=end_geometry,
    )


def validate_template_file(
    template_path: str | Path,
    eval_config: dict[str, Any],
    *,
    pose_backend: str,
) -> dict[str, Any]:
    path = Path(template_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "required": str(pose_backend).lower() == "rknn",
            "action_id": str(eval_config.get("action_id") or ""),
            "reason": "template_read_failed",
            "message": f"模板读取失败：{exc}",
            "errors": [{"code": "template_read_failed", "stage": "file", "message": str(exc)}],
            "missing_stages": ["file"],
        }
    return validate_template_payload(payload, eval_config, pose_backend=pose_backend)


def _start_end_geometry(frames: list[dict[str, Any]], fraction: float) -> tuple[float | None, float | None]:
    geometries = [_as_float(frame.get("selected_included_angle")) for frame in frames if isinstance(frame, dict)]
    geometries = [value for value in geometries if value is not None]
    if not geometries:
        return None, None
    window = max(3, min(len(geometries), int(round(len(geometries) * fraction))))
    return float(median(geometries[:window])), float(median(geometries[-window:]))


def _result(action_id: str, limits: dict[str, float], *, errors: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    ok = not errors
    missing_stages = list(dict.fromkeys(str(item.get("stage") or "unknown") for item in errors))
    return {
        "ok": ok,
        "required": True,
        "action_id": action_id,
        "reason": "ok" if ok else str(errors[0].get("code") or "invalid_template"),
        "message": "模板检查通过" if ok else "；".join(str(item.get("message") or "模板不合格") for item in errors),
        "errors": errors,
        "missing_stages": missing_stages,
        "limits": limits,
        **fields,
    }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
