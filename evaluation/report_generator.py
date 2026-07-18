from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rehab_app.services.report_paths import is_safe_keyframe_path

try:
    from .core.action_metrics import METRIC_VALUE_FIELD, extract_metric_sequence, extract_named_metric_sequence
    from .core.dtw_compare import dtw_score
    from .core.error_classifier import classify
    from .core.rom import compute_rom
    from .core.speed_check import check_speed
    from .core.tut import compute_tut
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from evaluation.core.action_metrics import METRIC_VALUE_FIELD, extract_metric_sequence, extract_named_metric_sequence  # type: ignore
    from evaluation.core.dtw_compare import dtw_score  # type: ignore
    from evaluation.core.error_classifier import classify  # type: ignore
    from evaluation.core.rom import compute_rom  # type: ignore
    from evaluation.core.speed_check import check_speed  # type: ignore
    from evaluation.core.tut import compute_tut  # type: ignore

try:
    from action_scoring.service import get_quality_model_status, score_rep
    from action_scoring.completion_calibrator import (
        calibrated_completion_details,
        calibrated_completion_percent,
        report_error_cap,
        should_filter_reentry_attempt,
    )
    from action_scoring.labels import quality_grade
except Exception:  # optional board dependency
    calibrated_completion_details = None  # type: ignore[assignment]
    calibrated_completion_percent = None  # type: ignore[assignment]
    report_error_cap = None  # type: ignore[assignment]
    should_filter_reentry_attempt = None  # type: ignore[assignment]
    quality_grade = None  # type: ignore[assignment]
    get_quality_model_status = None  # type: ignore[assignment]
    score_rep = None  # type: ignore[assignment]

EVALUATE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线动作评估 MVP：模板 JSON + 患者 JSON -> 结构化报告 JSON。")
    parser.add_argument("--template", required=True, help="标准动作模板 JSON。")
    parser.add_argument("--attempt", required=True, help="患者尝试动作 JSON。")
    parser.add_argument("--config", required=True, help="动作评估 YAML 配置。")
    parser.add_argument("--out", required=True, help="输出评估报告 JSON。")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = EVALUATE_DIR / path
    if candidate.exists() or path.parts and path.parts[0] in {"samples", "configs", "reports", "core"}:
        return candidate
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid config file: {path}")
    return config


def select_angle_field(frames: list[dict[str, Any]], config: dict[str, Any]) -> str:
    fields = config.get("angle_fields") or [config.get("angle_field")]
    for field in fields:
        if not field:
            continue
        if any(_as_float(frame.get(str(field))) is not None for frame in frames):
            return str(field)
    raise ValueError(f"no configured angle field found in frames: {fields}")


def extract_angles(frames: list[dict[str, Any]], angle_field: str) -> list[float]:
    angles: list[float] = []
    for frame in frames:
        value = _as_float(frame.get(angle_field))
        if value is not None:
            angles.append(value)
    if not angles:
        raise ValueError(f"no angles available for field: {angle_field}")
    return angles


def get_frames(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    frames = payload.get("template_frames")
    if not isinstance(frames, list):
        raise ValueError(f"{label} JSON missing template_frames list")
    return [frame for frame in frames if isinstance(frame, dict)]


def extract_report_keyframes(attempt_payload: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_meta = attempt_payload.get("runtime_meta")
    if not isinstance(runtime_meta, dict):
        return []
    raw_keyframes = runtime_meta.get("keyframes")
    if not isinstance(raw_keyframes, list):
        return []
    keyframes: list[dict[str, Any]] = []
    for item in raw_keyframes:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").replace("\\", "/").strip()
        if not is_safe_keyframe_path(image_path):
            continue
        row = {
            "session_id": item.get("session_id"),
            "action_id": item.get("action_id"),
            "action_name": item.get("action_name"),
            "rep_index": item.get("rep_index"),
            "kind": item.get("kind") or "best_peak",
            "image_path": image_path,
            "signal_value": item.get("signal_value"),
            "primary_metric": item.get("primary_metric"),
            "primary_metric_unit": item.get("primary_metric_unit"),
            "frame_index": item.get("frame_index"),
            "relative_time": item.get("relative_time"),
            "selected_side": item.get("selected_side"),
            "visibility_min": item.get("visibility_min"),
            "write_status": item.get("write_status"),
            "write_error": item.get("write_error"),
        }
        rehab_keypoints = item.get("rehab_keypoints")
        if isinstance(rehab_keypoints, dict):
            row["rehab_keypoints"] = rehab_keypoints
        keyframes.append(row)
    return keyframes


def extract_quality_attempt_segments(attempt_payload: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_meta = attempt_payload.get("runtime_meta")
    if not isinstance(runtime_meta, dict):
        return []
    for field in ("quality_attempt_segments", "rep_segments"):
        raw_segments = runtime_meta.get(field)
        if isinstance(raw_segments, list):
            return [segment for segment in raw_segments if isinstance(segment, dict)]
    return []



ATTEMPT_METRIC_FIELDS = (
    "rom",
    "rom_target",
    "rom_diff",
    "max_signal",
    "tut_seconds",
    "tut_target",
    "missing_seconds",
    "tut_ratio",
    "peak_speed",
    "speed_ratio",
    "duration_seconds",
    "required_rom",
    "dynamic_target",
    "template_rom",
    "tut_required_seconds",
    "speed_ratio_max",
    "completion_trigger",
    "watchdog_used",
    "visibility_recovery_used",
    "pose_backend",
    "training_logic_version",
)


def enrich_quality_segments_with_runtime_metrics(attempt_payload: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_meta = attempt_payload.get("runtime_meta") if isinstance(attempt_payload.get("runtime_meta"), dict) else {}
    runtime_rows: dict[int, dict[str, Any]] = {}
    for field in ("rep_results", "invalid_attempts"):
        raw_rows = runtime_meta.get(field)
        if not isinstance(raw_rows, list):
            continue
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            attempt_index = _as_int(item.get("attempt_index"))
            if attempt_index is not None:
                runtime_rows[attempt_index] = item
    enriched: list[dict[str, Any]] = []
    for segment in segments:
        row = dict(segment)
        row.setdefault("pose_backend", runtime_meta.get("actual_backend") or runtime_meta.get("pose_backend"))
        row.setdefault("training_logic_version", runtime_meta.get("training_logic_version"))
        source = runtime_rows.get(_as_int(row.get("attempt_index")) or -1)
        if isinstance(source, dict):
            for key in ATTEMPT_METRIC_FIELDS:
                if row.get(key) is None and source.get(key) is not None:
                    row[key] = source.get(key)
            if not row.get("screen_prompt") and source.get("screen_prompt"):
                row["screen_prompt"] = source.get("screen_prompt")
            if not row.get("reason") and source.get("not_counted_reason"):
                row["reason"] = source.get("not_counted_reason")
        enriched.append(row)
    return enriched


def build_selected_attempts(quality_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_correct": _select_best_correct_attempt(quality_attempts),
        "representative_wrong": _select_representative_wrong_attempt(quality_attempts),
    }


def build_report_card_metrics(metrics: dict[str, Any], selected_attempts: dict[str, Any]) -> dict[str, Any]:
    selected = selected_attempts.get("best_correct") if isinstance(selected_attempts.get("best_correct"), dict) else None
    source = "best_correct"
    if not selected:
        selected = selected_attempts.get("representative_wrong") if isinstance(selected_attempts.get("representative_wrong"), dict) else None
        source = "representative_wrong"
    if not selected:
        card = dict(metrics)
        card["source"] = "full_session"
        return card
    return {
        "source": source,
        "attempt_index": selected.get("attempt_index"),
        "rep_index": selected.get("rep_index"),
        "countable": selected.get("countable"),
        "primary_error": selected.get("primary_error"),
        "rom": {
            "target": _as_float(selected.get("rom_target")) if _as_float(selected.get("rom_target")) is not None else _nested_metric(metrics, "rom", "target"),
            "actual": _as_float(selected.get("rom")),
            "diff": _as_float(selected.get("rom_diff")),
        },
        "tut": {
            "target": _as_float(selected.get("tut_target")) if _as_float(selected.get("tut_target")) is not None else _nested_metric(metrics, "tut", "target"),
            "actual": _as_float(selected.get("tut_seconds")),
            "ratio": _as_float(selected.get("tut_ratio")),
            "missing_seconds": _as_float(selected.get("missing_seconds")),
        },
        "speed": {
            "attempt_peak": _as_float(selected.get("peak_speed")),
            "ratio": _as_float(selected.get("speed_ratio")),
            "template_peak": _nested_metric(metrics, "speed", "template_peak"),
        },
        "dtw": metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {},
    }


def _select_best_correct_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in attempts if isinstance(item, dict) and item.get("countable")]
    if not candidates:
        return None

    def score(item: dict[str, Any]) -> tuple[float, float, float, int]:
        quality = _as_float(item.get("quality_score"))
        tut_ratio = _as_float(item.get("tut_ratio"))
        rom_diff = _as_float(item.get("rom_diff"))
        attempt_index = _as_int(item.get("attempt_index")) or 0
        return (
            quality if quality is not None else -1.0,
            tut_ratio if tut_ratio is not None else 0.0,
            -(rom_diff if rom_diff is not None else 999999.0),
            -attempt_index,
        )

    return dict(max(candidates, key=score))


def _select_representative_wrong_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in attempts if isinstance(item, dict) and not item.get("countable")]
    if not candidates:
        return None
    priority = {"ROM_LOW": 0, "TUT_LOW": 1, "SHAPE_BAD": 2, "TOO_FAST": 3, "VISIBILITY_LOW": 4}

    def score(item: dict[str, Any]) -> tuple[int, float, int]:
        error = str(item.get("primary_error") or "")
        rom_diff = _as_float(item.get("rom_diff")) or 0.0
        missing = _as_float(item.get("missing_seconds")) or 0.0
        attempt_index = _as_int(item.get("attempt_index")) or 0
        return (priority.get(error, 9), -max(rom_diff, missing), -attempt_index)

    return dict(min(candidates, key=score))


def _nested_metric(metrics: dict[str, Any], group: str, key: str) -> float | None:
    value = metrics.get(group) if isinstance(metrics.get(group), dict) else {}
    return _as_float(value.get(key))


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None

def build_quality_attempt_reports(action_id: str, segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None, dict[str, Any]]:
    status = _quality_model_status(action_id)
    reports: list[dict[str, Any]] = []
    scores: list[float] = []
    for segment in segments:
        if _should_skip_startup_reentry_artifact(action_id, segment):
            continue
        raw_score = _as_float(segment.get("raw_quality_score"))
        if raw_score is None:
            raw_score = _as_float(segment.get("model_score"))
        if raw_score is None:
            raw_score = _as_float(segment.get("quality_score"))
        row = {
            "attempt_index": segment.get("attempt_index"),
            "rep_index": segment.get("rep_index"),
            "countable": bool(segment.get("countable", False)),
            "primary_error": segment.get("primary_error") or "OK",
            "all_errors": segment.get("all_errors") if isinstance(segment.get("all_errors"), list) else [],
            "reason": segment.get("reason") or segment.get("screen_prompt"),
            "frame_count": segment.get("frame_count"),
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "raw_quality_score": raw_score,
            "model_score": raw_score,
            "quality_grade": segment.get("quality_grade"),
            "quality_backend": segment.get("quality_backend"),
            "quality_model_path": segment.get("quality_model_path"),
        }
        for key in ATTEMPT_METRIC_FIELDS:
            if segment.get(key) is not None:
                row[key] = segment.get(key)
        if raw_score is None and score_rep is not None:
            try:
                result = score_rep(action_id, segment)
            except Exception as exc:
                status = dict(status)
                status["last_error"] = str(exc)
                result = None
            if isinstance(result, dict):
                raw_score = _as_float(result.get("score"))
                row["raw_quality_score"] = raw_score
                row["model_score"] = raw_score
                row["quality_grade"] = result.get("grade")
                row["quality_backend"] = result.get("backend")
                row["quality_model_path"] = result.get("model_path")
                row["quality_valid_frames"] = result.get("valid_frames")
        calibration = _calibrated_completion_details(action_id, row, raw_score)
        completion = _as_float(calibration.get("completion_percent"))
        if completion is not None:
            completion = round(completion, 2)
            row["quality_score"] = completion
            row["completion_percent"] = completion
            row["rule_score"] = calibration.get("rule_score")
            row["completion_error_cap"] = calibration.get("error_cap")
            row["completion_calibration_mode"] = calibration.get("calibration_mode")
            row["quality_grade"] = quality_grade(completion) if quality_grade is not None else row.get("quality_grade")
            scores.append(completion)
        elif raw_score is not None:
            row["raw_quality_score"] = round(raw_score, 2)
            row["model_score"] = round(raw_score, 2)
        reports.append(row)
    overall = round(sum(scores) / len(scores), 2) if scores else None
    npu_v4 = any(
        str(item.get("pose_backend") or "").lower() == "rknn"
        or str(item.get("training_logic_version") or "") == "npu_training_v4"
        for item in reports
    )
    average_cap = report_error_cap(reports) if npu_v4 and report_error_cap is not None else None
    if overall is not None and average_cap is not None:
        overall = round(min(overall, float(average_cap)), 2)
    status = _quality_model_status(action_id, fallback=status)
    if average_cap is not None:
        status = dict(status)
        status["average_error_cap"] = average_cap
        status["calibration_mode"] = "npu_rule_first_v4"
    return reports, overall, status


def _calibrated_completion(action_id: str, payload: dict[str, Any], raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if calibrated_completion_percent is None:
        return raw_score
    try:
        return calibrated_completion_percent(action_id, payload, raw_score)
    except Exception:
        return raw_score


def _calibrated_completion_details(action_id: str, payload: dict[str, Any], raw_score: float | None) -> dict[str, Any]:
    if calibrated_completion_details is None:
        return {"completion_percent": _calibrated_completion(action_id, payload, raw_score), "model_score": raw_score}
    try:
        details = calibrated_completion_details(action_id, payload, raw_score)
        return dict(details) if isinstance(details, dict) else {"completion_percent": raw_score, "model_score": raw_score}
    except Exception:
        return {"completion_percent": _calibrated_completion(action_id, payload, raw_score), "model_score": raw_score}


def _should_skip_startup_reentry_artifact(action_id: str, payload: dict[str, Any]) -> bool:
    if action_id != "seated_knee_raise":
        return False
    if _as_int(payload.get("attempt_index")) != 1:
        return False
    if should_filter_reentry_attempt is None:
        return False
    try:
        return should_filter_reentry_attempt(payload)
    except Exception:
        return False


def build_completion_summary(action_id: str, action_name: str, attempts: list[dict[str, Any]], average: float | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in attempts:
        score = _as_float(item.get("completion_percent"))
        if score is None:
            score = _as_float(item.get("quality_score"))
        rows.append(
            {
                "attempt_index": item.get("attempt_index"),
                "rep_index": item.get("rep_index"),
                "countable": bool(item.get("countable", False)),
                "primary_error": item.get("primary_error") or "OK",
                "reason": item.get("reason"),
                "completion_percent": round(score, 2) if score is not None else None,
            }
        )
    return {
        "action_id": action_id,
        "action_name": action_name,
        "attempts": rows,
        "average_completion": average,
    }


def _quality_model_status(action_id: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if get_quality_model_status is None:
        return {
            "available": False,
            "backend": None,
            "model_path": None,
            "action_id": action_id,
            "input_frames": 30,
            "score_range": [0, 100],
            "last_score_time_ms": None,
            "last_error": (fallback or {}).get("last_error") or "quality_model_service_unavailable",
        }
    try:
        status = get_quality_model_status(action_id)
    except Exception as exc:
        status = {
            "available": False,
            "backend": None,
            "model_path": None,
            "action_id": action_id,
            "input_frames": 30,
            "score_range": [0, 100],
            "last_score_time_ms": None,
            "last_error": str(exc),
        }
    if fallback and fallback.get("last_error") and not status.get("last_error"):
        status = dict(status)
        status["last_error"] = fallback.get("last_error")
    return status


def build_tut_range(rom_result: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
    zone = config.get("tut_work_zone", {})
    if zone.get("mode") != "relative_to_max":
        raise ValueError(f"unsupported tut_work_zone mode: {zone.get('mode')}")
    max_angle = float(rom_result["max"])
    low = max_angle + float(zone.get("offset_low", 0.0))
    high = max_angle + float(zone.get("offset_high", 0.0))
    return min(low, high), max(low, high)


def build_metrics(
    template_frames: list[dict[str, Any]],
    attempt_frames: list[dict[str, Any]],
    template_field: str,
    attempt_field: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    template_rom = compute_rom(template_frames, template_field)
    attempt_rom = compute_rom(attempt_frames, attempt_field)
    tut_range = build_tut_range(template_rom, config)
    template_tut_result = compute_tut(template_frames, tut_range, template_field)
    tut_result = compute_tut(attempt_frames, tut_range, attempt_field)
    dtw_result = dtw_score(extract_angles(template_frames, template_field), extract_angles(attempt_frames, attempt_field))
    template_speed = check_speed(template_frames, template_field)
    attempt_speed = check_speed(attempt_frames, attempt_field)

    rom_target = float(template_rom["rom"])
    rom_actual = float(attempt_rom["rom"])
    target_tut = float(template_tut_result["tut_seconds"])
    tut_actual = float(tut_result["tut_seconds"])
    template_peak = float(template_speed["peak_angular_velocity"])
    attempt_peak = float(attempt_speed["peak_angular_velocity"])

    return {
        "rom": {
            "target": rom_target,
            "actual": rom_actual,
            "diff": max(0.0, rom_target - rom_actual),
            "template": template_rom,
            "attempt": attempt_rom,
        },
        "tut": {
            "target": target_tut,
            "target_source": "template",
            "template_tut": target_tut,
            "actual": tut_actual,
            "ratio": _safe_ratio(tut_actual, target_tut),
            "target_range": list(tut_range),
            "template_in_range_frames": template_tut_result["in_range_frames"],
            "template_total_frames": template_tut_result["total_frames"],
            "in_range_frames": tut_result["in_range_frames"],
            "total_frames": tut_result["total_frames"],
            "method": tut_result.get("method", "unknown"),
        },
        "dtw": {
            "distance": dtw_result["distance"],
            "normalized_distance": dtw_result["normalized_distance"],
            "path_length": dtw_result["path_length"],
        },
        "speed": {
            "template_peak": template_peak,
            "attempt_peak": attempt_peak,
            "ratio": _safe_ratio(attempt_peak, template_peak),
            "template_mean": template_speed["mean_angular_velocity"],
            "attempt_mean": attempt_speed["mean_angular_velocity"],
        },
    }


def build_action_metric_payload(
    template_frames: list[dict[str, Any]],
    attempt_frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    template_metric = extract_metric_sequence(template_frames, config)
    attempt_metric = extract_metric_sequence(attempt_frames, config)
    metrics = build_metrics(
        template_metric["frames"],
        attempt_metric["frames"],
        METRIC_VALUE_FIELD,
        METRIC_VALUE_FIELD,
        config,
    )
    secondary_metrics = _build_secondary_metrics(template_frames, attempt_frames, config)
    if secondary_metrics:
        metrics["secondary_metrics"] = secondary_metrics
    metric_report = {
        "metric_name": template_metric["metric_name"],
        "metric_unit": template_metric["metric_unit"],
        "metric_description": template_metric["metric_description"],
        "value_field": METRIC_VALUE_FIELD,
        "template_sample_count": template_metric["sample_count"],
        "attempt_sample_count": attempt_metric["sample_count"],
        "secondary_metrics": secondary_metrics,
    }
    return metrics, metric_report


def _build_secondary_metrics(
    template_frames: list[dict[str, Any]],
    attempt_frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    secondary_metric = config.get("secondary_metric")
    if not secondary_metric:
        return {}
    metric_names = secondary_metric if isinstance(secondary_metric, list) else [secondary_metric]
    payload: dict[str, Any] = {}
    for metric_name_value in metric_names:
        metric_name = str(metric_name_value or "").strip()
        if not metric_name:
            continue
        template_metric = extract_named_metric_sequence(template_frames, metric_name, config)
        attempt_metric = extract_named_metric_sequence(attempt_frames, metric_name, config)
        template_rom = compute_rom(template_metric["frames"], METRIC_VALUE_FIELD)
        attempt_rom = compute_rom(attempt_metric["frames"], METRIC_VALUE_FIELD)
        payload[metric_name] = {
            "metric_name": metric_name,
            "metric_unit": template_metric["metric_unit"],
            "metric_description": template_metric["metric_description"],
            "rom": {
                "target": float(template_rom["rom"]),
                "actual": float(attempt_rom["rom"]),
                "diff": max(0.0, float(template_rom["rom"]) - float(attempt_rom["rom"])),
                "template": template_rom,
                "attempt": attempt_rom,
            },
        }
    return payload


def build_structured_feedback(errors: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    error_code = str(errors.get("primary_error") or "OK")
    if error_code == "ROM_LOW":
        params = {
            "diff_angle": metrics["rom"]["diff"],
            "target": metrics["rom"]["target"],
            "actual": metrics["rom"]["actual"],
        }
    elif error_code == "TUT_LOW":
        params = {
            "target": metrics["tut"]["target"],
            "actual": metrics["tut"]["actual"],
            "ratio": metrics["tut"]["ratio"],
        }
    elif error_code == "SHAPE_BAD":
        params = {"normalized_distance": metrics["dtw"]["normalized_distance"]}
    elif error_code == "TOO_FAST":
        params = {
            "template_peak": metrics["speed"]["template_peak"],
            "attempt_peak": metrics["speed"]["attempt_peak"],
            "ratio": metrics["speed"]["ratio"],
        }
    else:
        params = {}
    return {"error_code": error_code, "params": params}


def make_report(template_path: Path, attempt_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    template_payload = load_json(template_path)
    attempt_payload = load_json(attempt_path)
    template_frames = get_frames(template_payload, "template")
    attempt_frames = get_frames(attempt_payload, "attempt")
    metric_report: dict[str, Any] | None = None
    if config.get("primary_metric"):
        metrics, metric_report = build_action_metric_payload(template_frames, attempt_frames, config)
        template_field = METRIC_VALUE_FIELD
        attempt_field = METRIC_VALUE_FIELD
    else:
        template_field = select_angle_field(template_frames, config)
        attempt_field = select_angle_field(attempt_frames, config)
        metrics = build_metrics(template_frames, attempt_frames, template_field, attempt_field, config)
    errors = classify(metrics, config)
    action_id = str(attempt_payload.get("action_id") or config.get("action_id") or config_path.stem)
    quality_segments = enrich_quality_segments_with_runtime_metrics(attempt_payload, extract_quality_attempt_segments(attempt_payload))
    quality_attempts, overall_quality, quality_model = build_quality_attempt_reports(action_id, quality_segments)
    runtime_meta = attempt_payload.get("runtime_meta") if isinstance(attempt_payload.get("runtime_meta"), dict) else {}
    npu_v4 = str(runtime_meta.get("actual_backend") or runtime_meta.get("pose_backend") or "").lower() == "rknn" or str(
        runtime_meta.get("training_logic_version") or ""
    ) == "npu_training_v4"
    report_primary_error = str(errors.get("primary_error") or "OK") if isinstance(errors, dict) else "OK"
    report_cap = report_error_cap([{"primary_error": report_primary_error}]) if npu_v4 and report_error_cap is not None else None
    if overall_quality is not None and report_cap is not None:
        overall_quality = round(min(overall_quality, float(report_cap)), 2)
        quality_model = dict(quality_model)
        quality_model["report_error_cap"] = report_cap
        quality_model["report_primary_error"] = report_primary_error
    reps = [attempt for attempt in quality_attempts if attempt.get("countable")]
    selected_attempts = build_selected_attempts(quality_attempts)
    report_card_metrics = build_report_card_metrics(metrics, selected_attempts)
    action_name = str(config.get("action_name", ""))
    completion_summary = build_completion_summary(action_id, action_name, quality_attempts, overall_quality)
    return {
        "evaluated_at": datetime.now().replace(microsecond=0).isoformat(),
        "action_name": action_name,
        "template_file": str(template_path),
        "attempt_file": str(attempt_path),
        "config_file": str(config_path),
        "fields": {
            "template": template_field,
            "attempt": attempt_field,
        },
        "metric": metric_report,
        "keypoint_rule": config.get("keypoint_rule", {}),
        "metrics": metrics,
        "errors": errors,
        "structured_feedback": build_structured_feedback(errors, metrics),
        "report_card_metrics": report_card_metrics,
        "selected_attempts": selected_attempts,
        "quality_attempts": quality_attempts,
        "reps": reps,
        "overall_quality": overall_quality,
        "overall_completion": overall_quality,
        "completion_by_action": {action_id: completion_summary},
        "quality_model": quality_model,
        "keyframes": extract_report_keyframes(attempt_payload),
        "runtime_meta": runtime_meta,
        "clinical_baseline": attempt_payload.get("clinical_baseline") if isinstance(attempt_payload.get("clinical_baseline"), dict) else {},
    }


def main() -> int:
    args = parse_args()
    template_path = resolve_path(args.template)
    attempt_path = resolve_path(args.attempt)
    config_path = resolve_path(args.config)
    out_path = resolve_path(args.out)

    try:
        report = make_report(template_path, attempt_path, config_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 1

    print(f"评估完成：{out_path}")
    print(json.dumps(report["errors"], ensure_ascii=False))
    return 0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


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
    raise SystemExit(main())
