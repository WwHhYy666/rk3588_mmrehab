from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .core.action_metrics import METRIC_VALUE_FIELD, extract_metric_sequence, extract_named_metric_sequence
    from .core.dtw_compare import dtw_score
    from .core.error_classifier import classify
    from .core.rom import compute_rom
    from .core.speed_check import check_speed
    from .core.tut import compute_tut
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from evaluate.core.action_metrics import METRIC_VALUE_FIELD, extract_metric_sequence, extract_named_metric_sequence  # type: ignore
    from evaluate.core.dtw_compare import dtw_score  # type: ignore
    from evaluate.core.error_classifier import classify  # type: ignore
    from evaluate.core.rom import compute_rom  # type: ignore
    from evaluate.core.speed_check import check_speed  # type: ignore
    from evaluate.core.tut import compute_tut  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        if not _is_safe_keyframe_path(image_path):
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
        }
        rehab_keypoints = item.get("rehab_keypoints")
        if isinstance(rehab_keypoints, dict):
            row["rehab_keypoints"] = rehab_keypoints
        keyframes.append(row)
    return keyframes


def _is_safe_keyframe_path(value: str) -> bool:
    if not value or value.startswith("/") or value.startswith("\\"):
        return False
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return False
    normalized = path.as_posix()
    return normalized.startswith("evaluate/reports/keyframes/") and path.suffix.lower() in {".jpg", ".jpeg"}


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
    return {
        "evaluated_at": datetime.now().replace(microsecond=0).isoformat(),
        "action_name": config.get("action_name", ""),
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
        "keyframes": extract_report_keyframes(attempt_payload),
        "runtime_meta": attempt_payload.get("runtime_meta") if isinstance(attempt_payload.get("runtime_meta"), dict) else {},
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
