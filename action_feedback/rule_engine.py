from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rules(path: str | Path) -> dict[str, Any]:
    rules = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise ValueError(f"invalid feedback rule file: {path}")
    return rules


def build_feedback_from_files(report_path: str | Path, rule_path: str | Path) -> dict[str, Any]:
    return build_feedback(load_json(report_path), load_rules(rule_path))


def build_feedback(report: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    error_code = _resolve_error_code(report)
    rule = rules.get(error_code) or rules.get("OK")
    if not isinstance(rule, dict):
        raise ValueError(f"missing feedback rule for error code: {error_code}")

    raw_params = _extract_raw_params(report)
    screen_rule = _dict_value(rule, "screen")
    tts_rule = _dict_value(rule, "tts")
    motor_rule = _dict_value(rule, "motor")

    return {
        "error_code": error_code,
        "raw_params": raw_params,
        "screen": {
            "color": str(screen_rule.get("color", "green")),
            "title": str(screen_rule.get("title", error_code)),
            "message": _render_template(str(screen_rule.get("template", "")), raw_params),
        },
        "tts": {
            "text": _render_template(str(tts_rule.get("template", "")), raw_params),
        },
        "motor": {
            "pattern": str(motor_rule.get("pattern", "success_once")),
        },
    }


def _resolve_error_code(report: dict[str, Any]) -> str:
    structured_feedback = report.get("structured_feedback")
    if isinstance(structured_feedback, dict):
        code = structured_feedback.get("error_code")
        if code:
            return str(code)

    errors = report.get("errors")
    if isinstance(errors, dict):
        code = errors.get("primary_error")
        if code:
            return str(code)

    return "OK"


def _extract_raw_params(report: dict[str, Any]) -> dict[str, float]:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    rom = metrics.get("rom") if isinstance(metrics.get("rom"), dict) else {}
    tut = metrics.get("tut") if isinstance(metrics.get("tut"), dict) else {}
    speed = metrics.get("speed") if isinstance(metrics.get("speed"), dict) else {}
    dtw = metrics.get("dtw") if isinstance(metrics.get("dtw"), dict) else {}

    tut_target = _as_float(tut.get("target")) or 0.0
    tut_actual = _as_float(tut.get("actual")) or 0.0
    return {
        "rom_diff": _as_float(rom.get("diff")) or 0.0,
        "missing_seconds": max(0.0, tut_target - tut_actual),
        "speed_ratio": _as_float(speed.get("ratio")) or 0.0,
        "dtw_normalized_distance": _as_float(dtw.get("normalized_distance")) or 0.0,
    }


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _render_template(template: str, params: dict[str, float]) -> str:
    try:
        return template.format(**params)
    except (KeyError, ValueError):
        return template


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
    rules = {
        "ROM_LOW": {
            "screen": {"color": "red", "title": "动作不到位", "template": "还差 {rom_diff:.1f} 度"},
            "tts": {"template": "还差 {rom_diff:.0f} 度"},
            "motor": {"pattern": "short_double"},
        },
        "TUT_LOW": {
            "screen": {"color": "orange", "title": "保持不足", "template": "还差 {missing_seconds:.1f} 秒"},
            "tts": {"template": "保持住"},
            "motor": {"pattern": "interval"},
        },
        "SHAPE_BAD": {
            "screen": {"color": "yellow", "title": "轨迹偏离", "template": "DTW {dtw_normalized_distance:.2f}"},
            "tts": {"template": "轨迹不标准"},
            "motor": {"pattern": "short_once"},
        },
        "TOO_FAST": {
            "screen": {"color": "yellow", "title": "过快", "template": "速度倍率 {speed_ratio:.2f}"},
            "tts": {"template": "请放慢速度"},
            "motor": {"pattern": "long_once"},
        },
        "OK": {
            "screen": {"color": "green", "title": "完成", "template": "很好"},
            "tts": {"template": "很好"},
            "motor": {"pattern": "success_once"},
        },
    }
    base_report = {
        "metrics": {
            "rom": {"diff": 8.0},
            "tut": {"target": 3.0, "actual": 1.8},
            "speed": {"ratio": 1.7},
            "dtw": {"normalized_distance": 0.3},
        }
    }
    for code in ("ROM_LOW", "TUT_LOW", "SHAPE_BAD", "TOO_FAST", "OK"):
        report = {**base_report, "structured_feedback": {"error_code": code}}
        feedback = build_feedback(report, rules)
        assert feedback["error_code"] == code
        assert set(feedback["raw_params"]) == {"rom_diff", "missing_seconds", "speed_ratio", "dtw_normalized_distance"}
        assert feedback["screen"]["message"]
        assert feedback["tts"]["text"]
        assert feedback["motor"]["pattern"]

    fallback_report = {**base_report, "errors": {"primary_error": "TUT_LOW"}}
    assert build_feedback(fallback_report, rules)["error_code"] == "TUT_LOW"
    assert build_feedback(base_report, rules)["error_code"] == "OK"
    print("rule_engine inline tests passed")
