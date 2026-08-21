"""Runtime feedback mapping for realtime rep-level results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RULE_PATH = Path(__file__).resolve().parents[1] / "feedback" / "rules" / "knee_flexion_feedback.yaml"


def load_rules(path: str | Path = DEFAULT_RULE_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def process_prompt(state: str, *, visible: bool, angle: float | None, target_low: float | None) -> str:
    if not visible:
        return "请站到摄像头前"
    if state == "BASELINE":
        return "请保持静止，正在校准"
    if state == "IDLE":
        return "准备开始下一遍"
    if state == "RISING":
        if target_low is not None and angle is not None and angle < target_low:
            return "再抬高一点"
        return "继续抬高"
    if state == "HOLDING":
        return "保持住"
    if state == "RETURNING":
        return "慢慢放下"
    return "很好，准备下一遍"


def rep_feedback(rep_result: dict[str, Any], rules: dict[str, Any] | None = None, *, action_id: str | None = None) -> dict[str, Any]:
    rules = rules or load_rules()
    code = str(rep_result.get("primary_error") or "OK")
    rule = rules.get(code) or rules.get("OK") or {}
    screen_rule = _dict(rule.get("screen"))
    tts_rule = _dict(rule.get("tts"))
    motor_rule = _dict(rule.get("motor"))
    missing_seconds = float(rep_result.get("missing_seconds") or 0.0)
    params = {
        "rom_diff": float(rep_result.get("rom_diff") or 0.0),
        "missing_seconds": missing_seconds,
        "missing_seconds_ceil": int(math.ceil(missing_seconds)) if missing_seconds > 0 else 0,
        "speed_ratio": float(rep_result.get("speed_ratio") or 0.0),
        "dtw_normalized_distance": 0.0,
    }
    rotation_key = f"{action_id or ''}:{code}:{int(rep_result.get('attempt_index') or 0)}"
    return {
        "screen_prompt": _render(_select_template(screen_rule, rotation_key), params),
        "screen_title": str(screen_rule.get("title", code)),
        "screen_color": str(screen_rule.get("color", "green")),
        "tts_text": _render(_select_template(tts_rule, rotation_key), params),
        "tts_priority": str(tts_rule.get("priority") or "low"),
        "motor_mock_pattern": str(motor_rule.get("pattern", "success_once")),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _render(template: str, params: dict[str, float | int]) -> str:
    try:
        return template.format(**params)
    except (KeyError, ValueError):
        return template


def _select_template(rule: dict[str, Any], rotation_key: str) -> str:
    templates = rule.get("templates")
    if isinstance(templates, list):
        choices = [str(item).strip() for item in templates if str(item).strip()]
        if choices:
            return choices[_stable_index(rotation_key, len(choices))]
    return str(rule.get("template", ""))


def _stable_index(value: str, length: int) -> int:
    if length <= 1:
        return 0
    return sum(ord(ch) for ch in value) % length
