from __future__ import annotations

from typing import Any


DEFAULT_GRADE_THRESHOLDS = (
    (85.0, "优秀"),
    (70.0, "良好"),
    (50.0, "一般"),
    (0.0, "需改进"),
)


def quality_grade(score: float | None) -> str | None:
    if score is None:
        return None
    value = max(0.0, min(100.0, float(score)))
    for threshold, label in DEFAULT_GRADE_THRESHOLDS:
        if value >= threshold:
            return label
    return "需改进"


def label_from_attempt_segment(segment: dict[str, Any], config: dict[str, Any] | None = None) -> float | None:
    """Build a 0-1 label directly from runtime_meta.quality_attempt_segments."""

    if not isinstance(segment, dict):
        return None
    cfg = config or {}
    penalties = cfg.get("penalties") if isinstance(cfg.get("penalties"), dict) else {}
    rom_actual = _value(segment.get("rom"), segment.get("max_signal"))
    rom_target = _value(segment.get("rom_target"))
    rom_diff = _value(segment.get("rom_diff"))
    tut_ratio = _value(segment.get("tut_ratio"))
    tut_actual = _value(segment.get("tut_seconds"))
    tut_target = _value(segment.get("tut_target"))
    speed_ratio = _value(segment.get("speed_ratio"))

    rom_score = _ratio_score(rom_actual, rom_target)
    if rom_score is None and rom_diff is not None and rom_target is not None and abs(rom_target) > 1e-6:
        rom_score = _clip01(1.0 - abs(rom_diff) / abs(rom_target))
    if rom_score is None:
        rom_score = 1.0

    if tut_ratio is None:
        tut_ratio = _ratio_score(tut_actual, tut_target)
    tut_score = _clip01(tut_ratio if tut_ratio is not None else 1.0)

    speed_score = 1.0
    if speed_ratio is not None and speed_ratio > 1.0:
        speed_score = _clip01(1.0 - max(0.0, speed_ratio - 1.5) / 1.5)

    score = 0.55 * rom_score + 0.35 * tut_score + 0.10 * speed_score
    error_codes = [str(segment.get("primary_error") or "OK")]
    all_errors = segment.get("all_errors")
    if isinstance(all_errors, list):
        error_codes.extend(str(item) for item in all_errors)
    for error_code in sorted(set(code for code in error_codes if code and code != "OK")):
        score -= float(penalties.get(error_code, _default_penalty(error_code)))
    return _clip01(score)


def pseudo_label_from_report_rep(rep: dict[str, Any], metrics: dict[str, Any] | None, config: dict[str, Any] | None = None) -> float | None:
    """Backward-compatible label helper for old callers."""

    if not isinstance(rep, dict):
        return None
    cfg = config or {}
    penalties = cfg.get("penalties") if isinstance(cfg.get("penalties"), dict) else {}
    threshold_max = _nested_float(cfg, "dtw_normalized_max") or _nested_float(metrics or {}, "dtw", "normalized_distance") or 1.0
    rom_actual = _value(rep.get("rom"), rep.get("max_signal"))
    rom_target = _value(rep.get("rom_target"), _nested_float(metrics or {}, "rom", "target"))
    tut_actual = _value(rep.get("tut_seconds"), _nested_float(metrics or {}, "tut", "actual"))
    tut_target = _value(rep.get("tut_target"), _nested_float(metrics or {}, "tut", "target"))
    dtw_value = _value(rep.get("dtw_normalized_distance"), _nested_float(metrics or {}, "dtw", "normalized_distance"))
    if rom_actual is None or rom_target is None or tut_actual is None or tut_target is None or dtw_value is None:
        return label_from_attempt_segment(rep, config)
    rom_score = _clip01(rom_actual / rom_target) if rom_target > 1e-6 else 0.0
    tut_score = _clip01(tut_actual / tut_target) if tut_target > 1e-6 else 0.0
    dtw_score = _clip01(1.0 - (dtw_value / max(threshold_max, 1e-6)))
    score = 0.4 * rom_score + 0.4 * tut_score + 0.2 * dtw_score
    error_code = str(rep.get("primary_error") or "")
    if error_code:
        score -= float(penalties.get(error_code, 0.0))
    return _clip01(score)


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _value(current)


def _value(*candidates: Any) -> float | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (int, float)):
            return float(candidate)
        try:
            return float(str(candidate).strip())
        except ValueError:
            continue
    return None


def _ratio_score(actual: Any, target: Any) -> float | None:
    actual_value = _value(actual)
    target_value = _value(target)
    if actual_value is None or target_value is None or abs(target_value) <= 1e-6:
        return None
    return _clip01(actual_value / target_value)


def _default_penalty(error_code: str) -> float:
    if error_code == "ROM_LOW":
        return 0.18
    if error_code == "TUT_LOW":
        return 0.14
    if error_code in {"TOO_FAST", "EARLY_RETURN", "SHAPE_BAD"}:
        return 0.10
    return 0.08


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
