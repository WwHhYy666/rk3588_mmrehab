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


def pseudo_label_from_report_rep(rep: dict[str, Any], metrics: dict[str, Any] | None, config: dict[str, Any] | None = None) -> float | None:
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
        return None
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
