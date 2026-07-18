from __future__ import annotations

from typing import Any


ERROR_SCORE_CAPS = {
    "ROM_LOW": 70.0,
    "TUT_LOW": 78.0,
    "SHAPE_BAD": 75.0,
    "TOO_FAST": 82.0,
    "VISIBILITY_LOW": 50.0,
}


def calibrated_completion_details(
    action_id: str,
    payload: dict[str, Any],
    model_score: float | None,
) -> dict[str, Any]:
    """Return the displayed completion score and its rule/model components."""

    raw_score = _clip_score(model_score)
    error_code = str(payload.get("primary_error") or "OK").upper()
    if not _is_npu_v4(payload):
        legacy = _legacy_completion(error_code, payload, raw_score)
        return {
            "completion_percent": legacy,
            "rule_score": None,
            "model_score": raw_score,
            "error_cap": None,
            "calibration_mode": "legacy",
        }

    if error_code == "OK":
        rule_score = _ok_rule_score(payload)
        completion = rule_score if raw_score is None else rule_score * 0.75 + raw_score * 0.25
        return {
            "completion_percent": round(min(96.0, max(70.0, completion)), 2),
            "rule_score": round(rule_score, 2),
            "model_score": raw_score,
            "error_cap": 96.0,
            "calibration_mode": "npu_rule_first_v4",
        }

    cap = ERROR_SCORE_CAPS.get(error_code, 75.0)
    rule_score = _error_rule_score(error_code, payload)
    return {
        "completion_percent": round(min(cap, rule_score), 2),
        "rule_score": round(rule_score, 2),
        "model_score": raw_score,
        "error_cap": cap,
        "calibration_mode": "npu_rule_first_v4",
    }


def calibrated_completion_percent(action_id: str, payload: dict[str, Any], model_score: float | None) -> float | None:
    return _as_float(calibrated_completion_details(action_id, payload, model_score).get("completion_percent"))


def should_filter_reentry_attempt(payload: dict[str, Any]) -> bool:
    """Return True for short post-offscreen pose-jump attempts that should not be reported."""

    if str(payload.get("primary_error") or "").upper() != "ROM_LOW":
        return False
    if bool(payload.get("countable", False)):
        return False
    duration = _as_float(payload.get("duration_seconds"))
    if duration is None or duration >= 2.0:
        return False
    tut_ratio = _as_float(payload.get("tut_ratio"))
    if tut_ratio is not None and tut_ratio > 0.01:
        return False
    rom_ratio = _rom_ratio(payload)
    return rom_ratio is not None and rom_ratio < 0.35


def report_error_cap(payloads: list[dict[str, Any]]) -> float | None:
    caps = [
        ERROR_SCORE_CAPS[error]
        for payload in payloads
        for error in [str(payload.get("primary_error") or "OK").upper()]
        if error in ERROR_SCORE_CAPS
    ]
    return min(caps) if caps else None


def _ok_rule_score(payload: dict[str, Any]) -> float:
    score = 90.0
    rom_ratio = _rom_ratio(payload)
    if rom_ratio is not None:
        score += min(3.0, max(0.0, rom_ratio - 1.0) * 12.0)
    tut_ratio = _tut_ratio(payload)
    if tut_ratio is not None:
        score += min(3.0, max(0.0, tut_ratio - 1.0) * 12.0)

    speed_ratio = _as_float(payload.get("speed_ratio"))
    speed_limit = _as_float(payload.get("speed_ratio_max")) or 1.5
    if speed_ratio is not None and speed_ratio > 1.0:
        denominator = max(0.1, speed_limit - 1.0)
        score -= min(10.0, (speed_ratio - 1.0) / denominator * 10.0)
    if bool(payload.get("watchdog_used")) or str(payload.get("completion_trigger") or "").startswith("watchdog_"):
        score -= 8.0
    if bool(payload.get("visibility_recovery_used")) or str(payload.get("completion_trigger") or "") == "visibility_interrupted":
        score -= 12.0
    return min(96.0, max(70.0, score))


def _error_rule_score(error_code: str, payload: dict[str, Any]) -> float:
    if error_code == "ROM_LOW":
        ratio = _rom_ratio(payload)
        if ratio is None:
            return 45.0
        if ratio < 0.60:
            return _lerp_clamped(ratio, 0.0, 0.60, 20.0, 45.0)
        if ratio < 0.80:
            return _lerp_clamped(ratio, 0.60, 0.80, 45.0, 62.0)
        return _lerp_clamped(ratio, 0.80, 1.0, 62.0, 70.0)
    if error_code == "TUT_LOW":
        ratio = _tut_ratio(payload)
        return 60.0 if ratio is None else _lerp_clamped(ratio, 0.0, 1.0, 45.0, 78.0)
    if error_code == "SHAPE_BAD":
        return 65.0
    if error_code == "TOO_FAST":
        speed_ratio = _as_float(payload.get("speed_ratio")) or 1.0
        return max(45.0, 82.0 - max(0.0, speed_ratio - 1.0) * 20.0)
    if error_code == "VISIBILITY_LOW":
        return 40.0
    return 60.0


def _legacy_completion(error_code: str, payload: dict[str, Any], raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    if error_code == "OK":
        return round(max(88.0, raw_score), 2)
    if error_code == "ROM_LOW":
        ratio = _rom_ratio(payload)
        if ratio is None:
            return round(raw_score, 2)
        if ratio < 0.60:
            return round(_lerp_clamped(ratio, 0.0, 0.60, 20.0, 45.0), 2)
        if ratio < 0.80:
            return round(_lerp_clamped(ratio, 0.60, 0.80, 45.0, 65.0), 2)
        return round(_lerp_clamped(ratio, 0.80, 1.0, 60.0, 82.0), 2)
    if error_code == "TUT_LOW":
        ratio = _tut_ratio(payload)
        return round(min(max(raw_score, 55.0), 82.0) if ratio is None else _lerp_clamped(ratio, 0.0, 1.0, 55.0, 82.0), 2)
    return round(raw_score, 2)


def _is_npu_v4(payload: dict[str, Any]) -> bool:
    return str(payload.get("pose_backend") or "").lower() == "rknn" or str(
        payload.get("training_logic_version") or ""
    ) == "npu_training_v4"


def _rom_ratio(payload: dict[str, Any]) -> float | None:
    rom = _as_float(payload.get("rom"))
    target = _as_float(payload.get("required_rom"))
    if target is None:
        target = _as_float(payload.get("rom_target"))
    if rom is None or target is None or abs(target) <= 1e-9:
        return None
    return max(0.0, rom / target)


def _tut_ratio(payload: dict[str, Any]) -> float | None:
    ratio = _as_float(payload.get("tut_ratio"))
    if ratio is not None:
        return max(0.0, ratio)
    actual = _as_float(payload.get("tut_seconds"))
    target = _as_float(payload.get("tut_required_seconds"))
    if target is None:
        target = _as_float(payload.get("tut_target"))
    if actual is None or target is None or target <= 1e-9:
        return None
    return max(0.0, actual / target)


def _clip_score(value: float | None) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return max(0.0, min(100.0, number))


def _lerp_clamped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max <= in_min:
        return out_min
    ratio = max(0.0, min(1.0, (value - in_min) / (in_max - in_min)))
    return out_min + (out_max - out_min) * ratio


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None
