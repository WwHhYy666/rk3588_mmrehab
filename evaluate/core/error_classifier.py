"""Rule-based error classification for structured evaluation metrics."""

from __future__ import annotations

from typing import Any


def classify(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Classify metrics by priority: ROM_LOW > TUT_LOW > SHAPE_BAD > TOO_FAST > OK."""
    thresholds = config.get("thresholds", {})
    errors: list[str] = []

    rom_diff = _as_float(_safe_get(metrics, "rom", "diff")) or 0.0
    if rom_diff > float(thresholds.get("rom_diff_max", 0.0)):
        errors.append("ROM_LOW")

    tut_ratio = _as_float(_safe_get(metrics, "tut", "ratio"))
    if tut_ratio is not None and tut_ratio < float(thresholds.get("tut_ratio_min", 0.0)):
        errors.append("TUT_LOW")

    normalized_distance = _as_float(_safe_get(metrics, "dtw", "normalized_distance"))
    if normalized_distance is not None and normalized_distance > float(thresholds.get("dtw_normalized_max", float("inf"))):
        errors.append("SHAPE_BAD")

    speed_ratio = _as_float(_safe_get(metrics, "speed", "ratio"))
    if speed_ratio is not None and speed_ratio > float(thresholds.get("speed_ratio_max", float("inf"))):
        errors.append("TOO_FAST")

    primary_error = errors[0] if errors else "OK"
    return {
        "all_errors": errors,
        "primary_error": primary_error,
        "is_ok": not errors,
    }


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


if __name__ == "__main__":
    sample_metrics = {
        "rom": {"diff": 12.0},
        "tut": {"ratio": 0.4},
        "dtw": {"normalized_distance": 0.1},
        "speed": {"ratio": 2.0},
    }
    sample_config = {
        "thresholds": {
            "rom_diff_max": 10.0,
            "tut_ratio_min": 0.7,
            "dtw_normalized_max": 0.25,
            "speed_ratio_max": 1.5,
        }
    }
    result = classify(sample_metrics, sample_config)
    assert result["all_errors"] == ["ROM_LOW", "TUT_LOW", "TOO_FAST"]
    assert result["primary_error"] == "ROM_LOW"
    assert result["is_ok"] is False
    print("error_classifier inline tests passed")
