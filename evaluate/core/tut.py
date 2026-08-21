"""Time-under-tension calculation for configured angle ranges."""

from __future__ import annotations

from typing import Any


def compute_tut(
    frames: list[dict[str, Any]],
    target_range: tuple[float, float] | list[float],
    angle_field: str = "flexion_angle",
) -> dict[str, Any]:
    """Estimate time in target_range with linear interpolation between frames."""
    if len(target_range) != 2:
        raise ValueError("target_range must contain low and high values")
    low = float(min(target_range))
    high = float(max(target_range))

    points: list[tuple[float, float]] = []
    for frame in frames:
        angle = _as_float(frame.get(angle_field))
        time_value = _as_float(frame.get("relative_time"))
        if angle is None or time_value is None:
            continue
        points.append((time_value, angle))

    points.sort(key=lambda item: item[0])
    in_range_frames = sum(1 for _, angle in points if low <= angle <= high)

    tut_seconds = 0.0
    for index in range(1, len(points)):
        previous_time, previous_angle = points[index - 1]
        current_time, current_angle = points[index]
        tut_seconds += _segment_tut_seconds(previous_time, previous_angle, current_time, current_angle, low, high)

    return {
        "tut_seconds": tut_seconds,
        "in_range_frames": in_range_frames,
        "total_frames": len(points),
        "method": "linear_interpolation",
    }


def _segment_tut_seconds(t0: float, a0: float, t1: float, a1: float, low: float, high: float) -> float:
    duration = t1 - t0
    if duration <= 0.0:
        return 0.0

    delta_angle = a1 - a0
    if abs(delta_angle) <= 1e-12:
        return duration if low <= a0 <= high else 0.0

    if delta_angle > 0:
        enter_fraction = (low - a0) / delta_angle
        exit_fraction = (high - a0) / delta_angle
    else:
        enter_fraction = (high - a0) / delta_angle
        exit_fraction = (low - a0) / delta_angle

    inside_start = max(0.0, min(enter_fraction, exit_fraction))
    inside_end = min(1.0, max(enter_fraction, exit_fraction))
    if inside_end <= inside_start:
        return 0.0
    return (inside_end - inside_start) * duration


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    assert abs(actual - expected) <= tolerance, f"expected {expected}, got {actual}"


if __name__ == "__main__":
    inside = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 45}, {"relative_time": 2.0, "flexion_angle": 55}],
        (40, 60),
    )
    _assert_close(inside["tut_seconds"], 2.0)

    entering = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 20}, {"relative_time": 2.0, "flexion_angle": 60}],
        (40, 80),
    )
    _assert_close(entering["tut_seconds"], 1.0)

    leaving = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 50}, {"relative_time": 2.0, "flexion_angle": 90}],
        (40, 70),
    )
    _assert_close(leaving["tut_seconds"], 1.0)

    crossing = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 20}, {"relative_time": 4.0, "flexion_angle": 100}],
        (40, 60),
    )
    _assert_close(crossing["tut_seconds"], 1.0)

    flat_inside = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 50}, {"relative_time": 3.0, "flexion_angle": 50}],
        (40, 60),
    )
    _assert_close(flat_inside["tut_seconds"], 3.0)

    flat_outside = compute_tut(
        [{"relative_time": 0.0, "flexion_angle": 20}, {"relative_time": 3.0, "flexion_angle": 20}],
        (40, 60),
    )
    _assert_close(flat_outside["tut_seconds"], 0.0)

    non_increasing = compute_tut(
        [{"relative_time": 1.0, "flexion_angle": 50}, {"relative_time": 1.0, "flexion_angle": 50}],
        (40, 60),
    )
    _assert_close(non_increasing["tut_seconds"], 0.0)

    skipped = compute_tut(
        [
            {"relative_time": 0.0, "flexion_angle": 50},
            {"relative_time": 1.0},
            {"flexion_angle": 55},
            {"relative_time": 2.0, "flexion_angle": 50},
        ],
        (60, 40),
    )
    _assert_close(skipped["tut_seconds"], 2.0)
    assert skipped["in_range_frames"] == 2
    assert skipped["total_frames"] == 2
    assert skipped["method"] == "linear_interpolation"

    print("tut inline tests passed")
