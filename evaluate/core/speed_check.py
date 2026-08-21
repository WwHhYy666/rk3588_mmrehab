"""Angular speed checks for frame sequences."""

from __future__ import annotations

from typing import Any


def check_speed(frames: list[dict[str, Any]], angle_field: str = "flexion_angle") -> dict[str, float]:
    """Return peak and mean angular velocity from adjacent frame angle changes."""
    points: list[tuple[float, float]] = []
    for frame in frames:
        angle = _as_float(frame.get(angle_field))
        time_value = _as_float(frame.get("relative_time"))
        if angle is None or time_value is None:
            continue
        points.append((time_value, angle))

    velocities: list[float] = []
    for index in range(1, len(points)):
        previous_time, previous_angle = points[index - 1]
        current_time, current_angle = points[index]
        delta_time = current_time - previous_time
        if delta_time <= 1e-12:
            continue
        velocities.append(abs((current_angle - previous_angle) / delta_time))

    if not velocities:
        return {"peak_angular_velocity": 0.0, "mean_angular_velocity": 0.0}

    return {
        "peak_angular_velocity": max(velocities),
        "mean_angular_velocity": sum(velocities) / len(velocities),
    }


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
    sample_frames = [
        {"relative_time": 0.0, "flexion_angle": 10},
        {"relative_time": 1.0, "flexion_angle": 40},
        {"relative_time": 3.0, "flexion_angle": 50},
    ]
    result = check_speed(sample_frames)
    assert result["peak_angular_velocity"] == 30.0
    assert result["mean_angular_velocity"] == 17.5
    print("speed_check inline tests passed")
