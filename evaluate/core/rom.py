"""Range-of-motion calculation for frame sequences."""

from __future__ import annotations

from typing import Any


def compute_rom(frames: list[dict[str, Any]], angle_field: str = "flexion_angle") -> dict[str, Any]:
    """Return min, max, ROM, and frame indices for the configured angle field."""
    values: list[tuple[int, float]] = []
    for position, frame in enumerate(frames):
        value = _as_float(frame.get(angle_field))
        if value is None:
            continue
        frame_index = int(_as_float(frame.get("frame_index")) or position)
        values.append((frame_index, value))

    if not values:
        raise ValueError(f"no usable angle values found for field: {angle_field}")

    frame_at_min, min_angle = min(values, key=lambda item: item[1])
    frame_at_max, max_angle = max(values, key=lambda item: item[1])
    return {
        "min": min_angle,
        "max": max_angle,
        "rom": max_angle - min_angle,
        "frame_at_max": frame_at_max,
        "frame_at_min": frame_at_min,
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
        {"frame_index": 0, "flexion_angle": 10},
        {"frame_index": 1, "flexion_angle": 30},
        {"frame_index": 2, "flexion_angle": 18},
    ]
    result = compute_rom(sample_frames)
    assert result == {"min": 10.0, "max": 30.0, "rom": 20.0, "frame_at_max": 1, "frame_at_min": 0}
    print("rom inline tests passed")
