"""Angle helpers used as fallbacks when frame-level angle fields are missing."""

from __future__ import annotations

import math
from collections.abc import Sequence


def angle_from_3_points(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Return the included angle at point b in degrees using the cosine rule."""
    if len(a) != len(b) or len(b) != len(c):
        raise ValueError("points must have the same dimension")
    if len(a) not in (2, 3):
        raise ValueError("points must be 2D or 3D")

    ba = [float(a[i]) - float(b[i]) for i in range(len(a))]
    bc = [float(c[i]) - float(b[i]) for i in range(len(c))]
    ba_len = math.sqrt(sum(value * value for value in ba))
    bc_len = math.sqrt(sum(value * value for value in bc))
    if ba_len <= 1e-12 or bc_len <= 1e-12:
        raise ValueError("angle is undefined for zero-length vectors")

    dot = sum(ba[i] * bc[i] for i in range(len(ba)))
    cos_value = max(-1.0, min(1.0, dot / (ba_len * bc_len)))
    return math.degrees(math.acos(cos_value))


if __name__ == "__main__":
    right_angle = angle_from_3_points((1, 0), (0, 0), (0, 1))
    straight_angle = angle_from_3_points((-1, 0, 0), (0, 0, 0), (1, 0, 0))
    assert round(right_angle, 6) == 90.0
    assert round(straight_angle, 6) == 180.0
    print("angle_utils inline tests passed")
