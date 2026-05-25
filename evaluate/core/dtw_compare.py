"""DTW comparison helpers for template and attempt angle sequences."""

from __future__ import annotations

import math
from typing import Any

try:
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover - kept as a portable fallback.
    fastdtw = None


def dtw_score(
    template_angles: list[float],
    attempt_angles: list[float],
    window_ratio: float | None = 0.25,
    smooth_window: int = 1,
) -> dict[str, Any]:
    """Return DTW distance, normalized distance, and path length for two sequences."""
    left = _moving_average([float(value) for value in template_angles], smooth_window)
    right = _moving_average([float(value) for value in attempt_angles], smooth_window)
    if not left or not right:
        raise ValueError("template_angles and attempt_angles must not be empty")

    if window_ratio is None and fastdtw is not None:
        # fastdtw is kept for the unrestricted path. Windowed DTW uses _plain_dtw
        # so Sakoe-Chiba constraints behave predictably for demos and tests.
        distance, path = fastdtw(left, right, dist=lambda a, b: abs(a - b))
    else:
        distance, path = _plain_dtw(left, right, window_ratio=window_ratio)

    path_length = max(1, len(path))
    return {
        "distance": float(distance),
        "normalized_distance": float(distance) / path_length,
        "path_length": path_length,
    }


def _moving_average(values: list[float], smooth_window: int) -> list[float]:
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    if smooth_window == 1 or not values:
        return values

    radius_left = (smooth_window - 1) // 2
    radius_right = smooth_window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius_left)
        end = min(len(values), index + radius_right + 1)
        window = values[start:end]
        smoothed.append(sum(window) / len(window))
    return smoothed


def _plain_dtw(
    left: list[float],
    right: list[float],
    window_ratio: float | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    rows = len(left)
    cols = len(right)
    if window_ratio is not None and window_ratio < 0:
        raise ValueError("window_ratio must be >= 0 or None")

    window = max(rows, cols) if window_ratio is None else max(abs(rows - cols), math.ceil(max(rows, cols) * window_ratio))
    costs = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
    parents: list[list[tuple[int, int] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    costs[0][0] = 0.0

    for i in range(1, rows + 1):
        start_j = max(1, i - window)
        end_j = min(cols, i + window)
        for j in range(start_j, end_j + 1):
            cost = abs(left[i - 1] - right[j - 1])
            candidates = (
                (costs[i - 1][j], i - 1, j),
                (costs[i][j - 1], i, j - 1),
                (costs[i - 1][j - 1], i - 1, j - 1),
            )
            previous_cost, parent_i, parent_j = min(candidates, key=lambda item: item[0])
            costs[i][j] = cost + previous_cost
            parents[i][j] = (parent_i, parent_j)

    if math.isinf(costs[rows][cols]):
        raise ValueError("DTW path is unreachable with the configured window_ratio")

    path: list[tuple[int, int]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        elif i > 0:
            path.append((i - 1, 0))
        elif j > 0:
            path.append((0, j - 1))

        parent = parents[i][j]
        if parent is None:
            break
        next_i, next_j = parent
        if next_i == i and next_j == j:
            break
        i, j = next_i, next_j

    path.reverse()
    return costs[rows][cols], path


if __name__ == "__main__":
    same = dtw_score([0, 10, 20], [0, 10, 20])
    assert same["distance"] == 0.0
    assert same["normalized_distance"] == 0.0

    shifted = dtw_score([0, 10, 20], [0, 8, 18])
    assert shifted["distance"] > same["distance"]
    assert shifted["path_length"] >= 3

    smoothed = dtw_score([0, 10, 20, 30], [0, 12, 18, 30], smooth_window=3)
    assert set(smoothed) == {"distance", "normalized_distance", "path_length"}

    windowed = dtw_score([0, 5, 10, 15, 20], [0, 6, 11, 16, 21], window_ratio=0.2)
    assert windowed["distance"] >= 0.0
    assert windowed["normalized_distance"] >= 0.0
    assert windowed["path_length"] >= 5

    unrestricted = dtw_score([0, 10, 20], [0, 8, 18], window_ratio=None)
    assert unrestricted["distance"] > 0.0

    try:
        dtw_score([], [1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("empty sequence should raise ValueError")

    print("dtw_compare inline tests passed")
