"""Map COCO17 pose keypoints into this project's rehab keypoint names."""

from __future__ import annotations

from typing import Any


COCO17_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

REHAB_REQUIRED_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def coco17_to_rehab_keypoints(coco_keypoints: list[Any] | tuple[Any, ...]) -> dict[str, dict[str, float | None]]:
    rehab: dict[str, dict[str, float | None]] = {}
    for name in REHAB_REQUIRED_NAMES:
        index = COCO17_INDEX[name]
        point = coco_keypoints[index] if index < len(coco_keypoints) else None
        rehab[name] = _normalize_point(point)
    return rehab


def _normalize_point(point: Any) -> dict[str, float | None]:
    if point is None:
        return {"x": None, "y": None, "z": 0.0, "visibility": 0.0}
    if isinstance(point, dict):
        return {
            "x": _as_float(point.get("x")),
            "y": _as_float(point.get("y")),
            "z": _as_float(point.get("z")) or 0.0,
            "visibility": _as_float(point.get("visibility", point.get("score"))) or 0.0,
        }
    if isinstance(point, (list, tuple)):
        x = _as_float(point[0]) if len(point) > 0 else None
        y = _as_float(point[1]) if len(point) > 1 else None
        score = _as_float(point[2]) if len(point) > 2 else None
        return {"x": x, "y": y, "z": 0.0, "visibility": score if score is not None else 1.0}
    return {"x": None, "y": None, "z": 0.0, "visibility": 0.0}


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
    sample = [[0.0, 0.0, 0.0] for _ in range(17)]
    print(coco17_to_rehab_keypoints(sample))
