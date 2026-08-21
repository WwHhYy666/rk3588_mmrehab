from __future__ import annotations

from typing import Any


# Kept local to the extension package so its schema does not enlarge the
# stable lower-body REHAB_REQUIRED_NAMES contract.
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def normalize_coco17_points(points: Any) -> dict[str, dict[str, float]]:
    """Accept a COCO17 list or named map and return the extension point map."""
    if isinstance(points, dict):
        source = points
        return {name: dict(source[name]) for name in COCO17_NAMES if isinstance(source.get(name), dict)}
    if not isinstance(points, (list, tuple)):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for index, name in enumerate(COCO17_NAMES):
        item = points[index] if index < len(points) else None
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
            visibility = float(item[2]) if len(item) > 2 else 0.0
        except (TypeError, ValueError):
            continue
        normalized[name] = {"x": x, "y": y, "visibility": visibility}
    return normalized
