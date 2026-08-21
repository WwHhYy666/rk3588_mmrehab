"""Common pose result container for future MediaPipe/RKNN adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoseResult:
    backend: str
    fps: float | None
    keypoints: dict[str, dict[str, float | None]]
    raw: Any = None
    annotated_frame: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)
