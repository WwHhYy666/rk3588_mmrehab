"""Select the pose backend without changing the current MediaPipe default."""

from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_BACKENDS = {"mediapipe", "rknn", "auto"}
DEFAULT_BACKEND = "mediapipe"


@dataclass(frozen=True)
class PoseBackendSelection:
    requested: str
    backend: str
    available: bool
    fallback_used: bool
    message: str


def get_pose_backend() -> PoseBackendSelection:
    requested = os.environ.get("POSE_BACKEND", DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND
    if requested not in SUPPORTED_BACKENDS:
        return PoseBackendSelection(
            requested=requested,
            backend=DEFAULT_BACKEND,
            available=True,
            fallback_used=True,
            message=f"Unsupported POSE_BACKEND={requested!r}; fallback to mediapipe.",
        )
    if requested == "mediapipe":
        return PoseBackendSelection(
            requested=requested,
            backend="mediapipe",
            available=True,
            fallback_used=False,
            message="Using MediaPipe CPU pose backend.",
        )
    if requested == "auto":
        return PoseBackendSelection(
            requested=requested,
            backend="mediapipe",
            available=True,
            fallback_used=True,
            message="RKNN pose backend is not implemented yet; fallback to mediapipe.",
        )
    return PoseBackendSelection(
        requested=requested,
        backend="rknn",
        available=False,
        fallback_used=False,
        message="RKNN pose backend is not implemented yet. Run MediaPipe or the placeholder smoke test for now.",
    )


if __name__ == "__main__":
    print(get_pose_backend())
