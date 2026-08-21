"""Select the pose backend without changing the current MediaPipe default."""

from __future__ import annotations

import os
from dataclasses import dataclass


SUPPORTED_BACKENDS = {"mediapipe", "rknn", "auto"}
DEFAULT_BACKEND = "mediapipe"


@dataclass(frozen=True)
class PoseBackendSelection:
    requested_backend: str
    actual_backend: str
    available: bool
    fallback_used: bool
    backend_error_message: str | None
    message: str

    @property
    def requested(self) -> str:
        return self.requested_backend

    @property
    def backend(self) -> str:
        return self.actual_backend


def requested_pose_backend() -> str:
    requested = os.environ.get("POSE_BACKEND", DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND
    if requested not in SUPPORTED_BACKENDS:
        return DEFAULT_BACKEND
    return requested


def resolve_pose_backend(probe_rknn=None) -> PoseBackendSelection:
    raw_requested = os.environ.get("POSE_BACKEND", DEFAULT_BACKEND).strip().lower() or DEFAULT_BACKEND
    if raw_requested not in SUPPORTED_BACKENDS:
        return PoseBackendSelection(
            requested_backend=raw_requested,
            actual_backend=DEFAULT_BACKEND,
            available=True,
            fallback_used=True,
            backend_error_message=f"Unsupported POSE_BACKEND={raw_requested!r}",
            message=f"Unsupported POSE_BACKEND={raw_requested!r}; fallback to mediapipe.",
        )
    requested = raw_requested
    if requested == "mediapipe":
        return PoseBackendSelection(
            requested_backend=requested,
            actual_backend="mediapipe",
            available=True,
            fallback_used=False,
            backend_error_message=None,
            message="Using MediaPipe CPU pose backend.",
        )
    if requested == "auto":
        if probe_rknn is not None:
            try:
                probe_rknn()
                return PoseBackendSelection(
                    requested_backend=requested,
                    actual_backend="rknn",
                    available=True,
                    fallback_used=False,
                    backend_error_message=None,
                    message="Using RKNN pose backend selected by auto mode.",
                )
            except Exception as exc:
                return PoseBackendSelection(
                    requested_backend=requested,
                    actual_backend="mediapipe",
                    available=True,
                    fallback_used=True,
                    backend_error_message=str(exc),
                    message=f"RKNN unavailable in auto mode; fallback to mediapipe: {exc}",
                )
        return PoseBackendSelection(
            requested_backend=requested,
            actual_backend="mediapipe",
            available=True,
            fallback_used=True,
            backend_error_message="RKNN probe was not provided.",
            message="RKNN probe was not provided; fallback to mediapipe.",
        )
    if probe_rknn is not None:
        probe_rknn()
    return PoseBackendSelection(
        requested_backend=requested,
        actual_backend="rknn",
        available=True,
        fallback_used=False,
        backend_error_message=None,
        message="Using forced RKNN pose backend.",
    )


def get_pose_backend() -> PoseBackendSelection:
    return resolve_pose_backend()


if __name__ == "__main__":
    print(resolve_pose_backend())
