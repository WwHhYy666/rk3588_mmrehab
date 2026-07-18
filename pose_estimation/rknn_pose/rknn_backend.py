"""RKNNLite wrapper for the maintained NPU 8085 pose pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pose_estimation.rknn_pose.pose_result import PoseResult
from pose_estimation.rknn_pose.yolov5n_rtmpose_backend import YOLOv5nRTMPoseBackend


PIPELINE = "yolov5n_rtmpose"
SUPPORTED_PIPELINES = {PIPELINE}


class RKNNPoseBackend:
    """Expose the YOLOv5n raw + RTMPose cascade through the shared app API."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int | None = None,
        conf_thres: float | None = None,
        nms_thres: float | None = None,
        keypoint_thres: float | None = None,
    ) -> None:
        del input_size, conf_thres, nms_thres
        self.pipeline = _normalize_pipeline(os.environ.get("RKNN_POSE_PIPELINE", PIPELINE))
        self.backend = YOLOv5nRTMPoseBackend(
            det_model_path=model_path,
            keypoint_thres=keypoint_thres,
        )
        self.model_path = self.backend.model_path
        self.keypoint_thres = self.backend.keypoint_thres

    def load(self) -> None:
        self.backend.load()

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        return self.backend.infer(frame_bgr)

    def release(self) -> None:
        self.backend.release()

    def set_enabled_fn(self, enabled_fn: Callable[[], bool] | None) -> None:
        self.backend.set_enabled_fn(enabled_fn)

    def reset_tracking_state(self, reason: str = "manual") -> None:
        self.backend.reset_tracking_state(reason)

    def resource_snapshot(self) -> dict[str, Any]:
        return self.backend.resource_snapshot()

    def diagnostics_snapshot(self) -> dict[str, Any]:
        return self.backend.diagnostics_snapshot()


def _normalize_pipeline(value: str | None) -> str:
    text = str(value or PIPELINE).strip().lower() or PIPELINE
    if text not in SUPPORTED_PIPELINES:
        raise ValueError(
            f"Unsupported RKNN_POSE_PIPELINE={text!r}; "
            f"the latest branch only maintains {PIPELINE!r}."
        )
    return text
