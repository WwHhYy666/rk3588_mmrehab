"""RKNNLite backend wrapper for YOLOv8-Pose."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from vision.rknn_pose.pose_result import PoseResult
from vision.rknn_pose.yolov8_pose_postprocess import draw_poses, output_shapes, postprocess_yolov8_pose, prepare_input


DEFAULT_MODEL_PATH = "/home/elf/models/yolov8n-pose.rknn"


class RKNNPoseBackend:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int | None = None,
        conf_thres: float | None = None,
        nms_thres: float | None = None,
        keypoint_thres: float | None = None,
    ) -> None:
        self.model_path = str(model_path or os.environ.get("RKNN_POSE_MODEL", DEFAULT_MODEL_PATH))
        self.input_size = int(input_size or os.environ.get("RKNN_POSE_INPUT_SIZE", "640"))
        self.conf_thres = float(conf_thres if conf_thres is not None else os.environ.get("RKNN_POSE_CONF_THRES", "0.25"))
        self.nms_thres = float(nms_thres if nms_thres is not None else os.environ.get("RKNN_POSE_NMS_THRES", "0.45"))
        self.keypoint_thres = float(keypoint_thres if keypoint_thres is not None else os.environ.get("RKNN_POSE_KEYPOINT_THRES", "0.30"))
        self.top_k = int(os.environ.get("RKNN_POSE_TOPK", "100"))
        self.max_det = int(os.environ.get("RKNN_POSE_MAX_DET", "5"))
        self.rknn: Any | None = None

    def load(self) -> None:
        model = Path(self.model_path)
        if not model.exists():
            raise FileNotFoundError(f"RKNN model not found: {model}")
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:  # pragma: no cover - board only
            raise RuntimeError(f"RKNNLite import failed: {exc}") from exc

        rknn = RKNNLite()
        ret = rknn.load_rknn(str(model))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        self.rknn = rknn

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        if self.rknn is None:
            self.load()
        assert self.rknn is not None

        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        input_tensor, meta = prepare_input(frame_bgr, self.input_size)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[input_tensor])
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        shapes = output_shapes(outputs)

        postprocess_start = time.perf_counter()
        try:
            detections = postprocess_yolov8_pose(
                outputs,
                meta,
                conf_thres=self.conf_thres,
                nms_thres=self.nms_thres,
                keypoint_thres=self.keypoint_thres,
                top_k=self.top_k,
                max_det=self.max_det,
            )
            postprocess_error = None
        except Exception as exc:
            detections = []
            postprocess_error = f"{exc}; output_shapes={shapes}"
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

        draw_start = time.perf_counter()
        annotated = draw_poses(frame_bgr, detections, keypoint_thres=self.keypoint_thres)
        draw_ms = (time.perf_counter() - draw_start) * 1000.0
        total_pose_ms = (time.perf_counter() - total_start) * 1000.0
        performance_ms = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "draw_ms": round(draw_ms, 2),
            "total_pose_ms": round(total_pose_ms, 2),
        }
        return PoseResult(
            backend="rknn",
            fps=1000.0 / total_pose_ms if total_pose_ms > 0 else None,
            keypoints={},
            raw=outputs,
            annotated_frame=annotated,
            meta={
                "detections": detections,
                "postprocess_error": postprocess_error,
                "output_shapes": shapes,
                "performance_ms": performance_ms,
                "inference_ms": inference_ms,
                "preprocess_ms": preprocess_ms,
                "postprocess_ms": postprocess_ms,
                "draw_ms": draw_ms,
                "total_pose_ms": total_pose_ms,
                "model_path": self.model_path,
                "input_size": self.input_size,
                "conf_thres": self.conf_thres,
                "nms_thres": self.nms_thres,
                "keypoint_thres": self.keypoint_thres,
                "top_k": self.top_k,
                "max_det": self.max_det,
            },
        )

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
