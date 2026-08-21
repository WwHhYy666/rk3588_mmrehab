"""RTMDet + RTMPose RKNN cascade backend helpers.

This module is adapted from ``rknn/rknn_rtmdet_rtmpose_demo.py`` and keeps the
NPU-specific two-stage pose pipeline isolated from the 8082 training flow.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision.rknn_pose.pose_result import PoseResult
from vision.rknn_pose.yolov8_pose_postprocess import (
    postprocess_yolov8_pose_with_diagnostics,
    prepare_input as prepare_yolov8_input,
)


COCO17_EDGES = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

KEYPOINT_COLORS = (
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
)

LINK_COLORS = (
    (0, 255, 0),
    (0, 255, 0),
    (0, 128, 255),
    (0, 128, 255),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (0, 255, 0),
    (0, 128, 255),
    (0, 255, 0),
    (0, 128, 255),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
    (255, 153, 51),
)


class RKNNModel:
    """Small RKNNLite wrapper for one .rknn model."""

    def __init__(self, model_path: str | Path, *, core_mask: str = "auto") -> None:
        self.model_path = str(model_path)
        self.core_mask = str(core_mask or "auto")
        self.rknn: Any | None = None

    def load(self) -> None:
        if self.rknn is not None:
            return
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
            raise RuntimeError(f"load_rknn failed: {ret}; model={model}")
        runtime_kwargs: dict[str, Any] = {}
        if self.core_mask != "auto" and hasattr(RKNNLite, self.core_mask):
            runtime_kwargs["core_mask"] = getattr(RKNNLite, self.core_mask)
        elif hasattr(RKNNLite, "NPU_CORE_0_1_2"):
            runtime_kwargs["core_mask"] = RKNNLite.NPU_CORE_0_1_2
        ret = rknn.init_runtime(**runtime_kwargs)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}; model={model}")
        self.rknn = rknn

    def inference(self, inputs: list[np.ndarray]) -> list[Any]:
        if self.rknn is None:
            self.load()
        assert self.rknn is not None
        outputs = self.rknn.inference(inputs=inputs)
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        return list(outputs)

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


class RTMDetRTMPoseBackend:
    """Two-stage person detector + top-down pose backend."""

    def __init__(
        self,
        *,
        det_model_path: str | Path | None = None,
        pose_model_path: str | Path | None = None,
        det_score_thres: float | None = None,
        nms_thres: float | None = None,
        keypoint_thres: float | None = None,
        max_persons: int | None = None,
        input_layout: str | None = None,
        head_layout: str | None = None,
        bbox_decode_mode: str | None = None,
        simcc_score_mode: str | None = None,
        core_mask: str | None = None,
    ) -> None:
        self.det_model_path = str(det_model_path or os.environ.get("RKNN_DET_MODEL", "rknn/rtmdet_fp16.rknn"))
        self.pose_model_path = str(pose_model_path or os.environ.get("RKNN_RTMPOSE_MODEL", "rknn/rtmpose_fp16.rknn"))
        self.det_score_thres = float(det_score_thres if det_score_thres is not None else os.environ.get("RKNN_DET_SCORE_THRES", "0.50"))
        self.nms_thres = float(nms_thres if nms_thres is not None else os.environ.get("RKNN_DET_NMS_THRES", "0.45"))
        self.keypoint_thres = float(keypoint_thres if keypoint_thres is not None else os.environ.get("RKNN_POSE_KEYPOINT_THRES", "0.25"))
        self.max_persons = int(max_persons if max_persons is not None else os.environ.get("RKNN_MAX_POSE_PERSONS", "1"))
        self.input_layout = _choice(os.environ.get("RKNN_INPUT_LAYOUT") if input_layout is None else input_layout, {"nchw", "nhwc"}, "nchw")
        self.head_layout = _choice(os.environ.get("RKNN_RTMDET_HEAD_LAYOUT") if head_layout is None else head_layout, {"auto", "nchw", "nhwc"}, "auto")
        self.bbox_decode_mode = _choice(os.environ.get("RKNN_RTMDET_BBOX_DECODE_MODE") if bbox_decode_mode is None else bbox_decode_mode, {"auto", "stride", "none"}, "auto")
        self.simcc_score_mode = _choice(os.environ.get("RKNN_RTMPOSE_SIMCC_SCORE_MODE") if simcc_score_mode is None else simcc_score_mode, {"sqrt", "avg"}, "sqrt")
        self.nms_pre = int(os.environ.get("RKNN_DET_NMS_PRE", "100"))
        self.det_interval = max(1, int(os.environ.get("RKNN_DET_INTERVAL", "8")))
        self.det_cache_seconds = max(0.0, float(os.environ.get("RKNN_DET_CACHE_SECONDS", "1.5")))
        self.pose_interval = max(1, int(os.environ.get("RKNN_POSE_INTERVAL", "2")))
        self.pose_cache_seconds = max(0.0, float(os.environ.get("RKNN_POSE_CACHE_SECONDS", "1.0")))
        self.rtmpose_bbox_expand = max(1.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_EXPAND", "1.35")))
        self.rtmpose_bbox_top_expand = max(0.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_TOP_EXPAND", "0.20")))
        self.rtmpose_debug_crop_every = max(0, int(os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_EVERY", "0")))
        self.rtmpose_debug_crop_path = os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_PATH", "outputs/debug_rtmpose_crop.jpg")
        self.person_select = _choice(os.environ.get("RKNN_PERSON_SELECT"), {"score", "largest", "center", "largest_center"}, "largest_center")
        self.det_model = RKNNModel(self.det_model_path, core_mask=str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto")))
        self.pose_model = RKNNModel(self.pose_model_path, core_mask=str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto")))
        self._frame_index = 0
        self._cached_bboxes = np.zeros((0, 4), dtype=np.float32)
        self._cached_scores = np.zeros((0,), dtype=np.float32)
        self._cached_at = 0.0
        self._cached_frame_shape: tuple[int, ...] | None = None
        self._cached_det_shapes: list[tuple[int, ...]] = []
        self._cached_det_meta: dict[str, Any] = {}
        self._cached_pose_detections: list[dict[str, Any]] = []
        self._cached_pose_at = 0.0
        self._cached_pose_frame_shape: tuple[int, ...] | None = None

    @property
    def model_path(self) -> str:
        return f"{self.det_model_path};{self.pose_model_path}"

    def load(self) -> None:
        self.det_model.load()
        self.pose_model.load()

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        self.load()
        total_start = time.perf_counter()
        postprocess_error = None
        det_outputs: list[Any] = []
        pose_output_shapes: list[list[tuple[int, ...]]] = []
        frame_shape = tuple(frame_bgr.shape)
        frame_index = self._frame_index
        self._frame_index += 1
        now = time.perf_counter()
        cache_valid = self._cache_valid(frame_shape, now)
        run_detector = (not cache_valid) or self.det_interval <= 1 or frame_index % self.det_interval == 0
        det_cache_hit = False
        det_preprocess_ms = 0.0
        det_inference_ms = 0.0
        det_postprocess_ms = 0.0
        det_shapes: list[tuple[int, ...]] = []
        det_meta: dict[str, Any] = {}
        bboxes = np.zeros((0, 4), dtype=np.float32)
        bbox_scores = np.zeros((0,), dtype=np.float32)
        pose_bboxes = np.zeros((0, 4), dtype=np.float32)
        pose_bbox_debug: list[list[float]] = []

        if run_detector:
            preprocess_start = time.perf_counter()
            det_input, scale, pad_left, pad_top = preprocess_det(frame_bgr, self.input_layout)
            det_preprocess_ms = _elapsed_ms(preprocess_start)

            det_start = time.perf_counter()
            det_outputs = self.det_model.inference([det_input])
            det_inference_ms = _elapsed_ms(det_start)
            det_shapes = output_shapes(det_outputs)

            det_post_start = time.perf_counter()
            try:
                bboxes, bbox_scores, det_meta = postprocess_rtmdet(
                    det_outputs,
                    score_thr=self.det_score_thres,
                    nms_thr=self.nms_thres,
                    scale=scale,
                    pad_left=pad_left,
                    pad_top=pad_top,
                    frame_shape=frame_bgr.shape,
                    max_persons=self.max_persons,
                    bbox_decode_mode=self.bbox_decode_mode,
                    head_layout=self.head_layout,
                    nms_pre=self.nms_pre,
                    person_select=self.person_select,
                )
                if len(bboxes) > 0:
                    self._update_cache(bboxes, bbox_scores, frame_shape, now, det_shapes, det_meta)
                elif cache_valid:
                    bboxes, bbox_scores, det_meta, det_shapes = self._cached_detection()
                    det_cache_hit = True
            except Exception as exc:
                if cache_valid:
                    bboxes, bbox_scores, det_meta, det_shapes = self._cached_detection()
                    det_cache_hit = True
                else:
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    bbox_scores = np.zeros((0,), dtype=np.float32)
                    det_meta = {}
                    postprocess_error = f"RTMDet postprocess failed: {exc}; output_shapes={det_shapes}"
            det_postprocess_ms = _elapsed_ms(det_post_start)
        else:
            bboxes, bbox_scores, det_meta, det_shapes = self._cached_detection()
            det_cache_hit = True

        if postprocess_error is None and len(bboxes) > 0:
            pose_bboxes = expand_bboxes_for_rtmpose(
                bboxes,
                frame_bgr.shape,
                expand=self.rtmpose_bbox_expand,
                top_expand=self.rtmpose_bbox_top_expand,
            )
            pose_bbox_debug = [[round(float(value), 2) for value in bbox.tolist()] for bbox in pose_bboxes]

        pose_preprocess_ms = 0.0
        pose_inference_ms = 0.0
        pose_postprocess_ms = 0.0
        keypoints_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        pose_cache_hit = False
        pose_reused = False
        pose_cache_valid = self._pose_cache_valid(frame_shape, now)
        run_pose = (
            postprocess_error is None
            and len(bboxes) > 0
            and (
                not pose_cache_valid
                or self.pose_interval <= 1
                or frame_index % self.pose_interval == 0
                or not det_cache_hit
            )
        )

        if postprocess_error is None and run_pose:
            for bbox_index, bbox in enumerate(pose_bboxes):
                pose_pre_start = time.perf_counter()
                pose_input, inv_matrix, pose_crop = preprocess_pose(frame_bgr, bbox, self.input_layout, return_crop=True)
                pose_preprocess_ms += _elapsed_ms(pose_pre_start)

                pose_infer_start = time.perf_counter()
                pose_outputs = self.pose_model.inference([pose_input])
                pose_inference_ms += _elapsed_ms(pose_infer_start)
                pose_output_shapes.append(output_shapes(pose_outputs))

                pose_post_start = time.perf_counter()
                try:
                    keypoints, scores = decode_simcc(pose_outputs, inv_matrix, score_mode=self.simcc_score_mode)
                    keypoints_list.append(keypoints)
                    scores_list.append(scores)
                    if bbox_index == 0 and self.rtmpose_debug_crop_every > 0 and frame_index % self.rtmpose_debug_crop_every == 0:
                        save_rtmpose_debug_crop(
                            pose_crop,
                            pose_outputs,
                            self.rtmpose_debug_crop_path,
                            score_mode=self.simcc_score_mode,
                            keypoint_thres=self.keypoint_thres,
                        )
                except Exception as exc:
                    postprocess_error = f"RTMPose postprocess failed: {exc}; output_shapes={pose_output_shapes}"
                    keypoints_list = []
                    scores_list = []
                    break
                finally:
                    pose_postprocess_ms += _elapsed_ms(pose_post_start)

        if postprocess_error is None and len(bboxes) > 0 and not run_pose and pose_cache_valid:
            detections = self._cached_pose()
            pose_cache_hit = True
            pose_reused = True
        else:
            detections = _build_detections(bboxes, bbox_scores, keypoints_list, scores_list)
            if detections:
                self._update_pose_cache(detections, frame_shape, now)
        draw_start = time.perf_counter()
        annotated = draw_poses(frame_bgr, detections, keypoint_thres=self.keypoint_thres)
        draw_ms = _elapsed_ms(draw_start)
        total_pose_ms = _elapsed_ms(total_start)
        performance_ms = {
            "preprocess_ms": round(det_preprocess_ms + pose_preprocess_ms, 2),
            "inference_ms": round(det_inference_ms + pose_inference_ms, 2),
            "postprocess_ms": round(det_postprocess_ms + pose_postprocess_ms, 2),
            "draw_ms": round(draw_ms, 2),
            "total_pose_ms": round(total_pose_ms, 2),
            "det_preprocess_ms": round(det_preprocess_ms, 2),
            "det_inference_ms": round(det_inference_ms, 2),
            "det_postprocess_ms": round(det_postprocess_ms, 2),
            "pose_preprocess_ms": round(pose_preprocess_ms, 2),
            "pose_inference_ms": round(pose_inference_ms, 2),
            "pose_postprocess_ms": round(pose_postprocess_ms, 2),
        }
        return PoseResult(
            backend="rknn",
            fps=1000.0 / total_pose_ms if total_pose_ms > 0 else None,
            keypoints={},
            raw={"det_outputs": det_outputs},
            annotated_frame=annotated,
            meta={
                "detections": detections,
                "postprocess_error": postprocess_error,
                "output_shapes": det_shapes,
                "det_output_shapes": det_shapes,
                "pose_output_shapes": pose_output_shapes,
                "rknn_decoder": "rtmdet_rtmpose",
                "rknn_pipeline": "rtmdet_rtmpose",
                "keypoint_conf_range": _range_or_none([score for scores in scores_list for score in scores.tolist()]),
                "keypoint_xy_range": _xy_range_from_keypoints(keypoints_list),
                "keypoint_restored_xy_range": _xy_range_from_keypoints(keypoints_list),
                "performance_ms": performance_ms,
                "inference_ms": performance_ms["inference_ms"],
                "preprocess_ms": performance_ms["preprocess_ms"],
                "postprocess_ms": performance_ms["postprocess_ms"],
                "draw_ms": draw_ms,
                "total_pose_ms": total_pose_ms,
                "model_path": self.model_path,
                "det_model_path": self.det_model_path,
                "pose_model_path": self.pose_model_path,
                "det_score_thres": self.det_score_thres,
                "nms_thres": self.nms_thres,
                "keypoint_thres": self.keypoint_thres,
                "max_persons": self.max_persons,
                "input_layout": self.input_layout,
                "head_layout": det_meta.get("head_layout", self.head_layout),
                "bbox_decode_mode": self.bbox_decode_mode,
                "simcc_score_mode": self.simcc_score_mode,
                "rtmpose_expanded_bbox": pose_bbox_debug[0] if pose_bbox_debug else None,
                "rtmpose_bbox_expand": self.rtmpose_bbox_expand,
                "rtmpose_bbox_top_expand": self.rtmpose_bbox_top_expand,
                "rtmpose_crop_input_size": [288, 384],
                "rtmpose_debug_crop_path": self.rtmpose_debug_crop_path if self.rtmpose_debug_crop_every > 0 else None,
                "rtmdet_pairs": det_meta.get("pairs"),
                "det_cache_hit": det_cache_hit,
                "det_interval": self.det_interval,
                "det_cache_seconds": self.det_cache_seconds,
                "pose_cache_hit": pose_cache_hit,
                "pose_reused": pose_reused,
                "pose_interval": self.pose_interval,
                "pose_cache_seconds": self.pose_cache_seconds,
                "person_select": self.person_select,
            },
        )

    def release(self) -> None:
        self.det_model.release()
        self.pose_model.release()

    def _cache_valid(self, frame_shape: tuple[int, ...], now: float) -> bool:
        if len(self._cached_bboxes) == 0:
            return False
        if self._cached_frame_shape != frame_shape:
            return False
        if self.det_cache_seconds <= 0:
            return False
        return now - self._cached_at <= self.det_cache_seconds

    def _update_cache(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        frame_shape: tuple[int, ...],
        now: float,
        det_shapes: list[tuple[int, ...]],
        det_meta: dict[str, Any],
    ) -> None:
        self._cached_bboxes = bboxes.copy()
        self._cached_scores = scores.copy()
        self._cached_frame_shape = frame_shape
        self._cached_at = now
        self._cached_det_shapes = list(det_shapes)
        self._cached_det_meta = dict(det_meta)

    def _cached_detection(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[tuple[int, ...]]]:
        meta = dict(self._cached_det_meta)
        meta["cache_reused"] = True
        return self._cached_bboxes.copy(), self._cached_scores.copy(), meta, list(self._cached_det_shapes)

    def _pose_cache_valid(self, frame_shape: tuple[int, ...], now: float) -> bool:
        if not self._cached_pose_detections:
            return False
        if self._cached_pose_frame_shape != frame_shape:
            return False
        if self.pose_cache_seconds <= 0:
            return False
        return now - self._cached_pose_at <= self.pose_cache_seconds

    def _update_pose_cache(self, detections: list[dict[str, Any]], frame_shape: tuple[int, ...], now: float) -> None:
        self._cached_pose_detections = [_copy_detection(item) for item in detections]
        self._cached_pose_frame_shape = frame_shape
        self._cached_pose_at = now

    def _cached_pose(self) -> list[dict[str, Any]]:
        return [_copy_detection(item) for item in self._cached_pose_detections]


class YOLOv8RTMPoseBackend:
    """YOLOv8 person box + top-down RTMPose backend.

    This keeps YOLOv8 responsible for finding a tight person ROI and uses
    RTMPose for the COCO17 keypoints consumed by the realtime rehab pipeline.
    """

    def __init__(
        self,
        *,
        yolo_model_path: str | Path | None = None,
        pose_model_path: str | Path | None = None,
        conf_thres: float | None = None,
        nms_thres: float | None = None,
        keypoint_thres: float | None = None,
        input_size: int | None = None,
        input_layout: str | None = None,
        simcc_score_mode: str | None = None,
        max_persons: int | None = None,
        core_mask: str | None = None,
    ) -> None:
        self.yolo_model_path = str(yolo_model_path or os.environ.get("RKNN_POSE_MODEL", "/home/elf/models/yolov8n-pose.rknn"))
        self.pose_model_path = str(pose_model_path or os.environ.get("RKNN_RTMPOSE_MODEL", "rknn/rtmpose_fp16.rknn"))
        self.conf_thres = float(conf_thres if conf_thres is not None else os.environ.get("RKNN_POSE_CONF_THRES", "0.35"))
        self.nms_thres = float(nms_thres if nms_thres is not None else os.environ.get("RKNN_POSE_NMS_THRES", "0.45"))
        self.keypoint_thres = float(keypoint_thres if keypoint_thres is not None else os.environ.get("RKNN_POSE_KEYPOINT_THRES", "0.20"))
        self.input_size = int(input_size if input_size is not None else os.environ.get("RKNN_POSE_INPUT_SIZE", "640"))
        self.input_layout = _choice(os.environ.get("RKNN_INPUT_LAYOUT") if input_layout is None else input_layout, {"nchw", "nhwc"}, "nchw")
        self.simcc_score_mode = _choice(os.environ.get("RKNN_RTMPOSE_SIMCC_SCORE_MODE") if simcc_score_mode is None else simcc_score_mode, {"sqrt", "avg"}, "sqrt")
        self.max_persons = int(max_persons if max_persons is not None else os.environ.get("RKNN_MAX_POSE_PERSONS", "1"))
        self.person_select = _choice(os.environ.get("RKNN_PERSON_SELECT"), {"score", "largest", "center", "largest_center"}, "largest_center")
        self.top_k = int(os.environ.get("RKNN_POSE_TOPK", "100"))
        self.keypoint_decode_mode = os.environ.get("RKNN_KEYPOINT_DECODE_MODE", "auto").strip().lower() or "auto"
        self.keypoint_anchor_order = os.environ.get("RKNN_KEYPOINT_ANCHOR_ORDER", "auto").strip().lower() or "auto"
        self.det_interval = max(1, int(os.environ.get("RKNN_YOLO_DET_INTERVAL", os.environ.get("RKNN_DET_INTERVAL", "8"))))
        self.det_cache_seconds = max(0.0, float(os.environ.get("RKNN_YOLO_DET_CACHE_SECONDS", os.environ.get("RKNN_DET_CACHE_SECONDS", "2.0"))))
        self.pose_interval = max(1, int(os.environ.get("RKNN_POSE_INTERVAL", "1")))
        self.pose_cache_seconds = max(0.0, float(os.environ.get("RKNN_POSE_CACHE_SECONDS", "1.0")))
        self.rtmpose_bbox_expand = max(1.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_EXPAND", "1.35")))
        self.rtmpose_bbox_top_expand = max(0.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_TOP_EXPAND", "0.20")))
        self.rtmpose_debug_crop_every = max(0, int(os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_EVERY", "0")))
        self.rtmpose_debug_crop_path = os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_PATH", "outputs/debug_rtmpose_crop.jpg")
        self.yolo_model = RKNNModel(self.yolo_model_path, core_mask=str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto")))
        self.pose_model = RKNNModel(self.pose_model_path, core_mask=str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto")))
        self._frame_index = 0
        self._cached_bboxes = np.zeros((0, 4), dtype=np.float32)
        self._cached_scores = np.zeros((0,), dtype=np.float32)
        self._cached_yolo_detections: list[dict[str, Any]] = []
        self._cached_at = 0.0
        self._cached_frame_shape: tuple[int, ...] | None = None
        self._cached_det_shapes: list[tuple[int, ...]] = []
        self._cached_yolo_meta: dict[str, Any] = {}
        self._cached_pose_detections: list[dict[str, Any]] = []
        self._cached_pose_at = 0.0
        self._cached_pose_frame_shape: tuple[int, ...] | None = None

    @property
    def model_path(self) -> str:
        return f"{self.yolo_model_path};{self.pose_model_path}"

    def load(self) -> None:
        self.yolo_model.load()
        self.pose_model.load()

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        self.load()
        total_start = time.perf_counter()
        postprocess_error = None
        yolo_outputs: list[Any] = []
        yolo_output_shapes: list[tuple[int, ...]] = []
        pose_output_shapes: list[list[tuple[int, ...]]] = []
        keypoints_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        yolo_meta: dict[str, Any] = {}
        frame_shape = tuple(frame_bgr.shape)
        frame_index = self._frame_index
        self._frame_index += 1
        now = time.perf_counter()
        det_cache_valid = self._cache_valid(frame_shape, now)
        run_detector = (not det_cache_valid) or self.det_interval <= 1 or frame_index % self.det_interval == 0
        det_cache_hit = False
        det_preprocess_ms = 0.0
        det_inference_ms = 0.0
        det_postprocess_ms = 0.0
        yolo_detections: list[dict[str, Any]] = []
        bboxes = np.zeros((0, 4), dtype=np.float32)
        bbox_scores = np.zeros((0,), dtype=np.float32)
        pose_bboxes = np.zeros((0, 4), dtype=np.float32)
        pose_bbox_debug: list[list[float]] = []

        if run_detector:
            det_pre_start = time.perf_counter()
            yolo_input, yolo_letterbox = prepare_yolov8_input(frame_bgr, self.input_size)
            det_preprocess_ms = _elapsed_ms(det_pre_start)

            det_infer_start = time.perf_counter()
            yolo_outputs = self.yolo_model.inference([yolo_input])
            det_inference_ms = _elapsed_ms(det_infer_start)
            yolo_output_shapes = output_shapes(yolo_outputs)

            det_post_start = time.perf_counter()
            try:
                yolo_detections, yolo_meta = postprocess_yolov8_pose_with_diagnostics(
                    yolo_outputs,
                    yolo_letterbox,
                    conf_thres=self.conf_thres,
                    nms_thres=self.nms_thres,
                    keypoint_thres=self.keypoint_thres,
                    top_k=self.top_k,
                    max_det=max(self.max_persons, 1),
                    keypoint_decode_mode=self.keypoint_decode_mode,
                    keypoint_anchor_order=self.keypoint_anchor_order,
                )
                postprocess_error = yolo_meta.get("postprocess_error")
                bboxes, bbox_scores = _detections_to_selected_bboxes(
                    yolo_detections,
                    frame_bgr.shape,
                    max_persons=max(self.max_persons, 1),
                    person_select=self.person_select,
                )
                if postprocess_error is None and len(bboxes) > 0:
                    self._update_cache(
                        bboxes,
                        bbox_scores,
                        yolo_detections,
                        frame_shape,
                        now,
                        yolo_output_shapes,
                        yolo_meta,
                    )
                elif det_cache_valid:
                    bboxes, bbox_scores, yolo_detections, yolo_meta, yolo_output_shapes = self._cached_detection()
                    postprocess_error = None
                    det_cache_hit = True
            except Exception as exc:
                if det_cache_valid:
                    bboxes, bbox_scores, yolo_detections, yolo_meta, yolo_output_shapes = self._cached_detection()
                    postprocess_error = None
                    det_cache_hit = True
                else:
                    postprocess_error = f"YOLOv8 person box postprocess failed: {exc}; output_shapes={yolo_output_shapes}"
                    yolo_meta = {"postprocess_error": postprocess_error}
            det_postprocess_ms = _elapsed_ms(det_post_start)
        else:
            bboxes, bbox_scores, yolo_detections, yolo_meta, yolo_output_shapes = self._cached_detection()
            det_cache_hit = True

        if postprocess_error is None and len(bboxes) == 0:
            postprocess_error = "YOLOv8 did not detect a person box for RTMPose"
        elif postprocess_error is None:
            pose_bboxes = expand_bboxes_for_rtmpose(
                bboxes,
                frame_bgr.shape,
                expand=self.rtmpose_bbox_expand,
                top_expand=self.rtmpose_bbox_top_expand,
            )
            pose_bbox_debug = [[round(float(value), 2) for value in bbox.tolist()] for bbox in pose_bboxes]

        pose_preprocess_ms = 0.0
        pose_inference_ms = 0.0
        pose_postprocess_ms = 0.0
        pose_cache_hit = False
        pose_reused = False
        pose_cache_valid = self._pose_cache_valid(frame_shape, now)
        run_pose = (
            postprocess_error is None
            and len(bboxes) > 0
            and (
                not pose_cache_valid
                or self.pose_interval <= 1
                or frame_index % self.pose_interval == 0
                or not det_cache_hit
            )
        )

        if postprocess_error is None and run_pose:
            for bbox_index, bbox in enumerate(pose_bboxes):
                pose_pre_start = time.perf_counter()
                pose_input, inv_matrix, pose_crop = preprocess_pose(frame_bgr, bbox, self.input_layout, return_crop=True)
                pose_preprocess_ms += _elapsed_ms(pose_pre_start)

                pose_infer_start = time.perf_counter()
                pose_outputs = self.pose_model.inference([pose_input])
                pose_inference_ms += _elapsed_ms(pose_infer_start)
                pose_output_shapes.append(output_shapes(pose_outputs))

                pose_post_start = time.perf_counter()
                try:
                    keypoints, scores = decode_simcc(pose_outputs, inv_matrix, score_mode=self.simcc_score_mode)
                    if bbox_index == 0 and self.rtmpose_debug_crop_every > 0 and frame_index % self.rtmpose_debug_crop_every == 0:
                        save_rtmpose_debug_crop(
                            pose_crop,
                            pose_outputs,
                            self.rtmpose_debug_crop_path,
                            score_mode=self.simcc_score_mode,
                            keypoint_thres=self.keypoint_thres,
                        )
                    keypoints_list.append(keypoints)
                    scores_list.append(scores)
                except Exception as exc:
                    postprocess_error = f"RTMPose postprocess failed: {exc}; output_shapes={pose_output_shapes}"
                    keypoints_list = []
                    scores_list = []
                    break
                finally:
                    pose_postprocess_ms += _elapsed_ms(pose_post_start)

        pose_geometry = {
            "geometry_ok": True,
            "geometry_message": "关键点几何正常",
            "geometry_bad_keypoints": [],
            "geometry_valid_count": 0,
            "keypoint_bbox_ratio": None,
        }
        if postprocess_error is None and len(bboxes) > 0 and not run_pose and pose_cache_valid:
            detections = self._cached_pose()
            pose_cache_hit = True
            pose_reused = True
            if detections:
                pose_geometry = {
                    "geometry_ok": bool(detections[0].get("geometry_ok", True)),
                    "geometry_message": detections[0].get("geometry_message", "关键点几何正常"),
                    "geometry_bad_keypoints": detections[0].get("geometry_bad_keypoints") or [],
                    "geometry_valid_count": detections[0].get("geometry_valid_count", 0),
                    "keypoint_bbox_ratio": detections[0].get("keypoint_bbox_ratio"),
                }
        else:
            detections = [] if postprocess_error else _build_detections(bboxes, bbox_scores, keypoints_list, scores_list)
            filtered_detections = []
            geometry_items = []
            for detection in detections:
                filtered, geometry = _filter_detection_geometry(detection, frame_shape)
                filtered_detections.append(filtered)
                geometry_items.append(geometry)
            detections = filtered_detections
            if geometry_items:
                bad_keypoints = sorted({name for item in geometry_items for name in item.get("geometry_bad_keypoints", [])})
                valid_counts = [int(item.get("geometry_valid_count", 0)) for item in geometry_items]
                ratios = [float(item["keypoint_bbox_ratio"]) for item in geometry_items if item.get("keypoint_bbox_ratio") is not None]
                pose_geometry = {
                    "geometry_ok": not bad_keypoints,
                    "geometry_message": "关键点几何正常" if not bad_keypoints else f"关键点超出人体框/画面: {','.join(bad_keypoints[:6])}",
                    "geometry_bad_keypoints": bad_keypoints,
                    "geometry_valid_count": max(valid_counts) if valid_counts else 0,
                    "keypoint_bbox_ratio": round(float(max(ratios)), 3) if ratios else None,
                }
            if detections:
                self._update_pose_cache(detections, frame_shape, now)
        draw_start = time.perf_counter()
        annotated = draw_poses(frame_bgr, detections, keypoint_thres=self.keypoint_thres)
        draw_ms = _elapsed_ms(draw_start)
        total_pose_ms = _elapsed_ms(total_start)
        performance_ms = {
            "preprocess_ms": round(det_preprocess_ms + pose_preprocess_ms, 2),
            "inference_ms": round(det_inference_ms + pose_inference_ms, 2),
            "postprocess_ms": round(det_postprocess_ms + pose_postprocess_ms, 2),
            "draw_ms": round(draw_ms, 2),
            "total_pose_ms": round(total_pose_ms, 2),
            "det_preprocess_ms": round(det_preprocess_ms, 2),
            "det_inference_ms": round(det_inference_ms, 2),
            "det_postprocess_ms": round(det_postprocess_ms, 2),
            "pose_preprocess_ms": round(pose_preprocess_ms, 2),
            "pose_inference_ms": round(pose_inference_ms, 2),
            "pose_postprocess_ms": round(pose_postprocess_ms, 2),
        }
        return PoseResult(
            backend="rknn",
            fps=1000.0 / total_pose_ms if total_pose_ms > 0 else None,
            keypoints={},
            raw={"yolo_outputs": yolo_outputs},
            annotated_frame=annotated,
            meta={
                "detections": detections,
                "postprocess_error": postprocess_error,
                "output_shapes": yolo_output_shapes,
                "det_output_shapes": yolo_output_shapes,
                "pose_output_shapes": pose_output_shapes,
                "rknn_decoder": "yolov8_rtmpose",
                "rknn_pipeline": "yolov8_rtmpose",
                "keypoint_conf_range": _range_or_none([score for scores in scores_list for score in scores.tolist()]),
                "keypoint_xy_range": _xy_range_from_keypoints(keypoints_list),
                "keypoint_restored_xy_range": _xy_range_from_keypoints(keypoints_list),
                "performance_ms": performance_ms,
                "inference_ms": performance_ms["inference_ms"],
                "preprocess_ms": performance_ms["preprocess_ms"],
                "postprocess_ms": performance_ms["postprocess_ms"],
                "draw_ms": draw_ms,
                "total_pose_ms": total_pose_ms,
                "model_path": self.model_path,
                "det_model_path": self.yolo_model_path,
                "pose_model_path": self.pose_model_path,
                "yolo_model_path": self.yolo_model_path,
                "det_score_thres": self.conf_thres,
                "nms_thres": self.nms_thres,
                "keypoint_thres": self.keypoint_thres,
                "max_persons": self.max_persons,
                "input_size": self.input_size,
                "input_layout": self.input_layout,
                "simcc_score_mode": self.simcc_score_mode,
                "person_select": self.person_select,
                "yolo_decoder": yolo_meta.get("rknn_decoder"),
                "yolo_postprocess_error": yolo_meta.get("postprocess_error"),
                "yolo_detection_count": len(yolo_detections),
                "yolo_keypoint_conf_range": yolo_meta.get("keypoint_conf_range"),
                "selected_yolo_bbox": [round(float(value), 2) for value in bboxes[0].tolist()] if len(bboxes) else None,
                "rtmpose_expanded_bbox": pose_bbox_debug[0] if pose_bbox_debug else None,
                "rtmpose_bbox_expand": self.rtmpose_bbox_expand,
                "rtmpose_bbox_top_expand": self.rtmpose_bbox_top_expand,
                "rtmpose_crop_input_size": [288, 384],
                "rtmpose_keypoint_bbox_ratio": pose_geometry.get("keypoint_bbox_ratio"),
                "rtmpose_debug_crop_path": self.rtmpose_debug_crop_path if self.rtmpose_debug_crop_every > 0 else None,
                "det_cache_hit": det_cache_hit,
                "det_interval": self.det_interval,
                "det_cache_seconds": self.det_cache_seconds,
                "pose_cache_hit": pose_cache_hit,
                "pose_reused": pose_reused,
                "pose_interval": self.pose_interval,
                "pose_cache_seconds": self.pose_cache_seconds,
                "yolo_det_cache_hit": det_cache_hit,
                "yolo_det_interval": self.det_interval,
                "yolo_det_cache_seconds": self.det_cache_seconds,
                "pose_geometry_ok": pose_geometry["geometry_ok"],
                "pose_geometry_message": pose_geometry["geometry_message"],
                "pose_geometry_bad_keypoints": pose_geometry["geometry_bad_keypoints"],
                "pose_geometry_valid_count": pose_geometry["geometry_valid_count"],
                "pose_geometry_keypoint_bbox_ratio": pose_geometry.get("keypoint_bbox_ratio"),
            },
        )

    def release(self) -> None:
        self.yolo_model.release()
        self.pose_model.release()

    def _cache_valid(self, frame_shape: tuple[int, ...], now: float) -> bool:
        if len(self._cached_bboxes) == 0:
            return False
        if self._cached_frame_shape != frame_shape:
            return False
        if self.det_cache_seconds <= 0:
            return False
        return now - self._cached_at <= self.det_cache_seconds

    def _update_cache(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        yolo_detections: list[dict[str, Any]],
        frame_shape: tuple[int, ...],
        now: float,
        det_shapes: list[tuple[int, ...]],
        yolo_meta: dict[str, Any],
    ) -> None:
        self._cached_bboxes = bboxes.copy()
        self._cached_scores = scores.copy()
        self._cached_yolo_detections = [_copy_detection(item) for item in yolo_detections]
        self._cached_frame_shape = frame_shape
        self._cached_at = now
        self._cached_det_shapes = list(det_shapes)
        self._cached_yolo_meta = dict(yolo_meta)

    def _cached_detection(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], list[tuple[int, ...]]]:
        meta = dict(self._cached_yolo_meta)
        meta["cache_reused"] = True
        return (
            self._cached_bboxes.copy(),
            self._cached_scores.copy(),
            [_copy_detection(item) for item in self._cached_yolo_detections],
            meta,
            list(self._cached_det_shapes),
        )

    def _pose_cache_valid(self, frame_shape: tuple[int, ...], now: float) -> bool:
        if not self._cached_pose_detections:
            return False
        if self._cached_pose_frame_shape != frame_shape:
            return False
        if self.pose_cache_seconds <= 0:
            return False
        return now - self._cached_pose_at <= self.pose_cache_seconds

    def _update_pose_cache(self, detections: list[dict[str, Any]], frame_shape: tuple[int, ...], now: float) -> None:
        self._cached_pose_detections = [_copy_detection(item) for item in detections]
        self._cached_pose_frame_shape = frame_shape
        self._cached_pose_at = now

    def _cached_pose(self) -> list[dict[str, Any]]:
        return [_copy_detection(item) for item in self._cached_pose_detections]


class RTMPoseFixedBackend:
    """Single RTMPose backend that uses a fixed person ROI instead of RTMDet."""

    def __init__(
        self,
        *,
        pose_model_path: str | Path | None = None,
        fixed_bbox: str | list[float] | tuple[float, ...] | np.ndarray | None = None,
        keypoint_thres: float | None = None,
        input_layout: str | None = None,
        simcc_score_mode: str | None = None,
        core_mask: str | None = None,
    ) -> None:
        self.pose_model_path = str(pose_model_path or os.environ.get("RKNN_RTMPOSE_MODEL", "rknn/rtmpose_fp16.rknn"))
        self.fixed_bbox = parse_fixed_bbox(fixed_bbox if fixed_bbox is not None else os.environ.get("RKNN_RTMPOSE_FIXED_BBOX", "80,20,560,470"))
        self.fixed_bbox_mode = _choice(os.environ.get("RKNN_RTMPOSE_FIXED_BBOX_MODE"), {"absolute"}, "absolute")
        self.fixed_score = float(os.environ.get("RKNN_RTMPOSE_FIXED_SCORE", "1.0"))
        self.keypoint_thres = float(keypoint_thres if keypoint_thres is not None else os.environ.get("RKNN_POSE_KEYPOINT_THRES", "0.20"))
        self.input_layout = _choice(os.environ.get("RKNN_INPUT_LAYOUT") if input_layout is None else input_layout, {"nchw", "nhwc"}, "nchw")
        self.simcc_score_mode = _choice(os.environ.get("RKNN_RTMPOSE_SIMCC_SCORE_MODE") if simcc_score_mode is None else simcc_score_mode, {"sqrt", "avg"}, "sqrt")
        self.draw_annotated = os.environ.get("RKNN_RTMPOSE_DRAW", "1").strip() != "0"
        self.pose_model = RKNNModel(self.pose_model_path, core_mask=str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto")))

    @property
    def model_path(self) -> str:
        return self.pose_model_path

    def load(self) -> None:
        self.pose_model.load()

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        self.load()
        total_start = time.perf_counter()
        postprocess_error = None
        frame_shape = tuple(frame_bgr.shape)
        bbox = clamp_bbox_to_frame(self.fixed_bbox, frame_bgr.shape)
        pose_outputs: list[Any] = []
        pose_output_shapes: list[tuple[int, ...]] = []
        keypoints_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []

        pose_pre_start = time.perf_counter()
        pose_input, inv_matrix = preprocess_pose(frame_bgr, bbox, self.input_layout)
        pose_preprocess_ms = _elapsed_ms(pose_pre_start)

        pose_infer_start = time.perf_counter()
        pose_outputs = self.pose_model.inference([pose_input])
        pose_inference_ms = _elapsed_ms(pose_infer_start)
        pose_output_shapes = output_shapes(pose_outputs)

        pose_post_start = time.perf_counter()
        try:
            keypoints, scores = decode_simcc(pose_outputs, inv_matrix, score_mode=self.simcc_score_mode)
            keypoints_list.append(keypoints)
            scores_list.append(scores)
        except Exception as exc:
            postprocess_error = f"RTMPose fixed postprocess failed: {exc}; output_shapes={pose_output_shapes}"
        pose_postprocess_ms = _elapsed_ms(pose_post_start)

        if postprocess_error is None:
            bboxes = np.asarray([bbox], dtype=np.float32)
            bbox_scores = np.asarray([self.fixed_score], dtype=np.float32)
            detections = _build_detections(bboxes, bbox_scores, keypoints_list, scores_list)
        else:
            detections = []
        pose_geometry = {
            "geometry_ok": True,
            "geometry_message": "关键点几何正常",
            "geometry_bad_keypoints": [],
            "geometry_valid_count": 0,
        }
        if detections:
            detections, geometry_items = zip(*[_filter_detection_geometry(detection, frame_shape) for detection in detections])
            detections = list(detections)
            bad_keypoints = sorted({name for item in geometry_items for name in item.get("geometry_bad_keypoints", [])})
            valid_counts = [int(item.get("geometry_valid_count", 0)) for item in geometry_items]
            pose_geometry = {
                "geometry_ok": not bad_keypoints,
                "geometry_message": "关键点几何正常" if not bad_keypoints else f"关键点超出人体框/画面: {','.join(bad_keypoints[:6])}",
                "geometry_bad_keypoints": bad_keypoints,
                "geometry_valid_count": max(valid_counts) if valid_counts else 0,
            }

        draw_start = time.perf_counter()
        annotated = draw_poses(frame_bgr, detections, keypoint_thres=self.keypoint_thres) if self.draw_annotated else None
        draw_ms = _elapsed_ms(draw_start)
        total_pose_ms = _elapsed_ms(total_start)
        performance_ms = {
            "preprocess_ms": round(pose_preprocess_ms, 2),
            "inference_ms": round(pose_inference_ms, 2),
            "postprocess_ms": round(pose_postprocess_ms, 2),
            "draw_ms": round(draw_ms, 2),
            "total_pose_ms": round(total_pose_ms, 2),
            "det_preprocess_ms": 0.0,
            "det_inference_ms": 0.0,
            "det_postprocess_ms": 0.0,
            "pose_preprocess_ms": round(pose_preprocess_ms, 2),
            "pose_inference_ms": round(pose_inference_ms, 2),
            "pose_postprocess_ms": round(pose_postprocess_ms, 2),
        }
        return PoseResult(
            backend="rknn",
            fps=1000.0 / total_pose_ms if total_pose_ms > 0 else None,
            keypoints={},
            raw={"pose_outputs": pose_outputs},
            annotated_frame=annotated,
            meta={
                "detections": detections,
                "postprocess_error": postprocess_error,
                "output_shapes": pose_output_shapes,
                "det_output_shapes": [],
                "pose_output_shapes": [pose_output_shapes],
                "rknn_decoder": "rtmpose_fixed",
                "rknn_pipeline": "rtmpose_fixed",
                "keypoint_conf_range": _range_or_none([score for scores in scores_list for score in scores.tolist()]),
                "keypoint_xy_range": _xy_range_from_keypoints(keypoints_list),
                "keypoint_restored_xy_range": _xy_range_from_keypoints(keypoints_list),
                "performance_ms": performance_ms,
                "inference_ms": performance_ms["inference_ms"],
                "preprocess_ms": performance_ms["preprocess_ms"],
                "postprocess_ms": performance_ms["postprocess_ms"],
                "draw_ms": draw_ms,
                "total_pose_ms": total_pose_ms,
                "model_path": self.model_path,
                "pose_model_path": self.pose_model_path,
                "fixed_bbox": [round(float(value), 2) for value in bbox.tolist()],
                "fixed_bbox_requested": [round(float(value), 2) for value in self.fixed_bbox.tolist()],
                "fixed_bbox_mode": self.fixed_bbox_mode,
                "fixed_score": self.fixed_score,
                "frame_shape": frame_shape,
                "keypoint_thres": self.keypoint_thres,
                "input_layout": self.input_layout,
                "simcc_score_mode": self.simcc_score_mode,
                "rtmpose_draw_enabled": self.draw_annotated,
                "pose_geometry_ok": pose_geometry["geometry_ok"],
                "pose_geometry_message": pose_geometry["geometry_message"],
                "pose_geometry_bad_keypoints": pose_geometry["geometry_bad_keypoints"],
                "pose_geometry_valid_count": pose_geometry["geometry_valid_count"],
            },
        )

    def release(self) -> None:
        self.pose_model.release()


def letterbox(image: np.ndarray, target_size: tuple[int, int] = (640, 640), pad_val: int = 114) -> tuple[np.ndarray, float, int, int]:
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), pad_val, dtype=np.uint8)
    pad_left = 0
    pad_top = 0
    padded[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    return padded, float(scale), pad_left, pad_top


def preprocess_det(frame: np.ndarray, input_layout: str = "nchw") -> tuple[np.ndarray, float, int, int]:
    padded, scale, pad_left, pad_top = letterbox(frame, (640, 640), 114)
    image = padded.astype(np.float32)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    image = (image - mean) / std
    if input_layout == "nhwc":
        blob = image[None, ...].astype(np.float32)
    else:
        blob = image.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return blob, scale, pad_left, pad_top


def sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))


def infer_head_layout(outputs: list[Any] | tuple[Any, ...]) -> str:
    votes = {"nchw": 0, "nhwc": 0}
    for output in outputs:
        arr = np.asarray(output)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            continue
        first_is_head = arr.shape[0] in (4, 80)
        last_is_head = arr.shape[-1] in (4, 80)
        if first_is_head and not last_is_head:
            votes["nchw"] += 1
        elif last_is_head and not first_is_head:
            votes["nhwc"] += 1
    if votes["nhwc"] > votes["nchw"]:
        return "nhwc"
    return "nchw"


def normalize_pred_map(output: Any, head_layout: str = "auto") -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D/4D prediction map, got shape {arr.shape}")
    if head_layout == "nchw":
        return arr.astype(np.float32)
    if head_layout == "nhwc":
        return arr.transpose(2, 0, 1).astype(np.float32)
    if arr.shape[0] in (4, 80):
        return arr.astype(np.float32)
    if arr.shape[-1] in (4, 80):
        return arr.transpose(2, 0, 1).astype(np.float32)
    raise ValueError(f"Cannot infer NCHW/NHWC layout for prediction map shape {arr.shape}")


def pair_rtmdet_outputs(outputs: list[Any] | tuple[Any, ...], input_size: int = 640, head_layout: str = "auto") -> tuple[list[tuple[int, int, int, np.ndarray, np.ndarray]], str]:
    resolved_layout = infer_head_layout(outputs) if head_layout == "auto" else head_layout
    cls_maps = []
    bbox_maps = []
    for idx, output in enumerate(outputs):
        arr = normalize_pred_map(output, resolved_layout)
        channels, h, w = arr.shape
        if channels == 80:
            cls_maps.append((h, w, idx, arr))
        elif channels == 4:
            bbox_maps.append((h, w, idx, arr))
    if not cls_maps or not bbox_maps:
        shapes = [np.asarray(output).shape for output in outputs]
        raise RuntimeError(f"Cannot find cls/bbox RTMDet heads from output shapes: {shapes}")

    pairs = []
    for h, w, cls_idx, cls_map in cls_maps:
        match = next((item for item in bbox_maps if item[0] == h and item[1] == w), None)
        if match is None:
            raise RuntimeError(f"No bbox map matches cls map shape {(h, w)}")
        _, _, bbox_idx, bbox_map = match
        stride_h = input_size / h
        stride_w = input_size / w
        stride = int(round((stride_h + stride_w) * 0.5))
        pairs.append((stride, cls_idx, bbox_idx, cls_map, bbox_map))
    return sorted(pairs, key=lambda item: item[0]), resolved_layout


def generate_priors(h: int, w: int, stride: int) -> np.ndarray:
    shifts_x = (np.arange(w, dtype=np.float32) + 0.5) * stride
    shifts_y = (np.arange(h, dtype=np.float32) + 0.5) * stride
    grid_x, grid_y = np.meshgrid(shifts_x, shifts_y)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def decode_bboxes(priors: np.ndarray, bbox_pred: np.ndarray, stride: int, bbox_decode_mode: str) -> np.ndarray:
    distances = bbox_pred.astype(np.float32)
    if bbox_decode_mode == "stride":
        distances = distances * stride
    elif bbox_decode_mode == "auto":
        positive = distances[distances > 0]
        scale_probe = np.percentile(positive, 95) if positive.size else 0.0
        if scale_probe < 64:
            distances = distances * stride
    x1 = priors[:, 0] - distances[:, 0]
    y1 = priors[:, 1] - distances[:, 1]
    x2 = priors[:, 0] + distances[:, 2]
    y2 = priors[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def postprocess_rtmdet(
    outputs: list[Any],
    *,
    score_thr: float,
    nms_thr: float,
    scale: float,
    pad_left: int,
    pad_top: int,
    frame_shape: tuple[int, ...],
    max_persons: int,
    bbox_decode_mode: str = "auto",
    head_layout: str = "auto",
    nms_pre: int = 1000,
    person_select: str = "largest_center",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_bboxes = []
    all_scores = []
    pairs, resolved_layout = pair_rtmdet_outputs(outputs, 640, head_layout)
    pair_meta = []
    for stride, cls_idx, bbox_idx, cls_map, bbox_map in pairs:
        _, h, w = cls_map.shape
        person_scores = sigmoid(cls_map[0]).reshape(-1)
        keep = np.where(person_scores >= score_thr)[0]
        if keep.size == 0:
            pair_meta.append({"stride": stride, "cls_out": cls_idx, "bbox_out": bbox_idx, "kept": 0})
            continue
        if keep.size > nms_pre:
            top = np.argpartition(person_scores[keep], -nms_pre)[-nms_pre:]
            keep = keep[top]
        bbox_flat = bbox_map.transpose(1, 2, 0).reshape(-1, 4)
        priors = generate_priors(h, w, stride)
        bboxes = decode_bboxes(priors[keep], bbox_flat[keep], stride, bbox_decode_mode)
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, 640)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, 640)
        valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        all_bboxes.append(bboxes[valid])
        all_scores.append(person_scores[keep][valid])
        pair_meta.append({"stride": stride, "cls_out": cls_idx, "bbox_out": bbox_idx, "kept": int(valid.sum())})

    if not all_bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), {"head_layout": resolved_layout, "pairs": pair_meta}

    bboxes = np.concatenate(all_bboxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    keep_indices = nms(bboxes, scores, nms_thr)
    bboxes = map_bboxes_to_original(bboxes[keep_indices], scale, pad_left, pad_top, frame_shape)
    scores = scores[keep_indices]
    valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
    bboxes = bboxes[valid]
    scores = scores[valid]
    selected_strategy = person_select
    if max_persons > 0 and len(bboxes) > max_persons:
        selected = select_person_indices(bboxes, scores, frame_shape, max_persons, person_select)
        bboxes = bboxes[selected]
        scores = scores[selected]
    return bboxes, scores, {"head_layout": resolved_layout, "pairs": pair_meta, "person_select": selected_strategy}


def select_person_indices(
    bboxes: np.ndarray,
    scores: np.ndarray,
    frame_shape: tuple[int, ...],
    max_persons: int,
    strategy: str,
) -> np.ndarray:
    if len(bboxes) <= max_persons:
        return np.arange(len(bboxes), dtype=np.int64)
    h, w = frame_shape[:2]
    widths = np.maximum(bboxes[:, 2] - bboxes[:, 0], 1.0)
    heights = np.maximum(bboxes[:, 3] - bboxes[:, 1], 1.0)
    areas = widths * heights
    frame_area = max(float(w * h), 1.0)
    centers_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    centers_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    center_dist = np.sqrt(((centers_x - w * 0.5) / max(w * 0.5, 1.0)) ** 2 + ((centers_y - h * 0.5) / max(h * 0.5, 1.0)) ** 2)
    center_score = np.clip(1.0 - center_dist / 1.4142, 0.0, 1.0)
    area_score = np.clip(areas / (frame_area * 0.35), 0.0, 1.0)
    aspect = heights / widths
    aspect_score = np.clip(1.0 - np.abs(np.log(np.maximum(aspect, 1e-6) / 2.0)) / 1.2, 0.0, 1.0)

    mode = str(strategy or "largest_center").strip().lower()
    if mode == "score":
        ranking = scores
    elif mode == "largest":
        ranking = area_score
    elif mode == "center":
        ranking = center_score
    else:
        ranking = (scores * 0.45) + (area_score * 0.30) + (center_score * 0.20) + (aspect_score * 0.05)
    order = np.argsort(ranking)[::-1]
    return order[:max_persons].astype(np.int64)


def map_bboxes_to_original(bboxes: np.ndarray, scale: float, pad_left: int, pad_top: int, frame_shape: tuple[int, ...]) -> np.ndarray:
    if len(bboxes) == 0:
        return bboxes
    h, w = frame_shape[:2]
    mapped = bboxes.copy().astype(np.float32)
    mapped[:, 0::2] = (mapped[:, 0::2] - pad_left) / scale
    mapped[:, 1::2] = (mapped[:, 1::2] - pad_top) / scale
    mapped[:, 0::2] = np.clip(mapped[:, 0::2], 0, w - 1)
    mapped[:, 1::2] = np.clip(mapped[:, 1::2], 0, h - 1)
    return mapped


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def rotate_point(point: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([point[0] * cs - point[1] * sn, point[0] * sn + point[1] * cs], dtype=np.float32)


def get_affine_transform(center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int], inv: bool = False) -> np.ndarray:
    dst_w, dst_h = output_size
    src_w = scale[0]
    src_dir = rotate_point(np.array([0, src_w * -0.5], dtype=np.float32), 0.0)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)


def bbox_xyxy_to_center_scale(bbox: np.ndarray, input_size: tuple[int, int] = (288, 384), padding: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return center, np.array([w * padding, h * padding], dtype=np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), ones], axis=1) @ matrix.T


def topdown_affine_crop(frame: np.ndarray, bbox: np.ndarray, input_size: tuple[int, int] = (288, 384)) -> tuple[np.ndarray, np.ndarray]:
    center, scale = bbox_xyxy_to_center_scale(bbox, input_size)
    matrix = get_affine_transform(center, scale, input_size)
    inv_matrix = get_affine_transform(center, scale, input_size, inv=True)
    crop = cv2.warpAffine(frame, matrix, input_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return crop, inv_matrix


def preprocess_pose(
    frame: np.ndarray,
    bbox: np.ndarray,
    input_layout: str = "nchw",
    *,
    return_crop: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop, inv_matrix = topdown_affine_crop(frame, bbox, (288, 384))
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image = (image - mean) / std
    if input_layout == "nhwc":
        blob = image[None, ...].astype(np.float32)
    else:
        blob = image.transpose(2, 0, 1)[None, ...].astype(np.float32)
    if return_crop:
        return blob, inv_matrix, crop
    return blob, inv_matrix


def expand_bboxes_for_rtmpose(
    bboxes: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    expand: float,
    top_expand: float,
) -> np.ndarray:
    if len(bboxes) == 0:
        return bboxes
    height, width = frame_shape[:2]
    expanded = np.asarray(bboxes, dtype=np.float32).copy()
    for index, bbox in enumerate(expanded):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        new_w = box_w * float(expand)
        new_h = box_h * float(expand)
        nx1 = cx - new_w * 0.5
        nx2 = cx + new_w * 0.5
        ny1 = cy - new_h * 0.5 - box_h * float(top_expand)
        ny2 = cy + new_h * 0.5
        expanded[index] = [
            max(0.0, min(float(width - 1), nx1)),
            max(0.0, min(float(height - 1), ny1)),
            max(0.0, min(float(width - 1), nx2)),
            max(0.0, min(float(height - 1), ny2)),
        ]
    return expanded


def save_rtmpose_debug_crop(
    crop_bgr: np.ndarray,
    outputs: list[Any] | tuple[Any, ...],
    out_path: str,
    *,
    score_mode: str,
    keypoint_thres: float,
) -> None:
    try:
        simcc_x, simcc_y = normalize_simcc_outputs(outputs)
        x_locs = np.argmax(simcc_x, axis=1).astype(np.float32) / 2.0
        y_locs = np.argmax(simcc_y, axis=1).astype(np.float32) / 2.0
        if score_mode == "avg":
            scores = (np.max(simcc_x, axis=1) + np.max(simcc_y, axis=1)) * 0.5
        else:
            scores = np.sqrt(np.maximum(np.max(simcc_x, axis=1), 0) * np.maximum(np.max(simcc_y, axis=1), 0))
        points = np.stack([x_locs, y_locs], axis=1)
        debug = draw_crop_keypoints(crop_bgr, points, scores, keypoint_thres)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), debug)
    except Exception:
        return


def draw_crop_keypoints(crop_bgr: np.ndarray, points: np.ndarray, scores: np.ndarray, keypoint_thres: float) -> np.ndarray:
    output = crop_bgr.copy()
    visible = [float(score) >= keypoint_thres for score in scores]
    for idx, (start, end) in enumerate(COCO17_EDGES):
        if start >= len(points) or end >= len(points) or not (visible[start] and visible[end]):
            continue
        cv2.line(output, tuple(np.round(points[start]).astype(int)), tuple(np.round(points[end]).astype(int)), LINK_COLORS[idx], 2, cv2.LINE_AA)
    for idx, point in enumerate(points):
        if idx < len(visible) and visible[idx]:
            cv2.circle(output, tuple(np.round(point).astype(int)), 4, KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
    return output


def parse_fixed_bbox(value: str | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError(f"RKNN_RTMPOSE_FIXED_BBOX must be x1,y1,x2,y2, got: {value!r}")
        coords = [float(part) for part in parts]
    else:
        coords = [float(part) for part in list(value)]
        if len(coords) != 4:
            raise ValueError(f"fixed bbox must have 4 values, got: {coords!r}")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"fixed bbox must satisfy x2>x1 and y2>y1, got: {coords!r}")
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def clamp_bbox_to_frame(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    height, width = frame_shape[:2]
    clamped = np.asarray(bbox, dtype=np.float32).copy()
    clamped[0::2] = np.clip(clamped[0::2], 0.0, max(width - 1, 0))
    clamped[1::2] = np.clip(clamped[1::2], 0.0, max(height - 1, 0))
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"fixed bbox is outside frame after clipping: bbox={bbox.tolist()}, frame_shape={frame_shape}")
    return clamped


def normalize_simcc_outputs(outputs: list[Any] | tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for idx, output in enumerate(outputs):
        arr = np.asarray(output, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            continue
        if arr.shape[-1] in (576, 768):
            candidates.append((idx, arr.shape[-1], arr))
        elif arr.shape[0] in (576, 768):
            candidates.append((idx, arr.shape[0], arr.T))
    simcc_x = next((arr for _, length, arr in candidates if length == 576), None)
    simcc_y = next((arr for _, length, arr in candidates if length == 768), None)
    if simcc_x is None or simcc_y is None:
        shapes = [np.asarray(output).shape for output in outputs]
        raise RuntimeError(f"Cannot find simcc_x/simcc_y from RTMPose output shapes: {shapes}")
    return simcc_x, simcc_y


def decode_simcc(outputs: list[Any] | tuple[Any, ...], inv_matrix: np.ndarray, split_ratio: float = 2.0, score_mode: str = "sqrt") -> tuple[np.ndarray, np.ndarray]:
    simcc_x, simcc_y = normalize_simcc_outputs(outputs)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    max_x = np.max(simcc_x, axis=1)
    max_y = np.max(simcc_y, axis=1)
    if score_mode == "avg":
        scores = (max_x + max_y) * 0.5
    else:
        scores = np.sqrt(np.maximum(max_x, 0) * np.maximum(max_y, 0))
    crop_keypoints = np.stack([x_locs, y_locs], axis=1).astype(np.float32) / split_ratio
    crop_keypoints[scores <= 0] = -1
    keypoints = transform_points(crop_keypoints, inv_matrix).astype(np.float32)
    return keypoints, scores.astype(np.float32)


def draw_poses(frame_bgr: np.ndarray, detections: list[dict[str, Any]], *, keypoint_thres: float = 0.30) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        bbox = np.asarray(detection.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 255), 2)
        cv2.putText(output, f"{float(detection.get('score', 0.0)):.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 255), 1)
        keypoints = detection.get("keypoints") or []
        visible = [len(point) >= 3 and float(point[2]) >= keypoint_thres for point in keypoints]
        for idx, (start, end) in enumerate(COCO17_EDGES):
            if start >= len(keypoints) or end >= len(keypoints) or not (visible[start] and visible[end]):
                continue
            pt1 = tuple(np.round(keypoints[start][:2]).astype(int))
            pt2 = tuple(np.round(keypoints[end][:2]).astype(int))
            cv2.line(output, pt1, pt2, LINK_COLORS[idx], 2, cv2.LINE_AA)
        for idx, point in enumerate(keypoints):
            if idx < len(visible) and visible[idx]:
                center = tuple(np.round(point[:2]).astype(int))
                cv2.circle(output, center, 4, KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
                cv2.circle(output, center, 5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def nms(bboxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(bboxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = bboxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def output_shapes(outputs: list[Any] | tuple[Any, ...]) -> list[tuple[int, ...]]:
    return [tuple(np.asarray(output).shape) for output in outputs if output is not None]


def _build_detections(bboxes: np.ndarray, bbox_scores: np.ndarray, keypoints_list: list[np.ndarray], scores_list: list[np.ndarray]) -> list[dict[str, Any]]:
    detections = []
    for bbox, bbox_score, keypoints, scores in zip(bboxes, bbox_scores, keypoints_list, scores_list):
        points = [[float(x), float(y), _normalize_score(float(score))] for (x, y), score in zip(keypoints, scores)]
        detections.append({"bbox": [float(value) for value in bbox], "score": float(bbox_score), "keypoints": points})
    return detections


def _filter_detection_geometry(
    detection: dict[str, Any],
    frame_shape: tuple[int, ...],
    *,
    bbox_margin_ratio: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    filtered = _copy_detection(detection)
    bbox = np.asarray(filtered.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox.tolist()
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    margin_x = box_w * float(bbox_margin_ratio)
    margin_y = box_h * float(bbox_margin_ratio)
    allowed_x1 = max(0.0, x1 - margin_x)
    allowed_y1 = max(0.0, y1 - margin_y)
    allowed_x2 = min(float(width - 1), x2 + margin_x)
    allowed_y2 = min(float(height - 1), y2 + margin_y)

    bad_keypoints: list[str] = []
    keypoints = filtered.get("keypoints") or []
    for index, point in enumerate(keypoints):
        name = COCO17_NAMES[index] if index < len(COCO17_NAMES) else f"kp_{index}"
        if len(point) < 3:
            bad_keypoints.append(name)
            point[:] = [0.0, 0.0, 0.0]
            continue
        x = float(point[0])
        y = float(point[1])
        score = float(point[2])
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(score):
            bad_keypoints.append(name)
            point[:] = [0.0, 0.0, 0.0]
            continue
        if x < allowed_x1 or x > allowed_x2 or y < allowed_y1 or y > allowed_y2:
            bad_keypoints.append(name)
            point[2] = 0.0
    filtered["keypoints"] = keypoints

    visible_scores = [float(point[2]) for point in keypoints if len(point) >= 3 and np.isfinite(point[2]) and float(point[2]) > 0.0]
    visible_points = [
        (float(point[0]), float(point[1]))
        for point in keypoints
        if len(point) >= 3 and np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2]) and float(point[2]) > 0.0
    ]
    keypoint_bbox_ratio = None
    if visible_points:
        ys = [point[1] for point in visible_points]
        keypoint_bbox_ratio = (max(ys) - min(ys)) / max(box_h, 1.0)
        if len(visible_points) >= 6 and keypoint_bbox_ratio < 0.45:
            bad_keypoints.append("skeleton_too_small")
    geometry_ok = len(bad_keypoints) == 0
    geometry_message = "关键点几何正常" if geometry_ok else f"关键点超出人体框/画面: {','.join(bad_keypoints[:6])}"
    filtered["geometry_ok"] = geometry_ok
    filtered["geometry_message"] = geometry_message
    filtered["geometry_bad_keypoints"] = bad_keypoints
    filtered["geometry_valid_count"] = len(visible_scores)
    filtered["keypoint_bbox_ratio"] = round(float(keypoint_bbox_ratio), 3) if keypoint_bbox_ratio is not None else None
    return filtered, {
        "geometry_ok": geometry_ok,
        "geometry_message": geometry_message,
        "geometry_bad_keypoints": bad_keypoints,
        "geometry_valid_count": len(visible_scores),
        "keypoint_bbox_ratio": round(float(keypoint_bbox_ratio), 3) if keypoint_bbox_ratio is not None else None,
    }


def _copy_detection(detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox": [float(value) for value in detection.get("bbox", [])],
        "score": float(detection.get("score", 0.0)),
        "keypoints": [[float(value) for value in point] for point in detection.get("keypoints", [])],
    }


def _detections_to_selected_bboxes(
    detections: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
    *,
    max_persons: int,
    person_select: str,
) -> tuple[np.ndarray, np.ndarray]:
    bboxes: list[list[float]] = []
    scores: list[float] = []
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
    if not bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    bbox_array = np.asarray(bboxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    selected = select_person_indices(
        bbox_array,
        score_array,
        frame_shape,
        max(1, int(max_persons)),
        person_select,
    )
    return bbox_array[selected], score_array[selected]


def _normalize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return float(value)
    return float(1.0 / (1.0 + np.exp(-value)))


def _xy_range_from_keypoints(keypoints_list: list[np.ndarray]) -> dict[str, list[float] | None]:
    xs = []
    ys = []
    for keypoints in keypoints_list:
        if keypoints.size == 0:
            continue
        xs.extend([float(value) for value in keypoints[:, 0]])
        ys.extend([float(value) for value in keypoints[:, 1]])
    return {"x": _range_or_none(xs), "y": _range_or_none(ys)}


def _range_or_none(values: list[float]) -> list[float] | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return None
    return [round(min(clean), 4), round(max(clean), 4)]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _choice(value: str | None, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
