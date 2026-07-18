"""YOLOv5n raw-head person detector + top-down RTMPose backend.

This backend replaces the pure RTMPose ROI tracker used by the 8084 demo.
YOLOv5n provides person boxes and RTMPose only receives a resized person ROI.
"""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pose_estimation.rknn_pose.pose_result import PoseResult


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

YOLOV5_STRIDES = (8, 16, 32)
YOLOV5_ANCHORS = np.asarray(
    (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    ),
    dtype=np.float32,
)
YOLOV5_NUM_CLASSES = 80
YOLOV5_ATTRS_PER_ANCHOR = YOLOV5_NUM_CLASSES + 5
YOLOV5_COMBINED_CHANNELS = 3 * YOLOV5_ATTRS_PER_ANCHOR
YOLOV5_RAW_CHANNELS = {240: "cls", 12: "bbox", 3: "obj"}

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


@dataclass(frozen=True)
class LetterboxMeta:
    input_width: int
    input_height: int
    original_width: int
    original_height: int
    scale: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class RestoreInfo:
    image_bbox: np.ndarray
    input_size: tuple[int, int]

    @property
    def width(self) -> float:
        return max(float(self.image_bbox[2] - self.image_bbox[0]), 1.0)

    @property
    def height(self) -> float:
        return max(float(self.image_bbox[3] - self.image_bbox[1]), 1.0)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        input_w, input_h = self.input_size
        output = np.asarray(points, dtype=np.float32).copy()
        output[:, 0] = output[:, 0] / max(float(input_w), 1.0) * self.width + self.image_bbox[0]
        output[:, 1] = output[:, 1] / max(float(input_h), 1.0) * self.height + self.image_bbox[1]
        return output


class RKNNModel:
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
        except Exception as exc:  # pragma: no cover - RK3588 only
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

    def inference(self, inputs: list[np.ndarray], *, data_format: str | None = None) -> list[Any]:
        if self.rknn is None:
            self.load()
        assert self.rknn is not None
        inference_kwargs: dict[str, Any] = {"inputs": inputs}
        # Keep the tensor shape and RKNNLite data_format explicit.  The RKNN
        # binaries used by this pipeline expose native NHWC inputs even though
        # their source ONNX graphs were NCHW.
        if data_format:
            inference_kwargs["data_format"] = str(data_format).lower()
        outputs = self.rknn.inference(**inference_kwargs)
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        return list(outputs)

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


class YOLOv5nRTMPoseBackend:
    """YOLOv5n person box detector followed by RTMPose keypoint inference."""

    def __init__(
        self,
        *,
        det_model_path: str | Path | None = None,
        pose_model_path: str | Path | None = None,
        keypoint_thres: float | None = None,
        core_mask: str | None = None,
    ) -> None:
        self.det_model_path = str(det_model_path or os.environ.get("RKNN_DET_MODEL", "models/vision/yolov5n_raw_fp.rknn"))
        self.pose_model_path = str(pose_model_path or os.environ.get("RKNN_RTMPOSE_MODEL", "models/vision/rtmpose_m_256x192_fp.rknn"))
        self.det_input_size = int(os.environ.get("RKNN_YOLOV5_INPUT_SIZE", os.environ.get("RKNN_DET_INPUT_SIZE", "640")))
        self.det_input_layout = _choice(os.environ.get("RKNN_YOLOV5_INPUT_LAYOUT", os.environ.get("RKNN_INPUT_LAYOUT")), {"nchw", "nhwc"}, "nhwc")
        self.det_input_mode = _choice(
            os.environ.get("RKNN_YOLOV5_INPUT_MODE", os.environ.get("RKNN_DET_INPUT_MODE")),
            {"rgb_0_1", "bgr_0_1", "rgb_uint8", "bgr_uint8", "rgb_norm", "bgr_norm"},
            "rgb_0_1",
        )
        self.det_score_thres = float(os.environ.get("RKNN_DET_SCORE_THRES", os.environ.get("RKNN_YOLOV5_SCORE_THRES", "0.25")))
        self.det_nms_thres = float(os.environ.get("RKNN_DET_NMS_THRES", os.environ.get("RKNN_YOLOV5_NMS_THRES", "0.65")))
        self.det_topk = int(os.environ.get("RKNN_DET_TOPK", os.environ.get("RKNN_YOLOV5_TOPK", "20")))
        self.det_score_mode = _choice(os.environ.get("RKNN_YOLOV5_SCORE_MODE"), {"auto", "raw", "sigmoid"}, "auto")
        self.det_box_format = _choice(os.environ.get("RKNN_YOLOV5_BOX_FORMAT"), {"xyxy", "xywh"}, "xyxy")
        self.det_interval = max(1, int(os.environ.get("RKNN_DET_INTERVAL", os.environ.get("RKNN_YOLOV5_DET_INTERVAL", "1"))))
        self.det_cache_seconds = max(0.0, float(os.environ.get("RKNN_DET_CACHE_SECONDS", os.environ.get("RKNN_YOLOV5_DET_CACHE_SECONDS", "0.5"))))
        self.adaptive_detector = os.environ.get("RKNN_ADAPTIVE_DETECTOR", "0").strip() == "1"
        self.det_refresh_seconds = max(0.1, float(os.environ.get("RKNN_DET_REFRESH_SECONDS", "0.75")))
        self.det_retry_seconds = max(0.05, float(os.environ.get("RKNN_DET_RETRY_SECONDS", "0.25")))
        self.det_bad_pose_frames = max(1, int(os.environ.get("RKNN_DET_BAD_POSE_FRAMES", "2")))
        self.tracker_margin = max(0.0, float(os.environ.get("RKNN_TRACKER_MARGIN", "0.20")))
        self.tracker_alpha = min(1.0, max(0.05, float(os.environ.get("RKNN_TRACKER_ALPHA", "0.35"))))
        self.tracker_min_points = max(3, int(os.environ.get("RKNN_TRACKER_MIN_POINTS", "5")))
        self.max_pose_padding_ratio = min(0.90, max(0.10, float(os.environ.get("RKNN_MAX_POSE_PADDING_RATIO", "0.55"))))
        self.draw_enabled = os.environ.get("RKNN_YOLOV5_BACKEND_DRAW", "1").strip() != "0"
        self.person_only_fast = os.environ.get("RKNN_YOLOV5_PERSON_ONLY_FAST", "0").strip() == "1"

        self.keypoint_thres = float(keypoint_thres if keypoint_thres is not None else os.environ.get("RKNN_POSE_KEYPOINT_THRES", "0.18"))
        self.pose_input_w = int(os.environ.get("RKNN_RTMPOSE_INPUT_WIDTH", "192"))
        self.pose_input_h = int(os.environ.get("RKNN_RTMPOSE_INPUT_HEIGHT", "256"))
        self.pose_input_size = (self.pose_input_w, self.pose_input_h)
        self.pose_input_layout = _choice(os.environ.get("RKNN_RTMPOSE_INPUT_LAYOUT", os.environ.get("RKNN_INPUT_LAYOUT")), {"nchw", "nhwc"}, "nhwc")
        self.simcc_score_mode = _choice(os.environ.get("RKNN_RTMPOSE_SIMCC_SCORE_MODE"), {"sqrt", "avg"}, "sqrt")
        self.simcc_split_ratio = float(os.environ.get("RKNN_RTMPOSE_SIMCC_SPLIT_RATIO", "2.0"))
        self.max_persons = max(1, int(os.environ.get("RKNN_MAX_POSE_PERSONS", "1")))
        self.person_select = _choice(os.environ.get("RKNN_PERSON_SELECT"), {"score", "largest", "center", "largest_center"}, "largest_center")
        self.rtmpose_bbox_expand = max(1.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_EXPAND", "1.25")))
        self.rtmpose_bbox_top_expand = max(0.0, float(os.environ.get("RKNN_RTMPOSE_BBOX_TOP_EXPAND", "0.10")))
        self.rtmpose_wide_bbox_ratio = max(0.1, float(os.environ.get("RKNN_RTMPOSE_WIDE_BBOX_RATIO", "0.65")))
        self.rtmpose_wide_bbox_expand = max(
            self.rtmpose_bbox_expand,
            float(os.environ.get("RKNN_RTMPOSE_WIDE_BBOX_EXPAND", "1.50")),
        )
        self.debug_crop_every = max(0, int(os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_EVERY", "30")))
        self.debug_crop_path = os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_PATH", "outputs/debug_yolov5n_rtmpose_crop.jpg")

        core = str(core_mask or os.environ.get("RKNN_CORE_MASK", "auto"))
        det_core = str(os.environ.get("RKNN_DET_CORE_MASK", core))
        pose_core = str(os.environ.get("RKNN_POSE_CORE_MASK", core))
        self.det_model = RKNNModel(self.det_model_path, core_mask=det_core)
        self.pose_model = RKNNModel(self.pose_model_path, core_mask=pose_core)
        self._frame_index = 0
        self._cached_dets = np.zeros((0, 5), dtype=np.float32)
        self._cached_frame_shape: tuple[int, ...] | None = None
        self._cached_at = 0.0
        self._cached_output_shapes: list[tuple[int, ...]] = []
        self._cached_meta: dict[str, Any] = {}
        self._last_detector_at = 0.0
        self._bad_pose_frames = 0
        self._tracker_quality = "empty"
        self._detector_trigger_reason = "initial"
        self._enabled_fn: Callable[[], bool] | None = None
        self._resource_lock = threading.RLock()
        self._resource_state = "qwen_available"
        self._resource_owner = "qwen"
        self._last_loaded_at: float | None = None
        self._last_released_at: float | None = None
        self._last_resource_error: str | None = None
        self._last_diagnostics: dict[str, Any] = {}
        self._tracking_reset_count = 0
        self._last_tracking_reset_reason: str | None = None
        self._last_tracking_reset_at: float | None = None

    @property
    def model_path(self) -> str:
        return f"{self.det_model_path};{self.pose_model_path}"

    def set_enabled_fn(self, enabled_fn: Callable[[], bool] | None) -> None:
        self._enabled_fn = enabled_fn

    def resource_snapshot(self) -> dict[str, Any]:
        requested = self._pose_requested()
        with self._resource_lock:
            try:
                det_model_size_bytes = Path(self.det_model_path).stat().st_size
            except OSError:
                det_model_size_bytes = None
            det_model_deprecated = Path(self.det_model_path).name == "yolov5n_nonms_fp.rknn" or (
                det_model_size_bytes is not None and det_model_size_bytes >= 100 * 1024 * 1024
            )
            return {
                "state": self._resource_state,
                "owner": self._resource_owner,
                "requested": requested,
                "models_loaded": self.det_model.rknn is not None or self.pose_model.rknn is not None,
                "det_model_loaded": self.det_model.rknn is not None,
                "pose_model_loaded": self.pose_model.rknn is not None,
                "det_model_path": self.det_model_path,
                "det_model_size_bytes": det_model_size_bytes,
                "det_model_deprecated": det_model_deprecated,
                "pose_model_path": self.pose_model_path,
                "core_mask": self.det_model.core_mask if self.det_model.core_mask == self.pose_model.core_mask else "split",
                "det_core_mask": self.det_model.core_mask,
                "pose_core_mask": self.pose_model.core_mask,
                "last_loaded_at": self._last_loaded_at,
                "last_released_at": self._last_released_at,
                "last_error": self._last_resource_error,
                "tracking_reset_count": self._tracking_reset_count,
                "last_tracking_reset_reason": self._last_tracking_reset_reason,
                "last_tracking_reset_at": self._last_tracking_reset_at,
            }

    def diagnostics_snapshot(self) -> dict[str, Any]:
        with self._resource_lock:
            return dict(self._last_diagnostics)

    def reset_tracking_state(self, reason: str = "manual") -> None:
        """Drop cross-frame tracking state without unloading either RKNN model."""
        with self._resource_lock:
            self._cached_dets = np.zeros((0, 5), dtype=np.float32)
            self._cached_frame_shape = None
            self._cached_at = 0.0
            self._cached_output_shapes = []
            self._cached_meta = {}
            self._last_detector_at = 0.0
            self._bad_pose_frames = 0
            self._tracker_quality = "empty"
            self._detector_trigger_reason = "tracking_reset"
            self._tracking_reset_count += 1
            self._last_tracking_reset_reason = str(reason or "manual")
            self._last_tracking_reset_at = time.time()
            self._last_diagnostics.update(
                {
                    "detector_trigger_reason": self._detector_trigger_reason,
                    "detector_age_ms": None,
                    "tracker_roi": None,
                    "tracker_quality": self._tracker_quality,
                    "tracker_visible_points": 0,
                    "bad_pose_frames": 0,
                    "tracking_reset_count": self._tracking_reset_count,
                    "last_tracking_reset_reason": self._last_tracking_reset_reason,
                    "last_tracking_reset_at": self._last_tracking_reset_at,
                }
            )

    def load(self) -> None:
        with self._resource_lock:
            self._resource_state = "loading"
            self._resource_owner = "pose"
            try:
                self.det_model.load()
                self.pose_model.load()
            except Exception as exc:
                self._resource_state = "error"
                self._last_resource_error = str(exc)
                raise
            self._resource_state = "pose_active"
            self._last_loaded_at = time.time()
            self._last_resource_error = None

    def release(self) -> None:
        with self._resource_lock:
            if self.det_model.rknn is None and self.pose_model.rknn is None:
                self._resource_state = "qwen_available"
                self._resource_owner = "qwen"
                return
            self._resource_state = "releasing"
            try:
                self.det_model.release()
                self.pose_model.release()
            finally:
                self.reset_tracking_state("released")
                self._detector_trigger_reason = "released"
                self._resource_state = "qwen_available"
                self._resource_owner = "qwen"
                self._last_released_at = time.time()

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        if not self._pose_requested():
            self.release()
            preview = frame_bgr.copy()
            cv2.putText(
                preview,
                "NPU pose released - Qwen available",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )
            return PoseResult(
                backend="rknn",
                fps=None,
                keypoints={},
                annotated_frame=preview,
                meta={
                    "detections": [],
                    "postprocess_error": None,
                    "rknn_decoder": "yolov5_raw_head_rtmpose",
                    "rknn_pipeline": "yolov5n_raw_rtmpose",
                    "npu_idle": True,
                    "npu_resource": self.resource_snapshot(),
                    "performance_ms": {"total_pose_ms": 0.0},
                },
            )

        with self._resource_lock:
            self._resource_state = "loading" if self.det_model.rknn is None else "pose_active"
            self._resource_owner = "pose"
            try:
                result = self._infer_active(frame_bgr)
            except Exception as exc:
                self._resource_state = "error"
                self._last_resource_error = str(exc)
                raise
            self._resource_state = "pose_active"
            self._last_loaded_at = self._last_loaded_at or time.time()
            self._last_resource_error = None
            self._last_diagnostics = {
                key: value
                for key, value in result.meta.items()
                if key
                in {
                    "detections",
                    "postprocess_error",
                    "det_output_shapes",
                    "pose_output_shapes",
                    "rknn_decoder",
                    "rknn_pipeline",
                    "keypoint_conf_range",
                    "performance_ms",
                    "det_model_path",
                    "pose_model_path",
                    "det_model_loaded",
                    "pose_model_loaded",
                    "det_score_thres",
                    "det_nms_thres",
                    "det_raw_count",
                    "det_score_pass_count",
                    "det_person_candidate_count",
                    "det_valid_bbox_count",
                    "det_keep_count",
                    "det_person_top_scores",
                    "det_best_class_id",
                    "det_best_class_score",
                    "det_decoder",
                    "det_feature_map_shapes",
                    "det_output_contract",
                    "det_model_contract_warning",
                    "detector_contract_error",
                    "selected_yolo_bbox",
                    "selected_yolo_score",
                    "rtmpose_wide_bbox_ratio",
                    "rtmpose_wide_bbox_expand",
                    "rtmpose_applied_x_scales",
                    "rtmpose_restore_bboxes",
                    "rtmpose_roi_fallbacks",
                    "rtmpose_padding_ratios",
                    "adaptive_detector",
                    "detector_trigger_reason",
                    "detector_age_ms",
                    "det_cache_hit",
                    "det_cache_seconds",
                    "det_cache_valid",
                    "det_cache_age_ms",
                    "yolo_detection_count",
                    "tracker_roi",
                    "tracker_quality",
                    "tracker_visible_points",
                    "bad_pose_frames",
                }
            }
            result.meta["npu_resource"] = self.resource_snapshot()
            return result

    def _infer_active(self, frame_bgr: np.ndarray) -> PoseResult:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected a BGR HxWx3 frame, got {getattr(frame_bgr, 'shape', None)}")
        if frame_bgr.size == 0:
            raise ValueError("Expected a non-empty camera frame")
        # RTMPose is loaded lazily only after the detector has produced a
        # usable person ROI.
        self.det_model.load()
        total_start = time.perf_counter()
        frame_shape = tuple(frame_bgr.shape)
        frame_index = self._frame_index
        self._frame_index += 1
        now = time.perf_counter()

        postprocess_error = None
        detection_miss_reason = None
        det_outputs: list[Any] = []
        det_output_shapes: list[tuple[int, ...]] = []
        pose_output_shapes: list[list[tuple[int, ...]]] = []
        det_meta: dict[str, Any] = {}
        det_cache_hit = False
        det_pre_ms = det_infer_ms = det_post_ms = 0.0
        pose_pre_ms = pose_infer_ms = pose_post_ms = 0.0

        if self.adaptive_detector:
            run_detector, detector_trigger_reason = adaptive_detector_decision(
                has_tracker=self._cache_valid(frame_shape, now),
                detector_age_seconds=max(0.0, now - self._last_detector_at) if self._last_detector_at > 0 else None,
                refresh_seconds=self.det_refresh_seconds,
                retry_seconds=self.det_retry_seconds,
                bad_pose_frames=self._bad_pose_frames,
                bad_pose_limit=self.det_bad_pose_frames,
            )
        else:
            run_detector = self.det_interval <= 1 or frame_index % self.det_interval == 0 or not self._cache_valid(frame_shape, now)
            detector_trigger_reason = "fixed_interval" if run_detector else "fixed_interval_cache"
        self._detector_trigger_reason = detector_trigger_reason
        if run_detector:
            # Throttle detector retries even when inference or postprocessing
            # misses. Otherwise a single miss turns every following frame into
            # a 170-200ms detector frame and collapses the pose FPS.
            self._last_detector_at = now
            det_pre_start = time.perf_counter()
            det_input, letterbox_meta = preprocess_yolov5(frame_bgr, self.det_input_size, self.det_input_layout, self.det_input_mode)
            det_pre_ms = elapsed_ms(det_pre_start)

            det_infer_start = time.perf_counter()
            det_outputs = self.det_model.inference([det_input], data_format=self.det_input_layout)
            det_infer_ms = elapsed_ms(det_infer_start)
            det_output_shapes = output_shapes_of(det_outputs)

            det_post_start = time.perf_counter()
            try:
                dets, det_meta = postprocess_yolov5_person(
                    det_outputs,
                    letterbox_meta,
                    score_thr=self.det_score_thres,
                    iou_thr=self.det_nms_thres,
                    topk=self.det_topk,
                    score_mode=self.det_score_mode,
                    box_format=self.det_box_format,
                    person_only_fast=self.person_only_fast,
                )
                if len(dets) > 0:
                    self._update_cache(dets, frame_shape, now, det_output_shapes, det_meta)
                elif self._cache_valid(frame_shape, now):
                    dets, det_output_shapes, det_meta = self._cached_detection()
                    det_cache_hit = True
            except Exception as exc:
                if self._cache_valid(frame_shape, now):
                    dets, det_output_shapes, det_meta = self._cached_detection()
                    det_cache_hit = True
                else:
                    dets = np.zeros((0, 5), dtype=np.float32)
                    postprocess_error = f"YOLOv5 detector output contract error: {exc}; output_shapes={det_output_shapes}"
                    det_meta = {
                        "postprocess_error": postprocess_error,
                        "detector_contract_error": str(exc),
                    }
            det_post_ms = elapsed_ms(det_post_start)
        else:
            if self._cache_valid(frame_shape, now):
                dets, det_output_shapes, det_meta = self._cached_detection()
                det_cache_hit = True
            else:
                dets = np.zeros((0, 5), dtype=np.float32)
                det_meta = {"cache_reused": False, "cache_expired": True}

        selected = select_person_dets(dets, frame_bgr.shape, max_persons=self.max_persons, mode=self.person_select)
        pose_bbox_fallbacks: list[bool] = []
        pose_bbox_padding_ratios: list[float] = []
        pose_bbox_values: list[np.ndarray] = []
        for det in selected:
            expanded_bbox = expand_bbox_for_pose(
                det[:4],
                frame_bgr.shape,
                scale=self.rtmpose_bbox_expand,
                top_expand=self.rtmpose_bbox_top_expand,
                wide_ratio=self.rtmpose_wide_bbox_ratio,
                wide_scale=self.rtmpose_wide_bbox_expand,
            )
            safe_bbox, used_fallback, padding_ratio = bound_pose_bbox_for_frame(
                expanded_bbox,
                det[:4],
                frame_bgr.shape,
                self.pose_input_size,
                max_padding_ratio=self.max_pose_padding_ratio,
            )
            pose_bbox_values.append(safe_bbox)
            pose_bbox_fallbacks.append(used_fallback)
            pose_bbox_padding_ratios.append(padding_ratio)
        pose_bboxes = np.asarray(pose_bbox_values, dtype=np.float32)
        pose_x_scales = [
            adaptive_pose_x_scale(
                det[:4],
                base_scale=self.rtmpose_bbox_expand,
                wide_ratio=self.rtmpose_wide_bbox_ratio,
                wide_scale=self.rtmpose_wide_bbox_expand,
            )
            for det in selected
        ]
        if postprocess_error is None and len(selected) == 0:
            max_person_score = det_meta.get("person_score_range")
            max_person_score = max_person_score[1] if isinstance(max_person_score, list) and len(max_person_score) == 2 else None
            score_passed = max_person_score is not None and float(max_person_score) > self.det_score_thres
            reason = (
                "person score passed threshold but every bbox decode attempt was invalid"
                if score_passed
                else "no person score passed threshold"
            )
            detection_miss_reason = (
                f"YOLOv5n did not produce a person box for RTMPose: {reason}; "
                f"max_person_score={max_person_score}, threshold={self.det_score_thres}, "
                f"data_format={self.det_input_layout}(explicit)"
            )

        # Materialize YOLO detections before running RTMPose.  A pose model
        # load/decode failure must never hide a valid person detector result.
        detections: list[dict[str, Any]] = [
            {
                "bbox": [float(value) for value in det[:4].tolist()],
                "score": float(det[4]),
                "pose_bbox": [float(value) for value in pose_bbox.tolist()],
                "keypoints": [],
            }
            for det, pose_bbox in zip(selected, pose_bboxes)
        ]
        keypoint_arrays: list[np.ndarray] = []
        score_arrays: list[np.ndarray] = []
        pose_restore_bboxes: list[list[float]] = []
        pose_crop_ranges: list[dict[str, list[float] | None]] = []
        pose_decode_meta: list[dict[str, Any]] = []
        if postprocess_error is None:
            for idx, (det, pose_bbox) in enumerate(zip(selected, pose_bboxes)):
                pose_pre_start = time.perf_counter()
                pose_input, restore, crop = preprocess_rtmpose(frame_bgr, pose_bbox, self.pose_input_size, self.pose_input_layout)
                actual_crop_bbox = [float(value) for value in restore.image_bbox.tolist()]
                detections[idx]["rtmpose_crop_bbox"] = actual_crop_bbox
                pose_restore_bboxes.append([round(value, 2) for value in actual_crop_bbox])
                pose_pre_ms += elapsed_ms(pose_pre_start)

                pose_infer_start = time.perf_counter()
                pose_outputs = self.pose_model.inference([pose_input], data_format=self.pose_input_layout)
                pose_infer_ms += elapsed_ms(pose_infer_start)
                pose_output_shapes.append(output_shapes_of(pose_outputs))

                pose_post_start = time.perf_counter()
                try:
                    keypoints, scores, crop_points, decode_meta = decode_simcc(
                        pose_outputs,
                        restore,
                        input_size=self.pose_input_size,
                        split_ratio=self.simcc_split_ratio,
                        score_mode=self.simcc_score_mode,
                    )
                    if idx == 0 and self.debug_crop_every > 0 and frame_index % self.debug_crop_every == 0:
                        save_debug_crop(crop, crop_points, scores, self.debug_crop_path, self.keypoint_thres)
                    keypoint_arrays.append(keypoints)
                    score_arrays.append(scores)
                    pose_crop_ranges.append(xy_range(crop_points))
                    pose_decode_meta.append(decode_meta)
                    detections[idx]["keypoints"] = [
                        [float(x), float(y), normalize_score(float(score))]
                        for (x, y), score in zip(keypoints, scores)
                    ]
                except Exception as exc:
                    postprocess_error = f"RTMPose postprocess failed: {exc}; pose_output_shapes={pose_output_shapes}"
                    keypoint_arrays = []
                    score_arrays = []
                    break
                finally:
                    pose_post_ms += elapsed_ms(pose_post_start)

        tracker_bbox = None
        tracker_visible_points = 0
        if keypoint_arrays and score_arrays and len(selected) > 0:
            tracker_bbox, tracker_visible_points = tracker_bbox_from_keypoints(
                keypoint_arrays[0],
                score_arrays[0],
                frame_bgr.shape,
                score_threshold=min(self.keypoint_thres, 0.16),
                margin=self.tracker_margin,
            )
        if self.adaptive_detector:
            if tracker_bbox is not None and tracker_visible_points >= self.tracker_min_points:
                tracked = blend_detection_bbox(selected[0], tracker_bbox, alpha=self.tracker_alpha)
                self._update_tracked_cache_bbox(tracked[None, :], frame_shape)
                self._bad_pose_frames = 0
                self._tracker_quality = "pose_keypoints"
            else:
                self._bad_pose_frames += 1
                self._tracker_quality = "weak_pose" if len(selected) else "no_person"

        draw_start = time.perf_counter()
        annotated = draw_poses(frame_bgr, detections, keypoint_thres=self.keypoint_thres) if self.draw_enabled else frame_bgr
        draw_ms = elapsed_ms(draw_start)
        total_ms = elapsed_ms(total_start)
        performance_ms = {
            "preprocess_ms": round(det_pre_ms + pose_pre_ms, 2),
            "inference_ms": round(det_infer_ms + pose_infer_ms, 2),
            "postprocess_ms": round(det_post_ms + pose_post_ms, 2),
            "draw_ms": round(draw_ms, 2),
            "total_pose_ms": round(total_ms, 2),
            "det_preprocess_ms": round(det_pre_ms, 2),
            "det_inference_ms": round(det_infer_ms, 2),
            "det_postprocess_ms": round(det_post_ms, 2),
            "pose_preprocess_ms": round(pose_pre_ms, 2),
            "pose_inference_ms": round(pose_infer_ms, 2),
            "pose_postprocess_ms": round(pose_post_ms, 2),
        }
        return PoseResult(
            backend="rknn",
            fps=1000.0 / total_ms if total_ms > 0 else None,
            keypoints={},
            raw={"det_outputs": det_outputs},
            annotated_frame=annotated,
            meta={
                "detections": detections,
                "postprocess_error": postprocess_error,
                "detection_miss_reason": detection_miss_reason,
                "output_shapes": det_output_shapes,
                "det_output_shapes": det_output_shapes,
                "pose_output_shapes": pose_output_shapes,
                "keypoint_decode_meta": pose_decode_meta[0] if pose_decode_meta else None,
                "keypoint_decode_meta_all": pose_decode_meta,
                "rknn_decoder": det_meta.get("rknn_decoder", "yolov5_raw_head"),
                "rknn_pipeline": "yolov5n_raw_rtmpose",
                "keypoint_decode_mode": "simcc",
                "keypoint_conf_range": range_or_none([score for scores in score_arrays for score in scores.tolist()]),
                "keypoint_xy_range": xy_range(np.concatenate(keypoint_arrays, axis=0)) if keypoint_arrays else {"x": None, "y": None},
                "keypoint_restored_xy_range": xy_range(np.concatenate(keypoint_arrays, axis=0)) if keypoint_arrays else {"x": None, "y": None},
                "performance_ms": performance_ms,
                "inference_ms": performance_ms["inference_ms"],
                "preprocess_ms": performance_ms["preprocess_ms"],
                "postprocess_ms": performance_ms["postprocess_ms"],
                "draw_ms": round(draw_ms, 2),
                "total_pose_ms": round(total_ms, 2),
                "model_path": self.model_path,
                "det_model_path": self.det_model_path,
                "pose_model_path": self.pose_model_path,
                "det_model_loaded": self.det_model.rknn is not None,
                "pose_model_loaded": self.pose_model.rknn is not None,
                "frame_shape": frame_shape,
                "det_input_size": self.det_input_size,
                "det_input_layout": self.det_input_layout,
                "det_data_format_explicit": True,
                "det_input_mode": self.det_input_mode,
                "det_score_thres": self.det_score_thres,
                "det_nms_thres": self.det_nms_thres,
                "det_score_mode": "sigmoid(cls)*sigmoid(objectness)" if len(det_outputs) in {3, 9} else self.det_score_mode,
                "det_box_format": "xyxy_input_pixels",
                "det_letterbox": det_meta.get("letterbox"),
                "det_raw_count": det_meta.get("raw_count"),
                "det_score_pass_count": det_meta.get("score_pass_count"),
                "det_person_candidate_count": det_meta.get("person_candidate_count"),
                "det_valid_bbox_count": det_meta.get("valid_bbox_count"),
                "det_scored_width_range": det_meta.get("scored_width_range"),
                "det_scored_height_range": det_meta.get("scored_height_range"),
                "det_keep_count": det_meta.get("keep_count"),
                "det_person_score_range": det_meta.get("person_score_range"),
                "det_person_score_mean": det_meta.get("person_score_mean"),
                "det_person_top_scores": det_meta.get("person_top_scores"),
                "det_best_class_id": det_meta.get("best_class_id"),
                "det_best_class_score": det_meta.get("best_class_score"),
                "det_box_decode_mode": det_meta.get("box_decode_mode"),
                "det_box_decode_fallback_used": det_meta.get("box_decode_fallback_used"),
                "det_inferred_coordinate_mode": det_meta.get("inferred_coordinate_mode"),
                "det_coordinate_decode_fallback_used": det_meta.get("coordinate_decode_fallback_used"),
                "det_box_decode_attempts": det_meta.get("box_decode_attempts"),
                "det_raw_top_candidates": det_meta.get("raw_top_candidates"),
                "det_decoder": det_meta.get("rknn_decoder"),
                "det_feature_map_shapes": det_meta.get("feature_map_shapes"),
                "det_output_contract": det_meta.get("output_contract"),
                "det_model_contract_warning": det_meta.get("model_contract_warning"),
                "detector_contract_error": det_meta.get("detector_contract_error"),
                "det_cache_hit": det_cache_hit,
                "det_interval": self.det_interval,
                "det_cache_seconds": self.det_cache_seconds,
                "det_cache_valid": self._cache_valid(frame_shape, now),
                "det_cache_age_ms": round(max(0.0, now - self._cached_at) * 1000.0, 2) if self._cached_at > 0 else None,
                "adaptive_detector": self.adaptive_detector,
                "detector_trigger_reason": detector_trigger_reason,
                "detector_age_ms": round(max(0.0, now - self._last_detector_at) * 1000.0, 2) if self._last_detector_at > 0 else None,
                "detector_retry_seconds": self.det_retry_seconds,
                "tracker_roi": [round(float(value), 2) for value in tracker_bbox.tolist()] if tracker_bbox is not None else None,
                "tracker_quality": self._tracker_quality,
                "tracker_visible_points": tracker_visible_points,
                "bad_pose_frames": self._bad_pose_frames,
                "backend_draw_enabled": self.draw_enabled,
                "person_only_fast": self.person_only_fast,
                "person_select": self.person_select,
                "selected_yolo_bbox": [round(float(value), 2) for value in selected[0, :4].tolist()] if len(selected) else None,
                "selected_yolo_score": round(float(selected[0, 4]), 4) if len(selected) else None,
                "yolo_detection_count": int(len(dets)),
                "rtmpose_crop_input_size": [self.pose_input_w, self.pose_input_h],
                "rtmpose_bbox_expand": self.rtmpose_bbox_expand,
                "rtmpose_bbox_top_expand": self.rtmpose_bbox_top_expand,
                "rtmpose_wide_bbox_ratio": self.rtmpose_wide_bbox_ratio,
                "rtmpose_wide_bbox_expand": self.rtmpose_wide_bbox_expand,
                "rtmpose_applied_x_scales": [round(float(value), 3) for value in pose_x_scales],
                "rtmpose_pose_bboxes": [[round(float(value), 2) for value in bbox.tolist()] for bbox in pose_bboxes],
                "rtmpose_roi_fallbacks": pose_bbox_fallbacks,
                "rtmpose_padding_ratios": [round(float(value), 4) for value in pose_bbox_padding_ratios],
                "rtmpose_restore_bboxes": pose_restore_bboxes,
                "rtmpose_crop_xy_ranges": pose_crop_ranges,
                "rtmpose_debug_crop_path": self.debug_crop_path if self.debug_crop_every > 0 else None,
                "keypoint_thres": self.keypoint_thres,
                "input_layout": self.pose_input_layout,
                "pose_data_format_explicit": True,
                "simcc_score_mode": self.simcc_score_mode,
            },
        )

    def _pose_requested(self) -> bool:
        if self._enabled_fn is None:
            return True
        try:
            return bool(self._enabled_fn())
        except Exception as exc:
            self._last_resource_error = f"enabled callback failed: {exc}"
            return False

    def _cache_valid(self, frame_shape: tuple[int, ...], now: float) -> bool:
        return len(self._cached_dets) > 0 and self._cached_frame_shape == frame_shape and self.det_cache_seconds > 0 and now - self._cached_at <= self.det_cache_seconds

    def _update_cache(self, dets: np.ndarray, frame_shape: tuple[int, ...], now: float, output_shapes: list[tuple[int, ...]], meta: dict[str, Any]) -> None:
        self._cached_dets = np.asarray(dets, dtype=np.float32).copy()
        self._cached_frame_shape = frame_shape
        self._cached_at = now
        self._cached_output_shapes = list(output_shapes)
        self._cached_meta = dict(meta)

    def _update_tracked_cache_bbox(self, dets: np.ndarray, frame_shape: tuple[int, ...]) -> None:
        """Update the tracked ROI without extending the last real YOLO hit."""
        if len(self._cached_dets) == 0 or self._cached_frame_shape != frame_shape:
            return
        self._cached_dets = np.asarray(dets, dtype=np.float32).copy()

    def _cached_detection(self) -> tuple[np.ndarray, list[tuple[int, ...]], dict[str, Any]]:
        meta = dict(self._cached_meta)
        meta["cache_reused"] = True
        return self._cached_dets.copy(), list(self._cached_output_shapes), meta


def letterbox(frame_bgr: np.ndarray, input_size: int) -> tuple[np.ndarray, LetterboxMeta]:
    original_h, original_w = frame_bgr.shape[:2]
    scale = min(float(input_size) / max(float(original_w), 1.0), float(input_size) / max(float(original_h), 1.0))
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, LetterboxMeta(input_size, input_size, original_w, original_h, scale, float(left), float(top))


def preprocess_yolov5(frame_bgr: np.ndarray, input_size: int, input_layout: str, input_mode: str) -> tuple[np.ndarray, LetterboxMeta]:
    boxed, meta = letterbox(frame_bgr, input_size)
    image = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB) if input_mode.startswith("rgb") else boxed
    if input_mode.endswith("uint8"):
        arr = np.ascontiguousarray(image, dtype=np.uint8)
    else:
        arr = image.astype(np.float32)
        if input_mode.endswith("0_1"):
            arr /= 255.0
        else:
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            if input_mode.startswith("bgr"):
                mean = mean[::-1]
                std = std[::-1]
            arr = (arr - mean) / std
    blob = arr[None, ...] if input_layout == "nhwc" else arr.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(blob), meta


def postprocess_yolov5_person(
    outputs: list[Any],
    meta: LetterboxMeta,
    *,
    score_thr: float,
    iou_thr: float,
    topk: int,
    score_mode: str | None = None,
    box_format: str | None = None,
    person_only_fast: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    del box_format  # Raw heads and the legacy rollback path have fixed xyxy semantics.
    arrays = [np.asarray(output) for output in outputs if output is not None]
    if len(arrays) == 3:
        raw_boxes, person_scores, decode_diagnostics = decode_yolov5_combined_heads(
            arrays,
            meta,
            person_only_fast=person_only_fast,
            candidate_score_thr=score_thr,
        )
        score_diagnostics = raw_score_diagnostics(person_scores, decode_diagnostics)
    elif len(arrays) == 9:
        raw_boxes, person_scores, decode_diagnostics = decode_yolov5_raw_heads(
            arrays,
            meta,
            person_only_fast=person_only_fast,
            candidate_score_thr=score_thr,
        )
        score_diagnostics = raw_score_diagnostics(person_scores, decode_diagnostics)
    elif len(arrays) == 2:
        raw_boxes, scores = normalize_legacy_yolov5_outputs(arrays)
        effective_score_mode = score_mode or "auto"
        person_scores = normalize_det_scores(scores[:, 0], effective_score_mode)
        person_scores = np.nan_to_num(person_scores, nan=0.0, posinf=0.0, neginf=0.0)
        score_diagnostics = detector_score_diagnostics(scores, person_scores, effective_score_mode)
        decode_diagnostics = {
            "rknn_decoder": "legacy_decoded_xyxy",
            "feature_map_shapes": None,
            "output_contract": "legacy [1,N,4] xyxy + [1,80,N] scores",
            "model_contract_warning": (
                "Legacy TopK/Gather detector output is deprecated and not trusted for 8085; "
                "use models/vision/yolov5n_raw_fp.rknn"
            ),
        }
    else:
        shapes = [tuple(arr.shape) for arr in arrays]
        raise RuntimeError(
            "expected 3 combined or 9 split YOLOv5 raw-head outputs; "
            f"legacy rollback accepts exactly 2 outputs, got {len(arrays)} with shapes={shapes}"
        )

    if not np.all(np.isfinite(raw_boxes)):
        raise RuntimeError("YOLOv5 boxes contain NaN/Inf")
    raw_count = int(decode_diagnostics.get("raw_count_total", len(raw_boxes)))
    score_keep = person_scores > float(score_thr)
    scored_boxes = raw_boxes[score_keep]
    scored_scores = person_scores[score_keep]
    raw_top_indices = top_indices(person_scores, 5)
    top_restored = restore_letterbox_boxes(raw_boxes[raw_top_indices], meta) if len(raw_top_indices) else np.zeros((0, 4), dtype=np.float32)
    raw_top_candidates = []
    for index, restored_box in zip(raw_top_indices, top_restored):
        width = float(restored_box[2] - restored_box[0])
        height = float(restored_box[3] - restored_box[1])
        raw_top_candidates.append(
            {
                "index": int(index),
                "location": yolov5_candidate_location(int(index), meta.input_width),
                "score": round(float(person_scores[index]), 6),
                "input_box": [round(float(value), 4) for value in raw_boxes[index].tolist()],
                "restored_box": [round(float(value), 2) for value in restored_box.tolist()],
                "restored_size": [round(width, 2), round(height, 2)],
                "valid_after_restore": bool(np.isfinite(restored_box).all() and width > 2.0 and height > 2.0),
            }
        )
    if len(scored_boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32), {
            "raw_count": raw_count,
            "keep_count": 0,
            "score_pass_count": 0,
            "person_candidate_count": 0,
            "valid_bbox_count": 0,
            "letterbox": letterbox_meta_dict(meta),
            "raw_top_candidates": raw_top_candidates,
            "box_decode_mode": "xyxy:input_pixels",
            "box_decode_fallback_used": False,
            "coordinate_decode_fallback_used": False,
            **decode_diagnostics,
            **score_diagnostics,
        }

    restored = restore_letterbox_boxes(scored_boxes, meta)
    widths = restored[:, 2] - restored[:, 0]
    heights = restored[:, 3] - restored[:, 1]
    valid = np.isfinite(restored).all(axis=1) & (widths > 2.0) & (heights > 2.0)
    valid_boxes = restored[valid]
    valid_scores = scored_scores[valid]
    if len(valid_boxes) > 0:
        keep_idx = nms_numpy(valid_boxes, valid_scores, iou_thr=iou_thr)
        valid_boxes = valid_boxes[keep_idx]
        valid_scores = valid_scores[keep_idx]
        order = valid_scores.argsort()[::-1][: int(topk)]
        dets = np.concatenate([valid_boxes[order], valid_scores[order][:, None]], axis=1).astype(np.float32)
    else:
        dets = np.zeros((0, 5), dtype=np.float32)

    return dets, {
        "raw_count": raw_count,
        "keep_count": int(len(dets)),
        "score_pass_count": int(np.count_nonzero(score_keep)),
        "person_candidate_count": int(np.count_nonzero(score_keep)),
        "valid_bbox_count": int(np.count_nonzero(valid)),
        "scored_width_range": array_range_or_none(widths),
        "scored_height_range": array_range_or_none(heights),
        "letterbox": letterbox_meta_dict(meta),
        "box_decode_mode": "xyxy:input_pixels",
        "box_decode_fallback_used": False,
        "coordinate_decode_fallback_used": False,
        "raw_top_candidates": raw_top_candidates,
        **decode_diagnostics,
        **score_diagnostics,
    }


def raw_score_diagnostics(person_scores: np.ndarray, decode_diagnostics: dict[str, Any]) -> dict[str, Any]:
    person = np.asarray(person_scores, dtype=np.float32)
    indices = top_indices(person, 5)
    top = person[indices] if len(indices) else np.zeros((0,), dtype=np.float32)
    return {
        "person_score_range": array_range_or_none(person),
        "person_score_mean": round(float(np.mean(person)), 6) if person.size else None,
        "person_top_scores": [round(float(value), 6) for value in top.tolist()],
        "best_class_id": decode_diagnostics.get("best_class_id"),
        "best_class_score": decode_diagnostics.get("best_class_score"),
    }


def decode_yolov5_combined_heads(
    outputs: list[np.ndarray],
    meta: LetterboxMeta,
    *,
    person_only_fast: bool = False,
    candidate_score_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(outputs) != 3:
        raise RuntimeError(f"YOLOv5 combined raw decoder requires exactly 3 outputs, got {len(outputs)}")
    by_grid: dict[int, tuple[np.ndarray, list[int]]] = {}
    for index, output in enumerate(outputs):
        arr = np.asarray(output, dtype=np.float32)
        nchw = raw_tensor_to_nchw(arr, YOLOV5_COMBINED_CHANNELS)
        if nchw is None:
            raise RuntimeError(f"combined raw output[{index}] must contain 255 channels, got shape={arr.shape}")
        height, width = int(nchw.shape[2]), int(nchw.shape[3])
        if height != width or height in by_grid:
            raise RuntimeError(f"invalid or duplicate combined raw feature map {height}x{width}")
        by_grid[height] = (nchw, [int(value) for value in arr.shape])

    if person_only_fast:
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        combined_shapes: list[dict[str, Any]] = []
        raw_count_total = 0
        for level_index, stride in enumerate(YOLOV5_STRIDES):
            grid_size = meta.input_width // stride
            item = by_grid.get(grid_size)
            if item is None:
                raise RuntimeError(f"missing combined raw tensor for stride {stride} ({grid_size}x{grid_size})")
            nchw, source_shape = item
            prediction = nchw.reshape(3, YOLOV5_ATTRS_PER_ANCHOR, grid_size, grid_size)
            bbox = prediction[:, 0:4].transpose(0, 2, 3, 1)
            objectness = sigmoid(prediction[:, 4])
            person_scores = sigmoid(prediction[:, 5]) * objectness
            flat_scores = person_scores.reshape(-1)
            candidate_mask = flat_scores > float(candidate_score_thr)
            if int(np.count_nonzero(candidate_mask)) < 5:
                candidate_mask[top_indices(flat_scores, 5)] = True
            candidate_indices = np.flatnonzero(candidate_mask)
            flat_bbox = bbox.reshape(-1, 4)[candidate_indices]
            grid = np.broadcast_to(yolov5_grid(grid_size), (3, grid_size, grid_size, 2)).reshape(-1, 2)[candidate_indices]
            anchors = np.broadcast_to(
                YOLOV5_ANCHORS[level_index][:, None, None, :],
                (3, grid_size, grid_size, 2),
            ).reshape(-1, 2)[candidate_indices]
            centers = (sigmoid(flat_bbox[:, :2]) * 2.0 - 0.5 + grid) * float(stride)
            sizes = np.square(sigmoid(flat_bbox[:, 2:4]) * 2.0) * anchors
            all_boxes.append(np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=-1))
            all_scores.append(flat_scores[candidate_indices])
            raw_count_total += 3 * grid_size * grid_size
            combined_shapes.append({"stride": stride, "grid": [grid_size, grid_size], "combined": source_shape})
        return (
            np.concatenate(all_boxes, axis=0).astype(np.float32),
            np.concatenate(all_scores, axis=0).astype(np.float32),
            {
                "rknn_decoder": "yolov5_combined_raw_head",
                "feature_map_shapes": combined_shapes,
                "output_contract": "3 combined raw tensors [1,255,H,W] at strides 8/16/32",
                "best_class_id": 0,
                "best_class_score": round(float(max(np.max(scores) for scores in all_scores)), 6),
                "person_only_fast": True,
                "raw_count_total": raw_count_total,
                "model_contract_warning": None,
            },
        )

    split_outputs: list[np.ndarray] = []
    combined_shapes: list[dict[str, Any]] = []
    for stride in YOLOV5_STRIDES:
        grid_size = meta.input_width // stride
        item = by_grid.get(grid_size)
        if item is None:
            raise RuntimeError(f"missing combined raw tensor for stride {stride} ({grid_size}x{grid_size})")
        nchw, source_shape = item
        prediction = nchw.reshape(1, 3, YOLOV5_ATTRS_PER_ANCHOR, grid_size, grid_size)
        bbox = prediction[:, :, 0:4].reshape(1, 12, grid_size, grid_size)
        obj = prediction[:, :, 4:5].reshape(1, 3, grid_size, grid_size)
        cls = prediction[:, :, 5:].reshape(1, 240, grid_size, grid_size)
        split_outputs.extend((cls, bbox, obj))
        combined_shapes.append({"stride": stride, "grid": [grid_size, grid_size], "combined": source_shape})

    boxes, scores, diagnostics = decode_yolov5_raw_heads(
        split_outputs,
        meta,
        person_only_fast=person_only_fast,
        candidate_score_thr=candidate_score_thr,
    )
    diagnostics.update(
        {
            "rknn_decoder": "yolov5_combined_raw_head",
            "feature_map_shapes": combined_shapes,
            "output_contract": "3 combined raw tensors [1,255,H,W] at strides 8/16/32",
            "model_contract_warning": None,
        }
    )
    return boxes, scores, diagnostics


def decode_yolov5_raw_heads(
    outputs: list[np.ndarray],
    meta: LetterboxMeta,
    *,
    person_only_fast: bool = False,
    candidate_score_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if meta.input_width != meta.input_height or meta.input_width % YOLOV5_STRIDES[-1] != 0:
        raise RuntimeError(f"YOLOv5 raw decoder requires square stride-aligned input, got {meta.input_width}x{meta.input_height}")
    if len(outputs) != 9:
        raise RuntimeError(f"YOLOv5 raw decoder requires exactly 9 outputs, got {len(outputs)}")

    levels: dict[int, dict[str, np.ndarray]] = {}
    source_shapes: dict[int, dict[str, list[int]]] = {}
    for index, output in enumerate(outputs):
        arr = np.asarray(output, dtype=np.float32)
        tensor_type = None
        nchw = None
        for channels, name in YOLOV5_RAW_CHANNELS.items():
            normalized = raw_tensor_to_nchw(arr, channels)
            if normalized is not None:
                if tensor_type is not None:
                    raise RuntimeError(f"ambiguous raw output[{index}] shape={arr.shape}")
                tensor_type = name
                nchw = normalized
        if tensor_type is None or nchw is None:
            raise RuntimeError(f"raw output[{index}] has unsupported shape={arr.shape}")
        height, width = int(nchw.shape[2]), int(nchw.shape[3])
        if height != width or tensor_type in levels.get(height, {}):
            raise RuntimeError(f"invalid or duplicate {tensor_type} tensor for {height}x{width}")
        levels.setdefault(height, {})[tensor_type] = nchw
        source_shapes.setdefault(height, {})[tensor_type] = [int(value) for value in arr.shape]

    all_boxes: list[np.ndarray] = []
    all_person_scores: list[np.ndarray] = []
    class_maxima = np.zeros((YOLOV5_NUM_CLASSES,), dtype=np.float32)
    raw_count_total = 0
    feature_map_shapes: list[dict[str, Any]] = []
    for level_index, stride in enumerate(YOLOV5_STRIDES):
        grid_size = meta.input_width // stride
        tensors = levels.get(grid_size)
        if tensors is None or set(tensors) != {"cls", "bbox", "obj"}:
            raise RuntimeError(f"missing raw tensors for stride {stride} ({grid_size}x{grid_size})")
        cls = tensors["cls"][0].reshape(3, YOLOV5_NUM_CLASSES, grid_size, grid_size).transpose(0, 2, 3, 1)
        bbox = tensors["bbox"][0].reshape(3, 4, grid_size, grid_size).transpose(0, 2, 3, 1)
        obj = tensors["obj"][0].reshape(3, grid_size, grid_size)
        objectness = sigmoid(obj)
        if person_only_fast:
            person_scores = sigmoid(cls[..., 0]) * objectness
            class_maxima[0] = max(class_maxima[0], float(np.max(person_scores)))
            flat_scores = person_scores.reshape(-1)
            candidate_mask = flat_scores > float(candidate_score_thr)
            if int(np.count_nonzero(candidate_mask)) < 5:
                candidate_mask[top_indices(flat_scores, 5)] = True
            candidate_indices = np.flatnonzero(candidate_mask)
            flat_bbox = bbox.reshape(-1, 4)[candidate_indices]
            grid = np.broadcast_to(yolov5_grid(grid_size), (3, grid_size, grid_size, 2)).reshape(-1, 2)[candidate_indices]
            anchors = np.broadcast_to(
                YOLOV5_ANCHORS[level_index][:, None, None, :],
                (3, grid_size, grid_size, 2),
            ).reshape(-1, 2)[candidate_indices]
            centers = (sigmoid(flat_bbox[:, :2]) * 2.0 - 0.5 + grid) * float(stride)
            sizes = np.square(sigmoid(flat_bbox[:, 2:4]) * 2.0) * anchors
            boxes = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=-1)
            person_scores = flat_scores[candidate_indices]
        else:
            grid = yolov5_grid(grid_size)
            centers = (sigmoid(bbox[..., :2]) * 2.0 - 0.5 + grid) * float(stride)
            anchors = YOLOV5_ANCHORS[level_index][:, None, None, :]
            sizes = np.square(sigmoid(bbox[..., 2:4]) * 2.0) * anchors
            boxes = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=-1)
            class_scores = sigmoid(cls) * objectness[..., None]
            class_maxima = np.maximum(class_maxima, np.max(class_scores, axis=(0, 1, 2)))
            person_scores = class_scores[..., 0]
        all_boxes.append(boxes.reshape(-1, 4))
        all_person_scores.append(person_scores.reshape(-1))
        raw_count_total += 3 * grid_size * grid_size
        feature_map_shapes.append(
            {
                "stride": stride,
                "grid": [grid_size, grid_size],
                "cls": source_shapes[grid_size]["cls"],
                "bbox": source_shapes[grid_size]["bbox"],
                "objectness": source_shapes[grid_size]["obj"],
            }
        )

    best_class_id = int(np.argmax(class_maxima))
    return (
        np.concatenate(all_boxes, axis=0).astype(np.float32),
        np.concatenate(all_person_scores, axis=0).astype(np.float32),
        {
            "rknn_decoder": "yolov5_raw_head",
            "feature_map_shapes": feature_map_shapes,
            "output_contract": "9 split raw tensors: cls/bbox/objectness at strides 8/16/32",
            "best_class_id": best_class_id,
            "best_class_score": round(float(class_maxima[best_class_id]), 6),
            "person_only_fast": bool(person_only_fast),
            "raw_count_total": raw_count_total,
            "model_contract_warning": None,
        },
    )


@lru_cache(maxsize=8)
def yolov5_grid(grid_size: int) -> np.ndarray:
    grid_y, grid_x = np.meshgrid(
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
        indexing="ij",
    )
    grid = np.stack((grid_x, grid_y), axis=-1)[None, ...]
    grid.setflags(write=False)
    return grid


def raw_tensor_to_nchw(array: np.ndarray, expected_channels: int) -> np.ndarray | None:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 1:
        return None
    nchw_match = int(arr.shape[1]) == int(expected_channels)
    nhwc_match = int(arr.shape[-1]) == int(expected_channels)
    if nchw_match and nhwc_match:
        raise RuntimeError(f"ambiguous NCHW/NHWC tensor shape={arr.shape}, channels={expected_channels}")
    if nchw_match:
        return np.ascontiguousarray(arr)
    if nhwc_match:
        return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
    return None


def yolov5_candidate_location(index: int, input_width: int) -> dict[str, int] | None:
    remaining = int(index)
    for stride in YOLOV5_STRIDES:
        grid = int(input_width) // stride
        level_count = 3 * grid * grid
        if remaining < level_count:
            anchor = remaining // (grid * grid)
            cell = remaining % (grid * grid)
            return {"stride": int(stride), "anchor": int(anchor), "x": int(cell % grid), "y": int(cell // grid)}
        remaining -= level_count
    return None


def normalize_legacy_yolov5_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    boxes, scores = pick_yolov5_outputs(outputs)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores.ndim == 3:
        scores = scores[0]
    if scores.shape[0] == YOLOV5_NUM_CLASSES and scores.shape[1] == boxes.shape[0]:
        scores = scores.T
    elif scores.shape[-1] != YOLOV5_NUM_CLASSES and scores.shape[0] == boxes.shape[0]:
        scores = scores.reshape(boxes.shape[0], -1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError(f"unexpected legacy YOLOv5 box shape: {boxes.shape}")
    if scores.ndim != 2 or scores.shape[0] != boxes.shape[0] or scores.shape[1] != YOLOV5_NUM_CLASSES:
        raise RuntimeError(f"legacy YOLOv5 boxes/scores mismatch: boxes={boxes.shape}, scores={scores.shape}")
    return boxes, scores


def array_range_or_none(values: np.ndarray) -> list[float] | None:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return None
    return [round(float(np.min(array)), 6), round(float(np.max(array)), 6)]


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int64)
    limit = min(int(count), int(array.size))
    if limit == array.size:
        return np.argsort(array)[::-1]
    indices = np.argpartition(array, -limit)[-limit:]
    return indices[np.argsort(array[indices])[::-1]]


def infer_box_coordinate_mode(boxes: np.ndarray, meta: LetterboxMeta) -> str:
    finite = np.abs(np.asarray(boxes, dtype=np.float32))
    max_abs = float(np.max(finite)) if finite.size else 0.0
    if max_abs <= 2.0:
        return "normalized_input"
    if max_abs <= max(meta.input_width, meta.input_height) * 1.5:
        return "input_pixels"
    return "original_pixels"


def decode_person_box_candidates(
    boxes: np.ndarray,
    scores: np.ndarray,
    meta: LetterboxMeta,
    *,
    box_format: str,
    coordinate_mode: str,
    iou_thr: float,
    topk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    decoded = np.asarray(boxes, dtype=np.float32).copy()
    if box_format == "cxcywh":
        decoded = xywh_to_xyxy(decoded)
    elif box_format == "xyxy_sorted":
        decoded = sort_xyxy_endpoints(decoded)
    if coordinate_mode == "normalized_input":
        decoded *= np.asarray(
            [meta.input_width, meta.input_height, meta.input_width, meta.input_height],
            dtype=np.float32,
        )
    if coordinate_mode in {"normalized_input", "input_pixels"}:
        restored = restore_letterbox_boxes(decoded, meta)
    else:
        restored = clip_boxes_to_image(decoded, meta.original_width, meta.original_height)

    widths = restored[:, 2] - restored[:, 0]
    heights = restored[:, 3] - restored[:, 1]
    valid = np.isfinite(restored).all(axis=1) & (widths > 2.0) & (heights > 2.0)
    valid_boxes = restored[valid]
    valid_scores = np.asarray(scores, dtype=np.float32)[valid]
    if len(valid_boxes) > 0:
        keep_idx = nms_numpy(valid_boxes, valid_scores, iou_thr=iou_thr)
        valid_boxes = valid_boxes[keep_idx]
        valid_scores = valid_scores[keep_idx]
        order = valid_scores.argsort()[::-1][: int(topk)]
        valid_boxes = valid_boxes[order]
        valid_scores = valid_scores[order]
        dets = np.concatenate([valid_boxes, valid_scores[:, None]], axis=1).astype(np.float32)
    else:
        dets = np.zeros((0, 5), dtype=np.float32)

    sample_count = min(5, len(restored))
    attempt = {
        "mode": f"{box_format}:{coordinate_mode}",
        "candidate_count": int(len(boxes)),
        "valid_before_nms": int(np.count_nonzero(valid)),
        "keep_after_nms": int(len(dets)),
        "raw_box_range": range_or_none(np.asarray(boxes, dtype=np.float32).reshape(-1).tolist()),
        "restored_samples": [
            {
                "box": [round(float(value), 2) for value in restored[index].tolist()],
                "width": round(float(widths[index]), 2),
                "height": round(float(heights[index]), 2),
                "valid": bool(valid[index]),
            }
            for index in range(sample_count)
        ],
    }
    return dets, attempt


def clip_boxes_to_image(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(max(width - 1, 0)))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(max(height - 1, 0)))
    return out


def detector_score_diagnostics(scores: np.ndarray, person_scores: np.ndarray, score_mode: str) -> dict[str, Any]:
    person = np.asarray(person_scores, dtype=np.float32)
    raw = np.asarray(scores, dtype=np.float32)
    class_max = np.nanmax(raw, axis=0) if raw.ndim == 2 and raw.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    if score_mode == "sigmoid" or (score_mode == "auto" and class_max.size and (np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0)):
        class_max = sigmoid(class_max)
    class_max = np.nan_to_num(class_max, nan=0.0, posinf=0.0, neginf=0.0)
    best_class_id = int(np.argmax(class_max)) if class_max.size else None
    top = np.sort(person)[-5:][::-1] if person.size else np.zeros((0,), dtype=np.float32)
    return {
        "person_score_range": [round(float(np.min(person)), 6), round(float(np.max(person)), 6)] if person.size else None,
        "person_score_mean": round(float(np.mean(person)), 6) if person.size else None,
        "person_top_scores": [round(float(value), 6) for value in top.tolist()],
        "best_class_id": best_class_id,
        "best_class_score": round(float(class_max[best_class_id]), 6) if best_class_id is not None else None,
    }


def pick_yolov5_outputs(outputs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(output) for output in outputs if output is not None]
    boxes = None
    scores = None
    for arr in arrays:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 2:
            continue
        if squeezed.shape[-1] == 4 or squeezed.shape[0] == 4:
            candidate = squeezed if squeezed.shape[-1] == 4 else squeezed.T
            if boxes is None or candidate.shape[0] > boxes.shape[0]:
                boxes = candidate
        if 80 in squeezed.shape:
            candidate = squeezed
            if candidate.shape[0] == 80:
                candidate = candidate.T
            if scores is None or candidate.shape[0] > scores.shape[0]:
                scores = candidate
    if boxes is None or scores is None:
        shapes = [tuple(np.asarray(output).shape) for output in outputs if output is not None]
        raise RuntimeError(f"Cannot find YOLOv5 boxes/scores outputs from shapes={shapes}")
    return boxes, scores


def normalize_det_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if mode == "sigmoid" or (mode == "auto" and (np.nanmin(values) < 0.0 or np.nanmax(values) > 1.0)):
        return sigmoid(values)
    return values


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float32).copy()
    # Copy the columns: views would be mutated by the assignments below and
    # corrupt x2/y2 (for example x2 would use the already changed x1).
    cx = out[:, 0].copy()
    cy = out[:, 1].copy()
    w = out[:, 2].copy()
    h = out[:, 3].copy()
    out[:, 0] = cx - w * 0.5
    out[:, 1] = cy - h * 0.5
    out[:, 2] = cx + w * 0.5
    out[:, 3] = cy + h * 0.5
    return out


def sort_xyxy_endpoints(boxes: np.ndarray) -> np.ndarray:
    """Recover boxes whose two xyxy endpoints were emitted in reverse order."""
    out = np.asarray(boxes, dtype=np.float32).copy()
    x1 = np.minimum(out[:, 0], out[:, 2])
    y1 = np.minimum(out[:, 1], out[:, 3])
    x2 = np.maximum(out[:, 0], out[:, 2])
    y2 = np.maximum(out[:, 1], out[:, 3])
    out[:, 0] = x1
    out[:, 1] = y1
    out[:, 2] = x2
    out[:, 3] = y2
    return out


def restore_letterbox_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - meta.pad_x) / max(float(meta.scale), 1e-6)
    out[:, [1, 3]] = (out[:, [1, 3]] - meta.pad_y) / max(float(meta.scale), 1e-6)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, float(meta.original_width - 1))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, float(meta.original_height - 1))
    return out


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        remain = np.where(inter / union <= iou_thr)[0]
        order = order[remain + 1]
    return np.asarray(keep, dtype=np.int64)


def select_person_dets(dets: np.ndarray, frame_shape: tuple[int, ...], *, max_persons: int, mode: str) -> np.ndarray:
    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    frame_h, frame_w = frame_shape[:2]
    boxes = dets[:, :4]
    scores = dets[:, 4]
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
    frame_center = np.asarray([frame_w * 0.5, frame_h * 0.5], dtype=np.float32)
    dist = np.linalg.norm((centers - frame_center[None, :]) / np.asarray([frame_w, frame_h], dtype=np.float32), axis=1)
    if mode == "score":
        rank = scores
    elif mode == "largest":
        rank = areas
    elif mode == "center":
        rank = -dist
    else:
        rank = areas / max(float(frame_w * frame_h), 1.0) + scores * 0.25 - dist * 0.15
    order = np.argsort(rank)[::-1][:max_persons]
    return dets[order].astype(np.float32)


def adaptive_detector_decision(
    *,
    has_tracker: bool,
    detector_age_seconds: float | None,
    refresh_seconds: float,
    retry_seconds: float,
    bad_pose_frames: int,
    bad_pose_limit: int,
) -> tuple[bool, str]:
    if not has_tracker:
        if detector_age_seconds is None:
            return True, "initial_or_lost"
        if detector_age_seconds >= retry_seconds:
            return True, "lost_retry"
        return False, "lost_retry_cooldown"
    if bad_pose_frames >= bad_pose_limit:
        if detector_age_seconds is None or detector_age_seconds >= retry_seconds:
            return True, "pose_quality_drop"
        return False, "pose_quality_retry_cooldown"
    if detector_age_seconds is None or detector_age_seconds >= refresh_seconds:
        return True, "periodic_refresh"
    return False, "pose_tracker"


def tracker_bbox_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    score_threshold: float,
    margin: float,
) -> tuple[np.ndarray | None, int]:
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(confidence)
        & (confidence >= float(score_threshold))
        & (points[:, 0] >= 0.0)
        & (points[:, 1] >= 0.0)
    )
    visible_count = int(np.count_nonzero(valid))
    if visible_count < 3:
        return None, visible_count
    visible = points[valid]
    x1, y1 = np.min(visible, axis=0)
    x2, y2 = np.max(visible, axis=0)
    width = max(float(x2 - x1), 2.0)
    height = max(float(y2 - y1), 2.0)
    expand_x = width * float(margin)
    expand_y = height * float(margin)
    frame_h, frame_w = frame_shape[:2]
    bbox = np.asarray(
        [
            max(0.0, float(x1) - expand_x),
            max(0.0, float(y1) - expand_y),
            min(float(frame_w - 1), float(x2) + expand_x),
            min(float(frame_h - 1), float(y2) + expand_y),
        ],
        dtype=np.float32,
    )
    if bbox[2] - bbox[0] <= 2.0 or bbox[3] - bbox[1] <= 2.0:
        return None, visible_count
    return bbox, visible_count


def blend_detection_bbox(detection: np.ndarray, tracker_bbox: np.ndarray, *, alpha: float) -> np.ndarray:
    output = np.asarray(detection, dtype=np.float32).copy()
    blend = min(1.0, max(0.0, float(alpha)))
    output[:4] = output[:4] * (1.0 - blend) + np.asarray(tracker_bbox, dtype=np.float32) * blend
    return output


def bbox_padding_ratio(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in np.asarray(bbox).reshape(4)]
    total_area = max((x2 - x1) * (y2 - y1), 1.0)
    clipped_x1 = max(0.0, min(float(frame_w), x1))
    clipped_y1 = max(0.0, min(float(frame_h), y1))
    clipped_x2 = max(0.0, min(float(frame_w), x2))
    clipped_y2 = max(0.0, min(float(frame_h), y2))
    visible_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    return min(1.0, max(0.0, 1.0 - visible_area / total_area))


def bound_pose_bbox_for_frame(
    expanded_bbox: np.ndarray,
    detection_bbox: np.ndarray,
    frame_shape: tuple[int, ...],
    input_size: tuple[int, int],
    *,
    max_padding_ratio: float,
) -> tuple[np.ndarray, bool, float]:
    expanded_crop = aspect_bbox(np.asarray(expanded_bbox, dtype=np.float32), frame_shape, input_size)
    expanded_padding = bbox_padding_ratio(expanded_crop, frame_shape)
    frame_h, frame_w = frame_shape[:2]
    crop_area = max(float(expanded_crop[2] - expanded_crop[0]), 1.0) * max(float(expanded_crop[3] - expanded_crop[1]), 1.0)
    if expanded_padding <= float(max_padding_ratio) and crop_area <= float(frame_w * frame_h) * 2.0:
        return np.asarray(expanded_bbox, dtype=np.float32), False, expanded_padding

    raw = np.asarray(detection_bbox, dtype=np.float32)
    cx = float(raw[0] + raw[2]) * 0.5
    cy = float(raw[1] + raw[3]) * 0.5
    raw_h = max(float(raw[3] - raw[1]), float(frame_h) * 0.35)
    target_aspect = float(input_size[0]) / max(float(input_size[1]), 1.0)
    safe_h = min(float(frame_h) * 1.10, raw_h * 1.10)
    safe_w = min(float(frame_w) * 1.10, safe_h * target_aspect)
    safe = np.asarray([cx - safe_w * 0.5, cy - safe_h * 0.5, cx + safe_w * 0.5, cy + safe_h * 0.5], dtype=np.float32)
    safe_crop = aspect_bbox(safe, frame_shape, input_size)
    return safe, True, bbox_padding_ratio(safe_crop, frame_shape)


def adaptive_pose_x_scale(
    bbox: np.ndarray,
    *,
    base_scale: float,
    wide_ratio: float,
    wide_scale: float,
) -> float:
    """Increase horizontal context smoothly when the detected person is wide."""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    base_scale = max(1.0, float(base_scale))
    wide_scale = max(base_scale, float(wide_scale))
    wide_ratio = max(0.1, float(wide_ratio))
    blend_width = max(wide_ratio * 0.5, 0.1)
    blend = float(np.clip((w / h - wide_ratio) / blend_width, 0.0, 1.0))
    return base_scale + blend * (wide_scale - base_scale)


def expand_bbox_for_pose(
    bbox: np.ndarray,
    frame_shape: tuple[int, ...],
    *,
    scale: float,
    top_expand: float,
    wide_ratio: float = 0.65,
    wide_scale: float | None = None,
) -> np.ndarray:
    del frame_shape  # padded_crop() intentionally supports coordinates outside the image.
    x1, y1, x2, y2 = [float(value) for value in bbox]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x_scale = adaptive_pose_x_scale(
        bbox,
        base_scale=scale,
        wide_ratio=wide_ratio,
        wide_scale=scale if wide_scale is None else wide_scale,
    )
    new_w = w * x_scale
    new_h = h * float(scale)
    out = np.asarray(
        [
            cx - new_w * 0.5,
            cy - new_h * 0.5 - h * float(top_expand),
            cx + new_w * 0.5,
            cy + new_h * 0.5,
        ],
        dtype=np.float32,
    )
    return out


def preprocess_rtmpose(frame: np.ndarray, bbox: np.ndarray, input_size: tuple[int, int], input_layout: str) -> tuple[np.ndarray, RestoreInfo, np.ndarray]:
    crop_bbox = aspect_bbox(bbox, frame.shape, input_size)
    x1, y1, x2, y2 = [int(round(value)) for value in crop_bbox]
    restore = RestoreInfo(np.asarray([x1, y1, x2, y2], dtype=np.float32), input_size)
    crop = padded_crop(frame, x1, y1, x2, y2)
    crop = cv2.resize(crop, input_size, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    image = (image - mean) / std
    blob = image[None, ...] if input_layout == "nhwc" else image.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(blob.astype(np.float32)), restore, crop


def aspect_bbox(bbox: np.ndarray, frame_shape: tuple[int, ...], input_size: tuple[int, int]) -> np.ndarray:
    del frame_shape  # padded_crop() supplies black padding outside the camera image.
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if not np.isfinite([x1, y1, x2, y2]).all():
        raise ValueError(f"RTMPose bbox contains non-finite coordinates: {bbox}")
    x2 = max(x1 + 1.0, x2)
    y2 = max(y1 + 1.0, y2)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    target_ratio = float(input_size[0]) / max(float(input_size[1]), 1.0)
    if w / h > target_ratio:
        h = w / target_ratio
    else:
        w = h * target_ratio
    return np.asarray([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dtype=np.float32)


def padded_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    crop_w = max(1, int(x2 - x1))
    crop_h = max(1, int(y2 - y1))
    crop = np.zeros((crop_h, crop_w, frame.shape[2]), dtype=frame.dtype)
    src_x1 = max(0, min(frame_w, x1))
    src_y1 = max(0, min(frame_h, y1))
    src_x2 = max(0, min(frame_w, x2))
    src_y2 = max(0, min(frame_h, y2))
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return crop
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    crop[dst_y1 : dst_y1 + (src_y2 - src_y1), dst_x1 : dst_x1 + (src_x2 - src_x1)] = frame[src_y1:src_y2, src_x1:src_x2]
    return crop


def decode_simcc(
    outputs: list[Any],
    restore: RestoreInfo,
    *,
    input_size: tuple[int, int],
    split_ratio: float,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    simcc_x, simcc_y, meta = normalize_simcc_outputs(outputs, input_size, split_ratio)
    x_locs = np.argmax(simcc_x, axis=1).astype(np.float32) / float(split_ratio)
    y_locs = np.argmax(simcc_y, axis=1).astype(np.float32) / float(split_ratio)
    max_x = np.max(simcc_x, axis=1)
    max_y = np.max(simcc_y, axis=1)
    scores = (max_x + max_y) * 0.5 if score_mode == "avg" else np.sqrt(np.maximum(max_x, 0) * np.maximum(max_y, 0))
    crop_points = np.stack([x_locs, y_locs], axis=1).astype(np.float32)
    crop_points[scores <= 0] = -1
    keypoints = restore.to_image(crop_points)
    keypoints[scores <= 0] = -1
    meta.update({"restore_bbox": [round(float(value), 4) for value in restore.image_bbox.tolist()]})
    meta.update(
        {
            "simcc_x_shape": list(simcc_x.shape),
            "simcc_y_shape": list(simcc_y.shape),
            "split_ratio": float(split_ratio),
            "restore_mode": "aspect_ratio_padded_roi_resize",
        }
    )
    return keypoints.astype(np.float32), scores.astype(np.float32), crop_points.astype(np.float32), meta


def normalize_simcc_outputs(outputs: list[Any], input_size: tuple[int, int], split_ratio: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = []
    for idx, output in enumerate(outputs):
        arr = np.squeeze(np.asarray(output, dtype=np.float32))
        if arr.ndim != 2:
            continue
        if arr.shape[0] == 17 and arr.shape[1] > 17:
            candidates.append((idx, arr.shape[1], arr))
        elif arr.shape[1] == 17 and arr.shape[0] > 17:
            candidates.append((idx, arr.shape[0], arr.T))
    expected_x = int(round(input_size[0] * split_ratio))
    expected_y = int(round(input_size[1] * split_ratio))
    simcc_x = next((arr for _, length, arr in candidates if length == expected_x), None)
    simcc_y = next((arr for _, length, arr in candidates if length == expected_y), None)
    if simcc_x is None or simcc_y is None:
        shapes = [tuple(np.asarray(output).shape) for output in outputs if output is not None]
        raise RuntimeError(f"Cannot find SimCC outputs for input={input_size}, shapes={shapes}")
    return simcc_x, simcc_y, {"expected_x": expected_x, "expected_y": expected_y}


def draw_poses(frame_bgr: np.ndarray, detections: list[dict[str, Any]], *, keypoint_thres: float) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        bbox = np.asarray(detection.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
        x1, y1, x2, y2 = np.round(bbox).astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 255), 2)
        cv2.putText(output, f"person {float(detection.get('score', 0.0)):.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 255), 2, cv2.LINE_AA)
        pose_bbox = detection.get("pose_bbox")
        if isinstance(pose_bbox, list) and len(pose_bbox) == 4:
            px1, py1, px2, py2 = np.round(np.asarray(pose_bbox, dtype=np.float32)).astype(int)
            cv2.rectangle(output, (px1, py1), (px2, py2), (120, 245, 255), 1)
        keypoints = detection.get("keypoints") or []
        visible = [len(point) >= 3 and float(point[2]) >= keypoint_thres for point in keypoints]
        for idx, (start, end) in enumerate(COCO17_EDGES):
            if start >= len(keypoints) or end >= len(keypoints) or not (visible[start] and visible[end]):
                continue
            cv2.line(output, tuple(np.round(keypoints[start][:2]).astype(int)), tuple(np.round(keypoints[end][:2]).astype(int)), LINK_COLORS[idx], 2, cv2.LINE_AA)
        for idx, point in enumerate(keypoints):
            if idx < len(visible) and visible[idx]:
                cv2.circle(output, tuple(np.round(point[:2]).astype(int)), 4, KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
                cv2.circle(output, tuple(np.round(point[:2]).astype(int)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def save_debug_crop(crop: np.ndarray, crop_points: np.ndarray, scores: np.ndarray, path: str, keypoint_thres: float) -> None:
    try:
        detections = [{"bbox": [0, 0, crop.shape[1] - 1, crop.shape[0] - 1], "score": 1.0, "keypoints": [[float(x), float(y), normalize_score(float(s))] for (x, y), s in zip(crop_points, scores)]}]
        debug = draw_poses(crop, detections, keypoint_thres=keypoint_thres)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), debug)
    except Exception:
        return


def output_shapes_of(outputs: list[Any]) -> list[tuple[int, ...]]:
    return [tuple(np.asarray(output).shape) for output in outputs if output is not None]


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def normalize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return float(value)
    return float(1.0 / (1.0 + np.exp(-value)))


def range_or_none(values: list[float]) -> list[float] | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return None
    return [round(min(clean), 4), round(max(clean), 4)]


def xy_range(points: np.ndarray) -> dict[str, list[float] | None]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 2:
        return {"x": None, "y": None}
    return {"x": range_or_none(arr[:, 0].tolist()), "y": range_or_none(arr[:, 1].tolist())}


def letterbox_meta_dict(meta: LetterboxMeta) -> dict[str, float | int]:
    return {
        "input_width": meta.input_width,
        "input_height": meta.input_height,
        "original_width": meta.original_width,
        "original_height": meta.original_height,
        "scale": round(float(meta.scale), 6),
        "pad_x": round(float(meta.pad_x), 2),
        "pad_y": round(float(meta.pad_y), 2),
    }


def _choice(value: str | None, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
