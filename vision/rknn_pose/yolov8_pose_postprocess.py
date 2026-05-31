"""YOLOv8-Pose preprocessing, postprocessing, and drawing helpers.

The public RKNN Model Zoo YOLOv8-Pose examples use letterbox preprocessing,
confidence filtering, NMS, and COCO17 keypoints. This module keeps those
semantics local to the RKNN branch and returns keypoints restored to the
original frame pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


COCO17_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
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


def letterbox(frame_bgr: np.ndarray, input_size: int | tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    if isinstance(input_size, int):
        input_width = input_height = int(input_size)
    else:
        input_width, input_height = int(input_size[0]), int(input_size[1])
    original_height, original_width = frame_bgr.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_x = (input_width - resized_width) / 2.0
    pad_y = (input_height - resized_height) / 2.0

    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, LetterboxMeta(
        input_width=input_width,
        input_height=input_height,
        original_width=original_width,
        original_height=original_height,
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )


def prepare_input(frame_bgr: np.ndarray, input_size: int | tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    boxed, meta = letterbox(frame_bgr, input_size)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0), meta


def postprocess_yolov8_pose(
    outputs: list[Any] | tuple[Any, ...],
    meta: LetterboxMeta,
    *,
    conf_thres: float = 0.25,
    nms_thres: float = 0.45,
    keypoint_thres: float = 0.30,
    top_k: int = 100,
    max_det: int = 5,
) -> list[dict[str, Any]]:
    predictions = _coerce_predictions(outputs)
    if predictions.size == 0:
        return []
    if predictions.shape[1] < 56:
        return []

    scores = predictions[:, 4].astype(np.float32, copy=False)
    keep = np.flatnonzero(scores >= conf_thres)
    if keep.size == 0:
        return []
    if top_k > 0 and keep.size > top_k:
        top_order = np.argpartition(scores[keep], -top_k)[-top_k:]
        keep = keep[top_order]
    keep = keep[np.argsort(scores[keep])[::-1]]

    detections: list[dict[str, Any]] = []
    for row in predictions[keep]:
        score = float(row[4])
        x_center, y_center, width, height = [float(value) for value in row[:4]]
        box = _restore_box_xywh(x_center, y_center, width, height, meta)
        keypoints = []
        keypoint_values = row[5:56]
        for index in range(17):
            offset = index * 3
            x_value = float(keypoint_values[offset])
            y_value = float(keypoint_values[offset + 1])
            confidence = float(keypoint_values[offset + 2])
            x_pixel, y_pixel = restore_point(x_value, y_value, meta)
            keypoints.append([x_pixel, y_pixel, confidence if confidence >= keypoint_thres else 0.0])
        detections.append({"bbox": box, "score": score, "keypoints": keypoints})

    keep = nms(detections, nms_thres)
    if max_det > 0:
        keep = keep[:max_det]
    return [detections[index] for index in keep]


def restore_point(x_value: float, y_value: float, meta: LetterboxMeta) -> tuple[float, float]:
    x_pixel = (x_value - meta.pad_x) / meta.scale
    y_pixel = (y_value - meta.pad_y) / meta.scale
    return (
        float(np.clip(x_pixel, 0.0, max(meta.original_width - 1, 0))),
        float(np.clip(y_pixel, 0.0, max(meta.original_height - 1, 0))),
    )


def draw_poses(frame_bgr: np.ndarray, detections: list[dict[str, Any]], *, keypoint_thres: float = 0.30) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.get("bbox", [0, 0, 0, 0])]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(output, f"{float(detection.get('score', 0.0)):.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        keypoints = detection.get("keypoints") or []
        for start, end in COCO17_EDGES:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            start_point = keypoints[start]
            end_point = keypoints[end]
            if float(start_point[2]) < keypoint_thres or float(end_point[2]) < keypoint_thres:
                continue
            cv2.line(output, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (40, 220, 120), 2)
        for point in keypoints:
            if float(point[2]) >= keypoint_thres:
                cv2.circle(output, (int(point[0]), int(point[1])), 3, (80, 180, 255), -1)
    return output


def nms(detections: list[dict[str, Any]], iou_threshold: float) -> list[int]:
    if not detections:
        return []
    boxes = np.asarray([item["bbox"] for item in detections], dtype=np.float32)
    scores = np.asarray([item["score"] for item in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        overlaps = _iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return keep


def _coerce_predictions(outputs: list[Any] | tuple[Any, ...]) -> np.ndarray:
    arrays = [np.asarray(output) for output in outputs if output is not None]
    if not arrays:
        return np.empty((0, 56), dtype=np.float32)

    candidates: list[np.ndarray] = []
    for array in arrays:
        squeezed = np.squeeze(array)
        if squeezed.ndim == 1 and squeezed.shape[0] >= 56:
            candidates.append(squeezed.reshape(1, -1))
        elif squeezed.ndim == 2:
            candidate = _normalize_prediction_matrix(squeezed)
            if candidate is not None:
                candidates.append(candidate)
        elif squeezed.ndim == 3:
            for batch_index in range(squeezed.shape[0]):
                candidate = _normalize_prediction_matrix(squeezed[batch_index])
                if candidate is not None:
                    candidates.append(candidate)

    valid = [
        candidate.astype(np.float32)
        for candidate in candidates
        if candidate.ndim == 2 and 56 <= candidate.shape[1] <= 128
    ]
    if not valid:
        shapes = ", ".join(str(array.shape) for array in arrays)
        raise ValueError(f"unsupported YOLOv8-Pose output shape(s): {shapes}")
    feature_dims = {candidate.shape[1] for candidate in valid}
    if len(feature_dims) > 1:
        target_dim = min(feature_dims)
        valid = [candidate[:, :target_dim] for candidate in valid]
    return np.concatenate(valid, axis=0)


def output_shapes(outputs: list[Any] | tuple[Any, ...]) -> list[tuple[int, ...]]:
    return [tuple(np.asarray(output).shape) for output in outputs if output is not None]


def _normalize_prediction_matrix(matrix: np.ndarray) -> np.ndarray | None:
    if matrix.ndim != 2:
        return None
    rows, cols = matrix.shape
    min_feature_dim = 56
    max_feature_dim = 128
    if min_feature_dim <= cols <= max_feature_dim:
        return matrix
    if min_feature_dim <= rows <= max_feature_dim:
        return matrix.T
    return None


def _restore_box_xywh(x_center: float, y_center: float, width: float, height: float, meta: LetterboxMeta) -> list[float]:
    x1, y1 = restore_point(x_center - width / 2.0, y_center - height / 2.0, meta)
    x2, y2 = restore_point(x_center + width / 2.0, y_center + height / 2.0, meta)
    return [x1, y1, x2, y2]


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_box + area_boxes - intersection, 1e-9)
