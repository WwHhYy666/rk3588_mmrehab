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
    keypoint_decode_mode: str = "auto",
    keypoint_anchor_order: str = "auto",
) -> list[dict[str, Any]]:
    detections, _ = postprocess_yolov8_pose_with_diagnostics(
        outputs,
        meta,
        conf_thres=conf_thres,
        nms_thres=nms_thres,
        keypoint_thres=keypoint_thres,
        top_k=top_k,
        max_det=max_det,
        keypoint_decode_mode=keypoint_decode_mode,
        keypoint_anchor_order=keypoint_anchor_order,
    )
    return detections


def postprocess_yolov8_pose_with_diagnostics(
    outputs: list[Any] | tuple[Any, ...],
    meta: LetterboxMeta,
    *,
    conf_thres: float = 0.25,
    nms_thres: float = 0.45,
    keypoint_thres: float = 0.30,
    top_k: int = 100,
    max_det: int = 5,
    keypoint_decode_mode: str = "auto",
    keypoint_anchor_order: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "output_shapes": output_shapes(outputs),
        "keypoint_raw_shape": tuple(np.asarray(outputs[3]).shape) if len(outputs) > 3 and outputs[3] is not None else None,
        "keypoint_decode_override": keypoint_decode_mode if keypoint_decode_mode in {"absolute", "anchor_stride"} else None,
        "keypoint_anchor_order_setting": keypoint_anchor_order,
    }
    arrays = [np.asarray(output) for output in outputs if output is not None]
    if _looks_like_model_zoo_split(arrays):
        diagnostics["rknn_decoder"] = "model_zoo_split"
        detections = _postprocess_model_zoo_split(
            arrays,
            meta,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            top_k=top_k,
            max_det=max_det,
            keypoint_decode_mode=keypoint_decode_mode,
            keypoint_anchor_order=keypoint_anchor_order,
        )
    else:
        diagnostics["rknn_decoder"] = "decoded_flat"
        detections = _postprocess_decoded_flat(
            outputs,
            meta,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            top_k=top_k,
            max_det=max_det,
        )

    diagnostics.update(_detection_ranges(detections))
    diagnostics.update(_detection_keypoint_decode_diagnostics(detections))
    sanity_error = _sanity_check_detections(detections, meta)
    if sanity_error:
        diagnostics["postprocess_error"] = sanity_error
        return [], diagnostics
    diagnostics["postprocess_error"] = None
    return detections, diagnostics


def _postprocess_decoded_flat(
    outputs: list[Any] | tuple[Any, ...],
    meta: LetterboxMeta,
    *,
    conf_thres: float,
    nms_thres: float,
    top_k: int,
    max_det: int,
) -> list[dict[str, Any]]:
    predictions = _coerce_predictions(outputs)
    if predictions.size == 0:
        return []
    if predictions.shape[1] < 56:
        return []

    scores = np.asarray([_normalize_confidence(value) for value in predictions[:, 4]], dtype=np.float32)
    keep = np.flatnonzero(scores >= conf_thres)
    if keep.size == 0:
        return []
    if top_k > 0 and keep.size > top_k:
        top_order = np.argpartition(scores[keep], -top_k)[-top_k:]
        keep = keep[top_order]
    keep = keep[np.argsort(scores[keep])[::-1]]

    detections: list[dict[str, Any]] = []
    for row in predictions[keep]:
        score = _normalize_confidence(float(row[4]))
        x_center, y_center, width, height = [float(value) for value in row[:4]]
        box = _restore_box_xywh(x_center, y_center, width, height, meta)
        keypoints = []
        keypoint_values = row[5:56]
        for index in range(17):
            offset = index * 3
            x_value = float(keypoint_values[offset])
            y_value = float(keypoint_values[offset + 1])
            confidence = _normalize_confidence(float(keypoint_values[offset + 2]))
            x_pixel, y_pixel = restore_point(x_value, y_value, meta)
            keypoints.append([x_pixel, y_pixel, confidence])
        detections.append({"bbox": box, "score": score, "keypoints": keypoints})

    keep = nms(detections, nms_thres)
    if max_det > 0:
        keep = keep[:max_det]
    return [detections[index] for index in keep]


def _postprocess_model_zoo_split(
    arrays: list[np.ndarray],
    meta: LetterboxMeta,
    *,
    conf_thres: float,
    nms_thres: float,
    top_k: int,
    max_det: int,
    keypoint_decode_mode: str,
    keypoint_anchor_order: str,
) -> list[dict[str, Any]]:
    branches = [_feature_to_chw(array) for array in arrays[:3]]
    if any(branch is None for branch in branches):
        shapes = ", ".join(str(array.shape) for array in arrays)
        raise ValueError(f"unsupported RKNN Model Zoo split output shape(s): {shapes}")
    typed_items = [(index, branch) for index, branch in enumerate(branches) if branch is not None]
    typed_branches = [branch for _, branch in typed_items]
    keypoint_matrix = _keypoints_to_51n(arrays[3])
    if keypoint_matrix is None:
        raise ValueError(f"unsupported RKNN keypoint output shape: {arrays[3].shape}")
    anchor_order_candidates = _anchor_offset_candidates(typed_items, keypoint_anchor_order)

    detections: list[dict[str, Any]] = []
    for branch_id, branch in typed_items:
        _, grid_h, grid_w = branch.shape
        stride_x = meta.input_width / float(grid_w)
        stride_y = meta.input_height / float(grid_h)
        matrix = branch.reshape(branch.shape[0], -1)
        distances = _decode_dfl(matrix[:64, :])
        scores = np.asarray([_normalize_confidence(value) for value in matrix[64, :]], dtype=np.float32)
        keep = np.flatnonzero(scores >= conf_thres)
        if keep.size == 0:
            continue
        if top_k > 0 and keep.size > top_k:
            top_order = np.argpartition(scores[keep], -top_k)[-top_k:]
            keep = keep[top_order]
        keep = keep[np.argsort(scores[keep])[::-1]]

        for anchor_index in keep:
            anchor_index = int(anchor_index)
            row = anchor_index // grid_w
            col = anchor_index % grid_w
            left, top, right, bottom = [float(value) for value in distances[:, anchor_index]]
            x1 = (col + 0.5 - left) * stride_x
            y1 = (row + 0.5 - top) * stride_y
            x2 = (col + 0.5 + right) * stride_x
            y2 = (row + 0.5 + bottom) * stride_y
            x1_pixel, y1_pixel = restore_point(x1, y1, meta)
            x2_pixel, y2_pixel = restore_point(x2, y2, meta)
            bbox = [
                min(x1_pixel, x2_pixel),
                min(y1_pixel, y2_pixel),
                max(x1_pixel, x2_pixel),
                max(y1_pixel, y2_pixel),
            ]
            candidate = _select_keypoint_candidate(
                keypoint_matrix,
                branch_id=branch_id,
                anchor_index=anchor_index,
                col=col,
                row=row,
                stride_x=stride_x,
                stride_y=stride_y,
                meta=meta,
                bbox=bbox,
                keypoint_decode_mode=keypoint_decode_mode,
                anchor_order_candidates=anchor_order_candidates,
            )
            if candidate is None:
                continue
            detections.append(
                {
                    "bbox": bbox,
                    "score": float(scores[anchor_index]),
                    "keypoints": candidate["keypoints"],
                    "keypoint_decode_mode": candidate["decode_mode"],
                    "keypoint_anchor_order": candidate["anchor_order"],
                    "keypoint_global_index": candidate["global_index"],
                    "keypoint_geometry_score": candidate["geometry_score"],
                    "keypoint_candidate_count": candidate["candidate_count"],
                    "keypoint_branch_diagnostics": _branch_diagnostics(typed_items, candidate["anchor_order"], anchor_order_candidates),
                    "keypoint_raw_xy_range": _raw_keypoint_xy_range(candidate["raw_keypoints"]),
                }
            )

    keep_indices = nms(detections, nms_thres)
    if max_det > 0:
        keep_indices = keep_indices[:max_det]
    return [detections[index] for index in keep_indices]


def restore_point(x_value: float, y_value: float, meta: LetterboxMeta) -> tuple[float, float]:
    x_pixel = (x_value - meta.pad_x) / meta.scale
    y_pixel = (y_value - meta.pad_y) / meta.scale
    return (
        float(np.clip(x_pixel, 0.0, max(meta.original_width - 1, 0))),
        float(np.clip(y_pixel, 0.0, max(meta.original_height - 1, 0))),
    )


def _anchor_offset_candidates(
    typed_items: list[tuple[int, np.ndarray]],
    keypoint_anchor_order: str,
) -> list[dict[str, Any]]:
    forced = str(keypoint_anchor_order or "auto").strip().lower()
    orders: list[tuple[str, list[tuple[int, np.ndarray]]]] = []
    if forced == "output":
        orders = [("output", typed_items)]
    elif forced == "grid_asc":
        orders = [("grid_asc", sorted(typed_items, key=lambda item: item[1].shape[1] * item[1].shape[2]))]
    elif forced == "output_reverse":
        orders = [("output_reverse", list(reversed(typed_items)))]
    elif forced == "grid_desc":
        orders = [("grid_desc", sorted(typed_items, key=lambda item: item[1].shape[1] * item[1].shape[2], reverse=True))]
    else:
        orders = [
            ("grid_desc", sorted(typed_items, key=lambda item: item[1].shape[1] * item[1].shape[2], reverse=True)),
            ("output", typed_items),
            ("grid_asc", sorted(typed_items, key=lambda item: item[1].shape[1] * item[1].shape[2])),
            ("output_reverse", list(reversed(typed_items))),
        ]

    candidates: list[dict[str, Any]] = []
    seen = set()
    for name, ordered_items in orders:
        offsets: dict[int, int] = {}
        offset = 0
        signature = []
        for branch_id, branch in ordered_items:
            offsets[branch_id] = offset
            _, grid_h, grid_w = branch.shape
            signature.append((branch_id, grid_h, grid_w, offset))
            offset += grid_h * grid_w
        key = tuple(signature)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"name": name, "offsets": offsets, "signature": signature})
    return candidates


def _select_keypoint_candidate(
    keypoint_matrix: np.ndarray,
    *,
    branch_id: int,
    anchor_index: int,
    col: int,
    row: int,
    stride_x: float,
    stride_y: float,
    meta: LetterboxMeta,
    bbox: list[float],
    keypoint_decode_mode: str,
    anchor_order_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    decode_setting = str(keypoint_decode_mode or "auto").strip().lower()
    forced_decode = [decode_setting] if decode_setting in {"absolute", "anchor_stride"} else []
    best: dict[str, Any] | None = None
    candidate_count = 0
    for order in anchor_order_candidates:
        offsets = order.get("offsets") or {}
        if branch_id not in offsets:
            continue
        global_index = int(offsets[branch_id]) + int(anchor_index)
        if global_index < 0 or global_index >= keypoint_matrix.shape[1]:
            continue
        raw_keypoints = keypoint_matrix[:, global_index].reshape(17, 3)
        decode_modes = forced_decode or _candidate_decode_modes(raw_keypoints, meta)
        for decode_mode in decode_modes:
            keypoints = _decode_keypoints_for_anchor(
                raw_keypoints,
                decode_mode=decode_mode,
                col=col,
                row=row,
                stride_x=stride_x,
                stride_y=stride_y,
                meta=meta,
            )
            geometry_score = _pose_geometry_score(keypoints, bbox, meta)
            candidate_count += 1
            candidate = {
                "keypoints": keypoints,
                "decode_mode": decode_mode,
                "anchor_order": order["name"],
                "global_index": global_index,
                "geometry_score": geometry_score,
                "raw_keypoints": raw_keypoints,
                "candidate_count": 0,
            }
            if best is None or geometry_score > float(best["geometry_score"]):
                best = candidate
    if best is not None:
        best["candidate_count"] = candidate_count
    return best


def _candidate_decode_modes(raw_keypoints: np.ndarray, meta: LetterboxMeta) -> list[str]:
    preferred = _keypoint_decode_mode(raw_keypoints, meta)
    other = "anchor_stride" if preferred == "absolute" else "absolute"
    return [preferred, other]


def _decode_keypoints_for_anchor(
    raw_keypoints: np.ndarray,
    *,
    decode_mode: str,
    col: int,
    row: int,
    stride_x: float,
    stride_y: float,
    meta: LetterboxMeta,
) -> list[list[float]]:
    keypoints = []
    for x_value, y_value, confidence in raw_keypoints:
        conf = _normalize_confidence(float(confidence))
        if float(x_value) == 0.0 and float(y_value) == 0.0:
            conf = 0.0
        if decode_mode == "anchor_stride":
            x_decoded = (float(x_value) * 2.0 + float(col) - 0.5) * stride_x
            y_decoded = (float(y_value) * 2.0 + float(row) - 0.5) * stride_y
        else:
            x_decoded = float(x_value)
            y_decoded = float(y_value)
        x_pixel, y_pixel = restore_point(x_decoded, y_decoded, meta)
        keypoints.append([x_pixel, y_pixel, conf])
    return keypoints


def _pose_geometry_score(keypoints: list[list[float]], bbox: list[float], meta: LetterboxMeta) -> float:
    required = [5, 6, 11, 12, 13, 14, 15, 16]
    required_points = [keypoints[index] for index in required if index < len(keypoints)]
    if not required_points:
        return -999.0
    conf_values = [max(0.0, min(1.0, float(point[2]))) for point in required_points]
    conf_score = sum(conf_values) / len(conf_values)
    edge_ratio = _edge_point_ratio(required_points, meta)
    bbox_inside_ratio = _bbox_inside_ratio(required_points, bbox)
    body_score = _body_plausibility_score(keypoints, meta)
    return (conf_score * 3.0) + (bbox_inside_ratio * 1.5) + body_score - (edge_ratio * 3.0)


def _body_plausibility_score(keypoints: list[list[float]], meta: LetterboxMeta) -> float:
    diag = max(1.0, float(np.hypot(meta.original_width, meta.original_height)))

    def point(index: int) -> tuple[float, float, float] | None:
        if index >= len(keypoints):
            return None
        x, y, conf = keypoints[index]
        if float(conf) < 0.05:
            return None
        return float(x), float(y), float(conf)

    pairs = [(5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    lengths = []
    score = 0.0
    for start, end in pairs:
        a = point(start)
        b = point(end)
        if a is None or b is None:
            continue
        length = float(np.hypot(a[0] - b[0], a[1] - b[1])) / diag
        lengths.append(length)
        if 0.03 <= length <= 0.65:
            score += 0.3
        else:
            score -= 0.5

    left_shoulder = point(5)
    right_shoulder = point(6)
    left_hip = point(11)
    right_hip = point(12)
    if left_shoulder and left_hip and left_hip[1] > left_shoulder[1] - meta.original_height * 0.08:
        score += 0.4
    if right_shoulder and right_hip and right_hip[1] > right_shoulder[1] - meta.original_height * 0.08:
        score += 0.4
    if lengths:
        spread = max(lengths) - min(lengths)
        if spread > 0.75:
            score -= 1.0
    return score


def _edge_point_ratio(points: list[list[float]], meta: LetterboxMeta) -> float:
    edge_count = 0
    for x_value, y_value, conf in points:
        if float(conf) < 0.05:
            continue
        x = float(x_value)
        y = float(y_value)
        if x <= 2.0 or y <= 2.0 or x >= meta.original_width - 3 or y >= meta.original_height - 3:
            edge_count += 1
    return edge_count / max(len(points), 1)


def _bbox_inside_ratio(points: list[list[float]], bbox: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    pad_x = max(8.0, (x2 - x1) * 0.15)
    pad_y = max(8.0, (y2 - y1) * 0.15)
    inside = 0
    considered = 0
    for x_value, y_value, conf in points:
        if float(conf) < 0.05:
            continue
        considered += 1
        x = float(x_value)
        y = float(y_value)
        if x1 - pad_x <= x <= x2 + pad_x and y1 - pad_y <= y <= y2 + pad_y:
            inside += 1
    return inside / max(considered, 1)


def _branch_diagnostics(
    typed_items: list[tuple[int, np.ndarray]],
    selected_order: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    branches = []
    for branch_id, branch in typed_items:
        _, grid_h, grid_w = branch.shape
        branches.append(
            {
                "branch_id": branch_id,
                "shape": tuple(int(value) for value in branch.shape),
                "grid": [int(grid_h), int(grid_w)],
                "anchors": int(grid_h * grid_w),
            }
        )
    selected = next((item for item in candidates if item.get("name") == selected_order), None)
    return {
        "selected_order": selected_order,
        "candidate_orders": [str(item.get("name")) for item in candidates],
        "selected_offsets": dict(selected.get("offsets") or {}) if selected else {},
        "branches": branches,
    }


def draw_poses(frame_bgr: np.ndarray, detections: list[dict[str, Any]], *, keypoint_thres: float = 0.30) -> np.ndarray:
    output = frame_bgr.copy()
    draw_threshold = min(float(keypoint_thres), 0.05)
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
            if float(start_point[2]) < draw_threshold or float(end_point[2]) < draw_threshold:
                continue
            cv2.line(output, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (40, 220, 120), 2)
        for point in keypoints:
            if float(point[2]) >= draw_threshold:
                cv2.circle(output, (int(point[0]), int(point[1])), 5, (80, 180, 255), -1)
                cv2.circle(output, (int(point[0]), int(point[1])), 6, (10, 20, 30), 1)
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


def _looks_like_model_zoo_split(arrays: list[np.ndarray]) -> bool:
    if len(arrays) < 4:
        return False
    branches = [_feature_to_chw(array) for array in arrays[:3]]
    if any(branch is None for branch in branches):
        return False
    keypoints = _keypoints_to_51n(arrays[3])
    if keypoints is None:
        return False
    branch_anchor_count = sum(int(branch.shape[1] * branch.shape[2]) for branch in branches if branch is not None)
    return keypoints.shape[0] == 51 and keypoints.shape[1] >= branch_anchor_count


def _feature_to_chw(array: np.ndarray) -> np.ndarray | None:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 3:
        return None
    if squeezed.shape[0] == 65:
        return squeezed.astype(np.float32, copy=False)
    if squeezed.shape[-1] == 65:
        return np.transpose(squeezed, (2, 0, 1)).astype(np.float32, copy=False)
    return None


def _keypoints_to_51n(array: np.ndarray) -> np.ndarray | None:
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        if squeezed.shape[0] == 51:
            return squeezed.astype(np.float32, copy=False)
        if squeezed.shape[1] == 51:
            return squeezed.T.astype(np.float32, copy=False)
        return None
    if squeezed.ndim == 3:
        if squeezed.shape[0] == 51:
            return squeezed.reshape(51, -1).astype(np.float32, copy=False)
        if squeezed.shape[0] == 17 and squeezed.shape[1] == 3:
            return squeezed.reshape(51, -1).astype(np.float32, copy=False)
        if squeezed.shape[1] == 17 and squeezed.shape[2] == 3:
            return np.transpose(squeezed, (1, 2, 0)).reshape(51, -1).astype(np.float32, copy=False)
        if squeezed.shape[-1] == 51:
            return np.transpose(squeezed, (2, 0, 1)).reshape(51, -1).astype(np.float32, copy=False)
    return None


def _keypoint_decode_mode(raw_keypoints: np.ndarray, meta: LetterboxMeta) -> str:
    xy = raw_keypoints[:, :2].astype(np.float32, copy=False)
    finite_xy = xy[np.isfinite(xy)]
    if finite_xy.size == 0:
        return "absolute"
    max_value = float(np.max(finite_xy))
    min_value = float(np.min(finite_xy))
    if -4.0 <= min_value and max_value <= 4.0:
        return "anchor_stride"
    if max_value <= max(meta.input_width, meta.input_height) * 1.5:
        return "absolute"
    return "anchor_stride"


def _raw_keypoint_xy_range(raw_keypoints: np.ndarray) -> dict[str, list[float] | None]:
    return {
        "x": _range_or_none([float(value) for value in raw_keypoints[:, 0]]),
        "y": _range_or_none([float(value) for value in raw_keypoints[:, 1]]),
    }


def _decode_dfl(position: np.ndarray) -> np.ndarray:
    distribution = position.reshape(4, 16, -1)
    distribution = _softmax(distribution, axis=1)
    bins = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    return np.sum(distribution * bins, axis=1)


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _normalize_confidence(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return float(value)
    return float(1.0 / (1.0 + np.exp(-value)))


def _detection_ranges(detections: list[dict[str, Any]]) -> dict[str, Any]:
    confidences: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for detection in detections:
        for point in detection.get("keypoints") or []:
            if len(point) < 3:
                continue
            xs.append(float(point[0]))
            ys.append(float(point[1]))
            confidences.append(float(point[2]))
    return {
        "keypoint_conf_range": _range_or_none(confidences),
        "keypoint_xy_range": {
            "x": _range_or_none(xs),
            "y": _range_or_none(ys),
        },
        "keypoint_restored_xy_range": {
            "x": _range_or_none(xs),
            "y": _range_or_none(ys),
        },
    }


def _detection_keypoint_decode_diagnostics(detections: list[dict[str, Any]]) -> dict[str, Any]:
    modes = []
    anchor_orders = []
    geometry_scores = []
    global_indices = []
    candidate_counts = []
    branch_diagnostics = None
    raw_xs: list[float] = []
    raw_ys: list[float] = []
    for detection in detections:
        mode = detection.get("keypoint_decode_mode")
        if mode and mode not in modes:
            modes.append(str(mode))
        order = detection.get("keypoint_anchor_order")
        if order and order not in anchor_orders:
            anchor_orders.append(str(order))
        if detection.get("keypoint_geometry_score") is not None:
            geometry_scores.append(float(detection["keypoint_geometry_score"]))
        if detection.get("keypoint_global_index") is not None:
            global_indices.append(float(detection["keypoint_global_index"]))
        if detection.get("keypoint_candidate_count") is not None:
            candidate_counts.append(float(detection["keypoint_candidate_count"]))
        if branch_diagnostics is None and isinstance(detection.get("keypoint_branch_diagnostics"), dict):
            branch_diagnostics = detection.get("keypoint_branch_diagnostics")
        raw_range = detection.get("keypoint_raw_xy_range")
        if isinstance(raw_range, dict):
            x_range = raw_range.get("x")
            y_range = raw_range.get("y")
            if isinstance(x_range, list) and len(x_range) == 2:
                raw_xs.extend([float(x_range[0]), float(x_range[1])])
            if isinstance(y_range, list) and len(y_range) == 2:
                raw_ys.extend([float(y_range[0]), float(y_range[1])])
    return {
        "keypoint_decode_mode": "+".join(modes) if modes else None,
        "keypoint_anchor_order": "+".join(anchor_orders) if anchor_orders else None,
        "keypoint_geometry_score_range": _range_or_none(geometry_scores),
        "keypoint_global_index_range": _range_or_none(global_indices),
        "keypoint_candidate_count_range": _range_or_none(candidate_counts),
        "keypoint_branch_diagnostics": branch_diagnostics,
        "keypoint_raw_xy_range": {
            "x": _range_or_none(raw_xs),
            "y": _range_or_none(raw_ys),
        },
    }


def _range_or_none(values: list[float]) -> list[float] | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return None
    return [round(min(clean), 4), round(max(clean), 4)]


def _sanity_check_detections(detections: list[dict[str, Any]], meta: LetterboxMeta) -> str | None:
    if not detections:
        return None
    high_conf_points: list[list[float]] = []
    for detection in detections:
        for point in detection.get("keypoints") or []:
            if len(point) >= 3 and float(point[2]) >= 0.05:
                high_conf_points.append(point)
    if not high_conf_points:
        return "RKNN postprocess decoded detections but no usable keypoints"
    edge_count = 0
    for x_value, y_value, _ in high_conf_points:
        x = float(x_value)
        y = float(y_value)
        if x <= 1.0 or y <= 1.0 or x >= meta.original_width - 2 or y >= meta.original_height - 2:
            edge_count += 1
    if edge_count / max(len(high_conf_points), 1) > 0.85:
        return "RKNN postprocess decoded keypoints mostly clipped to image border"
    return None


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
