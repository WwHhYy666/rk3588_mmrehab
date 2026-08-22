import argparse
import os
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np


warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import psutil
except ImportError:
    psutil = None


COCO_SKELETON = [
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
]

KEYPOINT_COLORS = [
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
]

LINK_COLORS = [
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
]


def resolve_model_path(path, fallbacks):
    model_path = Path(path)
    if model_path.exists():
        return model_path
    for fallback in fallbacks:
        fallback_path = Path(fallback)
        if fallback_path.exists():
            print(f"Model path {model_path} not found; using fallback {fallback_path}.")
            return fallback_path
    return model_path


def parse_source(source):
    if source.isdigit():
        return int(source)
    return source


def open_capture(source, width=None, height=None):
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def make_writer(output_path, fps, frame_size):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")
    return writer


class RKNNModel:
    def __init__(self, model_path, target="rk3588", device_id=None, core_mask="auto"):
        self.model_path = Path(model_path)
        self.target = target
        self.device_id = device_id
        self.core_mask = core_mask
        self.rknn = None
        self.backend = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        try:
            from rknnlite.api import RKNNLite

            self.backend = "rknnlite"
            self.rknn = RKNNLite()
            print(f"Loading RKNNLite model: {self.model_path}")
            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0:
                raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}: {self.model_path}")

            runtime_kwargs = {}
            if self.core_mask != "auto" and hasattr(RKNNLite, self.core_mask):
                runtime_kwargs["core_mask"] = getattr(RKNNLite, self.core_mask)
            elif hasattr(RKNNLite, "NPU_CORE_0_1_2"):
                runtime_kwargs["core_mask"] = RKNNLite.NPU_CORE_0_1_2
            ret = self.rknn.init_runtime(**runtime_kwargs)
            if ret != 0:
                raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}: {self.model_path}")
            return
        except ImportError:
            pass

        try:
            from rknn.api import RKNN

            self.backend = "rknn-toolkit2"
            self.rknn = RKNN(verbose=False)
            print(f"Loading RKNN toolkit model: {self.model_path}")
            ret = self.rknn.load_rknn(str(self.model_path))
            if ret != 0:
                raise RuntimeError(f"RKNN.load_rknn failed with code {ret}: {self.model_path}")

            runtime_kwargs = {}
            if self.target:
                runtime_kwargs["target"] = self.target
            if self.device_id:
                runtime_kwargs["device_id"] = self.device_id
            print("Using rknn.api on PC. Real RKNN runtime usually requires a connected RK3588 device.")
            ret = self.rknn.init_runtime(**runtime_kwargs)
            if ret != 0:
                raise RuntimeError(f"RKNN.init_runtime failed with code {ret}: {self.model_path}")
            return
        except ImportError as exc:
            raise ImportError(
                "RKNN runtime is not installed. On RK3588 install rknn-toolkit-lite2; "
                "on PC install rknn-toolkit2 and connect a target board for runtime."
            ) from exc

    def inference(self, inputs):
        outputs = self.rknn.inference(inputs=inputs)
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        return outputs

    def release(self):
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass
            self.rknn = None


def letterbox(image, target_size=(640, 640), pad_val=114):
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
    return padded, scale, pad_left, pad_top


def preprocess_det(frame, input_layout="nchw"):
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


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def infer_head_layout(outputs):
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
    if votes["nchw"] > votes["nhwc"]:
        return "nchw"
    return "auto"


def normalize_pred_map(output, head_layout="auto"):
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


def pair_rtmdet_outputs(outputs, input_size=640, head_layout="auto", debug=False):
    if head_layout == "auto":
        head_layout = infer_head_layout(outputs)
        if debug:
            print(f"Inferred RTMDet head layout: {head_layout}")
    cls_maps = []
    bbox_maps = []
    for idx, output in enumerate(outputs):
        arr = normalize_pred_map(output, head_layout)
        channels, h, w = arr.shape
        if debug:
            print(f"RTMDet output[{idx}] raw={np.asarray(output).shape}, normalized={arr.shape}")
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

    return sorted(pairs, key=lambda item: item[0])


def generate_priors(h, w, stride):
    shifts_x = (np.arange(w, dtype=np.float32) + 0.5) * stride
    shifts_y = (np.arange(h, dtype=np.float32) + 0.5) * stride
    grid_x, grid_y = np.meshgrid(shifts_x, shifts_y)
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def decode_bboxes(priors, bbox_pred, stride, bbox_decode_mode):
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


def nms(bboxes, scores, iou_thr):
    if len(bboxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = bboxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
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


def map_bboxes_to_original(bboxes, scale, pad_left, pad_top, frame_shape):
    if len(bboxes) == 0:
        return bboxes
    h, w = frame_shape[:2]
    mapped = bboxes.copy().astype(np.float32)
    mapped[:, 0::2] = (mapped[:, 0::2] - pad_left) / scale
    mapped[:, 1::2] = (mapped[:, 1::2] - pad_top) / scale
    mapped[:, 0::2] = np.clip(mapped[:, 0::2], 0, w - 1)
    mapped[:, 1::2] = np.clip(mapped[:, 1::2], 0, h - 1)
    return mapped


def postprocess_rtmdet(
    outputs,
    score_thr,
    nms_thr,
    scale,
    pad_left,
    pad_top,
    frame_shape,
    max_persons,
    bbox_decode_mode="auto",
    head_layout="auto",
    nms_pre=1000,
    debug=False,
):
    all_bboxes = []
    all_scores = []
    pairs = pair_rtmdet_outputs(outputs, 640, head_layout, debug)
    for stride, cls_idx, bbox_idx, cls_map, bbox_map in pairs:
        _, h, w = cls_map.shape
        person_scores = sigmoid(cls_map[0]).reshape(-1)
        keep = np.where(person_scores >= score_thr)[0]
        if keep.size == 0:
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
        if debug:
            print(f"stride={stride}, cls_out={cls_idx}, bbox_out={bbox_idx}, kept={int(valid.sum())}")

    if not all_bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    bboxes = np.concatenate(all_bboxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    keep = nms(bboxes, scores, nms_thr)
    if max_persons > 0:
        keep = keep[:max_persons]
    bboxes = map_bboxes_to_original(bboxes[keep], scale, pad_left, pad_top, frame_shape)
    scores = scores[keep]
    valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
    return bboxes[valid], scores[valid]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def rotate_point(point, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([point[0] * cs - point[1] * sn, point[0] * sn + point[1] * cs], dtype=np.float32)


def get_affine_transform(center, scale, output_size, inv=False):
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


def bbox_xyxy_to_center_scale(bbox, input_size=(288, 384), padding=1.25):
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return center, np.array([w * padding, h * padding], dtype=np.float32)


def transform_points(points, matrix):
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.concatenate([points.astype(np.float32), ones], axis=1) @ matrix.T


def topdown_affine_crop(frame, bbox, input_size=(288, 384)):
    center, scale = bbox_xyxy_to_center_scale(bbox, input_size)
    matrix = get_affine_transform(center, scale, input_size)
    inv_matrix = get_affine_transform(center, scale, input_size, inv=True)
    crop = cv2.warpAffine(
        frame,
        matrix,
        input_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop, inv_matrix


def preprocess_pose(frame, bbox, input_layout="nchw"):
    crop, inv_matrix = topdown_affine_crop(frame, bbox, (288, 384))
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image = (image - mean) / std
    if input_layout == "nhwc":
        blob = image[None, ...].astype(np.float32)
    else:
        blob = image.transpose(2, 0, 1)[None, ...].astype(np.float32)
    return blob, inv_matrix


def normalize_simcc_outputs(outputs, debug=False):
    candidates = []
    for idx, output in enumerate(outputs):
        arr = np.asarray(output, dtype=np.float32)
        if debug:
            print(f"RTMPose output[{idx}] shape={arr.shape}")
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


def decode_simcc(outputs, inv_matrix, split_ratio=2.0, score_mode="sqrt", debug=False):
    simcc_x, simcc_y = normalize_simcc_outputs(outputs, debug)
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


def draw_bbox(frame, bbox, score):
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 255), 2)
    label = f"person {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 8, y_text + th + 8), (40, 220, 255), -1)
    cv2.putText(frame, label, (x1 + 4, y_text + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_pose(frame, keypoints, scores, kpt_thr):
    visible = scores >= kpt_thr
    for idx, (start, end) in enumerate(COCO_SKELETON):
        if not (visible[start] and visible[end]):
            continue
        pt1 = tuple(np.round(keypoints[start]).astype(int))
        pt2 = tuple(np.round(keypoints[end]).astype(int))
        cv2.line(frame, pt1, pt2, LINK_COLORS[idx], 3, cv2.LINE_AA)
    for idx, point in enumerate(keypoints):
        if not visible[idx]:
            continue
        center = tuple(np.round(point).astype(int))
        cv2.circle(frame, center, 4, KEYPOINT_COLORS[idx], -1, cv2.LINE_AA)
        cv2.circle(frame, center, 5, (255, 255, 255), 1, cv2.LINE_AA)


def mb(value):
    return value / (1024 * 1024)


def read_rknpu_load():
    path = Path("/sys/kernel/debug/rknpu/load")
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not text:
        return None
    nums = re.findall(r"(\d+)\s*%", text)
    if nums:
        return "/".join(f"{num}%" for num in nums[:3])
    return text.replace("\n", " | ")


class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = max(interval, 0.05)
        self.last_sample_time = 0.0
        self.process = psutil.Process() if psutil else None
        self.cpu_count = psutil.cpu_count() if psutil else None
        self.metrics = {
            "process_mem_mb": None,
            "process_cpu_percent": None,
            "rknpu_load": None,
        }
        if self.process is not None:
            try:
                self.process.cpu_percent(interval=None)
            except Exception:
                pass

    def sample(self, force=False):
        now = time.perf_counter()
        if not force and now - self.last_sample_time < self.interval:
            return self.metrics
        self.last_sample_time = now
        if self.process is not None:
            try:
                mem = self.process.memory_info()
                self.metrics["process_mem_mb"] = mb(mem.rss)
                raw_cpu = self.process.cpu_percent(interval=None)
                self.metrics["process_cpu_percent"] = raw_cpu / self.cpu_count if self.cpu_count else raw_cpu
            except Exception:
                pass
        self.metrics["rknpu_load"] = read_rknpu_load()
        return self.metrics

    def close(self):
        pass


def format_mem(value):
    return "N/A" if value is None else f"{value:.0f} MB"


def format_percent(value):
    return "N/A" if value is None else f"{value:.1f}%"


def draw_runtime_overlay(frame, runtime):
    metrics = runtime["metrics"]
    lines = [
        "RTMDet -> RTMPose RKNN realtime",
        f"persons: {runtime['person_count']} | frame: {runtime['frame_idx']}",
        f"FPS inst/ema/avg: {runtime['inst_fps']:.1f} / {runtime['fps']:.1f} / {runtime['avg_fps']:.1f}",
        f"latency: {runtime['latency_ms']:.1f} ms | throughput: {runtime['avg_fps']:.1f} frame/s",
        f"Proc mem: {format_mem(metrics.get('process_mem_mb'))} | CPU: {format_percent(metrics.get('process_cpu_percent'))}",
        f"RKNPU load: {metrics.get('rknpu_load') or 'N/A'}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_h = 22
    width = min(max(500, int(frame.shape[1] * 0.34)), frame.shape[1] - 20)
    height = 18 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), (60, 220, 255), 1)
    y = 34
    for idx, line in enumerate(lines):
        color = (255, 255, 255) if idx else (60, 220, 255)
        cv2.putText(frame, line, (20, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h


def build_argparser():
    parser = argparse.ArgumentParser(description="Cascade RTMDet + RTMPose RKNN human pose demo.")
    parser.add_argument("--source", default="0", help="Input video path or camera index. Default: 0")
    parser.add_argument("--output", default="outputs/rknn_rtmdet_rtmpose_demo.mp4", help="Output demo video path.")
    parser.add_argument("--det-rknn", default="rknn/rtmdet_l_640x640_partition_fp16.rknn")
    parser.add_argument("--pose-rknn", default="rknn/rtmpose_l_384x288_fp16.rknn")
    parser.add_argument("--target", default="rk3588", help="rknn.api target when running from PC.")
    parser.add_argument("--device-id", default=None, help="Optional rknn.api device id for connected board.")
    parser.add_argument("--core-mask", default="auto", help="RKNNLite core mask name, e.g. NPU_CORE_0_1_2.")
    parser.add_argument("--det-score-thr", type=float, default=0.35)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--kpt-score-thr", type=float, default=0.3)
    parser.add_argument("--max-persons", type=int, default=5, help="0 means keep all NMS results.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process until source ends.")
    parser.add_argument("--show", action="store_true", help="Show a real-time preview window.")
    parser.add_argument("--width", type=int, default=0, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=0, help="Camera capture height.")
    parser.add_argument("--metrics-interval", type=float, default=0.5, help="Resource sampling interval in seconds.")
    parser.add_argument("--no-resource-overlay", action="store_true", help="Disable resource/throughput overlay.")
    parser.add_argument("--input-layout", choices=["nchw", "nhwc"], default="nchw", help="RKNN input tensor layout.")
    parser.add_argument(
        "--head-layout",
        choices=["auto", "nchw", "nhwc"],
        default="auto",
        help="RTMDet output head layout. Use this if [1,80,80,80] is ambiguous.",
    )
    parser.add_argument(
        "--bbox-decode-mode",
        choices=["auto", "stride", "none"],
        default="auto",
        help="RTMDet bbox distance scale. Use stride if bbox preds are stride units; none if already pixels.",
    )
    parser.add_argument("--simcc-score-mode", choices=["sqrt", "avg"], default="sqrt")
    parser.add_argument("--debug-shapes", action="store_true", help="Print RKNN output shapes and head pairing info.")
    return parser


def run_pose_for_bboxes(pose_model, frame, bboxes, input_layout, score_mode, debug_shapes):
    if len(bboxes) == 0:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
    keypoints_list = []
    scores_list = []
    for bbox in bboxes:
        pose_input, inv_matrix = preprocess_pose(frame, bbox, input_layout)
        outputs = pose_model.inference([pose_input])
        keypoints, scores = decode_simcc(outputs, inv_matrix, 2.0, score_mode, debug_shapes)
        keypoints_list.append(keypoints)
        scores_list.append(scores)
    return np.stack(keypoints_list, axis=0), np.stack(scores_list, axis=0)


def main():
    args = build_argparser().parse_args()
    det_path = resolve_model_path(args.det_rknn, ["rknn/rtmdet_fp16.rknn"])
    pose_path = resolve_model_path(args.pose_rknn, ["rknn/rtmpose_fp16.rknn"])

    print(f"Loading RTMDet RKNN from {det_path}")
    det_model = RKNNModel(det_path, args.target, args.device_id, args.core_mask)
    print(f"Loading RTMPose RKNN from {pose_path}")
    pose_model = RKNNModel(pose_path, args.target, args.device_id, args.core_mask)

    source = parse_source(args.source)
    cap = open_capture(source, args.width or None, args.height or None)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_for_writer = src_fps if src_fps and src_fps > 1 else 25.0

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Video source opened but produced no frames.")
    h, w = frame.shape[:2]
    writer = make_writer(args.output, fps_for_writer, (w, h))

    frame_idx = 0
    fps = 0.0
    last_time = time.perf_counter()
    start_time = last_time
    monitor = ResourceMonitor(args.metrics_interval)
    printed_shapes = False

    try:
        while ok:
            infer_start = time.perf_counter()
            det_input, scale, pad_left, pad_top = preprocess_det(frame, args.input_layout)
            det_outputs = det_model.inference([det_input])
            debug_this_frame = args.debug_shapes and not printed_shapes
            if debug_this_frame:
                for idx, output in enumerate(det_outputs):
                    print(f"RTMDet raw output[{idx}] shape={np.asarray(output).shape}")
            bboxes, bbox_scores = postprocess_rtmdet(
                det_outputs,
                args.det_score_thr,
                args.nms_thr,
                scale,
                pad_left,
                pad_top,
                frame.shape,
                args.max_persons,
                args.bbox_decode_mode,
                args.head_layout,
                debug=debug_this_frame,
            )
            keypoints, keypoint_scores = run_pose_for_bboxes(
                pose_model, frame, bboxes, args.input_layout, args.simcc_score_mode, debug_this_frame
            )
            printed_shapes = printed_shapes or debug_this_frame
            infer_end = time.perf_counter()

            canvas = frame.copy()
            for bbox, score in zip(bboxes, bbox_scores):
                draw_bbox(canvas, bbox, score)
            for points, scores in zip(keypoints, keypoint_scores):
                draw_pose(canvas, points, scores, args.kpt_score_thr)

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            fps = inst_fps if fps == 0 else fps * 0.9 + inst_fps * 0.1
            last_time = now
            elapsed = max(now - start_time, 1e-6)
            avg_fps = (frame_idx + 1) / elapsed
            metrics = monitor.sample()

            if not args.no_resource_overlay:
                draw_runtime_overlay(
                    canvas,
                    {
                        "metrics": metrics,
                        "person_count": len(bboxes),
                        "frame_idx": frame_idx + 1,
                        "inst_fps": inst_fps,
                        "fps": fps,
                        "avg_fps": avg_fps,
                        "latency_ms": (infer_end - infer_start) * 1000.0,
                    },
                )

            writer.write(canvas)
            if args.show:
                cv2.imshow("RTMDet + RTMPose RKNN Cascade", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Interrupted by user; finalizing recorded video.")
    finally:
        cap.release()
        writer.release()
        monitor.close()
        det_model.release()
        pose_model.release()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    print(f"Processed frames: {frame_idx}, average throughput: {frame_idx / elapsed:.2f} frame/s")
    print(f"Saved demo video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
