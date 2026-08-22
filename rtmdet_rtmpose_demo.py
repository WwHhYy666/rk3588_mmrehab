import argparse
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch


warnings.filterwarnings("ignore", category=FutureWarning, module=r"mmdet\..*")
warnings.filterwarnings("ignore", message=r"torch\.meshgrid: in an upcoming release.*")


def patch_torch_load_for_openmmlab_checkpoints():
    """PyTorch 2.6+ defaults to weights_only=True; old OpenMMLab ckpts need metadata."""
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load


patch_torch_load_for_openmmlab_checkpoints()

from mmdet.apis import inference_detector, init_detector  # noqa: E402
from mmengine.registry import DefaultScope  # noqa: E402
from mmpose.apis import inference_topdown, init_model  # noqa: E402


try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


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


def parse_cuda_index(device):
    if not device.startswith("cuda"):
        return 0
    parts = device.split(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def mb(value):
    return value / (1024 * 1024)


class ResourceMonitor:
    def __init__(self, device, interval=0.5):
        self.device = device
        self.interval = max(interval, 0.05)
        self.last_sample_time = 0.0
        self.process = psutil.Process() if psutil else None
        self.cpu_count = psutil.cpu_count() if psutil else None
        self.metrics = {
            "process_mem_mb": None,
            "process_cpu_percent": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "gpu_util_percent": None,
            "torch_alloc_mb": None,
            "torch_reserved_mb": None,
        }
        self.nvml_handle = None
        if pynvml is not None and device.startswith("cuda"):
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(parse_cuda_index(device))
            except Exception:
                self.nvml_handle = None

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

        if self.nvml_handle is not None:
            try:
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                self.metrics["gpu_mem_used_mb"] = mb(gpu_mem.used)
                self.metrics["gpu_mem_total_mb"] = mb(gpu_mem.total)
                self.metrics["gpu_util_percent"] = util.gpu
            except Exception:
                pass

        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                idx = parse_cuda_index(self.device)
                self.metrics["torch_alloc_mb"] = mb(torch.cuda.memory_allocated(idx))
                self.metrics["torch_reserved_mb"] = mb(torch.cuda.memory_reserved(idx))
            except Exception:
                pass

        return self.metrics

    def close(self):
        if pynvml is not None and self.nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def parse_source(source):
    if source.isdigit():
        return int(source)
    return source


def open_capture(source, width=None, height=None):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def get_person_bboxes(det_result, score_thr):
    instances = det_result.pred_instances.cpu().numpy()
    keep = (instances.labels == 0) & (instances.scores >= score_thr)
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return instances.bboxes[keep].astype(np.float32), instances.scores[keep].astype(np.float32)


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


def format_mem_pair(used, total):
    if used is None:
        return "N/A"
    if total:
        return f"{used:.0f}/{total:.0f} MB"
    return f"{used:.0f} MB"


def format_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def draw_runtime_overlay(frame, runtime):
    metrics = runtime["metrics"]
    lines = [
        "RTMDet -> RTMPose realtime",
        f"persons: {runtime['person_count']} | frame: {runtime['frame_idx']}",
        f"FPS inst/ema/avg: {runtime['inst_fps']:.1f} / {runtime['fps']:.1f} / {runtime['avg_fps']:.1f}",
        f"latency: {runtime['latency_ms']:.1f} ms | throughput: {runtime['avg_fps']:.1f} frame/s",
        f"GPU mem: {format_mem_pair(metrics.get('gpu_mem_used_mb'), metrics.get('gpu_mem_total_mb'))}",
        f"GPU util: {format_percent(metrics.get('gpu_util_percent'))}",
        f"Torch alloc/resv: {format_mem_pair(metrics.get('torch_alloc_mb'), None)} / {format_mem_pair(metrics.get('torch_reserved_mb'), None)}",
        f"Proc mem: {format_mem_pair(metrics.get('process_mem_mb'), None)} | CPU: {format_percent(metrics.get('process_cpu_percent'))}",
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


def make_writer(output_path, fps, frame_size):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")
    return writer


def build_argparser():
    parser = argparse.ArgumentParser(description="Cascade RTMDet + RTMPose human pose demo.")
    parser.add_argument("--source", default="0", help="Input video path or camera index. Default: 0")
    parser.add_argument("--output", default="outputs/rtmdet_rtmpose_demo.mp4", help="Output demo video path.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--det-config", default="configs/rtmdet_l_8xb32-300e_coco.py")
    parser.add_argument("--det-weight", default="pretrained/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth")
    parser.add_argument("--pose-config", default="configs/rtmpose-l_simcc-body7_420e-384x288.py")
    parser.add_argument(
        "--pose-weight",
        default="pretrained/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth",
    )
    parser.add_argument("--det-score-thr", type=float, default=0.35)
    parser.add_argument("--kpt-score-thr", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process until source ends.")
    parser.add_argument("--show", action="store_true", help="Show a real-time preview window.")
    parser.add_argument("--width", type=int, default=0, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=0, help="Camera capture height.")
    parser.add_argument("--metrics-interval", type=float, default=0.5, help="Resource sampling interval in seconds.")
    parser.add_argument("--no-resource-overlay", action="store_true", help="Disable resource/throughput overlay.")
    return parser


def main():
    args = build_argparser().parse_args()
    source = parse_source(args.source)

    print(f"Loading RTMDet from {args.det_weight}")
    det_model = init_detector(args.det_config, args.det_weight, device=args.device)
    print(f"Loading RTMPose from {args.pose_weight}")
    pose_model = init_model(args.pose_config, args.pose_weight, device=args.device)

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
    monitor = ResourceMonitor(args.device, args.metrics_interval)

    try:
        while ok:
            infer_start = time.perf_counter()
            with DefaultScope.overwrite_default_scope("mmdet"):
                det_result = inference_detector(det_model, frame)
            bboxes, bbox_scores = get_person_bboxes(det_result, args.det_score_thr)
            with DefaultScope.overwrite_default_scope("mmpose"):
                pose_results = inference_topdown(pose_model, frame, bboxes) if len(bboxes) else []
            infer_end = time.perf_counter()

            canvas = frame.copy()
            for bbox, score in zip(bboxes, bbox_scores):
                draw_bbox(canvas, bbox, score)

            for pose_sample in pose_results:
                pred = pose_sample.pred_instances
                keypoints = pred.keypoints[0]
                keypoint_scores = pred.keypoint_scores[0]
                draw_pose(canvas, keypoints, keypoint_scores, args.kpt_score_thr)

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
                cv2.imshow("RTMDet + RTMPose Cascade", canvas)
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
        cv2.destroyAllWindows()
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    print(f"Processed frames: {frame_idx}, average throughput: {frame_idx / elapsed:.2f} frame/s")
    print(f"Saved demo video: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
