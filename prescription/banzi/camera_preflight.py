from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def list_video_devices() -> list[str]:
    if os.name == "nt":
        return ["0", "1", "2"]
    return [str(path) for path in sorted(Path("/dev").glob("video*"))]


def build_camera_candidates(requested_device: str) -> list[str]:
    requested = str(requested_device or "auto").strip()
    candidates: list[str] = []
    if requested and requested.lower() != "auto":
        candidates.append(requested)
    if os.name == "nt":
        candidates.extend(["0", "1", "2"])
    else:
        for directory in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
            if directory.exists():
                candidates.extend(str(path) for path in sorted(directory.glob("*")))
        candidates.extend(str(path) for path in sorted(Path("/dev").glob("video*")))

    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def normalize_device(device: str) -> str | int:
    text = str(device).strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"/dev/video(\d+)", text)
    if match:
        return int(match.group(1))
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether OpenCV can open and continuously read camera frames.")
    parser.add_argument(
        "--device",
        default=os.environ.get("RK_CAMERA_DEVICE", "auto"),
        help="Camera device path, index, or auto. Defaults to RK_CAMERA_DEVICE or auto.",
    )
    parser.add_argument("--width", type=int, default=int(os.environ.get("RK_CAMERA_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("RK_CAMERA_HEIGHT", "360")))
    parser.add_argument(
        "--frames",
        type=int,
        default=int(os.environ.get("RK_CAMERA_PREFLIGHT_FRAMES", "3")),
        help="Number of frames to read for stability testing. Use 300 for a pressure test.",
    )
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        print(f"cv2 import failed: {exc}")
        return 2

    print(f"requested_device: {args.device}")
    print(f"requested_size: {args.width}x{args.height}")
    print(f"requested_frames: {args.frames}")
    print(f"visible_video_devices: {', '.join(list_video_devices()) or 'none'}")
    candidates = build_camera_candidates(str(args.device))
    print(f"candidate_devices: {', '.join(candidates) or 'none'}")

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    selected: str | None = None
    for candidate in candidates:
        result = probe_camera(cv2, normalize_device(candidate), backend, args.width, args.height, args.frames)
        print(
            "candidate: "
            f"{candidate} opened={result['opened']} read_frame={result['read_frame']} "
            f"frame_shape={result['frame_shape']} success_frames={result['success_frames']} "
            f"failure_frames={result['failure_frames']} avg_fps={result['avg_fps']} "
            f"avg_read_ms={result['avg_read_ms']} max_read_ms={result['max_read_ms']} "
            f"selected={result['selected']}"
        )
        if result["selected"]:
            selected = candidate
            break

    if selected is None:
        print("selected_device: none")
        return 1

    print(f"selected_device: {selected}")
    return 0


def probe_camera(cv2: Any, device: str | int, backend: int, width: int, height: int, frames: int) -> dict[str, object]:
    cap = cv2.VideoCapture(device, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    opened = cap.isOpened()
    if not opened:
        cap.release()
        return _probe_result(False, False, None, 0, max(1, frames), None, None, None, False)

    total_frames = max(1, int(frames))
    success_frames = 0
    failure_frames = 0
    read_times_ms: list[float] = []
    shape = None
    started = time.perf_counter()
    for _ in range(total_frames):
        read_started = time.perf_counter()
        ok, frame = cap.read()
        read_times_ms.append((time.perf_counter() - read_started) * 1000.0)
        if ok and frame is not None:
            success_frames += 1
            shape = tuple(frame.shape)
        else:
            failure_frames += 1
    cap.release()
    elapsed_s = max(1e-6, time.perf_counter() - started)
    avg_fps = success_frames / elapsed_s
    selected = success_frames > 0 and failure_frames == 0
    return _probe_result(
        opened,
        success_frames > 0,
        shape,
        success_frames,
        failure_frames,
        avg_fps,
        sum(read_times_ms) / len(read_times_ms) if read_times_ms else None,
        max(read_times_ms) if read_times_ms else None,
        selected,
    )


def _round(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _probe_result(
    opened: bool,
    read_frame: bool,
    frame_shape: tuple[int, ...] | None,
    success_frames: int,
    failure_frames: int,
    avg_fps: float | None,
    avg_read_ms: float | None,
    max_read_ms: float | None,
    selected: bool,
) -> dict[str, object]:
    return {
        "opened": opened,
        "read_frame": read_frame,
        "frame_shape": frame_shape,
        "success_frames": success_frames,
        "failure_frames": failure_frames,
        "avg_fps": _round(avg_fps),
        "avg_read_ms": _round(avg_read_ms),
        "max_read_ms": _round(max_read_ms),
        "selected": selected,
    }


if __name__ == "__main__":
    sys.exit(main())
