from __future__ import annotations

import argparse
import os
import sys
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
    return int(device) if str(device).isdigit() else device


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether OpenCV can open and read one camera frame.")
    parser.add_argument(
        "--device",
        default=os.environ.get("RK_CAMERA_DEVICE", "auto"),
        help="Camera device path, index, or auto. Defaults to RK_CAMERA_DEVICE or auto.",
    )
    parser.add_argument("--width", type=int, default=int(os.environ.get("RK_CAMERA_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("RK_CAMERA_HEIGHT", "360")))
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        print(f"cv2 import failed: {exc}")
        return 2

    print(f"requested_device: {args.device}")
    print(f"requested_size: {args.width}x{args.height}")
    print(f"visible_video_devices: {', '.join(list_video_devices()) or 'none'}")
    candidates = build_camera_candidates(str(args.device))
    print(f"candidate_devices: {', '.join(candidates) or 'none'}")

    backend = cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_DSHOW
    selected: str | None = None
    for candidate in candidates:
        result = probe_camera(cv2, normalize_device(candidate), backend, args.width, args.height)
        print(
            "candidate: "
            f"{candidate} opened={result['opened']} read_frame={result['read_frame']} "
            f"frame_shape={result['frame_shape']} selected={result['selected']}"
        )
        if result["selected"]:
            selected = candidate
            break

    if selected is None:
        print("selected_device: none")
        return 1

    print(f"selected_device: {selected}")
    return 0


def probe_camera(cv2: Any, device: str | int, backend: int, width: int, height: int) -> dict[str, object]:
    cap = cv2.VideoCapture(device, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    opened = cap.isOpened()
    if not opened:
        cap.release()
        return {"opened": False, "read_frame": False, "frame_shape": None, "selected": False}

    ok = False
    shape = None
    for _ in range(3):
        ok, frame = cap.read()
        if ok and frame is not None:
            shape = tuple(frame.shape)
            break
    cap.release()
    return {"opened": opened, "read_frame": bool(ok and shape is not None), "frame_shape": shape, "selected": bool(ok and shape is not None)}


if __name__ == "__main__":
    sys.exit(main())
