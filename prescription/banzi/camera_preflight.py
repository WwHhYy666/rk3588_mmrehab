from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def list_video_devices() -> list[str]:
    if os.name == "nt":
        return ["0", "1", "2"]
    return [str(path) for path in sorted(Path("/dev").glob("video*"))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether OpenCV can open and read one camera frame.")
    parser.add_argument(
        "--device",
        default=os.environ.get("RK_CAMERA_DEVICE", "/dev/video21"),
        help="Camera device path or index. Defaults to RK_CAMERA_DEVICE or /dev/video21.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        import cv2
    except Exception as exc:
        print(f"cv2 import failed: {exc}")
        return 2

    device: str | int = args.device
    if str(device).isdigit():
        device = int(device)

    print(f"requested_device: {args.device}")
    print(f"visible_video_devices: {', '.join(list_video_devices()) or 'none'}")

    backend = cv2.CAP_V4L2 if os.name != "nt" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(device, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    opened = cap.isOpened()
    print(f"opened: {opened}")
    if not opened:
        cap.release()
        return 1

    ok, frame = cap.read()
    shape = None if frame is None else tuple(frame.shape)
    print(f"read_frame: {ok}")
    print(f"frame_shape: {shape}")
    cap.release()
    return 0 if ok and frame is not None else 1


if __name__ == "__main__":
    sys.exit(main())
