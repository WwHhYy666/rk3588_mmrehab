"""浏览器版 MediaPipe Pose 预览脚本。

这个脚本的定位是：

- 在 X11 暂时没打通时，用浏览器快速查看骨骼识别结果
- 作为 `pose_mediapipe_demo.py` 的网页预览备选方案

它不是当前项目的正式低延迟主方案，但比最基础的“每个客户端自己读相机”的写法更适合先做远程验证。

当前做法是：

- 只开一个后台线程负责读摄像头、跑 MediaPipe Pose、画骨架、编码 JPEG
- HTTP 客户端只负责拿“最新一帧”结果

这样做的好处是：

- 不会因为浏览器多开几个页面，就重复占用摄像头
- 不会让每个浏览器连接都自己跑一次骨骼识别
- 更接近“只保留最新帧”的实时预览思路
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# 当前仍沿用板端已验证的摄像头参数。
CAMERA_DEVICE = "/dev/video21"
CAMERA_BACKEND = cv2.CAP_V4L2
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FOURCC = "MJPG"

# 为了减小网页端延迟，推理阶段单独降低输入尺寸。
INFER_WIDTH = 640
INFER_HEIGHT = 360

# 网页端 JPEG 质量越高越清晰，但编码耗时也更高。
JPEG_QUALITY = 70
PORT = 8081


def open_camera() -> cv2.VideoCapture:
    """按当前板端已验证的参数打开摄像头。"""
    cap = cv2.VideoCapture(CAMERA_DEVICE, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


class PoseStreamState:
    """保存最新一帧网页预览结果。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame_id = 0
        self.jpg_bytes: bytes | None = None
        self.last_status = "等待第一帧..."
        self.running = True

    def update(self, jpg_bytes: bytes, status: str) -> None:
        with self.condition:
            self.frame_id += 1
            self.jpg_bytes = jpg_bytes
            self.last_status = status
            self.condition.notify_all()


cap = open_camera()

if not cap.isOpened():
    raise SystemExit(f"无法打开摄像头: {CAMERA_DEVICE}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
state = PoseStreamState()


def pose_worker() -> None:
    """持续读取摄像头、做姿态识别，并更新最新网页帧。"""
    frame_count = 0

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while state.running:
            success, frame = cap.read()

            if not success:
                print("读取摄像头失败，0.1 秒后重试。")
                time.sleep(0.1)
                continue

            frame_count += 1

            # 推理时先缩小尺寸，尽量降低浏览器版骨骼识别的整体延迟。
            infer_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            output_frame = frame.copy()
            status = "未检测到人体姿态"

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    output_frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                landmarks = result.pose_landmarks.landmark
                left_knee = landmarks[25]
                status = (
                    f"已检测到人体姿态 | left_knee: "
                    f"x={left_knee.x:.3f}, y={left_knee.y:.3f}, "
                    f"visibility={left_knee.visibility:.3f}"
                )

                if frame_count % 30 == 0:
                    print(status)
            else:
                cv2.putText(
                    output_frame,
                    "未检测到人体",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            ok, jpg = cv2.imencode(
                ".jpg",
                output_frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not ok:
                continue

            state.update(jpg.tobytes(), status)
    finally:
        pose.close()


class PoseHTTPHandler(BaseHTTPRequestHandler):
    """浏览器访问入口。"""

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>浏览器版骨骼识别预览</title>
            </head>
            <body>
                <h2>ELF2 浏览器版骨骼识别预览</h2>
                <p>设备节点: {CAMERA_DEVICE}</p>
                <p>采集分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}</p>
                <p>推理分辨率: {INFER_WIDTH}x{INFER_HEIGHT}</p>
                <p>说明: 这是网页备选方案，不是当前项目的低延迟主方案。</p>
                <p>状态接口: <a href="/status" target="_blank">/status</a></p>
                <img src="/stream.mjpg" style="max-width: 100%; height: auto;">
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(state.last_status.encode("utf-8"))
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_sent_frame_id = -1

            while state.running:
                with state.condition:
                    state.condition.wait_for(
                        lambda: state.frame_id != last_sent_frame_id or not state.running
                    )
                    if not state.running:
                        break

                    last_sent_frame_id = state.frame_id
                    data = state.jpg_bytes

                if data is None:
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """关闭默认访问日志，避免终端刷屏。"""
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


worker = threading.Thread(target=pose_worker, daemon=True)
worker.start()

server = ThreadedHTTPServer(("0.0.0.0", PORT), PoseHTTPHandler)

print(f"摄像头已打开: {CAMERA_DEVICE}")
print(f"采集分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"推理分辨率: {INFER_WIDTH}x{INFER_HEIGHT}")
print(f"浏览器打开: http://板子IP:{PORT}")
print("定位: 这是浏览器版骨骼识别预览，用于 X11 暂时没打通时的备选验证。")
print("说明: 相比当前最基础的网页摄像头预览，这份写法更偏向“只保留最新帧”。")
print("按 Ctrl+C 退出")

try:
    server.serve_forever()
finally:
    state.running = False
    with state.condition:
        state.condition.notify_all()
    cap.release()
