"""浏览器预览版摄像头脚本。

这个脚本的定位是：

- 当 X11 暂时没打通时，用浏览器快速确认摄像头画面
- 作为 SSH 环境下的临时远程观察工具

它不是当前项目的正式低延迟预览方案。

原因是当前实现属于最基础的 MJPEG over HTTP：

- 每帧都要重新编码 JPEG
- 浏览器端按图片流方式显示
- 代码里还显式限制了 FPS

因此它适合“先看到画面”，不适合替代 `cv2.imshow()` 做主预览界面。
"""

import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# 当前仍沿用板端已验证的摄像头参数。
CAMERA_DEVICE = "/dev/video21"
WIDTH = 1280
HEIGHT = 720
FPS = 15
PORT = 8080

cap = cv2.VideoCapture(CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise SystemExit(f"无法打开摄像头: {CAMERA_DEVICE}")


class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>浏览器摄像头预览</title>
            </head>
            <body>
                <h2>ELF2 浏览器摄像头预览</h2>
                <p>设备节点: {CAMERA_DEVICE}</p>
                <p>说明: 这是临时网页预览，不是当前项目的低延迟主方案。</p>
                <img src="/stream.mjpg" style="max-width: 100%; height: auto;">
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("读取摄像头失败")
                    time.sleep(0.1)
                    continue

                # 这里每帧都要重新 JPEG 编码，是当前网页方案延迟较高的主要来源之一。
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue

                data = jpg.tobytes()

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    # 显式限帧有助于减轻板端压力，但也会进一步拉大预览延迟。
                    time.sleep(1 / FPS)
                except BrokenPipeError:
                    break
                except ConnectionResetError:
                    break
            return

        self.send_response(404)
        self.end_headers()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


server = ThreadedHTTPServer(("0.0.0.0", PORT), CameraHandler)

print(f"摄像头已打开: {CAMERA_DEVICE}")
print(f"分辨率: {WIDTH}x{HEIGHT}")
print(f"浏览器打开: http://192.168.137.232:{PORT}")
print("定位: 这是 X11 预览失败时的临时网页预览工具。")
print("说明: 当前实现不是低延迟主方案，出现约 1 秒延迟是可能的。")
print("按 Ctrl+C 退出")

try:
    server.serve_forever()
finally:
    cap.release()
