from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from prescription.common.result_storage import RESULTS_DIR, SUMMARIES_DIR, ensure_dirs, save_prescription_artifacts


HOST = "127.0.0.1"
PORT = 8090
latest_summary: dict[str, object] | None = None


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def write_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def make_json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status_code: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    write_cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


class LocalResultSinkHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        write_cors_headers(self)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            make_json_response(
                self,
                {
                    "ok": True,
                    "host": HOST,
                    "port": PORT,
                    "results_dir": str(RESULTS_DIR),
                    "summaries_dir": str(SUMMARIES_DIR),
                },
            )
            return

        if self.path == "/latest":
            if latest_summary is None:
                make_json_response(self, {"ok": False, "error": "还没有本地保存记录。"}, status_code=404)
                return
            make_json_response(self, {"ok": True, "latest": latest_summary})
            return

        self.send_response(404)
        write_cors_headers(self)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/save_result":
            self.send_response(404)
            write_cors_headers(self)
            self.end_headers()
            return

        try:
            payload = read_json_body(self)
        except json.JSONDecodeError:
            make_json_response(self, {"ok": False, "error": "请求体不是有效 JSON。"}, status_code=400)
            return

        prescription = payload.get("prescription")
        if not isinstance(prescription, dict):
            make_json_response(self, {"ok": False, "error": "缺少 prescription JSON。"}, status_code=400)
            return

        save_result = save_prescription_artifacts(
            prescription,
            board_ip=str(payload.get("board_ip", "unknown")),
            board_port=str(payload.get("board_port", "unknown")),
            source=str(payload.get("source", "unknown")),
        )

        global latest_summary
        latest_summary = save_result["summary"]

        make_json_response(
            self,
            {
                "ok": True,
                "saved_path": save_result["saved_path"],
                "summary_path": save_result["summary_path"],
                "summary": save_result["summary"],
            },
        )

    def log_message(self, format: str, *args) -> None:
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> None:
    ensure_dirs()
    server = ThreadedHTTPServer((HOST, PORT), LocalResultSinkHandler)
    print(f"本机结果接收器已启动: http://{HOST}:{PORT}")
    print(f"模板目录: {RESULTS_DIR}")
    print(f"摘要目录: {SUMMARIES_DIR}")
    print("健康检查: /health")
    print("最近一次保存摘要: /latest")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
