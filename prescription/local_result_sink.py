from __future__ import annotations

import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn


HOST = "127.0.0.1"
PORT = 8090
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = DOCS_DIR / "results"
SUMMARIES_DIR = DOCS_DIR / "summaries"
RESULTS_LOG_PATH = DOCS_DIR / "results_log.md"

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
latest_summary: dict[str, object] | None = None


def ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_LOG_PATH.exists():
        RESULTS_LOG_PATH.write_text(
            "# 处方结果索引\n\n"
            "这里记录每次本机保存的简要摘要，方便快速找到对应的模板 JSON 和中文摘要文件。\n"
            "真正占空间的通常不是这份索引，而是 `docs/results/` 里的模板 JSON。\n\n",
            encoding="utf-8",
        )


def safe_name(value: str, default: str) -> str:
    text = (value or "").strip()
    if not text:
        return default
    return SAFE_NAME_PATTERN.sub("_", text).strip("_") or default


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


def quality_text(rom_flexion: float | None) -> str:
    if rom_flexion is None:
        return "未能计算出有效 ROM，建议检查关键点可见度后重录。"
    if rom_flexion < 15:
        return "ROM 偏小，建议重新录制一版动作更标准的结果。"
    if rom_flexion < 30:
        return "ROM 勉强可用，如需做模板建议再录一版更稳定的结果。"
    return "ROM 较明显，这份结果可以作为后续模板参考。"


def render_summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# 处方录制摘要",
            "",
            f"- 保存时间：{summary['saved_at']}",
            f"- 患者编号：`{summary['patient_id']}`",
            f"- 动作名称：`{summary['action_name']}`",
            f"- 侧别模式：`{summary['side_mode']}`",
            f"- 有效帧数：`{summary['frame_count']}`",
            f"- 无效帧数：`{summary['invalid_frame_count']}`",
            f"- 动作时长：`{summary['duration_seconds']}` 秒",
            f"- 最小屈曲角：`{summary['min_knee_flexion_angle']}` 度",
            f"- 最大屈曲角：`{summary['max_knee_flexion_angle']}` 度",
            f"- ROM：`{summary['rom_flexion']}` 度",
            f"- 结果格式：`{summary['result_format']}`",
            f"- 来源板子：`{summary['board_ip']}:{summary['board_port']}`",
            f"- 模板文件：`{summary['saved_path']}`",
            f"- 摘要文件：`{summary['summary_path']}`",
            "",
            "## 结果判断",
            "",
            summary["quality_text"],
            "",
            "## 说明",
            "",
            "- `docs/results/` 里的 JSON 是模板文件，按帧保存，所以行数会比较多。",
            "- 日常查看请优先打开这份摘要，不需要直接翻长 JSON。",
        ]
    ) + "\n"


def append_results_log(summary: dict[str, object]) -> None:
    line = (
        f"- 保存时间：{summary['saved_at']} | 患者：`{summary['patient_id']}` | 动作：`{summary['action_name']}` | "
        f"帧数：`{summary['frame_count']}` | ROM：`{summary['rom_flexion']}` | "
        f"模板：`{Path(summary['saved_path']).name}` | 摘要：`{Path(summary['summary_path']).name}` | "
        f"板子：`{summary['board_ip']}`\n"
    )
    with RESULTS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)


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

        ensure_dirs()

        patient_id = safe_name(str(prescription.get("patient_id", "")), "patient")
        action_name = safe_name(str(prescription.get("action_name", "")), "action")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{patient_id}_{action_name}_{timestamp}.json"
        summary_name = f"{patient_id}_{action_name}_{timestamp}_summary.md"
        output_path = RESULTS_DIR / file_name
        summary_path = SUMMARIES_DIR / summary_name

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(prescription, handle, ensure_ascii=False, indent=2)

        clinical = prescription.get("clinical_baseline", {})
        runtime_meta = prescription.get("runtime_meta", {})
        summary = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": file_name,
            "saved_path": str(output_path),
            "summary_name": summary_name,
            "summary_path": str(summary_path),
            "patient_id": prescription.get("patient_id"),
            "action_name": prescription.get("action_name"),
            "frame_count": clinical.get("frame_count"),
            "duration_seconds": clinical.get("duration_seconds"),
            "min_knee_flexion_angle": clinical.get("min_knee_flexion_angle"),
            "max_knee_flexion_angle": clinical.get("max_knee_flexion_angle"),
            "rom_flexion": clinical.get("rom_flexion"),
            "invalid_frame_count": runtime_meta.get("invalid_frame_count"),
            "board_ip": payload.get("board_ip", "unknown"),
            "board_port": payload.get("board_port", "unknown"),
            "source": payload.get("source", "unknown"),
            "side_mode": runtime_meta.get("side_mode", "unknown"),
            "result_format": runtime_meta.get("result_format", "unknown"),
        }
        summary["quality_text"] = quality_text(summary["rom_flexion"])

        summary_path.write_text(render_summary_markdown(summary), encoding="utf-8")

        global latest_summary
        latest_summary = summary
        append_results_log(summary)

        make_json_response(
            self,
            {
                "ok": True,
                "saved_path": str(output_path),
                "summary_path": str(summary_path),
                "summary": summary,
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
