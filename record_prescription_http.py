from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from result_storage import save_prescription_artifacts


CAMERA_DEVICE = "/dev/video21"
CAMERA_BACKEND = cv2.CAP_V4L2
FOURCC = "MJPG"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
INFER_WIDTH = 640
INFER_HEIGHT = 360
JPEG_QUALITY = 70
PORT = 8082

VISIBILITY_THRESHOLD = 0.55
SMOOTH_WINDOW_SIZE = 5
PREFER_3D_WORLD_ANGLE = True
MODEL_COMPLEXITY = 1

SIDE_MODE_LABELS = {
    "auto": "自动",
    "left": "左腿",
    "right": "右腿",
}

ANGLE_SOURCE_LABELS = {
    "3d_world": "3D 世界坐标",
    "2d_image": "2D 图像坐标",
}

LEFT_KNEE_RULE = {
    "side": "left",
    "target_joint": "left_knee",
    "hip_index": 23,
    "knee_index": 25,
    "ankle_index": 27,
}

RIGHT_KNEE_RULE = {
    "side": "right",
    "target_joint": "right_knee",
    "hip_index": 24,
    "knee_index": 26,
    "ankle_index": 28,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calculate_angle(points: list[tuple[float, ...]]) -> float | None:
    if len(points) != 3:
        return None

    a, b, c = points
    if len(a) != len(b) or len(b) != len(c):
        return None

    ba = [a[i] - b[i] for i in range(len(a))]
    bc = [c[i] - b[i] for i in range(len(c))]

    dot_product = sum(ba[i] * bc[i] for i in range(len(ba)))
    ba_length = math.sqrt(sum(v * v for v in ba))
    bc_length = math.sqrt(sum(v * v for v in bc))

    if ba_length < 1e-8 or bc_length < 1e-8:
        return None

    cos_value = clamp(dot_product / (ba_length * bc_length), -1.0, 1.0)
    return math.degrees(math.acos(cos_value))


def knee_flexion_from_included_angle(included_angle: float | None) -> float | None:
    if included_angle is None:
        return None
    return clamp(180.0 - included_angle, 0.0, 180.0)


def get_landmark_tuple(landmarks, index: int, use_3d: bool = False) -> tuple[float, ...]:
    landmark = landmarks[index]
    if use_3d:
        return (landmark.x, landmark.y, landmark.z)
    return (landmark.x, landmark.y)


def get_visibility(landmarks, indices: list[int]) -> tuple[float, float]:
    values = [landmarks[i].visibility for i in indices]
    return min(values), sum(values) / len(values)


def compute_knee_angle(result, rule: dict[str, object]) -> dict[str, object]:
    if not result.pose_landmarks:
        return {"valid": False}

    image_landmarks = result.pose_landmarks.landmark
    indices = [rule["hip_index"], rule["knee_index"], rule["ankle_index"]]
    visibility_min, visibility_avg = get_visibility(image_landmarks, indices)

    included_2d = None
    flexion_2d = None
    included_3d = None
    flexion_3d = None

    if visibility_min >= VISIBILITY_THRESHOLD:
        points_2d = [
            get_landmark_tuple(image_landmarks, int(rule["hip_index"]), use_3d=False),
            get_landmark_tuple(image_landmarks, int(rule["knee_index"]), use_3d=False),
            get_landmark_tuple(image_landmarks, int(rule["ankle_index"]), use_3d=False),
        ]
        included_2d = calculate_angle(points_2d)
        flexion_2d = knee_flexion_from_included_angle(included_2d)

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks.landmark
            points_3d = [
                get_landmark_tuple(world_landmarks, int(rule["hip_index"]), use_3d=True),
                get_landmark_tuple(world_landmarks, int(rule["knee_index"]), use_3d=True),
                get_landmark_tuple(world_landmarks, int(rule["ankle_index"]), use_3d=True),
            ]
            included_3d = calculate_angle(points_3d)
            flexion_3d = knee_flexion_from_included_angle(included_3d)

    selected_source = None
    selected_included = None
    selected_flexion = None

    if PREFER_3D_WORLD_ANGLE and included_3d is not None:
        selected_source = "3d_world"
        selected_included = included_3d
        selected_flexion = flexion_3d
    elif included_2d is not None:
        selected_source = "2d_image"
        selected_included = included_2d
        selected_flexion = flexion_2d

    return {
        "valid": selected_flexion is not None,
        "side": rule["side"],
        "visibility_min": visibility_min,
        "visibility_avg": visibility_avg,
        "included_angle_2d": included_2d,
        "flexion_angle_2d": flexion_2d,
        "included_angle_3d": included_3d,
        "flexion_angle_3d": flexion_3d,
        "selected_included_angle": selected_included,
        "selected_flexion_angle": selected_flexion,
        "selected_source": selected_source,
    }


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.values = deque(maxlen=window_size)

    def update(self, value: float | None) -> float | None:
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self) -> None:
        self.values.clear()


def choose_knee_rule(mode: str, left_result: dict[str, object], right_result: dict[str, object]):
    if mode == "left":
        return LEFT_KNEE_RULE, left_result
    if mode == "right":
        return RIGHT_KNEE_RULE, right_result

    left_valid = bool(left_result.get("valid", False))
    right_valid = bool(right_result.get("valid", False))

    if left_valid and not right_valid:
        return LEFT_KNEE_RULE, left_result
    if right_valid and not left_valid:
        return RIGHT_KNEE_RULE, right_result
    if left_valid and right_valid:
        if right_result.get("visibility_avg", 0) > left_result.get("visibility_avg", 0):
            return RIGHT_KNEE_RULE, right_result
        return LEFT_KNEE_RULE, left_result

    return LEFT_KNEE_RULE, left_result


def build_compact_keypoints(landmarks, selected_rule: dict[str, object]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    points = {
        "hip": int(selected_rule["hip_index"]),
        "knee": int(selected_rule["knee_index"]),
        "ankle": int(selected_rule["ankle_index"]),
    }
    for name, index in points.items():
        landmark = landmarks[index]
        compact[name] = {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility,
        }
    return compact


def split_host_port(host_header: str) -> tuple[str, str]:
    text = host_header.strip()
    if not text:
        return "unknown", str(PORT)

    if text.startswith("[") and "]:" in text:
        host, port = text.rsplit(":", 1)
        return host.strip("[]"), port

    if text.count(":") == 1:
        host, port = text.rsplit(":", 1)
        return host, port

    return text.strip("[]"), str(PORT)


def build_prescription(
    patient_id: str,
    action_name: str,
    frames: list[dict[str, object]],
    knee_rule: dict[str, object],
    meta: dict[str, object],
) -> dict[str, object]:
    selected_flexion_angles = [
        frame["selected_flexion_angle_smoothed"]
        for frame in frames
        if frame.get("selected_flexion_angle_smoothed") is not None
    ]
    selected_included_angles = [
        frame["selected_included_angle"]
        for frame in frames
        if frame.get("selected_included_angle") is not None
    ]

    if len(frames) >= 2:
        duration_seconds = frames[-1]["relative_time"] - frames[0]["relative_time"]
    else:
        duration_seconds = 0.0

    return {
        "patient_id": patient_id,
        "action_name": action_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "这是一次根据标准膝关节屈曲动作录制生成的个体化处方结果。",
        "camera_instruction": "录制时请尽量侧身面对摄像头，保证髋、膝、踝三个关键点持续清晰可见；如果自动选腿不稳定，请改用左腿或右腿固定模式。",
        "algorithm_note": {
            "included_angle_meaning": "腿伸直时夹角接近 180 度；膝盖弯曲时夹角会变小。",
            "flexion_angle_meaning": "屈曲角 = 180 - 夹角。腿伸直时接近 0 度；弯曲越大，数值越大。",
            "angle_source_priority": "优先使用 MediaPipe 的 3D world landmarks，不可用时回退到 2D 图像 landmarks。",
            "smoothing": f"平滑窗口 = {SMOOTH_WINDOW_SIZE} 帧。",
            "warning": "单目 MediaPipe 角度适合演示和趋势反馈，不属于临床级测量。",
        },
        "runtime_meta": meta,
        "keypoint_rule": knee_rule,
        "clinical_baseline": {
            "frame_count": len(frames),
            "duration_seconds": duration_seconds,
            "min_selected_included_angle": min(selected_included_angles) if selected_included_angles else None,
            "max_selected_included_angle": max(selected_included_angles) if selected_included_angles else None,
            "min_knee_flexion_angle": min(selected_flexion_angles) if selected_flexion_angles else None,
            "max_knee_flexion_angle": max(selected_flexion_angles) if selected_flexion_angles else None,
            "rom_flexion": max(selected_flexion_angles) - min(selected_flexion_angles) if selected_flexion_angles else None,
        },
        "template_frames": frames,
    }


def sanitize_text(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(21, CAMERA_BACKEND)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


class RecorderState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.running = True
        self.frame_id = 0
        self.jpg_bytes: bytes | None = None
        self.last_status = "等待首帧画面"

        self.is_recording = False
        self.patient_id = "patient_001"
        self.action_name = "knee_flexion"
        self.side_mode = "auto"
        self.start_time: float | None = None
        self.frames: list[dict[str, object]] = []
        self.frame_index = 0
        self.invalid_frame_count = 0
        self.selected_rule_at_recording: dict[str, object] | None = None
        self.smoother = MovingAverage(SMOOTH_WINDOW_SIZE)

        self.selected_result: dict[str, object] = {"valid": False}
        self.selected_rule: dict[str, object] = LEFT_KNEE_RULE
        self.current_rom: float | None = None

        self.last_export_payload: dict[str, object] | None = None
        self.last_export_summary: dict[str, object] | None = None
        self.last_export_board_result: dict[str, object] | None = None
        self.last_export_error: str | None = None
        self.awaiting_ack = False

    def reset_recording(self) -> None:
        self.frames = []
        self.frame_index = 0
        self.invalid_frame_count = 0
        self.selected_rule_at_recording = None
        self.current_rom = None
        self.start_time = None
        self.is_recording = False
        self.smoother.clear()

    def clear_export(self) -> None:
        self.last_export_payload = None
        self.last_export_summary = None
        self.last_export_board_result = None
        self.last_export_error = None
        self.awaiting_ack = False

    def clear_all(self) -> None:
        self.reset_recording()
        self.clear_export()

    def update_frame(self, jpg_bytes: bytes, status: str) -> None:
        with self.condition:
            self.frame_id += 1
            self.jpg_bytes = jpg_bytes
            self.last_status = status
            self.condition.notify_all()

    def snapshot_status(self) -> dict[str, object]:
        smoothed = self.smoother.values[-1] if self.smoother.values else None
        return {
            "recording": self.is_recording,
            "patient_id": self.patient_id,
            "action_name": self.action_name,
            "side_mode": self.side_mode,
            "side_mode_label": SIDE_MODE_LABELS.get(self.side_mode, self.side_mode),
            "valid_frames": len(self.frames),
            "invalid_frames": self.invalid_frame_count,
            "selected_side": self.selected_rule.get("side"),
            "selected_side_label": SIDE_MODE_LABELS.get(str(self.selected_rule.get("side")), self.selected_rule.get("side")),
            "selected_source": self.selected_result.get("selected_source"),
            "selected_source_label": ANGLE_SOURCE_LABELS.get(str(self.selected_result.get("selected_source")), self.selected_result.get("selected_source")),
            "selected_flexion_angle": self.selected_result.get("selected_flexion_angle"),
            "smoothed_flexion_angle": smoothed,
            "visibility_min": self.selected_result.get("visibility_min"),
            "visibility_avg": self.selected_result.get("visibility_avg"),
            "current_rom": self.current_rom,
            "pending_export": self.last_export_payload is not None,
            "awaiting_ack": self.awaiting_ack,
            "last_export_error": self.last_export_error,
            "status": self.last_status,
        }


state = RecorderState()
cap = open_camera()

if not cap.isOpened():
    raise SystemExit(f"无法打开摄像头: {CAMERA_DEVICE}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def make_json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status_code: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def build_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RK3588 处方录制（双端同步保存）</title>
  <style>
    :root {
      --bg: #f2efe7;
      --panel: #fffdf8;
      --ink: #1b1b1b;
      --muted: #5a5a5a;
      --line: #d4cfc3;
      --accent: #146356;
      --warn: #b85c38;
      --ok: #2b6f3e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: linear-gradient(135deg, #ede8dc 0%, #f8f5ee 40%, #e7efe8 100%);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 20px;
    }
    .hero, .panel {
      background: rgba(255, 253, 248, 0.95);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 40px rgba(35, 35, 35, 0.08);
    }
    .hero { padding: 20px 24px; }
    .hero h1 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
    }
    .panel { padding: 18px; }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 18px;
    }
    img.stream {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
      display: block;
    }
    form {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 14px;
      color: var(--muted);
    }
    input, select, button {
      font: inherit;
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 11px 14px;
      cursor: pointer;
      color: white;
      background: var(--accent);
    }
    button.alt { background: #5c6b73; }
    button.warn { background: var(--warn); }
    button.ok { background: var(--ok); }
    button:disabled {
      opacity: 0.65;
      cursor: not-allowed;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
    }
    .stat b {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.04em;
    }
    .message {
      min-height: 72px;
      border-radius: 12px;
      padding: 12px 14px;
      background: #f7f3ea;
      border: 1px solid var(--line);
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .mono {
      font-family: Consolas, "Courier New", monospace;
      word-break: break-all;
    }
    .hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    @media (max-width: 920px) {
      .grid { grid-template-columns: 1fr; }
      .buttons, .status-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>RK3588 浏览器版处方录制（双端同步保存）</h1>
      <p>视频画面只保留原始图像和骨架，不再在画面里叠加文字。所有状态统一看右侧中文面板；点击保存后，板端会先写入 <span class="mono">docs/results/</span> 和 <span class="mono">docs/summaries/</span>，再同步一份到 Windows 本机。</p>
    </section>
    <div class="grid">
      <section class="panel">
        <h2>实时预览</h2>
        <img class="stream" src="/stream.mjpg" alt="实时预览">
        <div class="hint">
          建议尽量侧身面对摄像头，保证髋、膝、踝三个关键点持续可见。<br>
          如果自动选腿不稳定，请改成固定的 <span class="mono">left</span> 或 <span class="mono">right</span> 模式。
        </div>
      </section>
      <section class="panel">
        <h2>录制控制</h2>
        <form id="record-form">
          <label>患者编号
            <input id="patient_id" name="patient_id" value="patient_001">
          </label>
          <label>动作名称
            <input id="action_name" name="action_name" value="knee_flexion">
          </label>
          <label>侧别模式
            <select id="side_mode" name="side_mode">
              <option value="auto">auto（自动）</option>
              <option value="left">left（左腿）</option>
              <option value="right">right（右腿）</option>
            </select>
          </label>
        </form>
        <div class="buttons">
          <button id="start-btn">开始录制</button>
          <button id="save-btn" class="ok">双端同步保存</button>
          <button id="retry-btn" class="alt">重试导出最近结果</button>
          <button id="clear-btn" class="alt">清空缓存</button>
          <button id="cancel-btn" class="warn">取消本轮录制</button>
        </div>
        <div style="margin-top:14px">
          <div class="message" id="message">等待操作。</div>
        </div>
        <div class="status-grid" id="status-grid"></div>
      </section>
    </div>
  </div>
  <script>
    const statusGrid = document.getElementById("status-grid");
    const messageBox = document.getElementById("message");
    const form = document.getElementById("record-form");
    const startBtn = document.getElementById("start-btn");
    const saveBtn = document.getElementById("save-btn");
    const retryBtn = document.getElementById("retry-btn");
    const clearBtn = document.getElementById("clear-btn");
    const cancelBtn = document.getElementById("cancel-btn");

    function formatNumber(value, unit = "") {
      if (value == null || Number.isNaN(Number(value))) {
        return "-";
      }
      return `${Number(value).toFixed(1)}${unit}`;
    }

    function setMessage(text) {
      messageBox.textContent = text;
    }

    function setBusy(isBusy) {
      [startBtn, saveBtn, retryBtn, clearBtn, cancelBtn].forEach((button) => {
        button.disabled = isBusy;
      });
    }

    function displayStatus(status) {
      if (status.recording) return "录制中";
      if (status.awaiting_ack) return "等待板端确认";
      if (status.pending_export) return "等待本机保存";
      return status.status || "等待操作";
    }

    function displaySaveState(status) {
      if (status.awaiting_ack) return "本机已保存，等待板端确认";
      if (status.pending_export) return "板端已生成结果，等待本机保存";
      return "无待处理导出";
    }

    function renderStatus(status) {
      const rows = [
        ["当前状态", displayStatus(status)],
        ["保存状态", displaySaveState(status)],
        ["患者编号", status.patient_id || "-"],
        ["动作名称", status.action_name || "-"],
        ["侧别模式", status.side_mode_label || status.side_mode || "-"],
        ["已录有效帧", String(status.valid_frames ?? "-")],
        ["无效帧", String(status.invalid_frames ?? "-")],
        ["当前选腿", status.selected_side_label || status.selected_side || "-"],
        ["角度来源", status.selected_source_label || status.selected_source || "-"],
        ["最低可见度", formatNumber(status.visibility_min)],
        ["平均可见度", formatNumber(status.visibility_avg)],
        ["当前屈曲角", formatNumber(status.selected_flexion_angle, " 度")],
        ["平滑屈曲角", formatNumber(status.smoothed_flexion_angle, " 度")],
        ["当前 ROM", formatNumber(status.current_rom, " 度")],
        ["待重试导出", status.pending_export ? "有" : "无"],
        ["最近错误", status.last_export_error || "-"],
      ];
      statusGrid.innerHTML = rows.map(([label, value]) => `
        <div class="stat">
          <b>${label}</b>
          <span>${value}</span>
        </div>
      `).join("");
    }

    async function getStatus() {
      try {
        const response = await fetch("/status");
        const status = await response.json();
        renderStatus(status);
      } catch (error) {
        renderStatus({});
        setMessage("状态拉取失败，请确认板端服务仍在运行。");
      }
    }

    function collectPayload() {
      const data = new FormData(form);
      return {
        patient_id: String(data.get("patient_id") || "").trim(),
        action_name: String(data.get("action_name") || "").trim(),
        side_mode: String(data.get("side_mode") || "auto").trim(),
      };
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `请求失败: ${response.status}`);
      }
      return data;
    }

    async function pushToLocalSink(payload) {
      const response = await fetch("http://127.0.0.1:8090/api/save_result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || `本地保存失败: ${response.status}`);
      }
      return data;
    }

    async function ackBoardSaved() {
      return await postJson("/api/ack_saved", {});
    }

    function successMessage(localResult, title) {
      const summaryPath = localResult.summary_path ? `\n摘要：${localResult.summary_path}` : "";
      return `${title}\n模板：${localResult.saved_path}${summaryPath}`;
    }

    startBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/start", collectPayload());
        setMessage(result.message || "已开始录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    clearBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/clear", { clear_export: true });
        setMessage(result.message || "已清空缓存。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    cancelBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const result = await postJson("/api/cancel", {});
        setMessage(result.message || "已取消本轮录制。");
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    saveBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const boardResult = await postJson("/api/save", {});
        try {
          const localResult = await pushToLocalSink({
            board_ip: window.location.hostname,
            board_port: window.location.port || "8082",
            source: "record_prescription_http",
            prescription: boardResult.prescription,
          });
          try {
            await ackBoardSaved();
            const boardSummaryPath = boardResult.board_summary_path ? `
板端摘要：${boardResult.board_summary_path}` : "";
            const localSummaryPath = localResult.summary_path ? `
Windows摘要：${localResult.summary_path}` : "";
            setMessage(`双端保存完成。
板端模板：${boardResult.board_saved_path || "-"}${boardSummaryPath}
Windows模板：${localResult.saved_path || "-"}${localSummaryPath}`);
          } catch (ackError) {
            setMessage(`Windows 已保存，但板端确认清理失败。
板端模板：${boardResult.board_saved_path || "-"}
Windows模板：${localResult.saved_path || "-"}
${String(ackError.message || ackError)}
板端仍保留最近一次导出，可以稍后重试。`);
          }
        } catch (error) {
          setMessage(`板端已保底落盘，但 Windows 同步失败。
板端模板：${boardResult.board_saved_path || "-"}
板端摘要：${boardResult.board_summary_path || "-"}
${String(error.message || error)}
请先确认 Windows 本机的 local_result_sink.py 已启动，然后点击“重试导出最近结果”。`);
        }
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    retryBtn.addEventListener("click", async () => {
      setBusy(true);
      try {
        const boardResult = await fetch("/api/export_last");
        const exportPayload = await boardResult.json();
        if (!boardResult.ok || exportPayload.ok === false) {
          throw new Error(exportPayload.error || "没有待导出的结果。");
        }
        const localResult = await pushToLocalSink({
          board_ip: window.location.hostname,
          board_port: window.location.port || "8082",
          source: "record_prescription_http_retry",
          prescription: exportPayload.prescription,
        });
        try {
          await ackBoardSaved();
          setMessage(successMessage(localResult, "重试成功。"));
        } catch (ackError) {
          setMessage(`${successMessage(localResult, "本机已保存，但板端确认失败。")}\n${String(ackError.message || ackError)}\n板端仍保留最近一次导出，可继续重试。`);
        }
      } catch (error) {
        setMessage(String(error.message || error));
      } finally {
        setBusy(false);
        getStatus();
      }
    });

    getStatus();
    setInterval(getStatus, 1000);
  </script>
</body>
</html>
"""


def pose_worker() -> None:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while state.running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            infer_frame = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))
            rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            output_frame = frame.copy()
            selected_result: dict[str, object] = {"valid": False}
            selected_rule = LEFT_KNEE_RULE

            if result.pose_landmarks:
                output_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in result.pose_landmarks.landmark:
                    output_landmarks.landmark.add(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility,
                    )
                mp_drawing.draw_landmarks(output_frame, output_landmarks, mp_pose.POSE_CONNECTIONS)

                left_result = compute_knee_angle(result, LEFT_KNEE_RULE)
                right_result = compute_knee_angle(result, RIGHT_KNEE_RULE)
                selected_rule, selected_result = choose_knee_rule(state.side_mode, left_result, right_result)

            raw_flexion = selected_result.get("selected_flexion_angle")
            smoothed_flexion = state.smoother.update(raw_flexion)

            if state.is_recording and state.selected_rule_at_recording is None:
                state.selected_rule_at_recording = selected_rule

            if state.is_recording:
                if selected_result.get("valid", False) and result.pose_landmarks:
                    now = time.time()
                    landmarks = result.pose_landmarks.landmark
                    frame_data = {
                        "frame_index": state.frame_index,
                        "relative_time": (now - state.start_time) if state.start_time is not None else 0,
                        "selected_side": selected_rule["side"],
                        "selected_source": selected_result.get("selected_source"),
                        "visibility_min": selected_result.get("visibility_min"),
                        "visibility_avg": selected_result.get("visibility_avg"),
                        "selected_included_angle": selected_result.get("selected_included_angle"),
                        "selected_flexion_angle_raw": raw_flexion,
                        "selected_flexion_angle_smoothed": smoothed_flexion,
                        "included_angle_2d": selected_result.get("included_angle_2d"),
                        "flexion_angle_2d": selected_result.get("flexion_angle_2d"),
                        "included_angle_3d": selected_result.get("included_angle_3d"),
                        "flexion_angle_3d": selected_result.get("flexion_angle_3d"),
                        "left_knee_angle": selected_result.get("selected_included_angle"),
                        "keypoints": build_compact_keypoints(landmarks, selected_rule),
                    }
                    state.frames.append(frame_data)
                    state.frame_index += 1
                else:
                    state.invalid_frame_count += 1

            recorded_angles = [
                frame_item["selected_flexion_angle_smoothed"]
                for frame_item in state.frames
                if frame_item.get("selected_flexion_angle_smoothed") is not None
            ]
            state.current_rom = max(recorded_angles) - min(recorded_angles) if recorded_angles else None

            state.selected_result = selected_result
            state.selected_rule = selected_rule

            if selected_result.get("valid", False):
                side = str(selected_result.get("side", ""))
                status = f"已检测到{SIDE_MODE_LABELS.get(side, side)}"
            elif state.awaiting_ack and state.last_export_payload is not None:
                status = "等待保存确认"
            elif state.is_recording:
                status = "录制中"
            else:
                status = "未检测到可靠关键点"

            ok, jpg = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                state.update_frame(jpg.tobytes(), status)
    finally:
        pose.close()


class PrescriptionHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            body = build_page_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/status":
            make_json_response(self, state.snapshot_status())
            return

        if parsed.path == "/api/export_last":
            if state.last_export_payload is None:
                make_json_response(self, {"ok": False, "error": "没有待重试导出的结果。"}, status_code=404)
                return
            response_payload = {
                "ok": True,
                "prescription": state.last_export_payload,
                "summary": state.last_export_summary,
            }
            if state.last_export_board_result is not None:
                response_payload["board_saved_path"] = state.last_export_board_result.get("saved_path")
                response_payload["board_summary_path"] = state.last_export_board_result.get("summary_path")
                response_payload["board_summary"] = state.last_export_board_result.get("summary")
            make_json_response(self, response_payload)
            return

        if parsed.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            last_sent_frame_id = -1
            while state.running:
                with state.condition:
                    state.condition.wait_for(lambda: state.frame_id != last_sent_frame_id or not state.running)
                    if not state.running:
                        break
                    last_sent_frame_id = state.frame_id
                    data = state.jpg_bytes

                if data is None:
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        try:
            payload = read_json_body(self)
        except json.JSONDecodeError:
            make_json_response(self, {"ok": False, "error": "请求体不是有效 JSON。"}, status_code=400)
            return

        if self.path == "/api/start":
            state.patient_id = sanitize_text(payload.get("patient_id"), "patient_001")
            state.action_name = sanitize_text(payload.get("action_name"), "knee_flexion")
            side_mode = sanitize_text(payload.get("side_mode"), "auto").lower()
            state.side_mode = side_mode if side_mode in {"left", "right", "auto"} else "auto"
            state.reset_recording()
            state.start_time = time.time()
            state.is_recording = True
            state.last_export_error = None
            make_json_response(self, {"ok": True, "message": "已开始录制，请在预览画面前完成标准动作。"})
            return

        if self.path == "/api/clear":
            clear_export = bool(payload.get("clear_export", False))
            state.reset_recording()
            if clear_export:
                state.clear_export()
            state.last_export_error = None
            make_json_response(self, {"ok": True, "message": "已清空录制缓存。"})
            return

        if self.path == "/api/cancel":
            state.clear_all()
            make_json_response(self, {"ok": True, "message": "已取消当前录制，并清空板端缓存。"})
            return

        if self.path == "/api/save":
            if not state.frames:
                state.last_export_error = "没有录到有效骨架数据，请重新录制。"
                make_json_response(self, {"ok": False, "error": state.last_export_error}, status_code=400)
                return

            state.is_recording = False
            meta = {
                "camera_device": CAMERA_DEVICE,
                "camera_backend": "cv2.CAP_V4L2",
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "infer_width": INFER_WIDTH,
                "infer_height": INFER_HEIGHT,
                "side_mode": state.side_mode,
                "prefer_3d_world_angle": PREFER_3D_WORLD_ANGLE,
                "model_complexity": MODEL_COMPLEXITY,
                "visibility_threshold": VISIBILITY_THRESHOLD,
                "smooth_window_size": SMOOTH_WINDOW_SIZE,
                "invalid_frame_count": state.invalid_frame_count,
                "result_format": "compact_v1",
            }
            prescription = build_prescription(
                state.patient_id,
                state.action_name,
                list(state.frames),
                state.selected_rule_at_recording or state.selected_rule,
                meta,
            )
            board_ip, board_port = split_host_port(self.headers.get("Host", ""))
            try:
                board_save_result = save_prescription_artifacts(
                    prescription,
                    board_ip=board_ip,
                    board_port=board_port,
                    source="record_prescription_http_board",
                )
            except OSError as error:
                state.last_export_error = f"?????????{error}"
                make_json_response(self, {"ok": False, "error": state.last_export_error}, status_code=500)
                return

            state.last_export_payload = prescription
            baseline = prescription["clinical_baseline"]
            state.last_export_summary = {
                "patient_id": prescription["patient_id"],
                "action_name": prescription["action_name"],
                "frame_count": baseline["frame_count"],
                "duration_seconds": baseline["duration_seconds"],
                "rom_flexion": baseline["rom_flexion"],
            }
            state.last_export_board_result = board_save_result
            state.last_export_error = None
            state.awaiting_ack = True
            make_json_response(
                self,
                {
                    "ok": True,
                    "prescription": prescription,
                    "summary": state.last_export_summary,
                    "board_saved_path": board_save_result["saved_path"],
                    "board_summary_path": board_save_result["summary_path"],
                    "board_summary": board_save_result["summary"],
                    "message": "???????????????????? Windows ???",
                },
            )
            return

        if self.path == "/api/ack_saved":
            if state.last_export_payload is None or not state.awaiting_ack:
                make_json_response(self, {"ok": False, "error": "当前没有待确认清理的导出结果。"}, status_code=400)
                return

            state.reset_recording()
            state.clear_export()
            make_json_response(self, {"ok": True, "message": "板端已确认本机保存成功，并清空本次录制缓存。"})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> None:
    worker = threading.Thread(target=pose_worker, daemon=True)
    worker.start()

    server = ThreadedHTTPServer(("0.0.0.0", PORT), PrescriptionHTTPHandler)

    print(f"8082 双端同步保存服务已启动: http://板子IP:{PORT}")
    print(f"摄像头设备: {CAMERA_DEVICE}")
    print(f"采集分辨率: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"推理分辨率: {INFER_WIDTH}x{INFER_HEIGHT}")
    print("说明: 保存时会先写入板端 docs/results/、docs/summaries/ 和 docs/results_log.md，再同步到 Windows。")
    print("提示: 如需 Windows 副本，请先在 Windows 本机启动 local_result_sink.py。")

    try:
        server.serve_forever()
    finally:
        state.running = False
        with state.condition:
            state.condition.notify_all()
        cap.release()


if __name__ == "__main__":
    main()
