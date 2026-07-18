"""NPU rehabilitation station on port 8085.

This entrypoint configures isolated NPU templates and reports plus the
maintained YOLOv5n raw + RTMPose RKNN backend.
"""

from __future__ import annotations

import os
import hashlib
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse


DIRECT_CAMERA_BY_ID = "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"

os.environ.setdefault("POSE_BACKEND", "rknn")
os.environ.setdefault("RKNN_POSE_PIPELINE", "yolov5n_rtmpose")
os.environ.setdefault("RKNN_DET_MODEL", "models/vision/yolov5n_raw_fp.rknn")
os.environ.setdefault("RKNN_RTMPOSE_MODEL", "models/vision/rtmpose_m_256x192_fp.rknn")
os.environ.setdefault("RKNN_CORE_MASK", "NPU_CORE_0_1_2")
os.environ.setdefault("RKNN_DET_CORE_MASK", os.environ["RKNN_CORE_MASK"])
os.environ.setdefault("RKNN_POSE_CORE_MASK", os.environ["RKNN_CORE_MASK"])
os.environ.setdefault("RKNN_POSE_KEYPOINT_THRES", "0.18")
os.environ.setdefault("RKNN_FIXED_LEG_VISIBILITY_THRESHOLD", "0.20")
os.environ.setdefault("RKNN_FIXED_DRAW_VISIBILITY_THRESHOLD", "0.20")
os.environ.setdefault("RKNN_STABILIZER_ALPHA", "0.55")
os.environ.setdefault("RKNN_STABILIZER_LOW_CONF_ALPHA", "0.32")
os.environ.setdefault("RKNN_STABILIZER_JUMP_SCALE", "0.55")
os.environ.setdefault("RKNN_STABILIZER_MAX_HOLD_FRAMES", "8")
os.environ.setdefault("RKNN_STABILIZER_LOCK_CONFIRM_FRAMES", "4")
os.environ.setdefault("RKNN_DISPLAY_ALPHA", "0.50")
os.environ.setdefault("RKNN_DISPLAY_LOW_CONF_ALPHA", "0.30")
os.environ.setdefault("RKNN_DISPLAY_JUMP_SCALE", "0.35")
os.environ.setdefault("RKNN_DISPLAY_MAX_HOLD_FRAMES", "4")
os.environ.setdefault("RKNN_DISPLAY_JUMP_CONFIRM_FRAMES", "2")
os.environ.setdefault("RKNN_DISPLAY_BBOX_ALPHA", "0.45")
os.environ.setdefault("RKNN_DISPLAY_BBOX_HOLD_FRAMES", "6")
os.environ.setdefault("RKNN_DISPLAY_HOLD_SECONDS", "0.25")
os.environ.setdefault("RKNN_DISPLAY_BBOX_HOLD_SECONDS", "0.35")
os.environ.setdefault("RKNN_DISPLAY_JUMP_CONFIRM_SECONDS", "0.20")
os.environ.setdefault("RKNN_DISPLAY_DISAPPEAR_RATIO", "0.65")
os.environ.setdefault("RKNN_DISPLAY_BBOX_IOU_JUMP", "0.35")
os.environ.setdefault("RKNN_DISPLAY_MAX_STALE_SECONDS", "0.50")
os.environ.setdefault("RKNN_RTMPOSE_DRAW", "0")
os.environ.setdefault("RKNN_YOLOV5_INPUT_LAYOUT", "nhwc")
os.environ.setdefault("RKNN_RTMPOSE_INPUT_LAYOUT", "nhwc")
os.environ["RK_CAMERA_SOURCE"] = "device"
os.environ.setdefault("RK_CAMERA_DEVICE", DIRECT_CAMERA_BY_ID)
os.environ.setdefault("RK_CAMERA_OPEN_MODE", "auto")
os.environ.setdefault("RK_CAMERA_GST_FORMAT", "MJPG")
os.environ.setdefault("RK_CAMERA_GST_JPEG_DECODER", "auto")
os.environ.setdefault("RK_CAMERA_GST_BACKEND", "gi")
os.environ.pop("RK_CAMERA_STREAM_URL", None)
os.environ.pop("RTM_POSE_STREAM_URL", None)
os.environ.setdefault("RK_CAMERA_WIDTH", "1280")
os.environ.setdefault("RK_CAMERA_HEIGHT", "720")
os.environ.setdefault("RK_CAMERA_FIXED_FPS", "1")
os.environ.setdefault("RKNN_PROCESS_WIDTH", "1280")
os.environ.setdefault("RKNN_PROCESS_HEIGHT", "720")
os.environ.setdefault("RKNN_STREAM_WIDTH", "960")
os.environ.setdefault("RKNN_STREAM_HEIGHT", "540")
os.environ.setdefault("RKNN_STREAM_FPS", "20")
os.environ.setdefault("RKNN_DIAGNOSTIC_SAMPLE_INTERVAL", "5")
os.environ.setdefault("RKNN_FAST_PREVIEW", "1")
os.environ.setdefault("RKNN_FAST_FRAME_DATA", "1")
os.environ.setdefault("REHAB_KEYFRAME_EVERY_N", "8")
os.environ.setdefault("RKNN_DET_INTERVAL", "3")
os.environ.setdefault("RKNN_DET_CACHE_SECONDS", "1.5")
os.environ.setdefault("RKNN_ADAPTIVE_DETECTOR", "1")
os.environ.setdefault("RKNN_DET_REFRESH_SECONDS", "0.75")
os.environ.setdefault("RKNN_DET_RETRY_SECONDS", "0.25")
os.environ.setdefault("RKNN_DET_BAD_POSE_FRAMES", "2")
os.environ.setdefault("RKNN_TRACKER_MARGIN", "0.20")
os.environ.setdefault("RKNN_TRACKER_ALPHA", "0.35")
os.environ.setdefault("RKNN_TRACKER_MIN_POINTS", "5")
os.environ.setdefault("RKNN_MAX_POSE_PADDING_RATIO", "0.55")
os.environ.setdefault("RKNN_ASYNC_PIPELINE", "1")
os.environ.setdefault("REHAB_SERVICE_MODE", "npu_rehab")
# Draw the stabilized rehab skeleton in the shared app instead of the raw
# backend output, otherwise one weak RTMPose frame makes the overlay flash.
os.environ.setdefault("RKNN_YOLOV5_BACKEND_DRAW", "0")
os.environ.setdefault("RKNN_YOLOV5_PERSON_ONLY_FAST", "1")
os.environ.setdefault("RKNN_RTMPOSE_DEBUG_CROP_EVERY", "0")
os.environ.setdefault("RK_JPEG_QUALITY", "72")
os.environ.setdefault("RKNN_RTMPOSE_WIDE_BBOX_RATIO", "0.65")
os.environ.setdefault("RKNN_RTMPOSE_WIDE_BBOX_EXPAND", "1.50")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rehab_app.server import rehab_http_server as app  # noqa: E402
from rehab_app.services import result_storage  # noqa: E402
from training import training_session as training_module  # noqa: E402


PORT = 8085
NPU_RUNTIME_DIR = PROJECT_ROOT / "runtime" / "npu"
NPU_DOCS_DIR = PROJECT_ROOT / "data" / "npu"
NPU_REPORTS_DIR = PROJECT_ROOT / "data" / "reports" / "npu"
NPU_PLAN = PROJECT_ROOT / "training" / "configs" / "rehab_demo_plan_npu.yaml"
NPU_REALTIME_CONFIG = PROJECT_ROOT / "training" / "configs" / "training_defaults_npu.yaml"
RUNTIME_STARTED_AT = time.time()
RUNTIME_SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "rehab_app" / "server" / "rehab_http_server.py",
    PROJECT_ROOT / "pose_estimation" / "rknn_pose" / "yolov5n_rtmpose_backend.py",
    PROJECT_ROOT / "pose_estimation" / "rknn_pose" / "pose_frame_adapter.py",
    PROJECT_ROOT / "training" / "training_session.py",
)


def _source_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


RUNTIME_SOURCE_HASHES = {str(path.relative_to(PROJECT_ROOT)): _source_hash(path) for path in RUNTIME_SOURCE_FILES}
RUNTIME_BUILD_ID = hashlib.sha256(
    "|".join(f"{name}:{value}" for name, value in sorted(RUNTIME_SOURCE_HASHES.items())).encode("utf-8")
).hexdigest()[:16]

NPU_CONFIG_BY_ACTION = {
    "sit_to_stand": "evaluation/configs/npu/sit_to_stand.yaml",
    "standing_hamstring_curl": "evaluation/configs/npu/standing_hamstring_curl.yaml",
    "seated_knee_raise": "evaluation/configs/npu/seated_knee_raise.yaml",
}

NPU_DEBUG_HEARTBEAT_SECONDS = 5.0
NPU_DEBUG_LEASE_SECONDS = 15.0
_npu_debug_lock = threading.RLock()
_npu_debug_enabled = False
_npu_debug_last_heartbeat_monotonic: float | None = None
_npu_debug_last_heartbeat_at: float | None = None
_npu_debug_last_error: str | None = None


def business_pose_active() -> bool:
    recording = bool(app.state.is_recording)
    status = str(app.realtime_session.snapshot().get("status") or "idle")
    return recording or status in app.ACTIVE_REALTIME_STATUSES


def npu_debug_active() -> bool:
    global _npu_debug_enabled
    with _npu_debug_lock:
        if not _npu_debug_enabled:
            return False
        now = time.monotonic()
        if _npu_debug_last_heartbeat_monotonic is None or now - _npu_debug_last_heartbeat_monotonic > NPU_DEBUG_LEASE_SECONDS:
            _npu_debug_enabled = False
            return False
        return True


def npu_debug_snapshot() -> dict[str, object]:
    active = npu_debug_active()
    with _npu_debug_lock:
        expires_at = (
            _npu_debug_last_heartbeat_at + NPU_DEBUG_LEASE_SECONDS
            if active and _npu_debug_last_heartbeat_at is not None
            else None
        )
        return {
            "active": active,
            "heartbeat_seconds": NPU_DEBUG_HEARTBEAT_SECONDS,
            "lease_seconds": NPU_DEBUG_LEASE_SECONDS,
            "lease_expires_at": expires_at,
            "last_heartbeat_at": _npu_debug_last_heartbeat_at,
            "last_error": _npu_debug_last_error,
        }


def start_npu_debug() -> dict[str, object]:
    global _npu_debug_enabled, _npu_debug_last_heartbeat_monotonic, _npu_debug_last_heartbeat_at, _npu_debug_last_error
    if qwen_busy():
        with _npu_debug_lock:
            _npu_debug_last_error = "小爱正在生成回答，请等待回答结束。"
        return {"ok": False, "error": _npu_debug_last_error}
    if business_pose_active():
        with _npu_debug_lock:
            _npu_debug_last_error = "医生录入或患者训练正在使用姿态模型。"
        return {"ok": False, "error": _npu_debug_last_error}
    now_monotonic = time.monotonic()
    now_wall = time.time()
    with _npu_debug_lock:
        _npu_debug_enabled = True
        _npu_debug_last_heartbeat_monotonic = now_monotonic
        _npu_debug_last_heartbeat_at = now_wall
        _npu_debug_last_error = None
    return {"ok": True, "npu_debug": npu_debug_snapshot()}


def heartbeat_npu_debug() -> dict[str, object]:
    global _npu_debug_last_heartbeat_monotonic, _npu_debug_last_heartbeat_at
    with _npu_debug_lock:
        if not _npu_debug_enabled:
            return {"ok": False, "error": "NPU 调试未启动。"}
        _npu_debug_last_heartbeat_monotonic = time.monotonic()
        _npu_debug_last_heartbeat_at = time.time()
    return {"ok": True, "npu_debug": npu_debug_snapshot()}


def stop_npu_debug(*, release: bool = True) -> dict[str, object]:
    global _npu_debug_enabled
    with _npu_debug_lock:
        _npu_debug_enabled = False
    if release and not business_pose_active() and app.rknn_backend is not None:
        app.rknn_backend.release()
    return {"ok": True, "npu_debug": npu_debug_snapshot()}


def npu_debug_watchdog() -> None:
    was_active = False
    while app.state.running:
        active = npu_debug_active()
        if was_active and not active and not business_pose_active() and app.rknn_backend is not None:
            app.rknn_backend.release()
        was_active = active
        time.sleep(1.0)


def save_npu_artifacts(prescription: dict[str, object], **kwargs):
    kwargs["board_port"] = str(PORT)
    kwargs["docs_dir"] = NPU_DOCS_DIR
    return result_storage.save_prescription_artifacts(prescription, **kwargs)


def qwen_busy() -> bool:
    return app.assistant_interaction_busy()


def release_pose_for_qwen() -> None:
    if pose_requested():
        return
    if app.rknn_backend is not None:
        app.rknn_backend.release()


def reset_npu_tracking_state() -> None:
    reset_npu_tracking_state_for("manual")


def reset_npu_tracking_state_for(reason: str) -> None:
    if app.rknn_backend is not None:
        reset_backend = getattr(app.rknn_backend, "reset_tracking_state", None)
        if callable(reset_backend):
            reset_backend(reason)
    if app.rknn_pose_stabilizer is not None:
        app.rknn_pose_stabilizer.reset()
    if app.rknn_display_stabilizer is not None:
        app.rknn_display_stabilizer.reset()
    if app.StablePersonSelector is not None:
        app.rknn_person_selector = app.StablePersonSelector()


class NpuRealtimeTrainingSession(training_module.RealtimeTrainingSession):
    def reset(self) -> None:
        super().reset()
        self.pose_backend = "rknn"

    def _training_logic_version(self) -> str:
        return "npu_training_v8_stage2_pipeline"

    def _qwen_block(self) -> dict[str, object] | None:
        if qwen_busy():
            return {"ok": False, "error": "小爱正在生成回答，请等待回答结束后再开始训练。"}
        return None

    def start(self, *args, **kwargs):
        block = self._qwen_block()
        if block:
            return block
        reset_npu_tracking_state()
        return super().start(*args, **kwargs)

    def start_playlist(self, *args, **kwargs):
        block = self._qwen_block()
        if block:
            return block
        reset_npu_tracking_state()
        return super().start_playlist(*args, **kwargs)

    def stop(self):
        result = super().stop()
        reset_npu_tracking_state()
        release_pose_for_qwen()
        return result

    def _start_playlist_action(self, index: int) -> None:
        previous_generation = self.action_generation
        super()._start_playlist_action(index)
        if self.action_generation != previous_generation:
            reset_npu_tracking_state_for(f"action_transition:{self.action_id}")

    def _enter_offscreen_wait(self) -> None:
        super()._enter_offscreen_wait()
        reset_npu_tracking_state_for("offscreen_wait")

    def _resume_running_after_offscreen(self) -> None:
        was_awaiting_return = self.status == "awaiting_return"
        super()._resume_running_after_offscreen()
        if was_awaiting_return and self.status != "awaiting_return":
            reset_npu_tracking_state_for("offscreen_reentry")

    def _complete_training(self) -> None:
        super()._complete_training()
        if self.status in {"completed", "error", "idle"}:
            reset_npu_tracking_state()
            release_pose_for_qwen()

    def _advance_after_rest(self) -> None:
        super()._advance_after_rest()
        if self.status == "completed":
            release_pose_for_qwen()


def configure_isolated_runtime() -> None:
    NPU_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    NPU_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    NPU_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    app.PORT = PORT
    app.RUNTIME_DIR = NPU_RUNTIME_DIR
    app.VOICE_RUNTIME_DIR = NPU_RUNTIME_DIR / "voice"
    app.ACTIVE_TEMPLATES_PATH = NPU_RUNTIME_DIR / "active_templates.json"
    app.EVALUATE_REPORTS_DIR = NPU_REPORTS_DIR
    app.LEGACY_RESULTS_DIR = NPU_DOCS_DIR / "results"
    app.DEFAULT_CONFIG_BY_ACTION = dict(NPU_CONFIG_BY_ACTION)
    app.RTMPOSE_PIPELINES = set(app.RTMPOSE_PIPELINES) | {
        "yolov5n_rtmpose",
        "yolov5n_raw_rtmpose",
        "yolov5n_nonms_rtmpose",
    }
    app.save_prescription_artifacts = save_npu_artifacts

    original_shell = app.build_app_shell

    def build_npu_app_shell(page_id: str, title: str, description: str) -> str:
        html = original_shell(page_id, title, description)
        return html.replace("</body>", '<script src="/assets/npu_status.js"></script>\n</body>')

    app.build_app_shell = build_npu_app_shell

    training_module.ACTIVE_TEMPLATES_PATH = app.ACTIVE_TEMPLATES_PATH
    training_module.DEFAULT_DEMO_PLAN = NPU_PLAN
    training_module.DEFAULT_REALTIME_CONFIG = NPU_REALTIME_CONFIG
    training_module.REPORTS_DIR = NPU_REPORTS_DIR
    training_module.KEYFRAMES_DIR = NPU_REPORTS_DIR / "keyframes"
    training_module.save_prescription_artifacts = save_npu_artifacts

    app.realtime_session = NpuRealtimeTrainingSession(NPU_REALTIME_CONFIG)
    app.realtime_session.set_keyframe_job_sink(app.enqueue_keyframe_job)
    if app.rknn_backend is not None:
        app.rknn_backend.set_enabled_fn(pose_requested)

    original_snapshot = app.RecorderState.snapshot_status

    def snapshot_status(self, include_heavy: bool = False):
        payload = original_snapshot(self, include_heavy=include_heavy)
        payload["service_port"] = PORT
        payload["service_mode"] = "npu_rehab"
        payload["npu_resource"] = (
            app.rknn_backend.resource_snapshot()
            if app.rknn_backend is not None
            else {"state": "error", "owner": None, "models_loaded": False, "last_error": "RKNN backend unavailable"}
        )
        payload["npu_debug"] = npu_debug_snapshot()
        payload["npu_pose_debug"] = app.rknn_backend.diagnostics_snapshot() if app.rknn_backend is not None else {}
        payload["camera_source"] = {
            "kind": "direct_device",
            "requested_device": app.CAMERA_DEVICE,
            "active_device": app.active_camera_device,
            "open_mode": app.active_camera_open_mode,
            "requested_resolution": [app.FRAME_WIDTH, app.FRAME_HEIGHT],
            "actual_resolution": [app.active_camera_actual_width, app.active_camera_actual_height],
            "actual_fps": app.active_camera_actual_fps,
            "uses_8082_stream": False,
        }
        payload["performance_profile"] = {
            "name": "npu_hd_low_latency",
            "fast_preview": bool(app.RKNN_FAST_PREVIEW),
            "fast_frame_data": bool(app.RKNN_FAST_FRAME_DATA),
            "camera_open_mode_requested": app.CAMERA_OPEN_MODE,
            "camera_open_mode_active": app.active_camera_open_mode,
            "camera_gst_format": app.CAMERA_GST_FORMAT,
            "camera_gst_jpeg_decoder": app.CAMERA_GST_JPEG_DECODER,
            "keyframe_every_n": app.REHAB_KEYFRAME_EVERY_N,
            "det_interval": int(os.environ.get("RKNN_DET_INTERVAL", "1")),
            "det_cache_seconds": float(os.environ.get("RKNN_DET_CACHE_SECONDS", "0.5")),
            "backend_draw_enabled": os.environ.get("RKNN_YOLOV5_BACKEND_DRAW", "1") != "0",
            "person_only_fast": os.environ.get("RKNN_YOLOV5_PERSON_ONLY_FAST", "0") == "1",
            "debug_crop_every": int(os.environ.get("RKNN_RTMPOSE_DEBUG_CROP_EVERY", "0")),
            "jpeg_quality": app.JPEG_QUALITY,
            "process_resolution": [app.RKNN_PROCESS_WIDTH, app.RKNN_PROCESS_HEIGHT],
            "stream_resolution": [app.RKNN_STREAM_WIDTH, app.RKNN_STREAM_HEIGHT],
            "stream_fps_limit": app.RKNN_STREAM_FPS,
            "display_stabilizer": {
                "alpha": float(os.environ.get("RKNN_DISPLAY_ALPHA", "0.50")),
                "low_conf_alpha": float(os.environ.get("RKNN_DISPLAY_LOW_CONF_ALPHA", "0.30")),
                "max_hold_frames": int(os.environ.get("RKNN_DISPLAY_MAX_HOLD_FRAMES", "4")),
                "bbox_hold_frames": int(os.environ.get("RKNN_DISPLAY_BBOX_HOLD_FRAMES", "6")),
                "hold_seconds": float(os.environ.get("RKNN_DISPLAY_HOLD_SECONDS", "0.25")),
                "bbox_hold_seconds": float(os.environ.get("RKNN_DISPLAY_BBOX_HOLD_SECONDS", "0.35")),
                "jump_confirm_seconds": float(os.environ.get("RKNN_DISPLAY_JUMP_CONFIRM_SECONDS", "0.20")),
                "disappear_ratio": float(os.environ.get("RKNN_DISPLAY_DISAPPEAR_RATIO", "0.65")),
                "bbox_iou_jump": float(os.environ.get("RKNN_DISPLAY_BBOX_IOU_JUMP", "0.35")),
                "max_stale_seconds": float(os.environ.get("RKNN_DISPLAY_MAX_STALE_SECONDS", "0.50")),
            },
            "async_pipeline": app.rknn_async_pipeline_enabled(),
            "adaptive_detector": os.environ.get("RKNN_ADAPTIVE_DETECTOR", "0") == "1",
            "det_refresh_seconds": float(os.environ.get("RKNN_DET_REFRESH_SECONDS", "0.75")),
            "det_retry_seconds": float(os.environ.get("RKNN_DET_RETRY_SECONDS", "0.25")),
        }
        payload["runtime"] = {
            "project_root": str(PROJECT_ROOT),
            "entrypoint": str(Path(__file__).resolve()),
            "pid": os.getpid(),
            "started_at": RUNTIME_STARTED_AT,
            "build_id": RUNTIME_BUILD_ID,
            "source_hashes": dict(RUNTIME_SOURCE_HASHES),
            "active_environment": {
                name: os.environ.get(name)
                for name in (
                    "POSE_BACKEND",
                    "RKNN_POSE_PIPELINE",
                    "RKNN_ASYNC_PIPELINE",
                    "REHAB_SERVICE_MODE",
                    "REHAB_AUDIO_OUTPUT_DEVICE",
                    "RKNN_ADAPTIVE_DETECTOR",
                    "RKNN_DET_REFRESH_SECONDS",
                    "RKNN_DET_RETRY_SECONDS",
                    "RKNN_DET_CACHE_SECONDS",
                    "RKNN_DET_SCORE_THRES",
                    "RKNN_CORE_MASK",
                    "RKNN_DET_CORE_MASK",
                    "RKNN_POSE_CORE_MASK",
                    "RK_CAMERA_OPEN_MODE",
                    "RK_CAMERA_GST_BACKEND",
                    "RK_CAMERA_WIDTH",
                    "RK_CAMERA_HEIGHT",
                    "RK_CAMERA_FIXED_FPS",
                    "RKNN_PROCESS_WIDTH",
                    "RKNN_PROCESS_HEIGHT",
                    "RKNN_STREAM_WIDTH",
                    "RKNN_STREAM_HEIGHT",
                    "RKNN_STREAM_FPS",
                    "RK_JPEG_QUALITY",
                )
            },
        }
        return payload

    app.RecorderState.snapshot_status = snapshot_status


def pose_requested() -> bool:
    return business_pose_active() or npu_debug_active()


def build_npu_debug_page() -> bytes:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>NPU 8085 姿态检测调试</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; background:#06111d; color:#e9f5ff; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; padding:18px; }}
    header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:14px; }}
    h1 {{ margin:0; font-size:24px; }} .muted {{ color:#8facbf; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; }}
    button,a {{ border:1px solid #2f6e91; background:#10324a; color:#e9f5ff; padding:10px 16px; border-radius:10px; text-decoration:none; cursor:pointer; }}
    button.primary {{ background:#087c65; border-color:#20c99b; }} button.stop {{ background:#6f2430; border-color:#e25a6d; }}
    .layout {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(310px,1fr); gap:14px; }}
    .panel {{ background:#0a1b2a; border:1px solid #173e55; border-radius:14px; padding:12px; min-width:0; }}
    .stream {{ width:100%; min-height:360px; max-height:76vh; object-fit:contain; background:#020812; border-radius:10px; }}
    .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#263849; }}
    .pill.good {{ background:#075c4b; }} .pill.warn {{ background:#6c4a10; }} .pill.bad {{ background:#722d36; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }}
    .card {{ background:#0d2638; border-radius:9px; padding:8px; overflow-wrap:anywhere; }}
    .card span {{ display:block; color:#8facbf; font-size:12px; }} pre {{ white-space:pre-wrap; word-break:break-word; max-height:32vh; overflow:auto; font-size:12px; }}
    #message {{ min-height:24px; margin:8px 0; color:#ffd67a; }}
    @media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .stream {{ min-height:260px; }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>NPU 8085 姿态检测调试</h1><div class="muted">YOLOv5n raw-head + RTMPose COCO-17</div></div>
    <div class="actions"><a href="/train">患者训练</a><a href="/doctor">医生录入</a></div>
  </header>
  <div class="actions">
    <button id="start" class="primary">开始 NPU 检测</button>
    <button id="stop" class="stop">停止并释放 NPU</button>
    <span id="state" class="pill">等待状态</span>
  </div>
  <div id="message">页面打开不会自动占用 NPU，请点击“开始 NPU 检测”。</div>
  <main class="layout">
    <section class="panel"><img class="stream" src="/stream.mjpg" alt="NPU pose stream"></section>
    <aside class="panel">
      <div id="metrics" class="grid"></div>
      <h3>原始调试状态</h3><pre id="raw">等待 /status</pre>
    </aside>
  </main>
  <script>
    const stateNode = document.getElementById('state');
    const messageNode = document.getElementById('message');
    const metricsNode = document.getElementById('metrics');
    const rawNode = document.getElementById('raw');
    let debugActive = false;
    function value(v) {{ return v === null || v === undefined || v === '' ? '-' : (typeof v === 'object' ? JSON.stringify(v) : String(v)); }}
    function card(label, v) {{ return `<div class="card"><span>${{label}}</span>${{value(v)}}</div>`; }}
    async function post(path, keepalive=false) {{
      const response = await fetch(path, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:'{{}}', keepalive}});
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${{response.status}}`);
      return data;
    }}
    async function refresh() {{
      try {{
        const [status, system] = await Promise.all([
          fetch('/status', {{cache:'no-store'}}).then(r => r.json()),
          fetch('/api/system/status', {{cache:'no-store'}}).then(r => r.json()).catch(() => ({{}})),
        ]);
        const debug = status.npu_debug || {{}};
        const resource = status.npu_resource || {{}};
        const diag = status.npu_pose_debug || {{}};
        const perf = diag.performance_ms || {{}};
        const npu = system.npu || {{}};
        debugActive = Boolean(debug.active);
        stateNode.textContent = debugActive ? 'NPU 检测中' : resource.models_loaded ? 'NPU 正在释放' : 'NPU 已释放';
        stateNode.className = `pill ${{debugActive ? 'good' : resource.last_error ? 'bad' : 'warn'}}`;
        metricsNode.innerHTML = [
          card('摄像头', status.camera_live_ok ? '正常' : '异常'),
          card('Pose FPS', status.pose_fps),
          card('人数', (diag.detections || []).length),
          card('关键点置信度', diag.keypoint_conf_range),
          card('检测模型', diag.det_model_path || resource.det_model_path),
          card('姿态模型', diag.pose_model_path || resource.pose_model_path),
          card('Decoder', diag.det_decoder || diag.rknn_decoder),
          card('输出契约', diag.det_output_contract),
          card('检测输出', diag.det_output_shapes),
          card('姿态输出', diag.pose_output_shapes),
          card('人体框', diag.selected_yolo_bbox),
          card('检测分数', diag.selected_yolo_score),
          card('检测耗时', `${{value(perf.det_inference_ms)}} ms`),
          card('姿态耗时', `${{value(perf.pose_inference_ms)}} ms`),
          card('总耗时', `${{value(perf.total_pose_ms)}} ms`),
          card('NPU 三核', npu.cores || npu.average_percent),
          card('后处理错误', diag.postprocess_error || diag.detector_contract_error || resource.last_error),
        ].join('');
        rawNode.textContent = JSON.stringify({{npu_debug:debug,npu_resource:resource,npu_pose_debug:diag,npu:npu}}, null, 2);
      }} catch (error) {{ messageNode.textContent = `状态读取失败：${{error}}`; }}
    }}
    document.getElementById('start').onclick = async () => {{
      try {{ await post('/api/npu/debug/start'); messageNode.textContent='NPU 模型正在加载，请等待人体框和骨架出现。'; await refresh(); }}
      catch (error) {{ messageNode.textContent=String(error.message || error); }}
    }};
    document.getElementById('stop').onclick = async () => {{
      try {{ await post('/api/npu/debug/stop'); messageNode.textContent='NPU 姿态已停止并释放，可使用 Qwen。'; await refresh(); }}
      catch (error) {{ messageNode.textContent=String(error.message || error); }}
    }};
    setInterval(async () => {{ if (debugActive) {{ try {{ await post('/api/npu/debug/heartbeat'); }} catch (_) {{}} }} }}, {int(NPU_DEBUG_HEARTBEAT_SECONDS * 1000)});
    setInterval(refresh, 1000); refresh();
    window.addEventListener('pagehide', () => {{
      if (debugActive && navigator.sendBeacon) navigator.sendBeacon('/api/npu/debug/stop', new Blob(['{{}}'], {{type:'application/json'}}));
    }});
  </script>
</body>
</html>""".encode("utf-8")


class NpuPrescriptionHTTPHandler(app.PrescriptionHTTPHandler):
    _POSE_START_PATHS = {"/api/start", "/api/realtime/start", "/api/realtime/start_playlist"}
    _POSE_RELEASE_PATHS = {"/api/save", "/api/cancel", "/api/clear", "/api/realtime/stop"}
    _QWEN_PATHS = {"/api/voice/ask", "/api/llm/report_summary", "/api/llm/ask"}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/npu-debug":
            body = build_npu_debug_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/npu/debug/start", "/api/npu/debug/stop", "/api/npu/debug/heartbeat"}:
            try:
                app.read_json_body(self)
            except Exception:
                pass
            if path == "/api/npu/debug/start":
                result = start_npu_debug()
                app.make_json_response(self, result, status_code=200 if result.get("ok") else 409)
            elif path == "/api/npu/debug/heartbeat":
                result = heartbeat_npu_debug()
                app.make_json_response(self, result, status_code=200 if result.get("ok") else 409)
            else:
                app.make_json_response(self, stop_npu_debug())
            return
        if path in self._POSE_START_PATHS and qwen_busy():
            app.make_json_response(
                self,
                {"ok": False, "error": "小爱正在生成回答，请等待回答结束后再开始录入或训练。"},
                status_code=409,
            )
            return
        if path in self._POSE_START_PATHS:
            stop_npu_debug(release=False)
        if path in self._QWEN_PATHS:
            stop_npu_debug(release=False)
            release_pose_for_qwen()
        try:
            super().do_POST()
        finally:
            if path in self._POSE_RELEASE_PATHS:
                release_pose_for_qwen()


def main() -> None:
    configure_isolated_runtime()
    backend_ready = app.cap is not None and app.rknn_backend is not None
    if app.cv2 is not None and backend_ready:
        threading.Thread(target=app.camera_capture_worker, name="npu-camera-capture", daemon=True).start()
        threading.Thread(target=app.pose_worker, name="npu-pose-worker", daemon=True).start()
        if app.rknn_async_pipeline_enabled():
            threading.Thread(target=app.render_worker, name="npu-render-worker", daemon=True).start()
            threading.Thread(target=app.keyframe_worker, name="npu-keyframe-worker", daemon=True).start()
        threading.Thread(target=npu_debug_watchdog, name="npu-debug-watchdog", daemon=True).start()
    else:
        app.state.last_status = f"NPU vision pipeline not ready: {app.vision_boot_error or 'unknown error'}"

    app.asr_worker.start()
    app.voice_llm_worker = app.make_voice_llm_worker()
    app.voice_llm_worker.start()
    server = app.ThreadedHTTPServer(("0.0.0.0", PORT), NpuPrescriptionHTTPHandler)

    print(f"NPU rehab station started: http://BOARD_IP:{PORT}/train", flush=True)
    print(f"doctor page: http://BOARD_IP:{PORT}/doctor", flush=True)
    print(f"NPU debug page: http://BOARD_IP:{PORT}/npu-debug", flush=True)
    print(f"pose pipeline: {app.state.rknn_pipeline}", flush=True)
    print(f"NPU data root: {NPU_DOCS_DIR}", flush=True)

    try:
        server.serve_forever()
    finally:
        app.state.running = False
        app.realtime_session.stop()
        app.asr_worker.stop()
        if app.voice_llm_worker is not None:
            app.voice_llm_worker.stop()
        app.llm_tts_worker.stop()
        with app.state.condition:
            app.state.condition.notify_all()
        if app.cap is not None:
            app.cap.release()
        if app.rknn_backend is not None:
            app.rknn_backend.release()


if __name__ == "__main__":
    main()
