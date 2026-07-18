from __future__ import annotations

import ast
import queue
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8-sig")


def load_function(path: str, function_name: str, globals_dict: dict[str, Any]):
    source = read_text(path)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    return namespace[function_name]


def test_npu_defaults_use_raw_detector_and_direct_hd_camera() -> None:
    launcher = read_text("scripts/start_npu_rehab_8085.sh")
    entrypoint = read_text("rehab_app/server/npu_rehab_server.py")

    assert 'DET_MODEL="${RKNN_DET_MODEL:-models/vision/yolov5n_raw_fp.rknn}"' in launcher
    assert 'POSE_MODEL="${RKNN_RTMPOSE_MODEL:-models/vision/rtmpose_m_256x192_fp.rknn}"' in launcher
    assert 'RKNN_DET_SCORE_THRES="${RKNN_DET_SCORE_THRES:-0.80}"' in launcher
    assert 'REHAB_ASSISTANT_TTS_GAIN="${REHAB_ASSISTANT_TTS_GAIN:-1.35}"' in launcher
    assert 'RK_CAMERA_SOURCE="device"' in launcher
    assert 'RK_CAMERA_DEVICE="${RK_CAMERA_DEVICE:-/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0}"' in launcher
    assert "unset RK_CAMERA_STREAM_URL RTM_POSE_STREAM_URL" in launcher
    assert 'RK_CAMERA_WIDTH="${RK_CAMERA_WIDTH:-1280}"' in launcher
    assert 'RK_CAMERA_HEIGHT="${RK_CAMERA_HEIGHT:-720}"' in launcher
    assert 'RK_CAMERA_FIXED_FPS="${RK_CAMERA_FIXED_FPS:-1}"' in launcher
    assert 'RK_CAMERA_OPEN_MODE="${RK_CAMERA_OPEN_MODE:-auto}"' in launcher
    assert 'RK_CAMERA_GST_FORMAT="${RK_CAMERA_GST_FORMAT:-MJPG}"' in launcher
    assert 'RK_CAMERA_GST_JPEG_DECODER="${RK_CAMERA_GST_JPEG_DECODER:-auto}"' in launcher
    assert 'RK_CAMERA_GST_BACKEND="${RK_CAMERA_GST_BACKEND:-gi}"' in launcher
    assert 'RKNN_PROCESS_WIDTH="${RKNN_PROCESS_WIDTH:-1280}"' in launcher
    assert 'RKNN_PROCESS_HEIGHT="${RKNN_PROCESS_HEIGHT:-720}"' in launcher
    assert 'RKNN_STREAM_WIDTH="${RKNN_STREAM_WIDTH:-960}"' in launcher
    assert 'RKNN_STREAM_HEIGHT="${RKNN_STREAM_HEIGHT:-540}"' in launcher
    assert 'RKNN_STREAM_FPS="${RKNN_STREAM_FPS:-20}"' in launcher
    assert 'RKNN_FAST_PREVIEW="${RKNN_FAST_PREVIEW:-1}"' in launcher
    assert 'RKNN_FAST_FRAME_DATA="${RKNN_FAST_FRAME_DATA:-1}"' in launcher
    assert 'RKNN_DET_INTERVAL="${RKNN_DET_INTERVAL:-3}"' in launcher
    assert 'RKNN_DET_CACHE_SECONDS="${RKNN_DET_CACHE_SECONDS:-1.5}"' in launcher
    assert 'RKNN_ASYNC_PIPELINE="${RKNN_ASYNC_PIPELINE:-1}"' in launcher
    assert 'REHAB_SERVICE_MODE="${REHAB_SERVICE_MODE:-npu_rehab}"' in launcher

    assert 'os.environ.setdefault("RKNN_DET_MODEL", "models/vision/yolov5n_raw_fp.rknn")' in entrypoint
    assert 'os.environ.setdefault("RKNN_RTMPOSE_MODEL", "models/vision/rtmpose_m_256x192_fp.rknn")' in entrypoint
    assert 'os.environ["RK_CAMERA_SOURCE"] = "device"' in entrypoint
    assert 'os.environ.setdefault("RK_CAMERA_DEVICE", DIRECT_CAMERA_BY_ID)' in entrypoint
    assert 'os.environ.setdefault("RKNN_PROCESS_WIDTH", "1280")' in entrypoint
    assert 'os.environ.setdefault("RKNN_PROCESS_HEIGHT", "720")' in entrypoint
    assert 'os.environ.setdefault("RKNN_ASYNC_PIPELINE", "1")' in entrypoint
    assert 'os.environ.setdefault("REHAB_SERVICE_MODE", "npu_rehab")' in entrypoint


def test_yolov5_rtmpose_uses_the_gated_fast_frame_path() -> None:
    app = read_text("rehab_app/server/rehab_http_server.py")
    npu_entry = read_text("rehab_app/server/npu_rehab_server.py")
    checker = read_text("scripts/check_npu_rehab_8085.sh")

    assert 'fast_frame_data = bool(RKNN_FAST_FRAME_DATA and rtmpose_pipeline)' in app
    assert 'if not backend_draw_enabled:' in app
    assert 'perf.infer_ms' in checker
    assert 'perf.jpeg_ms' in checker
    assert 'orientation.front_count' in checker
    assert 'orientation.mode' in checker
    assert 'start.guard_remaining' in checker
    assert 'start.ready' in checker
    assert 'perf.process_resolution' in checker
    assert 'perf.idle_fast_path' in checker
    assert 'def resize_rknn_process_frame(frame):' in app
    assert 'return ["mppjpegdec", "jpegdec"]' in app
    assert 'appsink drop=true max-buffers=1 sync=false' in app
    assert 'for mode, decoder in camera_open_variants()' in app
    assert 'def open_camera_gstreamer_gi' in app
    assert 'GStreamerGiCapture' in app
    assert 'camera_open_failures' in app
    assert 'STATIC_ASSET_VERSION' in app
    assert 'Cache-Control", "no-cache, must-revalidate"' in app

    common_js = read_text("rehab_app/server/static/common.js")
    assert 'status.stream_available ? "/stream.mjpg"' in common_js
    assert 'status.stream_ready ? "/stream.mjpg"' not in common_js
    assert 'draw_coco17_display_overlay' in app
    assert 'display_stabilizer=None if rknn_async_pipeline_enabled() else rknn_display_stabilizer' in app
    assert 'and os.environ.get("REHAB_SERVICE_MODE", "").strip() == "npu_rehab"' in app
    assert 'def render_worker()' in app
    assert 'def keyframe_worker()' in app
    assert 'keyframe_queue: queue.Queue[Any] = queue.Queue()' in app
    assert 'offer_keyframe_candidate' in app
    assert 'state.add_stream_client()' in app
    assert 'state.remove_stream_client()' in app
    assert 'if not state.has_stream_clients()' in app
    assert 'target=app.render_worker, name="npu-render-worker"' in npu_entry
    assert 'target=app.keyframe_worker, name="npu-keyframe-worker"' in npu_entry
    assert 'def reset_npu_tracking_state()' in npu_entry
    assert npu_entry.count("reset_npu_tracking_state()") >= 4


def test_latest_frame_queue_replaces_old_item_instead_of_dropping_new_item() -> None:
    put_latest = load_function(
        "rehab_app/server/rehab_http_server.py",
        "put_latest",
        {"queue": queue, "Any": Any},
    )
    target: queue.Queue[str] = queue.Queue(maxsize=1)
    target.put_nowait("old")
    drops: list[str] = []

    put_latest(target, "new", on_drop=lambda: drops.append("old"))

    assert target.get_nowait() == "new"
    assert drops == ["old"]


def test_benchmark_and_independent_npu_core_contract() -> None:
    entrypoint = read_text("rehab_app/server/npu_rehab_server.py")
    launcher = read_text("scripts/start_npu_rehab_8085.sh")
    backend = read_text("pose_estimation/rknn_pose/yolov5n_rtmpose_backend.py")
    wrapper = read_text("pose_estimation/rknn_pose/rknn_backend.py")
    benchmark = read_text("scripts/benchmark_npu_rehab_8085.py")

    assert "RKNN_DET_CORE_MASK" in entrypoint and "RKNN_POSE_CORE_MASK" in entrypoint
    assert 'RKNN_DET_CORE_MASK="${RKNN_DET_CORE_MASK:-${RKNN_CORE_MASK}}"' in launcher
    assert 'RKNN_POSE_CORE_MASK="${RKNN_POSE_CORE_MASK:-${RKNN_CORE_MASK}}"' in launcher
    assert 'os.environ.get("RKNN_DET_CORE_MASK", core)' in backend
    assert 'os.environ.get("RKNN_POSE_CORE_MASK", core)' in backend
    assert "det_model_size_bytes >= 100 * 1024 * 1024" in backend
    assert "SUPPORTED_PIPELINES = {PIPELINE}" in wrapper
    assert "YOLOv5nRTMPoseBackend" in wrapper
    assert 'choices=("idle", "npu-debug", "doctor", "train")' in benchmark
    assert '"p50"' in benchmark and '"p95"' in benchmark
    assert '"queue_wait_ms"' in benchmark and '"keyframe_candidate_copy_ms"' in benchmark
    assert '"render_total_ms"' in benchmark and '"capture_to_stream_age_ms"' in benchmark
    assert 'return "npu_training_v8_stage2_pipeline"' in entrypoint


def test_runtime_fingerprint_and_pipeline_toggle_contract() -> None:
    entrypoint = read_text("rehab_app/server/npu_rehab_server.py")
    checker = read_text("scripts/check_npu_rehab_8085.sh")
    toggle = read_text("scripts/set_npu_pose_execution_mode.sh")

    assert "RUNTIME_SOURCE_FILES" in entrypoint
    assert '"build_id": RUNTIME_BUILD_ID' in entrypoint
    assert '"source_hashes": dict(RUNTIME_SOURCE_HASHES)' in entrypoint
    assert "systemd.working_directory" in checker
    assert "systemd.exec_start" in checker
    assert "deployment.hashes_ok" in checker
    assert "running 8085 process does not match" in checker
    assert "Environment=RKNN_ASYNC_PIPELINE=%s" in toggle
    assert 'Usage: $0 async|sync' in toggle
    assert 'systemctl restart "$SERVICE_NAME"' in toggle


def test_8085_autostart_and_kiosk_do_not_reference_removed_modes() -> None:
    installer = read_text("scripts/install_npu_rehab_8085_autostart.sh")
    kiosk = read_text("scripts/open_npu_rehab_8085_kiosk.sh")
    debug_kiosk = read_text("scripts/open_npu_debug_8085_kiosk.sh")

    assert "rehab-station-npu-8085.service" in installer
    assert "start_npu_rehab_8085.sh" in installer
    assert "stop_npu_rehab_8085.sh" in installer
    assert "open_npu_rehab_8085_kiosk.sh" in installer
    assert "8082" not in installer
    assert "switch_rehab_mode" not in installer
    assert "http://127.0.0.1:8085/train?display=1" in kiosk
    assert '--user-data-dir="$PROFILE_DIR"' in kiosk
    assert "--kiosk" in kiosk
    assert "open_npu_rehab_8085_kiosk.sh" in debug_kiosk
