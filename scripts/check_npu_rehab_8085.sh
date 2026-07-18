#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${REHAB_NPU_SERVICE:-rehab-station-npu-8085.service}"
if command -v systemctl >/dev/null 2>&1; then
  echo "systemd.working_directory $(systemctl show "$SERVICE_NAME" -p WorkingDirectory --value 2>/dev/null || true)"
  echo "systemd.exec_start        $(systemctl show "$SERVICE_NAME" -p ExecStart --value 2>/dev/null || true)"
  MAIN_PID="$(systemctl show "$SERVICE_NAME" -p MainPID --value 2>/dev/null || true)"
  echo "systemd.main_pid          ${MAIN_PID:-0}"
  if [[ -n "${MAIN_PID:-}" && "$MAIN_PID" != "0" && -e "/proc/$MAIN_PID/cwd" ]]; then
    echo "systemd.process_cwd       $(readlink -f "/proc/$MAIN_PID/cwd" 2>/dev/null || true)"
  fi
fi

"${PYTHON_BIN:-python3}" - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def get_json(url):
    try:
        with urlopen(url, timeout=3.0) as response:
            return json.load(response)
    except (OSError, URLError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


status = get_json("http://127.0.0.1:8085/status")
resource = status.get("npu_resource") or {}
debug = status.get("npu_debug") or {}
pose_debug = status.get("npu_pose_debug") or {}
camera_source = status.get("camera_source") or {}
performance_profile = status.get("performance_profile") or {}
pose_performance = status.get("pose_performance") or {}
pose_quality = status.get("pose_quality") or {}
training = status.get("training") or {}
voice = status.get("voice") or {}
runtime = status.get("runtime") or {}
source_hashes = runtime.get("source_hashes") or {}
disk_hashes = {}
for relative_path in source_hashes:
    path = Path(relative_path)
    try:
        disk_hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        disk_hashes[relative_path] = None
runtime_root_ok = bool(runtime.get("project_root")) and Path(str(runtime.get("project_root"))).resolve() == Path.cwd().resolve()
source_hashes_ok = bool(source_hashes) and disk_hashes == source_hashes

print("service.mode             ", status.get("service_mode"))
print("service.port             ", status.get("service_port"))
print("pose.backend             ", status.get("actual_backend"))
print("pose.pipeline            ", status.get("rknn_pipeline"))
print("pose.fps                 ", status.get("pose_fps"))
print("pose.inference_fps       ", status.get("inference_fps"))
print("pose.training_fps        ", status.get("training_update_fps"))
print("pose.stream_fps          ", status.get("stream_fps"))
print("pose.capture_to_infer_ms ", status.get("capture_to_inference_age_ms"))
print("pose.capture_to_stream_ms", status.get("capture_to_stream_age_ms"))
print("pose.capture_id          ", status.get("latest_capture_id"))
print("pose.inference_id        ", status.get("latest_inference_id"))
print("pose.render_capture_id   ", status.get("latest_rendered_capture_id"))
print("pose.render_inference_id ", status.get("latest_rendered_inference_id"))
print("stream.clients           ", status.get("stream_client_count"))
print("camera.source            ", camera_source.get("kind"))
print("camera.device            ", camera_source.get("active_device") or camera_source.get("requested_device"))
print("camera.open_mode         ", camera_source.get("open_mode"))
print("camera.gst_backend       ", status.get("camera_gstreamer_backend_requested"))
print("camera.gi_available      ", status.get("camera_gstreamer_gi_available"))
print("camera.gi_error          ", status.get("camera_gstreamer_gi_error"))
print("camera.open_attempts     ", status.get("camera_open_attempts"))
print("camera.open_failures     ", status.get("camera_open_failures"))
print("camera.requested         ", camera_source.get("requested_resolution"))
print("camera.actual            ", camera_source.get("actual_resolution"))
print("camera.negotiated_fps    ", camera_source.get("actual_fps"))
print("camera.fixed_fps         ", (runtime.get("active_environment") or {}).get("RK_CAMERA_FIXED_FPS"))
print("camera.capture_fps       ", status.get("camera_capture_fps"))
print("camera.read_ms           ", status.get("camera_read_ms"))
print("camera.frame_age_ms      ", status.get("camera_frame_age_ms"))
print("camera.stream_age_ms     ", status.get("stream_frame_age_ms"))
print("camera.uses_8082_stream  ", camera_source.get("uses_8082_stream"))
print("perf.profile             ", performance_profile.get("name"))
print("perf.fast_preview        ", performance_profile.get("fast_preview"))
print("perf.fast_frame_data     ", performance_profile.get("fast_frame_data"))
print("perf.keyframe_every_n    ", performance_profile.get("keyframe_every_n"))
print("perf.det_interval        ", performance_profile.get("det_interval"))
print("perf.det_cache_seconds   ", performance_profile.get("det_cache_seconds"))
print("perf.async_pipeline      ", performance_profile.get("async_pipeline"))
print("perf.adaptive_detector   ", performance_profile.get("adaptive_detector"))
print("perf.det_refresh_seconds ", performance_profile.get("det_refresh_seconds"))
print("perf.det_retry_seconds   ", performance_profile.get("det_retry_seconds"))
print("perf.backend_draw        ", performance_profile.get("backend_draw_enabled"))
print("perf.person_only_fast    ", performance_profile.get("person_only_fast"))
print("perf.debug_crop_every    ", performance_profile.get("debug_crop_every"))
print("perf.jpeg_quality        ", performance_profile.get("jpeg_quality"))
print("perf.process_resolution  ", performance_profile.get("process_resolution"))
print("perf.stream_resolution   ", performance_profile.get("stream_resolution"))
print("perf.idle_fast_path      ", pose_performance.get("idle_fast_path"))
print("perf.process_resize_ms   ", pose_performance.get("process_resize_ms"))
print("perf.infer_ms            ", status.get("rknn_infer_call_ms"))
print("perf.pose_process_ms     ", status.get("pose_process_ms"))
print("perf.realtime_ms         ", pose_performance.get("realtime_process_ms"))
print("perf.jpeg_ms             ", pose_performance.get("jpeg_encode_ms"))
print("perf.loop_ms             ", status.get("pose_loop_ms"))
print("perf.capture_drops       ", status.get("capture_queue_drops"))
print("perf.render_drops        ", status.get("render_queue_drops"))
print("perf.render_no_client    ", status.get("render_skipped_no_client"))
print("perf.render_rate_limited ", status.get("render_rate_limited_drops"))
print("perf.keyframe_drops      ", status.get("keyframe_queue_drops"))
print("perf.keyframe_encoded    ", status.get("keyframe_encode_count"))
print("perf.keyframe_error      ", status.get("keyframe_encode_error"))
print("perf.keyframe_encode_ms  ", status.get("keyframe_encode_ms"))
print("perf.keyframe_write_ms   ", status.get("keyframe_write_ms"))
print("perf.stale_drops         ", status.get("stale_inference_drops"))
print("perf.percentiles         ", status.get("performance_percentiles"))
print("runtime.project_root     ", runtime.get("project_root"))
print("runtime.entrypoint       ", runtime.get("entrypoint"))
print("runtime.pid              ", runtime.get("pid"))
print("runtime.started_at       ", runtime.get("started_at"))
print("runtime.build_id         ", runtime.get("build_id"))
print("runtime.source_hashes    ", runtime.get("source_hashes"))
print("runtime.environment      ", runtime.get("active_environment"))
print("deployment.root_ok       ", runtime_root_ok)
print("deployment.hashes_ok     ", source_hashes_ok)
print("npu.state                ", resource.get("state"))
print("npu.owner                ", resource.get("owner"))
print("npu.models_loaded        ", resource.get("models_loaded"))
print("npu.det_model_loaded     ", resource.get("det_model_loaded"))
print("npu.pose_model_loaded    ", resource.get("pose_model_loaded"))
print("npu.det_model_size_bytes ", resource.get("det_model_size_bytes"))
print("npu.det_core_mask        ", resource.get("det_core_mask"))
print("npu.pose_core_mask       ", resource.get("pose_core_mask"))
print("npu.last_error           ", resource.get("last_error"))
print("debug.active            ", debug.get("active"))
print("debug.lease_expires_at  ", debug.get("lease_expires_at"))
print("det.decoder             ", pose_debug.get("det_decoder") or pose_debug.get("rknn_decoder"))
print("det.output_contract     ", pose_debug.get("det_output_contract"))
print("det.contract_error      ", pose_debug.get("detector_contract_error"))
print("det.selected_bbox       ", pose_debug.get("selected_yolo_bbox"))
print("det.score_threshold     ", (runtime.get("active_environment") or {}).get("RKNN_DET_SCORE_THRES") or pose_debug.get("det_score_thres"))
print("det.trigger_reason      ", status.get("detector_trigger_reason"))
print("det.age_ms             ", status.get("detector_age_ms"))
print("det.cache_hit           ", pose_debug.get("det_cache_hit"))
print("det.cache_valid         ", pose_debug.get("det_cache_valid"))
print("det.cache_age_ms        ", pose_debug.get("det_cache_age_ms"))
print("det.retry_seconds      ", pose_debug.get("detector_retry_seconds"))
print("diag.sampled           ", status.get("diagnostic_sampled"))
print("diag.sample_interval   ", status.get("diagnostic_sample_interval"))
print("tracker.roi            ", status.get("tracker_roi"))
print("tracker.quality        ", status.get("tracker_quality"))
print("tracker.visible_points ", status.get("tracker_visible_points"))
print("pose.keypoint_range     ", pose_debug.get("keypoint_conf_range"))
print("display.keypoint_count  ", pose_quality.get("display_keypoint_count"))
print("display.held_keypoints  ", pose_quality.get("display_held_keypoints"))
print("display.jump_pending    ", pose_quality.get("display_jump_pending"))
print("display.bbox_held       ", pose_quality.get("display_bbox_held"))
print("display.bbox_jump       ", pose_quality.get("display_bbox_jump_pending"))
print("pose.jump_pending       ", pose_quality.get("jump_pending"))
print("pose.jump_recovered     ", pose_quality.get("jump_recovery_accepted"))
print("pose.jump_counts        ", pose_quality.get("jump_reject_counts"))
print("training.status          ", training.get("status"))
print("training.logic_version   ", training.get("training_logic_version"))
print("training.action          ", training.get("action_id"))
print("training.completed       ", training.get("completed_reps"))
print("training.invalid         ", training.get("invalid_attempts"))
print("training.last_error      ", (training.get("last_invalid_attempt") or {}).get("primary_error"))
print("training.fixed_audio     ", training.get("training_fixed_audio_only"))
print("training.tts_backend     ", (training.get("tts") or {}).get("backend"))
print("training.tts_initialized ", (training.get("tts") or {}).get("real_tts_initialized"))
template_health = training.get("template_health") or {}
print("template.ok              ", template_health.get("ok"))
print("template.rom             ", template_health.get("rom"))
print("template.reason          ", template_health.get("reason"))
print("template.message         ", template_health.get("message"))
print("orientation.phase        ", training.get("orientation_phase"))
print("orientation.state        ", training.get("orientation_state"))
print("orientation.front_ok     ", training.get("front_view_ok"))
print("orientation.side_ok      ", training.get("side_view_ok"))
print("orientation.front_count  ", training.get("front_orientation_confirm_count"))
print("orientation.front_need   ", training.get("front_orientation_required_frames"))
print("orientation.side_count   ", training.get("orientation_confirm_count"))
print("orientation.side_need    ", training.get("orientation_required_frames"))
print("orientation.ratio        ", training.get("orientation_ratio"))
print("orientation.visibility   ", training.get("orientation_visibility"))
print("orientation.mode         ", training.get("rknn_orientation_mode"))
print("orientation.message      ", training.get("orientation_message"))
runtime_thresholds = training.get("runtime_thresholds") or {}
print("start.guard_remaining    ", training.get("action_guard_remaining_seconds"))
print("start.machine_state      ", training.get("machine_internal_state"))
print("start.baseline           ", training.get("baseline_angle"))
print("start.current_metric     ", training.get("current_metric"))
print("start.ready              ", runtime_thresholds.get("start_ready"))
print("start.motion_delta       ", runtime_thresholds.get("motion_delta"))
print("quality.rom_diff_max     ", runtime_thresholds.get("rom_diff_max"))
print("quality.min_rom_ratio    ", runtime_thresholds.get("min_rom_ratio"))
print("quality.min_rom_absolute ", runtime_thresholds.get("min_rom_absolute"))
print("quality.required_rom     ", runtime_thresholds.get("required_rom"))
print("quality.template_rom     ", runtime_thresholds.get("template_rom"))
print("quality.dynamic_target   ", runtime_thresholds.get("dynamic_target"))
print("quality.min_tut_seconds  ", runtime_thresholds.get("min_tut_seconds"))
print("quality.tut_required     ", runtime_thresholds.get("tut_required_seconds"))
print("presence.raw             ", training.get("presence_raw"))
print("presence.stable          ", training.get("presence_stable"))
print("presence.npu_hits        ", training.get("npu_presence_hits"))
print("presence.npu_misses      ", training.get("npu_presence_misses"))
print("start_pose.ready         ", training.get("start_pose_ready"))
print("start_pose.count         ", training.get("start_pose_confirm_count"))
print("start_pose.need          ", training.get("start_pose_required_frames"))
print("start_pose.reason        ", training.get("start_pose_reason"))
print("start_pose.motion        ", training.get("start_pose_motion"))
print("start_pose.geometry      ", training.get("start_pose_geometry"))
print("return_pose.ok           ", training.get("return_pose_ok"))
print("return_pose.stable       ", training.get("return_pose_stable"))
print("return_pose.count        ", training.get("return_pose_confirm_count"))
print("return_pose.need         ", training.get("return_pose_required_frames"))
print("return_pose.seconds      ", training.get("return_pose_stable_seconds"))
print("return.reversal_count    ", runtime_thresholds.get("return_reversal_confirm_count"))
print("return.reversal_ok       ", runtime_thresholds.get("return_reversal_confirmed"))
print("return.reversal_drop     ", runtime_thresholds.get("return_reversal_required_drop"))
print("rebaseline.state         ", training.get("rebaseline_state"))
print("rebaseline.pending       ", training.get("rebaseline_pending"))
print("rebaseline.cycles        ", training.get("rebaseline_cycle_count"))
print("watchdog.reason          ", training.get("last_watchdog_reason"))
print("watchdog.state           ", training.get("last_watchdog_state"))
print("watchdog.recoveries      ", training.get("watchdog_recovery_count"))
rep_audio = training.get("rep_audio_timing") or {}
print("count.completion_trigger ", rep_audio.get("completion_trigger"))
print("count.rep_settled_at     ", rep_audio.get("rep_settled_at"))
print("count.wav_queued_at      ", rep_audio.get("count_queued_at"))
print("count.wav_started_at     ", rep_audio.get("count_started_at"))
print("count.settle_delay_s     ", rep_audio.get("settle_to_audio_seconds"))
print("count.queue_delay_s      ", rep_audio.get("queue_to_audio_seconds"))
latest_quality = training.get("latest_quality") or {}
print("score.completion         ", latest_quality.get("completion_percent"))
print("score.rule               ", latest_quality.get("rule_score"))
print("score.onnx_raw           ", latest_quality.get("raw_quality_score"))
print("score.grade              ", latest_quality.get("grade"))
print("voice.qa_allowed         ", voice.get("qa_allowed"))
if resource.get("det_model_deprecated") or str(resource.get("det_model_path") or "").endswith("yolov5n_nonms_fp.rknn"):
    raise SystemExit("deprecated or oversized YOLOv5 detector is active; expected the ~4.9MB models/vision/yolov5n_raw_fp.rknn")
if runtime and (not runtime_root_ok or not source_hashes_ok):
    raise SystemExit("running 8085 process does not match the current project directory; restart the service after uploading")
PY
