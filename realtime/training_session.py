"""Realtime patient training session orchestration."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from evaluate.core.action_metrics import METRIC_VALUE_FIELD, extract_metric_from_frame, extract_metric_sequence
from evaluate.core.rom import compute_rom
from evaluate.core.speed_check import check_speed
from evaluate.core.tut import compute_tut
from evaluate.run_evaluate import build_tut_range, select_angle_field
from feedback.feedback_engine import build_feedback_from_files
from prescription.common.active_templates import get_active_template as registry_get_active_template, normalize_pose_backend
from prescription.common.result_storage import save_prescription_artifacts
from realtime.audio_player import RestAudioPlayer
from realtime.feedback_runtime import load_rules, process_prompt, rep_feedback
from realtime.knee_flexion import KneeFlexionRealtimeMachine, KneeFlexionTargets
from realtime.tts_worker import TTSWorker

try:
    from quality_model.service import get_quality_model_status, score_rep
    from quality_model.completion_calibrator import calibrated_completion_percent, should_filter_reentry_attempt
except Exception:  # optional board dependency
    calibrated_completion_percent = None  # type: ignore[assignment]
    get_quality_model_status = None  # type: ignore[assignment]
    score_rep = None  # type: ignore[assignment]
    should_filter_reentry_attempt = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TEMPLATES_PATH = PROJECT_ROOT / "runtime" / "active_templates.json"
DEFAULT_REALTIME_CONFIG = PROJECT_ROOT / "realtime" / "configs" / "knee_flexion_realtime.yaml"
DEFAULT_DEMO_PLAN = PROJECT_ROOT / "realtime" / "configs" / "rehab_demo_plan.yaml"
DEFAULT_TTS_PHRASES = PROJECT_ROOT / "realtime" / "configs" / "tts_phrases.yaml"
DEFAULT_FEEDBACK_RULE = PROJECT_ROOT / "feedback" / "rules" / "knee_flexion_feedback.yaml"
REPORTS_DIR = PROJECT_ROOT / "evaluate" / "reports"
KEYFRAMES_DIR = REPORTS_DIR / "keyframes"
PYTHON_EXE = Path("D:/anaconda/python.exe") if Path("D:/anaconda/python.exe").exists() else Path(sys.executable)

ACTIVE_STATUSES = {
    "running",
    "paused",
    "resting",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
    "awaiting_action_audio",
    "awaiting_rep_feedback",
}

DEFAULT_ORIENTATION_PROMPT = "请侧身对准镜头。"
DEFAULT_OFFSCREEN_PROMPT = "请回到画面中。"
DEFAULT_CARE_PROMPT = "累了吗？要休息吗？"
TRAINING_SIDE = "left"
ANGLE_RIGHT_PROMPT = "角度正确，我们继续训练吧"
INSCREEN_PROMPT = "回到画面了，我们继续"
STALE_TTS_EVENTS = {"orientation", "action_start", "correction", "offscreen"}


class RealtimeTrainingSession:
    def __init__(self, realtime_config_path: Path = DEFAULT_REALTIME_CONFIG) -> None:
        self.realtime_config_path = realtime_config_path
        self.lock = threading.RLock()
        self.tts_worker: TTSWorker | None = None
        self.rest_audio_player = RestAudioPlayer(PROJECT_ROOT)
        self.rules = load_rules(DEFAULT_FEEDBACK_RULE)
        self.reset()

    def reset(self) -> None:
        self.status = "idle"
        self.error: str | None = None
        self.patient_id = "patient_001"
        self.action_id = "knee_flexion"
        self.side_mode = TRAINING_SIDE
        self.pose_backend = "mediapipe"
        self.target_reps = 10
        self.session_id = self._new_session_id()
        self.start_time: float | None = None
        self.frame_index = 0
        self.frames: list[dict[str, Any]] = []
        self.rep_results: list[dict[str, Any]] = []
        self.keyframes: list[dict[str, Any]] = []
        self.keyframe_errors: list[str] = []
        self._keyframe_candidate: dict[str, Any] | None = None
        self.invalid_attempts: list[dict[str, Any]] = []
        self.last_invalid_attempt: dict[str, Any] | None = None
        self.invalid_streak = 0
        self.machine: KneeFlexionRealtimeMachine | None = None
        self.active_template: dict[str, Any] | None = None
        self.selected_rule: dict[str, Any] | None = None
        self.last_prompt = "等待开始训练"
        self.last_tts_text: str | None = None
        self.last_motor_mock_pattern: str | None = None
        self.last_machine_output: dict[str, Any] | None = None
        self.eval_config: dict[str, Any] | None = None
        self.metric_info: dict[str, Any] | None = None
        self.metric_baseline_hip_y: float | None = None
        self.metric_baseline_torso_height: float | None = None
        self.saved_attempt_file: str | None = None
        self.report_file: str | None = None
        self.report: dict[str, Any] | None = None
        self.feedback: dict[str, Any] | None = None
        self.playlist_mode = False
        self.playlist_actions: list[dict[str, Any]] = []
        self.playlist_index = 0
        self.playlist_reports: list[dict[str, Any]] = []
        self.rest_until: float | None = None
        self.rest_seconds = 10
        self.rest_context: str | None = None
        self.current_action_meta: dict[str, Any] | None = None
        self.current_realtime_config: dict[str, Any] = {}
        self.current_targets: KneeFlexionTargets | None = None
        self._feedback_attempt_sequence = 0
        self.pause_reason: str | None = None
        self.orientation_required = False
        self.orientation_ok = False
        self.front_view_ok = False
        self.side_view_ok = False
        self.orientation_phase = "idle"
        self.orientation_state = "idle"
        self.orientation_prompt_spoken = False
        self.orientation_prompt = DEFAULT_ORIENTATION_PROMPT
        self.target_template_side: str | None = None
        self.active_template_side: str | None = None
        self.initial_orientation_done = False
        self.last_frame_data: dict[str, Any] | None = None
        self.action_intro_tts = ""
        self.pending_action_start: dict[str, Any] | None = None
        self.pending_feedback_resume = False
        self.action_guidance_spoken = False
        self.orientation_confirm_count = 0
        self.front_orientation_confirm_count = 0
        self.return_confirm_count = 0
        self.front_orientation_confirm_frames = 2
        self.rknn_front_orientation_confirm_frames = 2
        self.orientation_confirm_frames = 8
        self.rknn_orientation_confirm_frames = 4
        self.offscreen_timeout_seconds = 5.0
        self.offscreen_since: float | None = None
        self.offscreen_seconds = 0.0
        self.offscreen_prompt_pending = False
        self.offscreen_prompt_spoken = False
        self.last_offscreen_resume_at: float | None = None
        self.offscreen_reentry_until: float | None = None
        self.reentry_state = "idle"
        self.reentry_ready = True
        self._reentry_samples: list[tuple[float, float]] = []
        self._filtered_reentry_attempts = 0
        self.care_prompt_invalid_streak = 5
        self.care_dialog = self._empty_care_dialog()
        self.rest_music = {"enabled": False, "file": "", "fade_seconds": 0.0}
        self.action_start_guard_seconds = 2.0
        self.offscreen_reentry_guard_seconds = 1.2
        self.action_guard_until: float | None = None
        self.tts_action_guard_extra_seconds = 0.5
        self.correction_tts_interval_seconds = 4.0
        self.last_correction_tts_at = 0.0
        self.last_correction_error_code: str | None = None
        self.quality_attempt_segments: list[dict[str, Any]] = []
        self._quality_pre_scored: dict[int, dict[str, Any]] = {}
        if not hasattr(self, "_quality_score_queue"):
            self._quality_score_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=16)
        else:
            self._drain_quality_score_queue()
        if not hasattr(self, "_quality_score_worker"):
            self._quality_score_worker: threading.Thread | None = None
        if not hasattr(self, "_quality_score_worker_stop"):
            self._quality_score_worker_stop = threading.Event()
        self._quality_last_error: str | None = None
        self._quality_last_scored_at: float | None = None
        self._quality_dropped_attempts = 0

    def start(
        self,
        patient_id: str = "patient_001",
        action_id: str = "knee_flexion",
        side_mode: str = "auto",
        target_reps: int | None = None,
        pose_backend: str = "mediapipe",
    ) -> dict[str, Any]:
        with self.lock:
            if self.status in ACTIVE_STATUSES:
                return {"ok": False, "error": "实时训练已经在运行中。"}

            try:
                demo_plan = self._load_demo_plan()
                action_meta = self._find_plan_action(action_id, demo_plan)
                realtime_config = self._load_yaml(
                    self._resolve_project_path(action_meta.get("realtime_config_file") or self.realtime_config_path)
                )
                self.pose_backend = normalize_pose_backend(pose_backend)
                active_template = self._get_active_template(action_id)
                if not active_template:
                    return {"ok": False, "error": f"请先用 {self.pose_backend} 后端录入医生标准动作，并保存为 active template。"}

                eval_config_path = self._resolve_project_path(
                    active_template.get("config_file") or action_meta.get("config_file") or "evaluate/configs/knee_flexion.yaml"
                )
                template_path = self._resolve_project_path(active_template.get("template_file") or "")
                if not template_path.exists():
                    return {"ok": False, "error": f"active template 文件不存在：{self._project_relative(template_path)}"}
                if not eval_config_path.exists():
                    return {"ok": False, "error": f"评估配置不存在：{self._project_relative(eval_config_path)}"}

                eval_config = self._load_yaml(eval_config_path)
                realtime_config = self._merge_realtime_config(realtime_config, eval_config)
                targets = self._build_targets(template_path, eval_config, realtime_config)
                if self.tts_worker is not None:
                    self.tts_worker.stop()
                self._stop_quality_score_worker(wait_seconds=0.5)

                self.reset()
                self.pose_backend = normalize_pose_backend(pose_backend)
                self.patient_id = str(patient_id or "patient_001").strip() or "patient_001"
                self.action_id = str(action_id or "knee_flexion").strip() or "knee_flexion"
                self.side_mode = TRAINING_SIDE
                self.active_template_side = self._template_side(template_path)
                self.target_template_side = TRAINING_SIDE
                self.target_reps = int(target_reps or realtime_config.get("target_reps", 10))
                self.active_template = active_template
                self.eval_config = eval_config
                self.metric_info = self._metric_info_from_config(eval_config)
                self.current_action_meta = dict(action_meta)
                self.current_realtime_config = dict(realtime_config)
                self.current_targets = targets
                feedback_rule_file = action_meta.get("feedback_rule_file") or self._feedback_rule_for_action(self.action_id)
                self.rules = load_rules(self._resolve_project_path(feedback_rule_file))
                self._load_plan_runtime_settings(demo_plan)
                self.action_intro_tts = str(action_meta.get("action_intro_tts") or "")
                self.orientation_required = bool(action_meta.get("require_side_view"))
                self.orientation_prompt = str(action_meta.get("orientation_prompt_tts") or DEFAULT_ORIENTATION_PROMPT)
                self.tts_worker = self._create_tts_worker(
                    global_cooldown=float(realtime_config.get("tts_global_cooldown_seconds", 3.0)),
                    same_text_cooldown=float(realtime_config.get("tts_same_text_cooldown_seconds", 5.0)),
                    natural_tts_options=self._natural_tts_options(realtime_config),
                )
                self.tts_worker.start()
                self._ensure_quality_score_worker()
                self._begin_action_after_audio(reset_timing=True, include_action_guidance=True, speak_orientation=True)
                return {"ok": True, "message": "已开始实时训练。", "training": self.snapshot()}
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                return {"ok": False, "error": str(exc)}

    def start_playlist(
        self,
        patient_id: str = "patient_001",
        side_mode: str = "auto",
        target_reps: int | None = None,
        pose_backend: str = "mediapipe",
    ) -> dict[str, Any]:
        with self.lock:
            if self.status in ACTIVE_STATUSES:
                return {"ok": False, "error": "实时训练已经在运行中。"}
            try:
                plan = self._load_demo_plan()
                actions = [action for action in plan.get("actions", []) if isinstance(action, dict)]
                if not actions:
                    return {"ok": False, "error": "rehab_demo_plan.yaml 中没有动作配置。"}
                self.pose_backend = normalize_pose_backend(pose_backend)
                missing = [
                    str(action.get("action_id"))
                    for action in actions
                    if action.get("action_id") and self._get_active_template(str(action.get("action_id"))) is None
                ]
                if missing:
                    return {"ok": False, "error": f"请先用 {self.pose_backend} 后端录入这些动作的医生模板：" + "、".join(missing), "missing": missing}

                if self.tts_worker is not None:
                    self.tts_worker.stop()
                self._stop_quality_score_worker(wait_seconds=0.5)

                self.reset()
                self.pose_backend = normalize_pose_backend(pose_backend)
                self.playlist_mode = True
                self.playlist_actions = actions
                self.playlist_index = 0
                self.patient_id = str(patient_id or "patient_001").strip() or "patient_001"
                self.side_mode = TRAINING_SIDE
                self.initial_orientation_done = False
                self.target_reps = int(target_reps or plan.get("default_target_reps", 3) or 3)
                self._load_plan_runtime_settings(plan)
                self.tts_worker = self._create_tts_worker(
                    global_cooldown=0.5,
                    same_text_cooldown=2.0,
                    natural_tts_options=self._natural_tts_options(plan),
                )
                self.tts_worker.start()
                self._ensure_quality_score_worker()
                welcome = str(plan.get("welcome_tts") or "康复训练即将开始，请坐稳并面向镜头。")
                if self._rep_audio_allowed():
                    self.tts_worker.speak(welcome, priority="high", event_type="welcome", phrase_key="welcome")
                self._start_playlist_action(0)
                return {"ok": True, "message": "已开始完整训练。", "training": self.snapshot()}
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                return {"ok": False, "error": str(exc)}

    def pause(self) -> dict[str, Any]:
        with self.lock:
            if self.status == "running":
                self.status = "paused"
                self.last_prompt = "训练已暂停"
            elif self.status == "paused":
                self.status = "running"
                self.last_prompt = "训练继续"
            return {"ok": True, "training": self.snapshot()}

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.rest_audio_player.stop()
            if self.tts_worker is not None:
                self.tts_worker.stop()
            self._stop_quality_score_worker(wait_seconds=0.5)
            if self.status in ACTIVE_STATUSES:
                self.status = "idle"
                self.pause_reason = None
                self.care_dialog = self._empty_care_dialog()
                self.last_prompt = "训练已停止"
            return {"ok": True, "training": self.snapshot()}

    def respond_to_care(self, needs_rest: bool) -> dict[str, Any]:
        with self.lock:
            if self.status != "awaiting_care_response":
                return {"ok": False, "error": "当前没有待响应的关怀提醒。"}
            self.care_dialog = self._empty_care_dialog()
            self.invalid_streak = 0
            if needs_rest:
                self._start_rest(
                    context="care_break",
                    prompt_text=f"我们休息 {self.rest_seconds} 秒",
                    tts_text=f"我们休息 {self.rest_seconds} 秒。",
                    event_type="care",
                )
            else:
                self._reset_active_motion_state()
                self._start_running_phase(
                    prefix_text="好，我们继续。",
                    reset_timing=False,
                    include_action_guidance=False,
                )
            return {"ok": True, "training": self.snapshot()}

    def process_frame(
        self,
        frame: dict[str, Any] | None,
        selected_rule: dict[str, Any] | None = None,
        keyframe_jpeg: bytes | None = None,
    ) -> None:
        with self.lock:
            now = time.time()
            if self.status == "resting":
                self.offscreen_seconds = 0.0
                if self.rest_until is not None and now >= self.rest_until:
                    self._advance_after_rest()
                return

            frame_data = dict(frame or {})
            person_visible = bool(frame_data.get("person_visible", frame_data.get("pose_detected")))
            if frame is not None and "person_visible" not in frame_data and "pose_detected" not in frame_data:
                person_visible = True
            frame_data["person_visible"] = person_visible
            frame_data["pose_detected"] = person_visible
            front_view_ok = bool(frame_data.get("front_view_ok"))
            side_view_ok = bool(frame_data.get("side_view_ok", frame_data.get("orientation_ok")))
            orientation_ok = side_view_ok or not self.orientation_required
            self.front_view_ok = front_view_ok if person_visible else False
            self.side_view_ok = side_view_ok if person_visible else False
            self.orientation_ok = orientation_ok if person_visible else False
            if selected_rule is not None:
                self.selected_rule = dict(selected_rule)

            self._update_offscreen_tracking(now, person_visible)

            if self.status == "awaiting_action_audio":
                if not person_visible and self.offscreen_seconds >= self.offscreen_timeout_seconds:
                    self._enter_offscreen_wait()
                else:
                    self._maybe_start_pending_action()
                return

            if self.status == "awaiting_rep_feedback":
                if not person_visible and self.offscreen_seconds >= self.offscreen_timeout_seconds:
                    self._enter_offscreen_wait()
                else:
                    self._maybe_resume_after_feedback()
                return

            if self.status in {"paused", "awaiting_care_response"}:
                return

            if self.status == "awaiting_orientation":
                self._process_orientation_gate(person_visible, orientation_ok, frame_data)
                return

            if self.status == "awaiting_return":
                self._maybe_speak_pending_offscreen_prompt()
                self._process_return_gate(person_visible, orientation_ok, frame_data)
                return

            if self.status != "running" or self.machine is None or self.start_time is None:
                return

            if not person_visible and self.offscreen_seconds >= self.offscreen_timeout_seconds:
                self._enter_offscreen_wait()
                return

            if self._reentry_calibration_active(now):
                self._warm_reentry_calibration(frame_data, now)
                return

            if self._in_action_guard(now):
                self._warm_baseline_during_action_guard(frame_data, now)
                self.last_prompt = "请保持静止，正在校准"
                return

            frame_data["reentry_strict_start"] = self._reentry_strict_start_active(now)
            frame_data["frame_index"] = self.frame_index
            frame_data["relative_time"] = now - self.start_time
            frame_data["selected_side"] = TRAINING_SIDE
            self.frame_index += 1

            frame_data = self._apply_action_metric(frame_data)
            self.last_frame_data = dict(frame_data)
            self._update_keyframe_candidate(frame_data, keyframe_jpeg)
            output = self.machine.process(frame_data)
            if not isinstance(output, dict):
                output = self._safe_machine_output(frame_data, reason="machine_returned_none")
            self.last_machine_output = output
            prompt = self._display_prompt_from_machine(output)
            if prompt:
                self.last_prompt = prompt
            elif (
                str(output.get("state") or "") == "IDLE"
                and self.last_prompt in {"请保持静止，正在校准", "请保持静止、等待校准"}
                and self._baseline_ready_from_output(output)
            ):
                self.last_prompt = "可以开始动作"
            if frame_data.get("target_angle_smoothed") is not None:
                self.frames.append(frame_data)

            rep_result = output.get("rep_result")
            if isinstance(rep_result, dict):
                self._handle_rep_done(rep_result)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            machine_state = self.last_machine_output or {}
            return {
                "status": self.status,
                "error": self.error,
                "patient_id": self.patient_id,
                "action_id": self.action_id,
                "side_mode": self.side_mode,
                "pose_backend": self.pose_backend,
                "target_reps": self.target_reps,
                "completed_reps": len(self.rep_results),
                "invalid_attempts": len(self.invalid_attempts),
                "last_invalid_attempt": self.last_invalid_attempt,
                "playlist_mode": self.playlist_mode,
                "playlist_index": self.playlist_index,
                "playlist_total": len(self.playlist_actions),
                "current_action_name": self.current_action_meta.get("action_name") if self.current_action_meta else None,
                "current_camera_prompt": self.current_action_meta.get("camera_prompt") if self.current_action_meta else None,
                "rest_remaining_seconds": self._rest_remaining_seconds(),
                "playlist_reports": list(self.playlist_reports),
                "current_state": machine_state.get("state"),
                "current_angle": machine_state.get("angle"),
                "current_metric": machine_state.get("angle"),
                "metric": self.metric_info,
                "metric_unit": (self.metric_info or {}).get("metric_unit"),
                "baseline_angle": machine_state.get("baseline_angle"),
                "target_range": machine_state.get("target_range"),
                "tut_seconds": machine_state.get("tut_seconds"),
                "tut_target": machine_state.get("tut_target"),
                "missing_seconds": machine_state.get("missing_seconds"),
                "runtime_thresholds": self._runtime_threshold_snapshot(machine_state),
                "prompt": self.last_prompt,
                "tts_text": self.last_tts_text,
                "motor_mock_pattern": self.last_motor_mock_pattern,
                "rep_results": list(self.rep_results),
                "active_template": self.active_template,
                "demo_plan": self._plan_snapshot(),
                "missing_plan_templates": self._missing_plan_templates(),
                "saved_attempt_file": self.saved_attempt_file,
                "report_file": self.report_file,
                "report": self.report,
                "feedback": self.feedback,
                "tts": self.tts_worker.snapshot() if self.tts_worker else None,
                "rest_audio": self.rest_audio_player.snapshot(),
                "pause_reason": self.pause_reason,
                "orientation_required": self.orientation_required,
                "orientation_ok": self.orientation_ok,
                "front_view_ok": self.front_view_ok,
                "side_view_ok": self.side_view_ok,
                "orientation_phase": self.orientation_phase,
                "orientation_state": self.orientation_state,
                "orientation_prompt_spoken": self.orientation_prompt_spoken,
                "orientation_prompt": self.orientation_prompt,
                "target_template_side": self.target_template_side,
                "active_template_side": self.active_template_side,
                "selected_side": self._selected_side_snapshot(),
                "target_leg_visibility": (self.last_frame_data or {}).get("target_leg_visibility"),
                "target_side_keypoint_visibility": (self.last_frame_data or {}).get("target_side_keypoint_visibility"),
                "target_leg_visibility_min": (self.last_frame_data or {}).get("target_leg_visibility_min"),
                "target_leg_visibility_ok": (self.last_frame_data or {}).get("target_leg_visibility_ok"),
                "initial_orientation_done": self.initial_orientation_done,
                "action_intro_tts": self.action_intro_tts,
                "pending_action_start": dict(self.pending_action_start) if isinstance(self.pending_action_start, dict) else None,
                "pending_feedback_resume": self.pending_feedback_resume,
                "offscreen_seconds": round(self.offscreen_seconds, 1),
                "care_dialog": dict(self.care_dialog),
                "rest_music": dict(self.rest_music),
                "quality_model": self._quality_status_snapshot(),
                "latest_quality": self._latest_quality_snapshot(),
                "quality_score_history": self._quality_score_history_snapshot(),
                "completion_by_action": self._completion_by_action_snapshot(),
                "overall_completion": self._overall_completion_snapshot(),
            }

    def _handle_rep_done(self, rep_result: dict[str, Any]) -> None:
        if self._should_filter_reentry_rep(rep_result):
            self._filtered_reentry_attempts += 1
            self.last_prompt = "请保持静止，正在校准"
            return
        self._feedback_attempt_sequence += 1
        rep_result = {**rep_result, "attempt_index": self._feedback_attempt_sequence}
        feedback = rep_feedback(rep_result, self.rules, action_id=self.action_id)
        countable = bool(rep_result.get("countable", True))
        rep_index = len(self.rep_results) + 1 if countable else None
        enriched = {
            **rep_result,
            "rep_index": rep_index,
            **feedback,
        }
        self.last_prompt = str(feedback.get("screen_prompt") or self.last_prompt)
        feedback_tts_text = str(feedback.get("tts_text") or "")
        self.last_motor_mock_pattern = str(feedback.get("motor_mock_pattern") or "")
        if self.last_motor_mock_pattern:
            print(f"[MOTOR MOCK] {self.last_motor_mock_pattern}")
        feedback_audio_queued = self._speak_rep_outcome(rep_result, rep_index, feedback_tts_text)

        segment = self._build_quality_attempt_segment(enriched)
        if segment is not None:
            self.quality_attempt_segments.append(segment)
            self._enqueue_quality_segment(segment)

        if not countable:
            self._keyframe_candidate = None
            self.invalid_streak += 1
            correction = str(enriched.get("screen_prompt") or "动作不到位")
            enriched["screen_prompt"] = correction
            enriched["not_counted_reason"] = self._error_reason(rep_result)
            self.last_prompt = enriched["screen_prompt"]
            self.last_invalid_attempt = enriched
            self.invalid_attempts.append(enriched)
            if self.invalid_streak >= self.care_prompt_invalid_streak:
                self._show_care_dialog()
            elif feedback_audio_queued:
                self._enter_feedback_wait()
            return

        self.invalid_streak = 0
        saved_keyframe = self._save_keyframe_candidate(int(rep_index)) if rep_index is not None else None
        if saved_keyframe is not None:
            enriched["keyframe"] = saved_keyframe
        self.rep_results.append(enriched)
        if feedback_audio_queued and len(self.rep_results) < self.target_reps:
            self._enter_feedback_wait()
            return
        if len(self.rep_results) >= self.target_reps:
            self._complete_training()


    def _should_filter_reentry_rep(self, rep_result: dict[str, Any]) -> bool:
        if should_filter_reentry_attempt is None:
            return False
        if self.offscreen_reentry_until is None or time.time() > self.offscreen_reentry_until:
            return False
        try:
            return should_filter_reentry_attempt(rep_result)
        except Exception:
            return False

    def _enter_feedback_wait(self) -> None:
        self.status = "awaiting_rep_feedback"
        self.pause_reason = "rep_feedback"
        self.pending_feedback_resume = True
        self.action_guard_until = None
        self.last_prompt = "请听完反馈后再做下一次。"

    def _maybe_resume_after_feedback(self) -> None:
        if self.tts_worker is not None and self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds):
            self.last_prompt = "请听完反馈后再做下一次。"
            return
        self.pending_feedback_resume = False
        self.pause_reason = None
        self.status = "running"
        self.action_guard_until = None
        self.last_prompt = ""

    def _build_quality_attempt_segment(self, enriched: dict[str, Any]) -> dict[str, Any] | None:
        start_time = _as_float(enriched.get("start_time"))
        end_time = _as_float(enriched.get("end_time"))
        if start_time is None or end_time is None:
            return None
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        selected_frames: list[dict[str, Any]] = []
        for frame in self.frames:
            relative_time = _as_float(frame.get("relative_time"))
            if relative_time is None or relative_time < start_time or relative_time > end_time:
                continue
            rehab_keypoints = frame.get("rehab_keypoints")
            compact_keypoints = _compact_rehab_keypoints(rehab_keypoints) if isinstance(rehab_keypoints, dict) else {}
            selected_frames.append(
                {
                    "frame_index": frame.get("frame_index"),
                    "relative_time": relative_time,
                    "selected_side": frame.get("selected_side"),
                    "visibility_min": frame.get("visibility_min") or frame.get("visibility"),
                    "target_angle_smoothed": frame.get("target_angle_smoothed"),
                    "primary_signal_smoothed": frame.get("primary_signal_smoothed"),
                    "rehab_keypoints": compact_keypoints,
                }
            )
        if not selected_frames:
            return None
        frame_indexes = [item.get("frame_index") for item in selected_frames if item.get("frame_index") is not None]
        attempt_index = int(enriched.get("attempt_index") or len(self.quality_attempt_segments) + 1)
        segment = {
            "attempt_index": attempt_index,
            "rep_index": enriched.get("rep_index"),
            "countable": bool(enriched.get("countable", False)),
            "session_id": self.session_id,
            "action_id": self.action_id,
            "action_name": self.current_action_meta.get("action_name") if self.current_action_meta else self.action_id,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_index": min(frame_indexes) if frame_indexes else selected_frames[0].get("frame_index"),
            "end_frame_index": max(frame_indexes) if frame_indexes else selected_frames[-1].get("frame_index"),
            "frame_count": len(selected_frames),
            "primary_error": enriched.get("primary_error") or "OK",
            "all_errors": list(enriched.get("all_errors") or []),
            "reason": enriched.get("not_counted_reason") or self._error_reason(enriched),
            "screen_prompt": enriched.get("screen_prompt"),
            "angle_curve": enriched.get("angle_curve") if isinstance(enriched.get("angle_curve"), list) else [],
            "skeleton_sequence": selected_frames,
        }
        for key in (
            "rom",
            "rom_target",
            "rom_diff",
            "max_signal",
            "tut_seconds",
            "tut_target",
            "missing_seconds",
            "tut_ratio",
            "peak_speed",
            "speed_ratio",
            "duration_seconds",
        ):
            if key in enriched:
                segment[key] = enriched.get(key)
        return segment

    def _drain_quality_score_queue(self) -> None:
        score_queue = getattr(self, "_quality_score_queue", None)
        if score_queue is None:
            return
        while True:
            try:
                score_queue.get_nowait()
            except queue.Empty:
                return
            try:
                score_queue.task_done()
            except ValueError:
                pass

    def _enqueue_quality_segment(self, segment: dict[str, Any]) -> None:
        if score_rep is None:
            self._quality_last_error = "quality_model_service_unavailable"
            return
        self._ensure_quality_score_worker()
        try:
            self._quality_score_queue.put_nowait(dict(segment))
        except queue.Full:
            self._quality_dropped_attempts += 1
            self._quality_last_error = "quality_score_queue_full"

    def _ensure_quality_score_worker(self) -> None:
        if score_rep is None:
            return
        worker = self._quality_score_worker
        if worker is not None and worker.is_alive():
            return
        self._quality_score_worker_stop.clear()
        self._quality_score_worker = threading.Thread(
            target=self._run_quality_score_worker,
            name="quality_score_worker",
            daemon=True,
        )
        self._quality_score_worker.start()

    def _stop_quality_score_worker(self, wait_seconds: float = 0.5) -> None:
        worker = self._quality_score_worker
        if worker is None:
            return
        self._quality_score_worker_stop.set()
        try:
            self._quality_score_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=max(0.0, wait_seconds))
        if not worker.is_alive():
            self._quality_score_worker = None

    def _run_quality_score_worker(self) -> None:
        while not self._quality_score_worker_stop.is_set():
            try:
                item = self._quality_score_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                action_id = str(item.get("action_id") or self.action_id)
                result = score_rep(action_id, item) if score_rep is not None else None
                if isinstance(result, dict):
                    attempt_index = int(item.get("attempt_index") or 0)
                    with self.lock:
                        if item.get("session_id") == self.session_id and item.get("action_id") == self.action_id:
                            self._quality_pre_scored[attempt_index] = result
                            self._quality_last_scored_at = time.time()
                            self._quality_last_error = None
                else:
                    with self.lock:
                        self._quality_last_error = "quality_score_unavailable"
            except Exception as exc:
                with self.lock:
                    self._quality_last_error = str(exc)
            finally:
                try:
                    self._quality_score_queue.task_done()
                except ValueError:
                    pass

    def _wait_for_quality_scores(self, max_seconds: float = 1.5) -> None:
        deadline = time.time() + max(0.0, max_seconds)
        while time.time() < deadline:
            if getattr(self._quality_score_queue, "unfinished_tasks", 0) <= 0:
                return
            time.sleep(0.03)

    def _completion_from_quality_result(self, segment: dict[str, Any], result: dict[str, Any]) -> float | None:
        raw_score = _as_float(result.get("score"))
        if calibrated_completion_percent is None:
            return raw_score
        try:
            return calibrated_completion_percent(self.action_id, segment, raw_score)
        except Exception:
            return raw_score

    def _quality_segments_with_scores(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for segment in self.quality_attempt_segments:
            row = dict(segment)
            attempt_index = int(row.get("attempt_index") or 0)
            result = self._quality_pre_scored.get(attempt_index)
            if isinstance(result, dict):
                raw_score = _as_float(result.get("score"))
                completion = self._completion_from_quality_result(row, result)
                row["quality_score"] = completion
                row["raw_quality_score"] = raw_score
                row["model_score"] = raw_score
                row["completion_percent"] = completion
                row["quality_grade"] = result.get("grade")
                row["quality_backend"] = result.get("backend")
                row["quality_model_path"] = result.get("model_path")
                row["quality_valid_frames"] = result.get("valid_frames")
            segments.append(row)
        return segments

    def _latest_quality_snapshot(self) -> dict[str, Any] | None:
        for segment in reversed(self.quality_attempt_segments):
            attempt_index = int(segment.get("attempt_index") or 0)
            result = self._quality_pre_scored.get(attempt_index)
            if not isinstance(result, dict):
                continue
            raw_score = _as_float(result.get("score"))
            completion = self._completion_from_quality_result(segment, result)
            return {
                "attempt_index": attempt_index,
                "rep_index": segment.get("rep_index"),
                "countable": bool(segment.get("countable", False)),
                "primary_error": segment.get("primary_error") or "OK",
                "score": completion,
                "completion_percent": completion,
                "quality_score": completion,
                "raw_quality_score": raw_score,
                "model_score": raw_score,
                "grade": result.get("grade"),
                "backend": result.get("backend"),
                "valid_frames": result.get("valid_frames"),
                "reason": segment.get("reason") or segment.get("screen_prompt"),
            }
        return None

    def _quality_score_history_snapshot(self, limit: int = 8) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for segment in reversed(self.quality_attempt_segments):
            attempt_index = int(segment.get("attempt_index") or 0)
            result = self._quality_pre_scored.get(attempt_index)
            row = {
                "attempt_index": attempt_index,
                "rep_index": segment.get("rep_index"),
                "countable": bool(segment.get("countable", False)),
                "primary_error": segment.get("primary_error") or "OK",
                "reason": segment.get("reason") or segment.get("screen_prompt"),
            }
            if isinstance(result, dict):
                raw_score = _as_float(result.get("score"))
                completion = self._completion_from_quality_result(segment, result)
                row.update(
                    {
                        "score": completion,
                        "completion_percent": completion,
                        "quality_score": completion,
                        "raw_quality_score": raw_score,
                        "model_score": raw_score,
                        "grade": result.get("grade"),
                        "backend": result.get("backend"),
                        "valid_frames": result.get("valid_frames"),
                    }
                )
            history.append(row)
            if len(history) >= limit:
                break
        return list(reversed(history))

    def _completion_summary_from_report(self, report: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(report, dict):
            return None
        runtime_meta = report.get("runtime_meta") if isinstance(report.get("runtime_meta"), dict) else {}
        action_id = str(report.get("action_id") or runtime_meta.get("action_id") or self.action_id)
        action_name = str(report.get("action_name") or action_id)
        completion_by_action = report.get("completion_by_action") if isinstance(report.get("completion_by_action"), dict) else {}
        summary = completion_by_action.get(action_id)
        if not isinstance(summary, dict) and completion_by_action:
            first_value = next(iter(completion_by_action.values()))
            summary = first_value if isinstance(first_value, dict) else None
        if isinstance(summary, dict):
            row = dict(summary)
            row.setdefault("action_id", action_id)
            row.setdefault("action_name", action_name)
            row.setdefault("average_completion", report.get("overall_completion") or report.get("overall_quality"))
        else:
            attempts = report.get("quality_attempts") if isinstance(report.get("quality_attempts"), list) else []
            rows: list[dict[str, Any]] = []
            scores: list[float] = []
            for item in attempts:
                if not isinstance(item, dict):
                    continue
                score = _as_float(item.get("completion_percent"))
                if score is None:
                    score = _as_float(item.get("quality_score"))
                if score is not None:
                    scores.append(score)
                rows.append(
                    {
                        "attempt_index": item.get("attempt_index"),
                        "rep_index": item.get("rep_index"),
                        "countable": bool(item.get("countable", False)),
                        "primary_error": item.get("primary_error") or "OK",
                        "reason": item.get("reason"),
                        "completion_percent": round(score, 2) if score is not None else None,
                    }
                )
            average = _as_float(report.get("overall_completion"))
            if average is None:
                average = round(sum(scores) / len(scores), 2) if scores else None
            row = {
                "action_id": action_id,
                "action_name": action_name,
                "attempts": rows,
                "average_completion": average,
            }
        if self.report_file:
            row["report_file"] = self.report_file
        return row

    def _completion_by_action_snapshot(self) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for item in self.playlist_reports:
            if not isinstance(item, dict):
                continue
            summary = item.get("completion_summary")
            if not isinstance(summary, dict):
                continue
            action_id = str(summary.get("action_id") or item.get("action_id") or "")
            if not action_id:
                continue
            row = dict(summary)
            if item.get("report_file") and not row.get("report_file"):
                row["report_file"] = item.get("report_file")
            grouped[action_id] = row
        current = self._completion_summary_from_report(self.report)
        if isinstance(current, dict):
            action_id = str(current.get("action_id") or self.action_id)
            grouped[action_id] = current
        return grouped

    def _overall_completion_snapshot(self) -> float | None:
        scores: list[float] = []
        for summary in self._completion_by_action_snapshot().values():
            if not isinstance(summary, dict):
                continue
            for item in summary.get("attempts") or []:
                if not isinstance(item, dict):
                    continue
                score = _as_float(item.get("completion_percent"))
                if score is not None:
                    scores.append(score)
        return round(sum(scores) / len(scores), 2) if scores else None

    def _quality_status_snapshot(self) -> dict[str, Any]:
        if get_quality_model_status is None:
            return {
                "available": False,
                "backend": None,
                "action_id": self.action_id,
                "model_path": None,
                "last_score_time_ms": None,
                "last_error": self._quality_last_error or "quality_model_service_unavailable",
                "queue_size": 0,
                "dropped_attempts": self._quality_dropped_attempts,
            }
        try:
            status = get_quality_model_status(self.action_id)
        except Exception as exc:
            status = {
                "available": False,
                "backend": None,
                "action_id": self.action_id,
                "model_path": None,
                "last_score_time_ms": None,
                "last_error": str(exc),
            }
        status = dict(status)
        status["queue_size"] = self._quality_score_queue.qsize()
        status["dropped_attempts"] = self._quality_dropped_attempts
        status["scored_attempts"] = len(self._quality_pre_scored)
        status["attempts"] = len(self.quality_attempt_segments)
        status["filtered_reentry_attempts"] = self._filtered_reentry_attempts
        status["worker_alive"] = self._quality_score_worker is not None and self._quality_score_worker.is_alive()
        if self._quality_last_error:
            status["last_error"] = self._quality_last_error
        return status

    def _rep_audio_allowed(self, *, rep_result: dict[str, Any] | None = None) -> bool:
        if rep_result is not None:
            return True
        state = str((self.last_machine_output or {}).get("state") or "")
        if state in {"RISING", "HOLDING", "RETURNING"}:
            return False
        if self.machine is not None:
            machine_state = str(getattr(getattr(self.machine, "state", None), "value", ""))
            if machine_state in {"RISING", "HOLDING", "RETURNING"}:
                return False
        return self.status in {"idle", "running", "resting", "completed", "awaiting_orientation", "awaiting_return", "awaiting_action_audio", "awaiting_care_response"}

    def _speak_rep_outcome(self, rep_result: dict[str, Any], rep_index: int | None, feedback_tts_text: str) -> bool:
        if self.tts_worker is None or not self._rep_audio_allowed(rep_result=rep_result):
            return False
        error_code = str(rep_result.get("primary_error") or "OK")
        if error_code == "OK" and rep_index is not None:
            self.last_correction_error_code = None
            text = self._count_text(int(rep_index)) or str(rep_index)
            queued = self.tts_worker.speak(text, priority="high", event_type="rep_count", phrase_key=f"count_{rep_index}")
            if queued:
                self.last_tts_text = text
            return queued
        if error_code == "VISIBILITY_LOW":
            return False
        if error_code in {"ROM_LOW", "TUT_LOW", "TOO_FAST", "EARLY_RETURN", "SHAPE_BAD"} and feedback_tts_text:
            if not self._should_speak_correction(error_code, rep_result):
                return False
            phrase_key = self._correction_phrase_key(error_code, feedback_tts_text, rep_result)
            queued = self.tts_worker.speak(feedback_tts_text, priority="high", event_type="correction", phrase_key=phrase_key)
            if queued:
                self.last_tts_text = feedback_tts_text
                self.last_correction_error_code = error_code
            return queued
        return False

    def _complete_training(self) -> None:
        try:
            attempt = self._build_attempt_payload()
            save_result = save_prescription_artifacts(
                attempt,
                board_ip="local",
                board_port="8082",
                source="realtime_training_board",
            )
            self.saved_attempt_file = self._project_relative(save_result["saved_path"])
            eval_result = self._run_evaluate(self.saved_attempt_file)
            if not eval_result.get("ok"):
                self.status = "error"
                self.error = str(eval_result.get("error"))
                return
            self.report_file = str(eval_result.get("report_file"))
            self.report = eval_result.get("report") if isinstance(eval_result.get("report"), dict) else None
            self.feedback = eval_result.get("feedback") if isinstance(eval_result.get("feedback"), dict) else None
            self.playlist_reports.append(
                {
                    "action_id": self.action_id,
                    "action_name": self.current_action_meta.get("action_name") if self.current_action_meta else self.action_id,
                    "attempt_file": self.saved_attempt_file,
                    "report_file": self.report_file,
                    "primary_error": (self.report or {}).get("errors", {}).get("primary_error")
                    if isinstance(self.report, dict)
                    else None,
                    "completion_summary": self._completion_summary_from_report(self.report),
                }
            )
            if self.playlist_mode and self.playlist_index < len(self.playlist_actions) - 1:
                text = str((self.current_action_meta or {}).get("set_done_tts") or "做得很好，请休息一下。")
                self._start_rest(
                    context="playlist_transition",
                    prompt_text=f"本组完成，休息 {self.rest_seconds} 秒",
                    tts_text=text,
                    event_type="set_done",
                )
                return
            self.status = "completed"
            self.pause_reason = None
            self.last_prompt = "全部训练完成" if self.playlist_mode else "本组训练完成"
            if self.tts_worker:
                plan = self._load_demo_plan()
                done_text = str(plan.get("finished_tts") or "今天的训练完成得很好，请注意休息。")
                self.last_tts_text = done_text
                if self._rep_audio_allowed():
                    self.tts_worker.speak(done_text, priority="high", event_type="training_finished", phrase_key="finished")
        finally:
            if self.tts_worker is not None and not self.playlist_mode:
                self.tts_worker.stop()

    def _start_playlist_action(self, index: int) -> None:
        action = self.playlist_actions[index]
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            raise ValueError("playlist 动作缺少 action_id")
        self.playlist_index = index
        self._clear_stale_tts({"orientation", "action_start", "correction", "offscreen"})
        self.current_action_meta = dict(action)
        self.action_id = action_id
        self.active_template = self._get_active_template(action_id)
        if not self.active_template:
            raise ValueError(f"缺少 active template: {action_id}")

        realtime_config = self._load_yaml(
            self._resolve_project_path(action.get("realtime_config_file") or self.realtime_config_path)
        )
        eval_config_path = self._resolve_project_path(
            self.active_template.get("config_file") or action.get("config_file") or "evaluate/configs/knee_flexion.yaml"
        )
        template_path = self._resolve_project_path(self.active_template.get("template_file") or "")
        eval_config = self._load_yaml(eval_config_path)
        realtime_config = self._merge_realtime_config(realtime_config, eval_config)
        targets = self._build_targets(template_path, eval_config, realtime_config)
        feedback_rule_file = action.get("feedback_rule_file") or self._feedback_rule_for_action(action_id)
        self.rules = load_rules(self._resolve_project_path(feedback_rule_file))
        self.current_realtime_config = dict(realtime_config)
        self.current_targets = targets
        self.eval_config = eval_config
        self.metric_info = self._metric_info_from_config(eval_config)
        self.target_reps = int(self.target_reps or realtime_config.get("target_reps", 3) or 3)
        self._drain_quality_score_queue()
        self.start_time = None
        self.frame_index = 0
        self.frames = []
        self.rep_results = []
        self._keyframe_candidate = None
        self.invalid_attempts = []
        self.quality_attempt_segments = []
        self._quality_pre_scored = {}
        self._quality_dropped_attempts = 0
        self._quality_last_error = None
        self.last_invalid_attempt = None
        self.invalid_streak = 0
        self.selected_rule = {"side": TRAINING_SIDE}
        self.active_template_side = self._template_side(template_path)
        self.target_template_side = TRAINING_SIDE
        self.last_machine_output = None
        self.metric_baseline_hip_y = None
        self.metric_baseline_torso_height = None
        self.saved_attempt_file = None
        self.report_file = None
        self.report = None
        self.feedback = None
        self.rest_until = None
        self.rest_context = None
        self._feedback_attempt_sequence = 0
        self.action_intro_tts = str(action.get("action_intro_tts") or "")
        self.action_guidance_spoken = False
        self.orientation_required = bool(action.get("require_side_view"))
        self.orientation_prompt = str(action.get("orientation_prompt_tts") or DEFAULT_ORIENTATION_PROMPT)
        self.care_dialog = self._empty_care_dialog()
        self._begin_action_after_audio(
            reset_timing=True,
            include_action_guidance=True,
            speak_orientation=False,
        )

    def _advance_after_rest(self) -> None:
        context = self.rest_context
        self.rest_until = None
        self.rest_context = None
        if context == "playlist_transition":
            next_index = self.playlist_index + 1
            if next_index >= len(self.playlist_actions):
                self.rest_audio_player.stop()
                self.status = "completed"
                self.pause_reason = None
                self.last_prompt = "全部训练完成"
                return
            self.rest_audio_player.stop()
            self._start_playlist_action(next_index)
            return

        self.invalid_streak = 0
        self.rest_audio_player.stop()
        self._reset_active_motion_state()
        self._clear_stale_tts()
        self._start_running_phase(
            prefix_text="休息结束，我们继续训练。",
            reset_timing=False,
            include_action_guidance=False,
        )

    def _rest_remaining_seconds(self) -> int | None:
        if self.status != "resting" or self.rest_until is None:
            return None
        return max(0, int(round(self.rest_until - time.time())))

    def _count_text(self, rep_index: int) -> str | None:
        plan = self._load_demo_plan()
        count_tts = plan.get("count_tts")
        if isinstance(count_tts, list) and 1 <= rep_index <= len(count_tts):
            return str(count_tts[rep_index - 1])
        fallback = ["一", "二", "三", "四", "五"]
        if 1 <= rep_index <= len(fallback):
            return fallback[rep_index - 1]
        return str(rep_index)

    def _build_attempt_payload(self) -> dict[str, Any]:
        target_angles = [
            _as_float(frame.get("target_angle_smoothed"))
            for frame in self.frames
            if _as_float(frame.get("target_angle_smoothed")) is not None
        ]
        included_angles = [
            _as_float(frame.get("selected_included_angle"))
            for frame in self.frames
            if _as_float(frame.get("selected_included_angle")) is not None
        ]
        duration = 0.0
        if len(self.frames) >= 2:
            duration = float(self.frames[-1].get("relative_time", 0.0)) - float(self.frames[0].get("relative_time", 0.0))
        pose_meta = self._pose_meta_from_frames()
        warning = (
            "RKNN 第一版使用 2D 图像角度，要求侧身固定机位，适合演示和趋势反馈，不属于临床级测量。"
            if pose_meta.get("actual_backend") == "rknn"
            else "单目 MediaPipe 角度适合演示和趋势反馈，不属于临床级测量。"
        )
        return {
            "patient_id": self.patient_id,
            "record_role": "patient_attempt",
            "action_id": self.action_id,
            "action_name": self.action_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "这是一次实时训练生成的患者动作结果。",
            "camera_instruction": "实时训练时请保持目标关节相关关键点持续可见。",
            "algorithm_note": {
                "result_source": "realtime_training_session",
                "warning": warning,
            },
            "runtime_meta": {
                **pose_meta,
                "result_format": "compact_v1",
                "record_role": "patient_attempt",
                "action_id": self.action_id,
                "side_mode": self.side_mode,
                "target_reps": self.target_reps,
                "completed_reps": len(self.rep_results),
                "rep_results": self.rep_results,
                "invalid_attempts": self.invalid_attempts,
                "filtered_reentry_attempts": self._filtered_reentry_attempts,
                "quality_attempt_segments": self._quality_segments_with_scores(),
                "rep_segments": [segment for segment in self._quality_segments_with_scores() if segment.get("countable")],
                "session_id": self.session_id,
                "keyframes": self._current_action_keyframes(),
                "keyframe_errors": list(self.keyframe_errors[-5:]),
                "active_template": self.active_template,
                "metric": self.metric_info,
            },
            "keypoint_rule": self.selected_rule or {},
            "clinical_baseline": {
                "frame_count": len(self.frames),
                "duration_seconds": duration,
                "min_selected_included_angle": min(included_angles) if included_angles else None,
                "max_selected_included_angle": max(included_angles) if included_angles else None,
                "min_target_angle": min(target_angles) if target_angles else None,
                "max_target_angle": max(target_angles) if target_angles else None,
                "rom_target_angle": max(target_angles) - min(target_angles) if target_angles else None,
                "min_primary_signal": min(target_angles) if target_angles else None,
                "max_primary_signal": max(target_angles) if target_angles else None,
                "rom_primary_signal": max(target_angles) - min(target_angles) if target_angles else None,
                "min_knee_flexion_angle": min(target_angles) if target_angles else None,
                "max_knee_flexion_angle": max(target_angles) if target_angles else None,
                "rom_flexion": max(target_angles) - min(target_angles) if target_angles else None,
            },
            "template_frames": self.frames,
        }

    def _update_keyframe_candidate(self, frame_data: dict[str, Any], keyframe_jpeg: bytes | None) -> None:
        if not keyframe_jpeg:
            return
        state_value = str(getattr(getattr(self.machine, "state", None), "value", ""))
        if state_value not in {"RISING", "HOLDING", "RETURNING"}:
            return
        value = _as_float(frame_data.get("primary_signal_smoothed"))
        if value is None:
            value = _as_float(frame_data.get("target_angle_smoothed"))
        if value is None:
            return
        current = self._keyframe_candidate
        current_value = _as_float(current.get("signal_value")) if isinstance(current, dict) else None
        direction = str((self.eval_config or {}).get("metric_direction") or "increase").lower()
        if current_value is not None:
            if direction == "decrease" and value >= current_value:
                return
            if direction != "decrease" and value <= current_value:
                return
        rehab_keypoints = frame_data.get("rehab_keypoints")
        self._keyframe_candidate = {
            "image_jpeg": keyframe_jpeg,
            "signal_value": value,
            "frame_index": frame_data.get("frame_index"),
            "relative_time": frame_data.get("relative_time"),
            "primary_metric": frame_data.get("primary_metric") or (self.metric_info or {}).get("metric_name"),
            "primary_metric_unit": frame_data.get("primary_metric_unit") or (self.metric_info or {}).get("metric_unit"),
            "selected_side": frame_data.get("selected_side"),
            "visibility_min": frame_data.get("visibility_min") or frame_data.get("visibility"),
            "rehab_keypoints": _compact_rehab_keypoints(rehab_keypoints) if isinstance(rehab_keypoints, dict) else {},
        }

    def _save_keyframe_candidate(self, rep_index: int) -> dict[str, Any] | None:
        candidate = self._keyframe_candidate
        self._keyframe_candidate = None
        if not isinstance(candidate, dict) or not candidate.get("image_jpeg"):
            return None
        try:
            session_id = self._safe_session_id(self.session_id)
            action_token = self._safe_file_token(self.action_id)
            out_dir = KEYFRAMES_DIR / session_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{action_token}_rep{rep_index}_best.jpg"
            out_path.write_bytes(candidate["image_jpeg"])
            keyframe = {
                "session_id": session_id,
                "action_id": self.action_id,
                "action_name": self.current_action_meta.get("action_name") if self.current_action_meta else self.action_id,
                "rep_index": rep_index,
                "kind": "best_peak",
                "image_path": self._project_relative(out_path),
                "signal_value": candidate.get("signal_value"),
                "primary_metric": candidate.get("primary_metric"),
                "primary_metric_unit": candidate.get("primary_metric_unit"),
                "frame_index": candidate.get("frame_index"),
                "relative_time": candidate.get("relative_time"),
                "selected_side": candidate.get("selected_side"),
                "visibility_min": candidate.get("visibility_min"),
                "rehab_keypoints": candidate.get("rehab_keypoints") if isinstance(candidate.get("rehab_keypoints"), dict) else {},
            }
            self.keyframes.append(keyframe)
            return keyframe
        except Exception as exc:
            self.keyframe_errors.append(f"keyframe_save_failed: {exc}")
            return None

    def _current_action_keyframes(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.keyframes if item.get("action_id") == self.action_id]

    def _run_evaluate(self, attempt_file: str) -> dict[str, Any]:
        active_template = self.active_template or {}
        template_file = str(active_template.get("template_file") or "")
        action_meta = self._find_plan_action(self.action_id, self._load_demo_plan())
        config_file = str(active_template.get("config_file") or action_meta.get("config_file") or "evaluate/configs/knee_flexion.yaml")
        report_path = REPORTS_DIR / f"report_{self.action_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        command = [
            str(PYTHON_EXE),
            "evaluate/banzi/run_evaluate.py",
            "--template",
            template_file,
            "--attempt",
            attempt_file,
            "--config",
            config_file,
            "--out",
            self._project_relative(report_path),
        ]
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), shell=False, timeout=30, capture_output=True, text=True)
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": "评估失败",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }
        report = json.loads(report_path.read_text(encoding="utf-8"))
        action_meta = self._find_plan_action(self.action_id, self._load_demo_plan())
        feedback_rule = self._resolve_project_path(action_meta.get("feedback_rule_file") or self._feedback_rule_for_action(self.action_id))
        if not feedback_rule.exists():
            feedback_rule = DEFAULT_FEEDBACK_RULE
        feedback = build_feedback_from_files(report_path, feedback_rule)
        return {
            "ok": True,
            "report_file": self._project_relative(report_path),
            "report": report,
            "feedback": feedback,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _apply_action_metric(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        eval_config = self.eval_config or {}
        if not eval_config.get("primary_metric"):
            return frame_data
        try:
            metric = extract_metric_from_frame(
                frame_data,
                eval_config,
                baseline_hip_y=self.metric_baseline_hip_y,
                baseline_torso_height=self.metric_baseline_torso_height,
            )
        except ValueError as exc:
            self.last_prompt = ""
            frame_data["target_angle_raw"] = None
            frame_data["target_angle_smoothed"] = None
            frame_data["metric_error"] = str(exc)
            return frame_data
        self.metric_baseline_hip_y = _as_float(metric.get("baseline_hip_y"))
        self.metric_baseline_torso_height = _as_float(metric.get("baseline_torso_height"))
        value = _as_float(metric.get("value"))
        if value is None:
            return frame_data
        frame_data["primary_metric"] = metric.get("metric_name")
        frame_data["primary_metric_unit"] = metric.get("metric_unit")
        frame_data["primary_signal_raw"] = value
        frame_data["primary_signal_smoothed"] = value
        frame_data["target_angle_raw"] = value
        frame_data["target_angle_smoothed"] = value
        return frame_data

    def _merge_realtime_config(self, realtime_config: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]:
        merged = dict(realtime_config)
        override = eval_config.get("realtime")
        if isinstance(override, dict):
            merged.update(override)
        return merged

    def _metric_info_from_config(self, eval_config: dict[str, Any]) -> dict[str, Any] | None:
        metric_name = str(eval_config.get("primary_metric") or "").strip()
        if not metric_name:
            return None
        return {
            "metric_name": metric_name,
            "metric_unit": eval_config.get("metric_unit"),
            "metric_direction": eval_config.get("metric_direction"),
            "secondary_metric": eval_config.get("secondary_metric"),
        }

    def _build_targets(
        self,
        template_path: Path,
        eval_config: dict[str, Any],
        realtime_config: dict[str, Any],
    ) -> KneeFlexionTargets:
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        frames = payload.get("template_frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("active template 中没有 template_frames。")
        if eval_config.get("primary_metric"):
            metric_sequence = extract_metric_sequence(frames, eval_config)
            metric_frames = metric_sequence["frames"]
            angle_field = METRIC_VALUE_FIELD
        else:
            metric_frames = frames
            angle_field = select_angle_field(frames, eval_config)
        template_rom = compute_rom(metric_frames, angle_field)
        target_range = build_tut_range(template_rom, eval_config)
        template_tut = compute_tut(metric_frames, target_range, angle_field)
        template_speed = check_speed(metric_frames, angle_field)
        thresholds = eval_config.get("thresholds", {}) if isinstance(eval_config.get("thresholds"), dict) else {}
        realtime_target_range = self._expand_target_range(target_range, realtime_config.get("tut_range_padding"))
        return KneeFlexionTargets(
            rom_target=float(template_rom.get("rom") or realtime_config.get("min_rom", 30.0)),
            tut_target=float(template_tut.get("tut_seconds") or 0.0),
            target_range=realtime_target_range,
            template_peak_speed=float(template_speed.get("peak_angular_velocity") or 0.0),
            rom_diff_max=float(thresholds.get("rom_diff_max", 10.0)),
            tut_ratio_min=float(realtime_config.get("tut_ratio_min", thresholds.get("tut_ratio_min", 0.8))),
            speed_ratio_max=float(thresholds.get("speed_ratio_max", 1.5)),
        )

    def _expand_target_range(self, target_range: tuple[float, float], padding_value: Any) -> tuple[float, float]:
        padding = max(0.0, float(padding_value or 0.0))
        low = float(target_range[0]) - padding
        high = float(target_range[1]) + padding
        return min(low, high), max(low, high)

    def _get_active_template(self, action_id: str) -> dict[str, Any] | None:
        return registry_get_active_template(action_id, self.pose_backend, ACTIVE_TEMPLATES_PATH)

    def _template_side(self, template_path: Path | None = None) -> str | None:
        candidate_path = template_path
        if candidate_path is None and isinstance(self.active_template, dict):
            raw_path = str(self.active_template.get("template_file") or "")
            if raw_path:
                candidate_path = self._resolve_project_path(raw_path)
        if candidate_path is None or not candidate_path.exists():
            return None
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        rule = payload.get("keypoint_rule") if isinstance(payload, dict) else None
        if isinstance(rule, dict):
            side = str(rule.get("side") or "").strip().lower()
            if side in {"left", "right"}:
                return side
        frames = payload.get("template_frames") if isinstance(payload, dict) else None
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    side = str(frame.get("selected_side") or "").strip().lower()
                    if side in {"left", "right"}:
                        return side
        return None

    def _load_demo_plan(self) -> dict[str, Any]:
        if not DEFAULT_DEMO_PLAN.exists():
            return {"actions": []}
        return self._load_yaml(DEFAULT_DEMO_PLAN)

    def _find_plan_action(self, action_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        actions = plan.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and action.get("action_id") == action_id:
                    return action
        return {"action_id": action_id}

    def _feedback_rule_for_action(self, action_id: str) -> str:
        return f"feedback/rules/{action_id}_feedback.yaml"

    def _plan_snapshot(self) -> dict[str, Any]:
        plan = self._load_demo_plan()
        actions = plan.get("actions")
        return {
            "plan_id": plan.get("plan_id"),
            "plan_name": plan.get("plan_name"),
            "default_target_reps": plan.get("default_target_reps"),
            "count_tts": plan.get("count_tts"),
            "rest_music_file": plan.get("rest_music_file"),
            "rest_music_fade_seconds": plan.get("rest_music_fade_seconds"),
            "actions": [
                {
                    "action_id": action.get("action_id"),
                    "action_name": action.get("action_name"),
                    "camera_prompt": action.get("camera_prompt"),
                    "config_file": action.get("config_file"),
                    "feedback_rule_file": action.get("feedback_rule_file"),
                    "require_side_view": action.get("require_side_view"),
                    "action_intro_tts": action.get("action_intro_tts"),
                    "has_active_template": self._get_active_template(str(action.get("action_id"))) is not None,
                }
                for action in actions
                if isinstance(action, dict)
            ]
            if isinstance(actions, list)
            else [],
        }

    def _runtime_threshold_snapshot(self, machine_state: dict[str, Any]) -> dict[str, Any]:
        config = self.current_realtime_config if isinstance(self.current_realtime_config, dict) else {}
        targets = self.current_targets
        start_delta = float(config.get("start_delta", 10.0))
        attempt_start_delta = float(config.get("attempt_start_delta", start_delta))
        min_attempt_delta = config.get("min_attempt_delta")
        if min_attempt_delta is None:
            min_attempt_delta = attempt_start_delta * 0.5
        target_range = list(targets.target_range) if targets is not None else list(machine_state.get("target_range") or [])
        tut_count_range = list(machine_state.get("tut_count_range") or target_range)
        last_attempt = self.last_invalid_attempt
        if not isinstance(last_attempt, dict) and self.rep_results:
            last_attempt = self.rep_results[-1]
        tut_ratio_min = float(targets.tut_ratio_min) if targets is not None else float(config.get("tut_ratio_min", 0.0) or 0.0)
        tut_target = float(targets.tut_target) if targets is not None else float(machine_state.get("tut_target") or 0.0)
        return {
            "baseline": machine_state.get("baseline_angle"),
            "current_metric": machine_state.get("angle"),
            "segment_mode": machine_state.get("segment_mode") or config.get("segment_mode"),
            "target_low": target_range[0] if len(target_range) >= 1 else None,
            "target_high": target_range[1] if len(target_range) >= 2 else None,
            "start_delta": start_delta,
            "attempt_start_delta": attempt_start_delta,
            "return_delta": float(config.get("return_delta", 6.0)),
            "min_attempt_delta": float(min_attempt_delta),
            "rom_diff_max": float(targets.rom_diff_max) if targets is not None else None,
            "tut_required_seconds": tut_target * tut_ratio_min,
            "tut_ratio_min": tut_ratio_min,
            "tut_count_mode": machine_state.get("tut_count_mode") or str(config.get("tut_count_mode") or "target_range"),
            "tut_count_low": tut_count_range[0] if len(tut_count_range) >= 1 else None,
            "tut_count_high": tut_count_range[1] if len(tut_count_range) >= 2 else None,
            "in_tut_zone": machine_state.get("in_tut_zone"),
            "last_rep_primary_error": last_attempt.get("primary_error") if isinstance(last_attempt, dict) else None,
            "last_rep_all_errors": last_attempt.get("all_errors") if isinstance(last_attempt, dict) else None,
            "count_by_peak_target": bool(config.get("count_by_peak_target", False)),
            "strict_quality_errors": bool(config.get("strict_quality_errors", True)),
            "motion_state_age_seconds": self._motion_state_age_seconds(machine_state),
            "rep_start_signal": machine_state.get("rep_start_signal"),
            "rep_peak_signal": machine_state.get("rep_peak_signal"),
            "rep_lowest_after_peak": machine_state.get("rep_lowest_after_peak"),
            "rep_last_signal": machine_state.get("rep_last_signal"),
            "return_close_to_start": machine_state.get("return_close_to_start"),
            "rest_anchor": machine_state.get("rest_anchor"),
            "rest_noise": machine_state.get("rest_noise"),
            "velocity": machine_state.get("velocity"),
            "motion_delta": machine_state.get("motion_delta"),
            "start_delta_used": machine_state.get("start_delta_used"),
            "start_ready": bool(machine_state.get("start_ready", False)),
            "reentry_state": self.reentry_state,
            "reentry_ready": bool(self.reentry_ready),
            "baseline_ready": self._baseline_ready_from_output(machine_state),
            "reentry_sample_count": len(self._reentry_samples),
            "peak_value": machine_state.get("peak_value"),
            "stable_return_seconds": machine_state.get("stable_return_seconds"),
            "rep_audio_suppressed": machine_state.get("rep_audio_suppressed") or not self._rep_audio_allowed(),
        }

    def _motion_state_age_seconds(self, machine_state: dict[str, Any]) -> float | None:
        started_at = _as_float(machine_state.get("rep_state_started_at") or machine_state.get("rep_started_at"))
        if started_at is None or self.start_time is None:
            return None
        return max(0.0, round((time.time() - self.start_time) - started_at, 2))

    def _missing_plan_templates(self) -> list[str]:
        actions = self._load_demo_plan().get("actions")
        if not isinstance(actions, list):
            return []
        missing: list[str] = []
        for action in actions:
            if isinstance(action, dict):
                action_id = str(action.get("action_id") or "")
                if action_id and self._get_active_template(action_id) is None:
                    missing.append(action_id)
        return missing

    def _begin_action_after_audio(
        self,
        *,
        reset_timing: bool,
        include_action_guidance: bool,
        speak_orientation: bool,
        prefix_text: str | None = None,
    ) -> None:
        self._reset_active_motion_state()
        wait_for_audio = False
        guidance_spoken = False
        if self.tts_worker is not None:
            for text_part, phrase_key in self._action_start_tts_parts(prefix_text, include_action_guidance):
                if self._rep_audio_allowed() and self.tts_worker.speak(text_part, priority="high", event_type="action_start", phrase_key=phrase_key):
                    wait_for_audio = True
                    guidance_spoken = guidance_spoken or bool(phrase_key and (phrase_key.startswith("start_") or phrase_key.startswith("intro_")))
                    self.last_tts_text = text_part
        self.action_guidance_spoken = self.action_guidance_spoken or guidance_spoken
        if wait_for_audio:
            self.status = "awaiting_action_audio"
            self.pause_reason = "action_audio"
            self.action_guard_until = None
            self.pending_action_start = {
                "reset_timing": bool(reset_timing),
                "speak_orientation": bool(speak_orientation),
            }
            self.last_prompt = "请听完提示后再开始动作。"
            return
        self._activate_action_after_audio(reset_timing=reset_timing, speak_orientation=speak_orientation)

    def _maybe_start_pending_action(self) -> None:
        if self.tts_worker is not None and self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds):
            self.last_prompt = "请听完提示后再开始动作。"
            return
        pending = self.pending_action_start if isinstance(self.pending_action_start, dict) else {}
        self._activate_action_after_audio(
            reset_timing=bool(pending.get("reset_timing", True)),
            speak_orientation=bool(pending.get("speak_orientation", False)),
        )

    def _activate_action_after_audio(self, *, reset_timing: bool, speak_orientation: bool) -> None:
        self.pending_action_start = None
        self.pause_reason = None
        self._reset_active_motion_state()
        if self._needs_initial_orientation_gate():
            self._enter_orientation_wait(speak=speak_orientation)
        else:
            self._start_running_phase(
                prefix_text=None,
                reset_timing=reset_timing,
                include_action_guidance=False,
                speak_action_audio=False,
            )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}

    def _natural_tts_options(self, config: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if "tts_sid" in config:
            options["sid"] = int(config["tts_sid"])
        if "tts_speed" in config:
            options["speed"] = float(config["tts_speed"])
        if "tts_silence_scale" in config:
            options["silence_scale"] = float(config["tts_silence_scale"])
        return options

    def _error_reason(self, rep_result: dict[str, Any]) -> str:
        code = str(rep_result.get("primary_error") or "")
        return {
            "ROM_LOW": "动作不到位",
            "EARLY_RETURN": "还没做到位就放下",
            "TUT_LOW": "保持时间不足",
            "TOO_FAST": "动作未完成",
            "SHAPE_BAD": "动作轨迹不标准",
        }.get(code, "动作不标准")

    def _resolve_project_path(self, value: object) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _project_relative(self, path: str | Path) -> str:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        try:
            return resolved.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _pose_meta_from_frames(self) -> dict[str, Any]:
        for frame in self.frames:
            backend = frame.get("actual_backend") or frame.get("pose_backend")
            if backend:
                return {
                    "requested_backend": frame.get("requested_backend"),
                    "actual_backend": backend,
                    "pose_backend": frame.get("pose_backend") or backend,
                    "pose_backend_version": frame.get("pose_backend_version"),
                    "pose_keypoint_schema": frame.get("pose_keypoint_schema"),
                    "rknn_model_path": frame.get("rknn_model_path"),
                }
        active_template = self.active_template if isinstance(self.active_template, dict) else {}
        backend = active_template.get("actual_backend") or active_template.get("pose_backend")
        return {
            "requested_backend": active_template.get("requested_backend"),
            "actual_backend": backend,
            "pose_backend": active_template.get("pose_backend") or backend,
            "pose_backend_version": active_template.get("pose_backend_version"),
            "pose_keypoint_schema": active_template.get("pose_keypoint_schema"),
            "rknn_model_path": active_template.get("rknn_model_path"),
        }

    def _empty_care_dialog(self) -> dict[str, Any]:
        return {
            "visible": False,
            "title": "温馨提示",
            "message": DEFAULT_CARE_PROMPT,
            "yes_label": "是",
            "no_label": "否",
        }

    def _in_action_guard(self, now: float) -> bool:
        return self.action_guard_until is not None and now < self.action_guard_until

    def _reentry_calibration_active(self, now: float) -> bool:
        if self.last_offscreen_resume_at is None or self.reentry_ready:
            return False
        return self.status == "running"

    def _reentry_strict_start_active(self, now: float) -> bool:
        return self.offscreen_reentry_until is not None and now <= self.offscreen_reentry_until

    def _warm_reentry_calibration(self, frame_data: dict[str, Any], now: float) -> None:
        if self.machine is None or self.start_time is None:
            return
        frame_data["frame_index"] = self.frame_index
        frame_data["relative_time"] = now - self.start_time
        frame_data["selected_side"] = TRAINING_SIDE
        self.frame_index += 1
        metric_frame = self._apply_action_metric(frame_data)
        self.last_frame_data = dict(metric_frame)
        angle = _as_float(metric_frame.get("target_angle_smoothed"))
        if angle is not None:
            self._reentry_samples.append((now, angle))
            window = max(0.5, float(self.offscreen_reentry_guard_seconds or 1.2))
            self._reentry_samples = [(ts, value) for ts, value in self._reentry_samples if now - ts <= window]
        machine_state = str(getattr(getattr(self.machine, "state", None), "value", ""))
        if machine_state == "BASELINE":
            output = self.machine.process(metric_frame)
            if isinstance(output, dict):
                self.last_machine_output = output
        if self._reentry_baseline_ready(now):
            self.reentry_state = "reentry_ready"
            self.reentry_ready = True
            self.action_guard_until = None
            self.last_prompt = "可以开始动作"
            return
        self.reentry_state = "reentry_calibrating"
        self.last_prompt = "请保持静止，正在校准"

    def _reentry_baseline_ready(self, now: float) -> bool:
        if self.last_offscreen_resume_at is None:
            return True
        guard_seconds = max(0.5, float(self.offscreen_reentry_guard_seconds or 1.2))
        if now - self.last_offscreen_resume_at < guard_seconds:
            return False
        if len(self._reentry_samples) < int((self.current_realtime_config or {}).get("reentry_min_stable_samples", 5) or 5):
            return False
        values = [value for _, value in self._reentry_samples]
        noise = max(values) - min(values) if values else 999.0
        elapsed = max(1e-3, self._reentry_samples[-1][0] - self._reentry_samples[0][0])
        velocity = abs((self._reentry_samples[-1][1] - self._reentry_samples[0][1]) / elapsed)
        rest_noise_max = float((self.current_realtime_config or {}).get("rest_noise_max", 0.025) or 0.025)
        rest_velocity_max = float((self.current_realtime_config or {}).get("rest_velocity_max", 0.06) or 0.06)
        baseline_ok = self._baseline_ready_from_output(self.last_machine_output or {})
        if baseline_ok and noise <= rest_noise_max and velocity <= rest_velocity_max:
            if self.machine is not None:
                stable_anchor = sum(values) / len(values)
                self.machine.baseline_angle = stable_anchor
                self.machine.rest_anchor = stable_anchor
                self.machine.rest_noise = noise
            return True
        return False

    def _baseline_ready_from_output(self, output: dict[str, Any]) -> bool:
        return output.get("baseline_angle") is not None or output.get("rest_anchor") is not None

    def _warm_baseline_during_action_guard(self, frame_data: dict[str, Any], now: float) -> None:
        if self.machine is None or self.start_time is None:
            return
        machine_state = str(getattr(getattr(self.machine, "state", None), "value", ""))
        if machine_state != "BASELINE":
            return
        frame_data["frame_index"] = self.frame_index
        frame_data["relative_time"] = now - self.start_time
        frame_data["selected_side"] = TRAINING_SIDE
        self.frame_index += 1
        metric_frame = self._apply_action_metric(frame_data)
        self.last_frame_data = dict(metric_frame)
        output = self.machine.process(metric_frame)
        if not isinstance(output, dict):
            current_output = getattr(self.machine, "_output", None)
            if callable(current_output):
                output = current_output(
                    visible=bool(metric_frame.get("person_visible", metric_frame.get("pose_detected", True))),
                    action_keypoints_valid=metric_frame.get("target_angle_smoothed") is not None,
                    angle=_as_float(metric_frame.get("target_angle_smoothed")),
                )
            else:
                output = self._safe_machine_output(metric_frame, reason="guard_baseline_warmup")
        self.last_machine_output = output

    def _safe_machine_output(self, frame_data: dict[str, Any], *, reason: str) -> dict[str, Any]:
        target_range = list(self.current_targets.target_range) if self.current_targets is not None else []
        return {
            "state": "RECOVERING",
            "visible": bool(frame_data.get("person_visible", frame_data.get("pose_detected", True))),
            "action_keypoints_valid": bool(frame_data.get("action_keypoints_valid", False)),
            "angle": _as_float(frame_data.get("target_angle_smoothed")),
            "baseline_angle": None,
            "target_range": target_range,
            "tut_seconds": 0.0,
            "tut_target": self.current_targets.tut_target if self.current_targets is not None else None,
            "missing_seconds": 0.0,
            "prompt": "识别状态恢复中",
            "rep_result": None,
            "recover_reason": reason,
        }

    def _tts_is_busy(self) -> bool:
        if self.tts_worker is None:
            return False
        return self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds)

    def _should_speak_correction(self, error_code: str | None = None, rep_result: dict[str, Any] | None = None) -> bool:
        now = time.time()
        normalized = str(error_code or "").strip()
        if not normalized:
            return False
        if isinstance(rep_result, dict):
            duration = _as_float(rep_result.get("duration_seconds"))
            min_seconds = float((self.current_realtime_config or {}).get("min_rep_seconds", 1.0) or 1.0)
            if duration is not None and duration < min_seconds:
                return False
        cooldown_seconds = float((self.current_realtime_config or {}).get("correction_tts_interval_seconds", self.correction_tts_interval_seconds) or 0.0)
        if now - self.last_correction_tts_at < cooldown_seconds:
            return False
        self.last_correction_tts_at = now
        self.last_correction_error_code = normalized
        return True

    def _display_prompt_from_machine(self, output: dict[str, Any]) -> str:
        raw_prompt = str(output.get("prompt") or "").strip()
        blocked = {
            "准备开始第一遍",
            "准备开始下一遍",
            "请保持静止，正在校准",
            "请保持目标腿关键点可见",
            "请保持关键点可见",
            "识别状态恢复中",
        }
        if not raw_prompt or raw_prompt in blocked:
            return ""
        if raw_prompt.startswith("准备开始") or "关键点" in raw_prompt:
            return ""
        return raw_prompt

    def _create_tts_worker(
        self,
        *,
        global_cooldown: float,
        same_text_cooldown: float,
        natural_tts_options: dict[str, Any] | None,
    ) -> TTSWorker:
        return TTSWorker(
            use_real_tts=True,
            lazy_real_tts_init=True,
            global_cooldown=global_cooldown,
            same_text_cooldown=same_text_cooldown,
            natural_tts_options=natural_tts_options,
            phrase_config_path=DEFAULT_TTS_PHRASES,
            project_root=PROJECT_ROOT,
        )

    def _clear_stale_tts(self, event_types: set[str] | None = None) -> None:
        if self.tts_worker is not None and hasattr(self.tts_worker, "clear_pending"):
            self.tts_worker.clear_pending(event_types or STALE_TTS_EVENTS)

    def _needs_initial_orientation_gate(self) -> bool:
        if not self.orientation_required:
            return False
        if not self.playlist_mode:
            return not self.initial_orientation_done
        return self.playlist_index == 0 and not self.initial_orientation_done

    def _selected_side_snapshot(self) -> str:
        frame_side = str((self.last_frame_data or {}).get("selected_side") or "").strip().lower()
        if frame_side in {"left", "right"}:
            return frame_side
        rule_side = str((self.selected_rule or {}).get("side") or "").strip().lower()
        if rule_side in {"left", "right"}:
            return rule_side
        return TRAINING_SIDE
    def _action_start_tts_parts(self, prefix_text: str | None, include_action_guidance: bool) -> list[tuple[str, str | None]]:
        parts: list[tuple[str, str | None]] = []
        if prefix_text:
            if prefix_text == INSCREEN_PROMPT:
                parts.append((prefix_text, "inscreen"))
            else:
                parts.append((prefix_text, "resume" if "继续" in prefix_text or "休息结束" in prefix_text else None))
        if include_action_guidance:
            action_id = self.action_id
            start_tts = str((self.current_action_meta or {}).get("start_tts") or "")
            if start_tts:
                parts.append((start_tts, f"start_{action_id}"))
            if self.action_intro_tts:
                parts.append((self.action_intro_tts, f"intro_{action_id}"))
        return parts

    def _event_phrase_key(self, event_type: str, text: str) -> str | None:
        if event_type == "set_done":
            return "set_done"
        if event_type == "resume":
            return "resume"
        if event_type == "care":
            return "care"
        if "休息" in text:
            return "rest"
        return None

    def _correction_phrase_key(self, error_code: str, text: str, rep_result: dict[str, Any]) -> str | None:
        if error_code == "TUT_LOW":
            missing = _as_float(rep_result.get("missing_seconds"))
            if missing is not None:
                return f"tut_{min(5, max(1, int(missing + 0.999)))}"
        action_suffix = {
            "seated_knee_extension": "extension",
            "sit_to_stand": "stand",
            "standing_hamstring_curl": "curl",
            "seated_knee_raise": "raise",
        }.get(self.action_id)
        if error_code in {"ROM_LOW", "TOO_FAST", "EARLY_RETURN"} and action_suffix:
            return f"rom_{action_suffix}"
        if error_code == "SHAPE_BAD" and action_suffix:
            return f"shape_{action_suffix}"
        return None

    def _new_session_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _safe_session_id(self, value: object) -> str:
        text = "".join(ch for ch in str(value or "") if ch.isalnum() or ch in {"_", "-"})
        return text or self._new_session_id()

    def _safe_file_token(self, value: object) -> str:
        text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "action"))
        return text.strip("_") or "action"

    def _load_plan_runtime_settings(self, plan: dict[str, Any]) -> None:
        self.rest_seconds = int(plan.get("rest_seconds", 10) or 10)
        self.offscreen_timeout_seconds = float(plan.get("offscreen_timeout_seconds", 5) or 5)
        self.care_prompt_invalid_streak = int(plan.get("care_prompt_invalid_streak", 5) or 5)
        self.action_start_guard_seconds = float(plan.get("action_start_guard_seconds", 2.0) or 2.0)
        self.tts_action_guard_extra_seconds = float(plan.get("tts_action_guard_extra_seconds", 0.5) or 0.5)
        self.correction_tts_interval_seconds = float(plan.get("correction_tts_interval_seconds", 4.0) or 4.0)
        self.offscreen_reentry_guard_seconds = float(plan.get("offscreen_reentry_guard_seconds", 1.2) or 1.2)
        self.front_orientation_confirm_frames = int(plan.get("front_orientation_confirm_frames", 2) or 2)
        self.rknn_front_orientation_confirm_frames = int(plan.get("rknn_front_orientation_confirm_frames", 2) or 2)
        self.rknn_orientation_confirm_frames = int(plan.get("rknn_orientation_confirm_frames", 4) or 4)
        self.rest_music = {
            "enabled": True,
            "file": str(plan.get("rest_music_file") or "/assets/rest_music.wav"),
            "fade_seconds": float(plan.get("rest_music_fade_seconds", 2.5) or 2.5),
            "playback": str(plan.get("rest_music_playback") or "backend"),
        }

    def _reset_active_motion_state(self) -> None:
        if self.current_targets is not None:
            self.machine = KneeFlexionRealtimeMachine(self.current_realtime_config, self.current_targets)
        self.last_machine_output = None
        self.metric_baseline_hip_y = None
        self.metric_baseline_torso_height = None
        self.orientation_confirm_count = 0
        self.front_orientation_confirm_count = 0
        self.return_confirm_count = 0
        self.front_view_ok = False
        self.side_view_ok = False
        self.offscreen_since = None
        self.offscreen_seconds = 0.0
        self.offscreen_prompt_pending = False
        self.offscreen_prompt_spoken = False
        self.action_guard_until = None
        self.last_correction_error_code = None

    def _enter_orientation_wait(self, *, speak: bool) -> None:
        self.status = "awaiting_orientation"
        self.pause_reason = "orientation"
        self.orientation_confirm_count = 0
        self.front_orientation_confirm_count = 0
        self.orientation_phase = "awaiting_front"
        self.orientation_state = "waiting_front_view"
        self.orientation_prompt_spoken = False
        self.action_guard_until = None
        self.last_prompt = "请先正对镜头。"

    def _start_running_phase(
        self,
        *,
        prefix_text: str | None,
        reset_timing: bool,
        include_action_guidance: bool,
        speak_action_audio: bool = True,
    ) -> None:
        if self.machine is None:
            self._reset_active_motion_state()
        if reset_timing or self.start_time is None:
            self.start_time = time.time()
        self.status = "running"
        self.pause_reason = None
        self.rest_audio_player.stop()
        self.care_dialog = self._empty_care_dialog()
        self.last_prompt = "请保持静止，正在校准"
        self.action_guard_until = time.time() + max(0.0, self.action_start_guard_seconds)
        spoken_parts: list[str] = []
        if prefix_text:
            spoken_parts.append(prefix_text)
        if include_action_guidance:
            start_tts = str((self.current_action_meta or {}).get("start_tts") or "")
            if start_tts:
                spoken_parts.append(start_tts)
            if self.action_intro_tts:
                spoken_parts.append(self.action_intro_tts)
        spoken_text = "".join(part for part in spoken_parts if part)
        if include_action_guidance and spoken_text:
            self.action_guidance_spoken = True
        if speak_action_audio and spoken_text and self.tts_worker and self._rep_audio_allowed():
            self.last_tts_text = spoken_text
            for text_part, phrase_key in self._action_start_tts_parts(prefix_text, include_action_guidance):
                if self._rep_audio_allowed():
                    self.tts_worker.speak(text_part, priority="high", event_type="action_start", phrase_key=phrase_key)

    def _update_offscreen_tracking(self, now: float, pose_detected: bool) -> None:
        if pose_detected:
            self.offscreen_since = None
            self.offscreen_seconds = 0.0
            return
        if self.offscreen_since is None:
            self.offscreen_since = now
        self.offscreen_seconds = now - self.offscreen_since

    def _orientation_confirm_frames_for_frame(self, frame_data: dict[str, Any]) -> int:
        if frame_data.get("actual_backend") == "rknn" or frame_data.get("pose_backend") == "rknn":
            return self.rknn_orientation_confirm_frames
        return self.orientation_confirm_frames

    def _front_orientation_confirm_frames_for_frame(self, frame_data: dict[str, Any]) -> int:
        if frame_data.get("actual_backend") == "rknn" or frame_data.get("pose_backend") == "rknn":
            return self.rknn_front_orientation_confirm_frames
        return self.front_orientation_confirm_frames

    def _process_orientation_gate(self, pose_detected: bool, orientation_ok: bool, frame_data: dict[str, Any]) -> None:
        front_view_ok = bool(frame_data.get("front_view_ok"))
        side_view_ok = bool(frame_data.get("side_view_ok", frame_data.get("orientation_ok")))
        front_required_frames = self._front_orientation_confirm_frames_for_frame(frame_data)
        side_required_frames = self._orientation_confirm_frames_for_frame(frame_data)
        if not pose_detected:
            self.orientation_confirm_count = 0
            self.front_orientation_confirm_count = 0
            self.orientation_state = "no_person"
            self.last_prompt = "请先回到画面中。"
            return
        if self.orientation_phase in {"idle", "awaiting_front"}:
            self.orientation_phase = "awaiting_front"
            if not front_view_ok:
                self.front_orientation_confirm_count = 0
                self.orientation_state = "waiting_front_view"
                self.last_prompt = "请先正对镜头。"
                return
            self.front_orientation_confirm_count += 1
            self.orientation_state = "front_view_confirming"
            self.last_prompt = "已检测到正对镜头。"
            if self.front_orientation_confirm_count < front_required_frames:
                return
            self.front_orientation_confirm_count = 0
            self.orientation_confirm_count = 0
            self.orientation_phase = "awaiting_side"
            self.orientation_state = "waiting_side_view"
            self.last_prompt = self.orientation_prompt
            if self.tts_worker and self._rep_audio_allowed() and not self.orientation_prompt_spoken:
                self.last_tts_text = self.orientation_prompt
                if self.tts_worker.speak(self.orientation_prompt, priority="high", event_type="orientation", phrase_key="orientation"):
                    self.orientation_prompt_spoken = True
            return
        if self.tts_worker is not None and self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds):
            self.orientation_state = "orientation_audio_playing"
            self.last_prompt = self.orientation_prompt
            return
        if not side_view_ok or not orientation_ok:
            self.orientation_confirm_count = 0
            self.orientation_state = "waiting_side_view"
            self.last_prompt = self.orientation_prompt
            return
        self.orientation_confirm_count += 1
        self.orientation_state = "side_view_confirming"
        self.last_prompt = "角度可以，马上开始。"
        if self.orientation_confirm_count < side_required_frames:
            return
        self.orientation_confirm_count = 0
        self.orientation_phase = "ready"
        self.orientation_state = "side_view_ok"
        self.initial_orientation_done = True
        if self.tts_worker and self._rep_audio_allowed():
            self.last_tts_text = ANGLE_RIGHT_PROMPT
            if self._rep_audio_allowed():
                self.tts_worker.speak(ANGLE_RIGHT_PROMPT, priority="high", event_type="orientation", phrase_key="angle_right")
        self._begin_action_after_audio(
            reset_timing=self.start_time is None,
            include_action_guidance=not self.action_guidance_spoken,
            speak_orientation=False,
        )


    def _maybe_speak_pending_offscreen_prompt(self) -> None:
        if not self.offscreen_prompt_pending:
            return
        self._speak_offscreen_prompt()

    def _speak_offscreen_prompt(self) -> bool:
        if self.tts_worker is None:
            return False
        if self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds):
            return False
        self.last_tts_text = DEFAULT_OFFSCREEN_PROMPT
        if self.tts_worker.speak(DEFAULT_OFFSCREEN_PROMPT, priority="high", event_type="offscreen", phrase_key="offscreen"):
            self.offscreen_prompt_pending = False
            self.offscreen_prompt_spoken = True
            return True
        return False

    def _process_return_gate(self, pose_detected: bool, orientation_ok: bool, frame_data: dict[str, Any]) -> None:
        if not pose_detected:
            self.return_confirm_count = 0
            self.last_prompt = DEFAULT_OFFSCREEN_PROMPT
            return
        if self.tts_worker is not None and self.tts_worker.is_busy(extra_guard_seconds=self.tts_action_guard_extra_seconds):
            self.last_prompt = DEFAULT_OFFSCREEN_PROMPT
            return
        metric_frame = self._apply_action_metric(dict(frame_data))
        keypoints_ok = bool(metric_frame.get("action_keypoints_valid", True)) and bool(metric_frame.get("target_angle_smoothed") is not None)
        if not keypoints_ok:
            self.return_confirm_count = 0
            self.last_prompt = "请完整入画并站稳。"
            return
        side_view_ok = bool(frame_data.get("side_view_ok", frame_data.get("orientation_ok"))) or not self.orientation_required
        self.return_confirm_count += 1
        if self.return_confirm_count < self._orientation_confirm_frames_for_frame(frame_data):
            self.last_prompt = "请保持稳定，马上继续。" if side_view_ok else self.orientation_prompt
            return
        self._resume_running_after_offscreen()

    def _resume_running_after_offscreen(self) -> None:
        now = time.time()
        self.return_confirm_count = 0
        self._reset_active_motion_state()
        self.status = "running"
        self.pause_reason = None
        self.start_time = now
        guard_seconds = max(0.0, float(self.offscreen_reentry_guard_seconds or 0.0))
        self.action_guard_until = now + guard_seconds
        self.last_offscreen_resume_at = now
        self.offscreen_reentry_until = now + guard_seconds + 2.0
        self.reentry_state = "reentry_calibrating"
        self.reentry_ready = False
        self._reentry_samples = []
        self.offscreen_since = None
        self.offscreen_seconds = 0.0
        self.offscreen_prompt_pending = False
        self.offscreen_prompt_spoken = False
        self.rest_audio_player.stop()
        self.last_prompt = "请保持静止，正在校准"
        if self.tts_worker and self._rep_audio_allowed():
            self.last_tts_text = INSCREEN_PROMPT
            self.tts_worker.speak(INSCREEN_PROMPT, priority="high", event_type="action_start", phrase_key="inscreen")

    def _enter_offscreen_wait(self) -> None:
        machine_state = str((self.last_machine_output or {}).get("state") or "")
        if not machine_state and self.machine is not None:
            machine_state = str(getattr(getattr(self.machine, "state", None), "value", ""))
        if bool((self.current_realtime_config or {}).get("reset_on_offscreen", False)):
            self._reset_active_motion_state()
        self.status = "awaiting_return"
        self.pause_reason = "offscreen"
        self.action_guard_until = None
        self.rest_audio_player.stop()
        self.return_confirm_count = 0
        self.offscreen_prompt_pending = True
        self.last_prompt = DEFAULT_OFFSCREEN_PROMPT
        self._speak_offscreen_prompt()

    def _show_care_dialog(self) -> None:
        self.status = "awaiting_care_response"
        self.pause_reason = "care_dialog"
        self.care_dialog = {
            "visible": True,
            "title": "温馨提示",
            "message": DEFAULT_CARE_PROMPT,
            "yes_label": "是",
            "no_label": "否",
        }
        self.last_prompt = DEFAULT_CARE_PROMPT
        if self.tts_worker:
            self.last_tts_text = DEFAULT_CARE_PROMPT
            if self._rep_audio_allowed():
                self.tts_worker.speak(DEFAULT_CARE_PROMPT, priority="high", event_type="care", phrase_key="care")

    def _start_rest(self, *, context: str, prompt_text: str, tts_text: str, event_type: str) -> None:
        self.status = "resting"
        self.pause_reason = None
        self.action_guard_until = None
        self.rest_context = context
        self.rest_until = time.time() + self.rest_seconds
        self.last_prompt = prompt_text
        self.care_dialog = self._empty_care_dialog()
        self._start_rest_music()
        if self.tts_worker:
            self.last_tts_text = tts_text
            if self._rep_audio_allowed():
                self.tts_worker.speak(tts_text, priority="high", event_type=event_type, phrase_key=self._event_phrase_key(event_type, tts_text))

    def _start_rest_music(self) -> None:
        music = self.rest_music if isinstance(self.rest_music, dict) else {}
        if not music.get("enabled") or str(music.get("playback") or "backend") != "backend":
            return
        delay_seconds = min(0.8, max(0.0, self.rest_seconds / 3.0))
        duration_seconds = max(0.1, float(self.rest_seconds) - delay_seconds)
        self.rest_audio_player.play(
            str(music.get("file") or "/assets/rest_music.wav"),
            duration_seconds=duration_seconds,
            fade_seconds=float(music.get("fade_seconds") or 0.0),
            delay_seconds=delay_seconds,
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _compact_rehab_keypoints(value: dict[str, Any]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, point in value.items():
        if not isinstance(point, dict):
            continue
        row: dict[str, float] = {}
        for field in ("x", "y", "z", "visibility"):
            number = _as_float(point.get(field))
            if number is not None:
                row[field] = number
        if "x" in row and "y" in row:
            compact[str(name)] = row
    return compact



