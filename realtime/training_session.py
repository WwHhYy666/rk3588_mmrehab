"""Realtime patient training session orchestration."""

from __future__ import annotations

import json
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
from prescription.common.result_storage import save_prescription_artifacts
from realtime.feedback_runtime import load_rules, process_prompt, rep_feedback
from realtime.knee_flexion import KneeFlexionRealtimeMachine, KneeFlexionTargets
from realtime.tts_worker import TTSWorker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TEMPLATES_PATH = PROJECT_ROOT / "runtime" / "active_templates.json"
DEFAULT_REALTIME_CONFIG = PROJECT_ROOT / "realtime" / "configs" / "knee_flexion_realtime.yaml"
DEFAULT_DEMO_PLAN = PROJECT_ROOT / "realtime" / "configs" / "rehab_demo_plan.yaml"
DEFAULT_FEEDBACK_RULE = PROJECT_ROOT / "feedback" / "rules" / "knee_flexion_feedback.yaml"
REPORTS_DIR = PROJECT_ROOT / "evaluate" / "reports"
PYTHON_EXE = Path("D:/anaconda/python.exe") if Path("D:/anaconda/python.exe").exists() else Path(sys.executable)


class RealtimeTrainingSession:
    def __init__(self, realtime_config_path: Path = DEFAULT_REALTIME_CONFIG) -> None:
        self.realtime_config_path = realtime_config_path
        self.lock = threading.RLock()
        self.tts_worker: TTSWorker | None = None
        self.rules = load_rules(DEFAULT_FEEDBACK_RULE)
        self.reset()

    def reset(self) -> None:
        self.status = "idle"
        self.error: str | None = None
        self.patient_id = "patient_001"
        self.action_id = "knee_flexion"
        self.side_mode = "auto"
        self.target_reps = 10
        self.start_time: float | None = None
        self.frame_index = 0
        self.frames: list[dict[str, Any]] = []
        self.rep_results: list[dict[str, Any]] = []
        self.invalid_attempts: list[dict[str, Any]] = []
        self.last_invalid_attempt: dict[str, Any] | None = None
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
        self.current_action_meta: dict[str, Any] | None = None

    def start(self, patient_id: str = "patient_001", action_id: str = "knee_flexion", side_mode: str = "auto", target_reps: int | None = None) -> dict[str, Any]:
        with self.lock:
            if self.status in {"running", "paused", "resting"}:
                return {"ok": False, "error": "实时训练已经在运行中。"}

            try:
                demo_plan = self._load_demo_plan()
                action_meta = self._find_plan_action(action_id, demo_plan)
                realtime_config = self._load_yaml(self._resolve_project_path(action_meta.get("realtime_config_file") or self.realtime_config_path))
                active_template = self._get_active_template(action_id)
                if not active_template:
                    return {"ok": False, "error": "请先录入医生标准动作，并保存为 active template。"}

                eval_config_path = self._resolve_project_path(active_template.get("config_file") or action_meta.get("config_file") or "evaluate/configs/knee_flexion.yaml")
                template_path = self._resolve_project_path(active_template.get("template_file") or "")
                if not template_path.exists():
                    return {"ok": False, "error": f"active template 文件不存在：{self._project_relative(template_path)}"}
                if not eval_config_path.exists():
                    return {"ok": False, "error": f"评估配置不存在：{self._project_relative(eval_config_path)}"}

                eval_config = self._load_yaml(eval_config_path)
                realtime_config = self._merge_realtime_config(realtime_config, eval_config)
                targets = self._build_targets(template_path, eval_config)
                if self.tts_worker is not None:
                    self.tts_worker.stop()
                self.reset()
                self.patient_id = str(patient_id or "patient_001").strip() or "patient_001"
                self.action_id = str(action_id or "knee_flexion").strip() or "knee_flexion"
                self.side_mode = side_mode if side_mode in {"auto", "left", "right"} else "auto"
                self.target_reps = int(target_reps or realtime_config.get("target_reps", 10))
                self.active_template = active_template
                self.eval_config = eval_config
                self.metric_info = self._metric_info_from_config(eval_config)
                feedback_rule_file = action_meta.get("feedback_rule_file") or self._feedback_rule_for_action(action_id)
                self.rules = load_rules(self._resolve_project_path(feedback_rule_file))
                self.machine = KneeFlexionRealtimeMachine(realtime_config, targets)
                self.start_time = time.time()
                self.status = "running"
                self.last_prompt = "请保持静止，正在校准"
                self.tts_worker = TTSWorker(
                    global_cooldown=float(realtime_config.get("tts_global_cooldown_seconds", 3.0)),
                    same_text_cooldown=float(realtime_config.get("tts_same_text_cooldown_seconds", 5.0)),
                    use_real_tts=True,
                )
                self.tts_worker.start()
                return {"ok": True, "message": "已开始实时训练。", "training": self.snapshot()}
            except Exception as exc:
                self.status = "error"
                self.error = str(exc)
                return {"ok": False, "error": str(exc)}

    def start_playlist(self, patient_id: str = "patient_001", side_mode: str = "auto", target_reps: int | None = None) -> dict[str, Any]:
        with self.lock:
            if self.status in {"running", "paused", "resting"}:
                return {"ok": False, "error": "实时训练已经在运行中。"}
            try:
                plan = self._load_demo_plan()
                actions = [action for action in plan.get("actions", []) if isinstance(action, dict)]
                if not actions:
                    return {"ok": False, "error": "rehab_demo_plan.yaml 中没有动作配置。"}
                missing = [
                    str(action.get("action_id"))
                    for action in actions
                    if action.get("action_id") and self._get_active_template(str(action.get("action_id"))) is None
                ]
                if missing:
                    return {"ok": False, "error": "请先录入这些动作的医生模板：" + "、".join(missing), "missing": missing}

                if self.tts_worker is not None:
                    self.tts_worker.stop()
                self.reset()
                self.playlist_mode = True
                self.playlist_actions = actions
                self.playlist_index = 0
                self.rest_seconds = int(plan.get("rest_seconds", 10) or 10)
                self.patient_id = str(patient_id or "patient_001").strip() or "patient_001"
                self.side_mode = side_mode if side_mode in {"auto", "left", "right"} else "auto"
                self.target_reps = int(target_reps or plan.get("default_target_reps", 3) or 3)
                self.tts_worker = TTSWorker(use_real_tts=True, global_cooldown=0.5, same_text_cooldown=2.0)
                self.tts_worker.start()
                welcome = str(plan.get("welcome_tts") or "康复训练即将开始，请坐稳并面向镜头。")
                self.tts_worker.speak(welcome, priority="high", event_type="welcome")
                self._start_playlist_action(0)
                return {"ok": True, "message": "已开始完整三动作训练。", "training": self.snapshot()}
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
            if self.tts_worker is not None:
                self.tts_worker.stop()
            if self.status in {"running", "paused", "resting"}:
                self.status = "idle"
                self.last_prompt = "训练已停止"
            return {"ok": True, "training": self.snapshot()}

    def process_frame(self, frame: dict[str, Any] | None, selected_rule: dict[str, Any] | None = None) -> None:
        with self.lock:
            if self.status == "resting":
                if self.rest_until is not None and time.time() >= self.rest_until:
                    self._advance_playlist_after_rest()
                return
            if self.status != "running" or self.machine is None or self.start_time is None:
                return

            now = time.time()
            frame_data = dict(frame or {})
            frame_data["frame_index"] = self.frame_index
            frame_data["relative_time"] = now - self.start_time
            self.frame_index += 1
            if selected_rule is not None:
                self.selected_rule = dict(selected_rule)

            frame_data = self._apply_action_metric(frame_data)
            output = self.machine.process(frame_data)
            self.last_machine_output = output
            prompt = str(output.get("prompt") or "") or process_prompt(
                str(output.get("state")),
                visible=bool(output.get("visible")),
                angle=_as_float(output.get("angle")),
                target_low=(output.get("target_range") or [None])[0],
            )
            self.last_prompt = prompt
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
                "baseline_angle": machine_state.get("baseline_angle"),
                "target_range": machine_state.get("target_range"),
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
            }

    def _handle_rep_done(self, rep_result: dict[str, Any]) -> None:
        feedback = rep_feedback(rep_result, self.rules)
        countable = bool(rep_result.get("countable", True))
        rep_index = len(self.rep_results) + 1 if countable else None
        enriched = {
            **rep_result,
            "rep_index": rep_index,
            **feedback,
        }
        self.last_prompt = str(feedback.get("screen_prompt") or self.last_prompt)
        self.last_tts_text = str(feedback.get("tts_text") or "")
        self.last_motor_mock_pattern = str(feedback.get("motor_mock_pattern") or "")
        if self.last_motor_mock_pattern:
            print(f"[MOTOR MOCK] {self.last_motor_mock_pattern}")
        if self.tts_worker:
            if countable:
                count_text = self._count_text(int(rep_index))
                self.last_tts_text = count_text or self.last_tts_text
                if count_text:
                    self.tts_worker.speak(count_text, priority="high", event_type="rep_count")
            elif self.last_tts_text:
                self.tts_worker.speak(self.last_tts_text, priority="low", event_type="correction")
        if not countable:
            correction = str(enriched.get("screen_prompt") or "动作不到位")
            enriched["screen_prompt"] = f"{correction}，未计数"
            enriched["not_counted_reason"] = "动作不到位，未计数"
            self.last_prompt = enriched["screen_prompt"]
            self.last_invalid_attempt = enriched
            self.invalid_attempts.append(enriched)
            return
        self.rep_results.append(enriched)
        if len(self.rep_results) >= self.target_reps:
            self._complete_training()

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
                    "primary_error": (self.report or {}).get("errors", {}).get("primary_error") if isinstance(self.report, dict) else None,
                }
            )
            if self.playlist_mode and self.playlist_index < len(self.playlist_actions) - 1:
                self.status = "resting"
                self.rest_until = time.time() + self.rest_seconds
                self.last_prompt = f"本组完成，休息 {self.rest_seconds} 秒"
                if self.tts_worker:
                    text = str((self.current_action_meta or {}).get("set_done_tts") or "做得很好，请休息一下。")
                    self.last_tts_text = text
                    self.tts_worker.speak(text, priority="high", event_type="set_done")
                return
            self.status = "completed"
            self.last_prompt = "全部训练完成" if self.playlist_mode else "本组训练完成"
            if self.tts_worker:
                plan = self._load_demo_plan()
                done_text = str(plan.get("finished_tts") or "今天的训练完成得很好，请注意休息。")
                self.last_tts_text = done_text
                self.tts_worker.speak(done_text, priority="high", event_type="training_finished")
        finally:
            if self.tts_worker is not None and not self.playlist_mode:
                self.tts_worker.stop()

    def _start_playlist_action(self, index: int) -> None:
        action = self.playlist_actions[index]
        action_id = str(action.get("action_id") or "").strip()
        if not action_id:
            raise ValueError("playlist 动作缺少 action_id")
        self.playlist_index = index
        self.current_action_meta = action
        self.action_id = action_id
        self.active_template = self._get_active_template(action_id)
        if not self.active_template:
            raise ValueError(f"缺少 active template: {action_id}")

        realtime_config = self._load_yaml(self._resolve_project_path(action.get("realtime_config_file") or self.realtime_config_path))
        eval_config_path = self._resolve_project_path(self.active_template.get("config_file") or action.get("config_file") or "evaluate/configs/knee_flexion.yaml")
        template_path = self._resolve_project_path(self.active_template.get("template_file") or "")
        eval_config = self._load_yaml(eval_config_path)
        realtime_config = self._merge_realtime_config(realtime_config, eval_config)
        targets = self._build_targets(template_path, eval_config)
        feedback_rule_file = action.get("feedback_rule_file") or self._feedback_rule_for_action(action_id)
        self.rules = load_rules(self._resolve_project_path(feedback_rule_file))
        self.machine = KneeFlexionRealtimeMachine(realtime_config, targets)
        self.eval_config = eval_config
        self.metric_info = self._metric_info_from_config(eval_config)
        self.target_reps = int(self.target_reps or realtime_config.get("target_reps", 3) or 3)
        self.start_time = time.time()
        self.frame_index = 0
        self.frames = []
        self.rep_results = []
        self.invalid_attempts = []
        self.last_invalid_attempt = None
        self.selected_rule = None
        self.last_machine_output = None
        self.metric_baseline_hip_y = None
        self.metric_baseline_torso_height = None
        self.saved_attempt_file = None
        self.report_file = None
        self.report = None
        self.feedback = None
        self.rest_until = None
        self.status = "running"
        self.last_prompt = str(action.get("camera_prompt") or "请保持关键点可见。")
        if self.tts_worker:
            start_tts = str(action.get("start_tts") or f"现在开始{action.get('action_name', action_id)}。")
            prompt = str(action.get("camera_prompt") or "")
            text = start_tts if not prompt else f"{start_tts}{prompt}"
            self.last_tts_text = text
            self.tts_worker.speak(text, priority="high", event_type="action_start")

    def _advance_playlist_after_rest(self) -> None:
        next_index = self.playlist_index + 1
        if next_index >= len(self.playlist_actions):
            self.status = "completed"
            self.rest_until = None
            return
        self._start_playlist_action(next_index)

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
        return {
            "patient_id": self.patient_id,
            "action_id": self.action_id,
            "action_name": self.action_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "这是一次实时训练生成的患者动作结果。",
            "camera_instruction": "实时训练时请保持目标关节相关关键点持续可见。",
            "algorithm_note": {
                "result_source": "realtime_training_session",
                "warning": "单目 MediaPipe 角度适合演示和趋势反馈，不属于临床级测量。",
            },
            "runtime_meta": {
                "result_format": "compact_v1",
                "record_role": "patient_attempt",
                "action_id": self.action_id,
                "side_mode": self.side_mode,
                "target_reps": self.target_reps,
                "completed_reps": len(self.rep_results),
                "rep_results": self.rep_results,
                "invalid_attempts": self.invalid_attempts,
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
            self.last_prompt = "请保持关键点可见"
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

    def _build_targets(self, template_path: Path, eval_config: dict[str, Any]) -> KneeFlexionTargets:
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
        realtime_config = self._load_yaml(self.realtime_config_path)
        return KneeFlexionTargets(
            rom_target=float(template_rom.get("rom") or realtime_config.get("min_rom", 30.0)),
            tut_target=float(template_tut.get("tut_seconds") or 0.0),
            target_range=(float(target_range[0]), float(target_range[1])),
            template_peak_speed=float(template_speed.get("peak_angular_velocity") or 0.0),
            rom_diff_max=float(thresholds.get("rom_diff_max", 10.0)),
            tut_ratio_min=float(thresholds.get("tut_ratio_min", 0.8)),
            speed_ratio_max=float(thresholds.get("speed_ratio_max", 1.5)),
        )

    def _get_active_template(self, action_id: str) -> dict[str, Any] | None:
        if not ACTIVE_TEMPLATES_PATH.exists():
            return None
        payload = json.loads(ACTIVE_TEMPLATES_PATH.read_text(encoding="utf-8"))
        entry = payload.get(action_id) if isinstance(payload, dict) else None
        return entry if isinstance(entry, dict) else None

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
            "actions": [
                {
                    "action_id": action.get("action_id"),
                    "action_name": action.get("action_name"),
                    "camera_prompt": action.get("camera_prompt"),
                    "config_file": action.get("config_file"),
                    "feedback_rule_file": action.get("feedback_rule_file"),
                    "has_active_template": self._get_active_template(str(action.get("action_id"))) is not None,
                }
                for action in actions
                if isinstance(action, dict)
            ] if isinstance(actions, list) else [],
        }

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

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}

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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None
