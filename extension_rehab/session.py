from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import ACTIONS, GROUP_ACTIONS, GROUP_PLANS
from .core import compute_extension_frame
from .template_store import ExtensionTemplateStore


class ExtendedTrainingSession:
    def __init__(
        self,
        *,
        template_store: ExtensionTemplateStore | None = None,
        report_dir: str | Path | None = None,
        rest_audio_player: object | None = None,
    ) -> None:
        self.template_store = template_store or ExtensionTemplateStore()
        self.report_dir = Path(report_dir) if report_dir is not None else Path("evaluate/reports/extensions")
        self.rest_audio_player = rest_audio_player
        self._lock = threading.RLock()
        self._reset("idle")

    def _reset(self, status: str) -> None:
        self.status = status
        self.extension_group: str | None = None
        self.extension_action: str | None = None
        self.patient_id = "patient_001"
        self.target_reps = 5
        self.count = 0
        self.invalid_attempts = 0
        self.phase = "ready"
        self.metric: float | None = None
        self.target_rom: float | None = None
        self.required_rom: float | None = None
        self.tut_seconds = 0.0
        self.speed = 0.0
        self.quality_score: float | None = None
        self.compensation_errors: list[str] = []
        self.missing_keypoints: list[str] = []
        self.components: dict[str, float] = {}
        self.selected_side: str | None = None
        self.template_health: dict[str, Any] = {"ok": False, "reason": "not_selected"}
        self.last_report_file: str | None = None
        self.report_files: list[str] = []
        self.playlist_action_ids: list[str] = []
        self.playlist_index: int | None = None
        self.rest_seconds = 0.0
        self.rest_until: float | None = None
        self.rest_music = {"file": "/assets/rest_music.wav", "playback": "backend", "fade_seconds": 1.0}
        self._baseline_metric: float | None = None
        self._baseline_components: dict[str, float] = {}
        self._last_metric: float | None = None
        self._attempt_started_at: float | None = None
        self._attempt_peak = 0.0
        self._attempt_errors: set[str] = set()
        self._rep_results: list[dict[str, Any]] = []
        self._template_frames: list[dict[str, Any]] = []

    def _template_duration_seconds(self) -> float:
        if len(self._template_frames) < 2:
            return 0.0
        started_at = float(self._template_frames[0]["timestamp"])
        latest_at = float(self._template_frames[-1]["timestamp"])
        return max(0.0, latest_at - started_at)

    def _validate_action(self, group: str, action_id: str) -> dict[str, Any] | None:
        config = ACTIONS.get(action_id)
        return config if config is not None and config.get("group") == group and action_id in GROUP_ACTIONS.get(group, ()) else None

    def _activate_action(self, action_id: str) -> None:
        config = ACTIONS[action_id]
        health = self.template_store.health(action_id)
        self.extension_action = action_id
        self.count = 0
        self.invalid_attempts = 0
        self.phase = "ready"
        self.metric = None
        self.target_rom = float(health.get("target_rom") or config["seed_rom"])
        self.required_rom = max(float(config["min_valid_rom"]), self.target_rom * 0.75)
        self.tut_seconds = 0.0
        self.speed = 0.0
        self.quality_score = None
        self.compensation_errors = []
        self.missing_keypoints = []
        self.components = {}
        self.selected_side = None
        self.template_health = health
        self._baseline_metric = None
        self._baseline_components = {}
        self._last_metric = None
        self._attempt_started_at = None
        self._attempt_peak = 0.0
        self._attempt_errors.clear()
        self._rep_results = []
        self.rest_until = None

    def _advance_rest_if_due(self, now: float | None = None) -> None:
        if self.status != "resting" or self.rest_until is None:
            return
        current = float(now if now is not None else time.time())
        if current < self.rest_until:
            return
        self._stop_rest_music()
        next_index = int(self.playlist_index or 0) + 1
        if next_index >= len(self.playlist_action_ids):
            self.status = "complete"
            self.rest_until = None
            return
        self.playlist_index = next_index
        self._activate_action(self.playlist_action_ids[next_index])
        self.status = "running"

    def _start_rest_music(self) -> None:
        player = self.rest_audio_player
        music = self.rest_music if isinstance(self.rest_music, dict) else {}
        if player is None or str(music.get("playback") or "backend") != "backend":
            return
        play = getattr(player, "play", None)
        if not callable(play):
            return
        delay_seconds = min(0.8, max(0.0, self.rest_seconds / 3.0))
        play(
            str(music.get("file") or "/assets/rest_music.wav"),
            duration_seconds=max(0.1, self.rest_seconds - delay_seconds),
            fade_seconds=float(music.get("fade_seconds") or 0.0),
            delay_seconds=delay_seconds,
        )

    def _stop_rest_music(self) -> None:
        stop = getattr(self.rest_audio_player, "stop", None)
        if callable(stop):
            stop()

    def start(
        self,
        *,
        group: str,
        action_id: str,
        patient_id: str = "patient_001",
        target_reps: int | None = None,
        require_template: bool | None = True,
    ) -> dict[str, Any]:
        with self._lock:
            self._stop_rest_music()
            config = self._validate_action(group, action_id)
            if config is None:
                return {"ok": False, "error": "unknown extension group/action"}
            health = self.template_store.health(action_id)
            if require_template is not False and not health.get("ok"):
                return {"ok": False, "error": "extension template is missing or unhealthy", "template_health": health}
            self._reset("running")
            self.extension_group = group
            self.extension_action = action_id
            self.patient_id = str(patient_id or "patient_001")
            self.target_reps = max(1, int(target_reps or 5))
            self._activate_action(action_id)
            return {"ok": True, "training": self.snapshot()}

    def start_group(
        self,
        *,
        group: str,
        patient_id: str = "patient_001",
        target_reps: int | None = None,
        require_template: bool | None = True,
    ) -> dict[str, Any]:
        with self._lock:
            self._stop_rest_music()
            plan = GROUP_PLANS.get(group)
            if plan is None:
                return {"ok": False, "error": "unknown extension group"}
            action_ids = list(plan["actions"])
            health_by_action = {action_id: self.template_store.health(action_id) for action_id in action_ids}
            unhealthy = {action_id: health for action_id, health in health_by_action.items() if not health.get("ok")}
            if require_template is not False and unhealthy:
                return {
                    "ok": False,
                    "error": "one or more extension templates are missing or unhealthy",
                    "template_health_by_action": unhealthy,
                }
            self._reset("running")
            self.extension_group = group
            self.patient_id = str(patient_id or "patient_001")
            self.target_reps = max(1, int(target_reps or plan["default_reps"]))
            self.playlist_action_ids = action_ids
            self.playlist_index = 0
            self.rest_seconds = float(plan["rest_seconds"])
            self.rest_music = {
                "file": plan["rest_music_file"],
                "playback": plan["rest_music_playback"],
                "fade_seconds": float(plan["rest_music_fade_seconds"]),
            }
            self._activate_action(action_ids[0])
            return {"ok": True, "training": self.snapshot()}

    def start_template(self, *, group: str, action_id: str) -> dict[str, Any]:
        with self._lock:
            self._stop_rest_music()
            config = self._validate_action(group, action_id)
            if config is None:
                return {"ok": False, "error": "unknown extension group/action"}
            self._reset("recording_template")
            self.extension_group = group
            self.extension_action = action_id
            self.template_health = self.template_store.health(action_id)
            return {"ok": True, "training": self.snapshot()}

    def process_frame(self, points: object, *, timestamp: float | None = None) -> dict[str, Any]:
        with self._lock:
            self._advance_rest_if_due(timestamp)
            if self.status not in {"running", "recording_template"} or self.extension_action is None:
                return self.snapshot()
            now = float(timestamp if timestamp is not None else time.time())
            result = compute_extension_frame(self.extension_action, points)
            self.metric = result.get("metric")
            self.components = dict(result.get("components") or {})
            self.selected_side = result.get("selected_side")
            self.compensation_errors = list(result.get("compensation_errors") or [])
            self.missing_keypoints = list(result.get("missing_keypoints") or [])
            self.quality_score = result.get("quality_score")
            if not result.get("valid") or self.metric is None:
                return self.snapshot()

            metric = float(self.metric)
            if self.status == "recording_template":
                self._template_frames.append({"timestamp": now, "metric": metric, "components": dict(self.components)})
                return self.snapshot()

            config = ACTIONS[self.extension_action]
            if self._baseline_metric is None:
                self._baseline_metric = metric
                self._baseline_components = dict(self.components)
                self._last_metric = metric
                return self.snapshot()

            dynamic_errors = list(self.compensation_errors)
            if self._last_metric is not None and abs(metric - self._last_metric) > float(config["max_frame_jump"]):
                dynamic_errors.append("sudden_jump")
            progress = metric - self._baseline_metric
            if progress < -float(config["attempt_start_delta"]):
                dynamic_errors.append("direction_error")
            if self.extension_action == "mini_squat" and progress >= float(config["attempt_start_delta"]):
                baseline_hip_y = self._baseline_components.get("hip_y")
                hip_y = self.components.get("hip_y")
                if baseline_hip_y is not None and hip_y is not None:
                    if hip_y - baseline_hip_y < float(config["compensation"]["min_hip_drop_ratio"]):
                        dynamic_errors.append("hip_drop")
            self.compensation_errors = list(dict.fromkeys(dynamic_errors))
            self._last_metric = metric
            if "sudden_jump" in self.compensation_errors or "direction_error" in self.compensation_errors:
                return self.snapshot()

            start_delta = float(config["attempt_start_delta"])
            return_delta = float(config["return_delta"])
            if self.phase == "ready" and progress >= start_delta and not self.compensation_errors:
                self.phase = "moving"
                self._attempt_started_at = now
                self._attempt_peak = progress
                self._attempt_errors.clear()
            elif self.phase == "moving":
                self._attempt_peak = max(self._attempt_peak, progress)
                self._attempt_errors.update(self.compensation_errors)
                if progress <= return_delta:
                    self._finish_attempt(now, config)
            return self.snapshot()

    def _finish_attempt(self, timestamp: float, config: dict[str, Any]) -> None:
        started = self._attempt_started_at if self._attempt_started_at is not None else timestamp
        duration = max(0.0, timestamp - started)
        peak = self._attempt_peak
        errors = sorted(self._attempt_errors)
        valid = bool(
            peak >= float(self.required_rom or config["min_valid_rom"])
            and float(config["min_duration"]) <= duration <= float(config["max_duration"])
            and not errors
        )
        result = {"duration_seconds": round(duration, 3), "rom": round(peak, 3), "valid": valid, "compensation_errors": errors}
        self._rep_results.append(result)
        self.tut_seconds = duration
        self.speed = peak / duration if duration > 0 else 0.0
        if valid:
            self.count += 1
        else:
            self.invalid_attempts += 1
        self.phase = "ready"
        self._attempt_started_at = None
        self._attempt_peak = 0.0
        self._attempt_errors.clear()
        if self.count >= self.target_reps:
            self.last_report_file = str(self._write_report())
            self.report_files.append(self.last_report_file)
            if self.playlist_action_ids and self.playlist_index is not None and self.playlist_index + 1 < len(self.playlist_action_ids):
                self.status = "resting"
                self.phase = "rest"
                self.rest_until = timestamp + self.rest_seconds
                self._start_rest_music()
            else:
                self.status = "complete"

    def _write_report(self) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.report_dir / f"extension_{self.extension_group}_{self.extension_action}_{timestamp}.json"
        payload = {
            "patient_id": self.patient_id,
            "extension_group": self.extension_group,
            "extension_action": self.extension_action,
            "playlist_action_ids": list(self.playlist_action_ids),
            "playlist_index": self.playlist_index,
            "playlist_total": len(self.playlist_action_ids),
            "count": self.count,
            "invalid_attempts": self.invalid_attempts,
            "target_rom": self.target_rom,
            "required_rom": self.required_rom,
            "last_tut_seconds": self.tut_seconds,
            "last_speed": self.speed,
            "repetitions": list(self._rep_results),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_template(self) -> dict[str, Any]:
        with self._lock:
            if self.status != "recording_template" or self.extension_action is None:
                return {"ok": False, "error": "extension template recording is not active"}
            if not self._template_frames:
                return {"ok": False, "error": "no valid extension template frames"}
            config = ACTIONS[self.extension_action]
            metrics = [float(frame["metric"]) for frame in self._template_frames]
            timestamps = [float(frame["timestamp"]) for frame in self._template_frames]
            minimum = min(metrics)
            maximum = max(metrics)
            rom = maximum - minimum
            returned = abs(metrics[-1] - metrics[0]) <= float(config["return_delta"])
            payload = {
                "valid_frames": len(metrics),
                "duration_seconds": max(0.0, timestamps[-1] - timestamps[0]),
                "minimum": minimum,
                "maximum": maximum,
                "rom": rom,
                "target_rom": rom,
                "direction": "increase" if metrics.index(maximum) > metrics.index(minimum) else "decrease",
                "returned": returned,
                "start_value": metrics[0],
                "end_value": metrics[-1],
                "frames": list(self._template_frames),
            }
            path = self.template_store.save(self.extension_action, payload)
            health = self.template_store.health(self.extension_action)
            self.template_health = health
            self.status = "idle"
            return {"ok": bool(health.get("ok")), "template_file": str(path), "template_health": health, "error": None if health.get("ok") else "extension template health check failed"}

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self.status == "running":
                self.status = "paused"
            elif self.status == "paused":
                self.status = "running"
            else:
                return {"ok": False, "error": "extension training is not running"}
            return {"ok": True, "training": self.snapshot()}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_rest_music()
            self.status = "stopped"
            self.phase = "ready"
            self._attempt_started_at = None
            return {"ok": True, "training": self.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._advance_rest_if_due()
            rest_remaining_seconds = max(0.0, self.rest_until - time.time()) if self.status == "resting" and self.rest_until is not None else 0.0
            rest_audio = getattr(self.rest_audio_player, "snapshot", None)
            return {
                "status": self.status,
                "extension_group": self.extension_group,
                "extension_action": self.extension_action,
                "playlist_action_ids": list(self.playlist_action_ids),
                "playlist_index": self.playlist_index,
                "playlist_total": len(self.playlist_action_ids),
                "rest_remaining_seconds": round(rest_remaining_seconds, 1),
                "rest_audio": rest_audio() if callable(rest_audio) else {},
                "patient_id": self.patient_id,
                "target_reps": self.target_reps,
                "count": self.count,
                "invalid_attempts": self.invalid_attempts,
                "phase": self.phase,
                "metric": self.metric,
                "target_rom": self.target_rom,
                "required_rom": self.required_rom,
                "tut_seconds": round(self.tut_seconds, 3),
                "speed": round(self.speed, 3),
                "quality_score": self.quality_score,
                "compensation_errors": list(self.compensation_errors),
                "missing_keypoints": list(self.missing_keypoints),
                "components": dict(self.components),
                "selected_side": self.selected_side,
                "template_health": dict(self.template_health),
                "template_valid_frames": len(self._template_frames),
                "template_duration_seconds": round(self._template_duration_seconds(), 3),
                "last_report_file": self.last_report_file,
                "report_files": list(self.report_files),
            }
