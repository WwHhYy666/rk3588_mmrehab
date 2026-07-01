"""Realtime knee-flexion state machine and per-rep metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import deque
from statistics import mean
from typing import Any

from evaluate.core.speed_check import check_speed
from evaluate.core.tut import compute_tut

from realtime.state_machine import ConsecutiveConfirm, MotionState

PERSON_OFFSCREEN_PROMPT = "请回到画面中"
TARGET_KEYPOINT_PROMPT = "请保持目标腿关键点可见"
ACTIVE_REP_STATES = {MotionState.RISING, MotionState.HOLDING, MotionState.RETURNING}


@dataclass
class KneeFlexionTargets:
    rom_target: float
    tut_target: float
    target_range: tuple[float, float]
    template_peak_speed: float
    rom_diff_max: float
    tut_ratio_min: float
    speed_ratio_max: float


class KneeFlexionRealtimeMachine:
    """Cycle detector for realtime rehab reps around a dynamic rest anchor."""

    def __init__(self, config: dict[str, Any], targets: KneeFlexionTargets) -> None:
        self.config = config
        self.targets = targets
        self.segment_mode = str(config.get("segment_mode") or "angle_curl")
        self.state = MotionState.BASELINE
        self.baseline_angle: float | None = None
        self.baseline_started_at: float | None = None
        self.baseline_samples: list[float] = []
        self.rep_started_at: float | None = None
        self.rep_state_started_at: float | None = None
        self.rep_start_angle: float | None = None
        self.rep_peak_angle: float | None = None
        self.rep_lowest_after_peak: float | None = None
        self.rep_last_angle: float | None = None
        self.rep_last_at: float | None = None
        self.rep_frames: list[dict[str, Any]] = []
        self.rest_samples: deque[tuple[float, float]] = deque()
        self.rest_anchor: float | None = None
        self.rest_noise = 0.0
        self.rest_started_at: float | None = None
        self.last_velocity = 0.0
        self.last_sample_angle: float | None = None
        self.last_sample_at: float | None = None
        self.stable_return_started_at: float | None = None
        self.visibility_lost_started_at: float | None = None
        self.reached_target = False
        self.lost_visibility_frames = 0
        self.metric_invalid_frames = 0
        self.last_prompt = "请保持静止，正在校准"
        self.last_error: str | None = None
        confirm_frames = int(config.get("confirm_frames", 3))
        self._start_confirm = ConsecutiveConfirm(confirm_frames)
        self._hold_confirm = ConsecutiveConfirm(confirm_frames)
        self._return_confirm = ConsecutiveConfirm(confirm_frames)
        self._finish_confirm = ConsecutiveConfirm(confirm_frames)
        finish_confirm_frames = int(config.get("finish_confirm_frames", config.get("return_stable_frames", confirm_frames)))
        self._return_stable_confirm = ConsecutiveConfirm(finish_confirm_frames)

    @property
    def target_low(self) -> float:
        return self.targets.target_range[0]

    @property
    def target_high(self) -> float:
        return self.targets.target_range[1]

    def process(self, frame: dict[str, Any]) -> dict[str, Any]:
        angle = _as_float(frame.get("target_angle_smoothed"))
        visibility_raw = frame.get("visibility_min")
        if visibility_raw is None:
            visibility_raw = frame.get("visibility")
        visibility = _as_float(visibility_raw)
        person_visible = bool(frame.get("person_visible", frame.get("pose_detected", True)))
        now = _as_float(frame.get("relative_time")) or 0.0
        visibility_threshold = float(frame.get("visibility_threshold") or self.config.get("visibility_threshold", 0.55))

        if angle is None or visibility is None or visibility < visibility_threshold:
            self.metric_invalid_frames += 1
            if not person_visible:
                self.last_prompt = PERSON_OFFSCREEN_PROMPT
            elif self.state == MotionState.BASELINE:
                self.last_prompt = ""
            if not person_visible:
                return self._output(visible=False, action_keypoints_valid=False, angle=angle)
            if self.state in ACTIVE_REP_STATES:
                self.lost_visibility_frames += 1
                if self.visibility_lost_started_at is None:
                    self.visibility_lost_started_at = now
                max_lost = int(self.config.get("max_lost_visibility_frames", 8))
                interrupt_reset_seconds = self.config.get("interrupt_reset_seconds")
                reset_by_time = (
                    interrupt_reset_seconds is not None
                    and now - self.visibility_lost_started_at >= float(interrupt_reset_seconds)
                )
                if self.lost_visibility_frames >= max_lost or reset_by_time:
                    self._reset_visibility_interrupted_attempt()
                    return self._output(
                        visible=person_visible,
                        action_keypoints_valid=False,
                        angle=angle,
                    )
            return self._output(visible=person_visible, action_keypoints_valid=False, angle=angle)
        self.lost_visibility_frames = 0
        self.visibility_lost_started_at = None
        self.metric_invalid_frames = 0
        self._update_motion_observation(now, angle)

        if self.state == MotionState.BASELINE:
            return self._process_baseline(now, angle)

        rep_result = None
        if self.state == MotionState.IDLE:
            self._process_idle(now, angle, frame)
        elif self.state == MotionState.RISING:
            self._append_rep_frame(frame)
            rep_result = self._process_rising(now, angle)
        elif self.state == MotionState.HOLDING:
            self._append_rep_frame(frame)
            self._process_holding(now, angle)
        elif self.state == MotionState.RETURNING:
            self._append_rep_frame(frame)
            rep_result = self._process_returning(now, angle)

        return self._output(visible=True, action_keypoints_valid=True, angle=angle, rep_result=rep_result)

    def _process_baseline(self, now: float, angle: float) -> dict[str, Any]:
        if self.baseline_started_at is None:
            self.baseline_started_at = now
        self.baseline_samples.append(angle)
        baseline_seconds = float(self.config.get("baseline_seconds", 2.0))
        if now - self.baseline_started_at >= baseline_seconds and self.baseline_samples:
            self.baseline_angle = mean(self.baseline_samples)
            self.rest_anchor = self.baseline_angle
            self.rest_noise = 0.0
            self.state = MotionState.IDLE
            self.last_prompt = ""
        else:
            self.last_prompt = ""
            return self._output(visible=True, action_keypoints_valid=True, angle=angle)

    def _process_idle(self, now: float, angle: float, frame: dict[str, Any]) -> None:
        baseline = self.rest_anchor if self.rest_anchor is not None else (self.baseline_angle if self.baseline_angle is not None else angle)
        start_delta = self._start_delta_from_rest()
        moving_up = self.last_velocity >= self._motion_start_velocity_min()
        if self._start_confirm.update(angle - baseline >= start_delta and moving_up):
            self.state = MotionState.RISING
            self.rep_started_at = now
            self.rep_state_started_at = now
            self.rep_frames = []
            self.rep_start_angle = angle
            self.rep_peak_angle = angle
            self.rep_lowest_after_peak = angle
            self.rep_last_angle = angle
            self.rep_last_at = now
            self.reached_target = False
            self.lost_visibility_frames = 0
            self._append_rep_frame(frame)
            self.last_prompt = str(self.config.get("raise_prompt") or "再抬高一点")
            self._hold_confirm.reset()
            self._return_confirm.reset()
            self._finish_confirm.reset()
            self._return_stable_confirm.reset()
        else:
            self.last_prompt = ""

    def _process_rising(self, now: float, angle: float) -> dict[str, Any] | None:
        if self._hold_confirm.update(angle >= self.target_low):
            self.reached_target = True
            self.state = MotionState.HOLDING
            self.rep_state_started_at = now
            self.last_prompt = "保持住"
            return None
        if self.rep_started_at is not None and now - self.rep_started_at >= float(self.config.get("min_rep_seconds", 1.0)):
            if self._returned_and_stable(now, angle):
                if self._is_noise_attempt():
                    self._reset_silent_attempt()
                    return None
                return self._finish_rep(countable=False)
            if self._should_timeout_finish(now, angle):
                if self._is_noise_attempt():
                    self._reset_silent_attempt()
                    return None
                return self._finish_rep(countable=False)
        self.last_prompt = str(self.config.get("raise_prompt") or "再抬高一点")
        return None

    def _process_holding(self, now: float, angle: float) -> None:
        if self._return_confirm.update(angle < self.target_low and self.last_velocity <= self._return_velocity_max()):
            self.state = MotionState.RETURNING
            self.rep_state_started_at = now
            self.last_prompt = self._returning_prompt()
        elif self._current_tut_status()["missing_seconds"] <= 0.0:
            self.last_prompt = self._hold_done_prompt()
        else:
            self.last_prompt = "保持住"

    def _process_returning(self, now: float, angle: float) -> dict[str, Any] | None:
        if self._returned_and_stable(now, angle):
            return self._finish_rep(countable=self.reached_target)
        if self._should_close_before_restart(angle):
            return self._finish_rep(countable=False, forced_error="TUT_LOW" if self.reached_target else "ROM_LOW")
        if self._should_timeout_finish(now, angle):
            return self._finish_rep(countable=self.reached_target)
        self.last_prompt = self._returning_prompt()
        return None

    def _returned_and_stable(self, now: float, angle: float) -> bool:
        returned = self._is_returned(angle)
        slow = abs(self.last_velocity) <= self._rest_velocity_max()
        stable = returned and slow
        if stable:
            if self.stable_return_started_at is None:
                self.stable_return_started_at = now
        else:
            self.stable_return_started_at = None
        finish_confirmed = self._finish_confirm.update(stable)
        stable_confirmed = self._return_stable_confirm.update(stable)
        stable_seconds = 0.0 if self.stable_return_started_at is None else now - self.stable_return_started_at
        return finish_confirmed and stable_confirmed and stable_seconds >= self._stable_return_seconds()

    def _is_returned(self, angle: float) -> bool:
        baseline = self.rest_anchor if self.rest_anchor is not None else (self.baseline_angle if self.baseline_angle is not None else angle)
        return_delta = self._return_delta_from_rest()
        returned_to_anchor = angle <= baseline + return_delta
        start_angle = self.rep_start_angle
        if start_angle is None:
            return returned_to_anchor
        return_to_start_delta = self._return_to_start_delta()
        returned_to_start = angle <= start_angle + return_to_start_delta
        return returned_to_anchor or returned_to_start

    def _update_motion_observation(self, now: float, angle: float) -> None:
        if self.last_sample_angle is not None and self.last_sample_at is not None:
            dt = max(1e-3, now - self.last_sample_at)
            self.last_velocity = (angle - self.last_sample_angle) / dt
        self.last_sample_angle = angle
        self.last_sample_at = now
        if self.state not in {MotionState.BASELINE, MotionState.IDLE}:
            return
        self.rest_samples.append((now, angle))
        window = self._rest_window_seconds()
        while self.rest_samples and now - self.rest_samples[0][0] > window:
            self.rest_samples.popleft()
        if len(self.rest_samples) < max(2, int(self.config.get("rest_min_samples", 3))):
            return
        values = [value for _, value in self.rest_samples]
        anchor = mean(values)
        noise = max(values) - min(values)
        if noise <= self._rest_noise_max() and abs(self.last_velocity) <= self._rest_velocity_max():
            self.rest_anchor = anchor
            self.baseline_angle = anchor
            self.rest_noise = noise
            if self.rest_started_at is None:
                self.rest_started_at = self.rest_samples[0][0]
        else:
            self.rest_started_at = None

    def _append_rep_frame(self, frame: dict[str, Any]) -> None:
        if not self.rep_frames or frame.get("frame_index") != self.rep_frames[-1].get("frame_index"):
            self.rep_frames.append(dict(frame))
        angle = _as_float(frame.get("target_angle_smoothed"))
        now = _as_float(frame.get("relative_time"))
        if angle is None:
            return
        if self.rep_start_angle is None:
            self.rep_start_angle = (
                self.rest_anchor
                if self.rest_anchor is not None
                else (self.baseline_angle if self.baseline_angle is not None else angle)
            )
        if self.rep_peak_angle is None or angle > self.rep_peak_angle:
            self.rep_peak_angle = angle
            self.rep_lowest_after_peak = angle
        elif self.rep_peak_angle is not None:
            if self.rep_lowest_after_peak is None or angle < self.rep_lowest_after_peak:
                self.rep_lowest_after_peak = angle
        self.rep_last_angle = angle
        if now is not None:
            self.rep_last_at = now

    def _should_timeout_finish(self, now: float, angle: float) -> bool:
        if self.rep_started_at is None:
            return False
        max_rep_seconds = self.config.get("max_rep_seconds")
        if max_rep_seconds is not None and now - self.rep_started_at >= float(max_rep_seconds):
            return self._has_clearly_returned(angle)
        if self.state == MotionState.RETURNING:
            return_timeout = self.config.get("return_timeout_seconds")
            if return_timeout is not None and self.rep_state_started_at is not None:
                if now - self.rep_state_started_at >= float(return_timeout):
                    return self._has_clearly_returned(angle)
        return False

    def _has_clearly_returned(self, angle: float) -> bool:
        peak = self.rep_peak_angle
        start = self.rep_start_angle
        if peak is None or start is None:
            return False
        min_attempt_delta = self.config.get("min_attempt_delta")
        if min_attempt_delta is None:
            min_attempt_delta = self._attempt_start_delta() * 0.5
        has_real_motion = peak - start >= float(min_attempt_delta)
        if not has_real_motion:
            return False
        if self._is_returned(angle):
            return True
        returned_fraction = float(self.config.get("return_completion_fraction", 0.70))
        return angle <= peak - (peak - start) * returned_fraction

    def _should_close_before_restart(self, angle: float) -> bool:
        peak = self.rep_peak_angle
        start = self.rep_start_angle
        lowest = self.rep_lowest_after_peak
        if peak is None or start is None or lowest is None:
            return False
        min_attempt_delta = self.config.get("min_attempt_delta")
        if min_attempt_delta is None:
            min_attempt_delta = self._attempt_start_delta() * 0.5
        has_real_motion = peak - start >= float(min_attempt_delta)
        returned_enough = lowest <= start + self._return_delta_from_rest()
        rising_again = angle - lowest >= self._attempt_start_delta()
        return has_real_motion and returned_enough and rising_again

    def _is_noise_attempt(self) -> bool:
        baseline = self.baseline_angle
        if baseline is None or not self.rep_frames:
            return False
        values = [_as_float(frame.get("target_angle_smoothed")) for frame in self.rep_frames]
        values = [value for value in values if value is not None]
        if not values:
            return False
        min_attempt_delta = self.config.get("min_attempt_delta")
        if min_attempt_delta is None:
            min_attempt_delta = self._attempt_start_delta() * 0.5
        return max(values) - baseline < float(min_attempt_delta)

    def _reset_silent_attempt(self) -> None:
        self.state = MotionState.IDLE
        self.rep_started_at = None
        self.rep_state_started_at = None
        self.rep_start_angle = None
        self.rep_peak_angle = None
        self.rep_lowest_after_peak = None
        self.rep_last_angle = None
        self.rep_last_at = None
        self.rep_frames = []
        self.stable_return_started_at = None
        self.visibility_lost_started_at = None
        self.reached_target = False
        self.lost_visibility_frames = 0
        self.metric_invalid_frames = 0
        self._start_confirm.reset()
        self._hold_confirm.reset()
        self._return_confirm.reset()
        self._finish_confirm.reset()
        self._return_stable_confirm.reset()
        self.last_prompt = ""

    def _finish_rep(self, *, countable: bool, forced_error: str | None = None) -> dict[str, Any]:
        result = self._evaluate_rep(self.rep_frames, forced_error=forced_error)
        countable = result.get("primary_error") == "OK"
        result["countable"] = countable
        self.state = MotionState.IDLE
        self.rep_started_at = None
        self.rep_state_started_at = None
        self.rep_start_angle = None
        self.rep_peak_angle = None
        self.rep_lowest_after_peak = None
        self.rep_last_angle = None
        self.rep_last_at = None
        self.rep_frames = []
        self.stable_return_started_at = None
        self.visibility_lost_started_at = None
        self.reached_target = False
        self.lost_visibility_frames = 0
        self.metric_invalid_frames = 0
        self._start_confirm.reset()
        self._hold_confirm.reset()
        self._return_confirm.reset()
        self._finish_confirm.reset()
        self._return_stable_confirm.reset()
        self.last_prompt = "" if countable else str(self.config.get("raise_prompt") or "再抬高一点")
        return result

    def _evaluate_rep(self, frames: list[dict[str, Any]], *, forced_error: str | None = None) -> dict[str, Any]:
        angles = [_as_float(frame.get("target_angle_smoothed")) for frame in frames]
        angles = [value for value in angles if value is not None]
        start_time = _as_float(frames[0].get("relative_time")) if frames else 0.0
        end_time = _as_float(frames[-1].get("relative_time")) if frames else start_time
        duration = max(0.0, (end_time or 0.0) - (start_time or 0.0))
        rom = (max(angles) - min(angles)) if angles else 0.0
        max_signal = max(angles) if angles else 0.0
        reached_target = max_signal >= self.target_low
        tut_count_range = self._tut_count_range()
        tut = compute_tut(frames, tut_count_range, "target_angle_smoothed")
        speed = check_speed(frames, "target_angle_smoothed")
        tut_actual = float(tut.get("tut_seconds", 0.0))
        peak_speed = float(speed.get("peak_angular_velocity", 0.0))
        rom_diff = max(0.0, self.targets.rom_target - rom)
        tut_ratio = _safe_ratio(tut_actual, self.targets.tut_target)
        tut_required = self.targets.tut_target * self.targets.tut_ratio_min
        speed_ratio = _safe_ratio(peak_speed, self.targets.template_peak_speed)

        strict_quality_errors = bool(self.config.get("strict_quality_errors", True))

        peak_ok = reached_target
        rom_ok = rom_diff <= self.targets.rom_diff_max
        tut_ok = tut_ratio >= self.targets.tut_ratio_min
        count_by_peak_target = bool(self.config.get("count_by_peak_target", False))
        rom_blocks_count = (not rom_ok) and not count_by_peak_target

        all_errors: list[str] = []
        if not peak_ok or rom_blocks_count:
            all_errors.append("ROM_LOW")
        elif not tut_ok:
            all_errors.append("TUT_LOW")

        if forced_error:
            primary_error = forced_error
            all_errors = [forced_error]
        else:
            primary_error = all_errors[0] if all_errors else "OK"

        return {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "rom": rom,
            "rom_target": self.targets.rom_target,
            "rom_diff": rom_diff,
            "max_signal": max_signal,
            "tut_seconds": tut_actual,
            "tut_target": self.targets.tut_target,
            "missing_seconds": max(0.0, tut_required - tut_actual),
            "tut_ratio": tut_ratio,
            "peak_speed": peak_speed,
            "speed_ratio": speed_ratio,
            "primary_error": primary_error,
            "all_errors": all_errors,
            "countable": primary_error == "OK",
            "reached_target": reached_target,
            "peak_ok": peak_ok,
            "rom_ok": rom_ok,
            "tut_ok": tut_ok,
            "tut_count_mode": self._tut_count_mode(),
            "tut_count_range": self._public_tut_count_range(tut_count_range),
            "count_by_peak_target": count_by_peak_target,
            "rom_blocks_count": rom_blocks_count,
            "angle_curve": [
                {
                    "relative_time": frame.get("relative_time"),
                    "target_angle_smoothed": frame.get("target_angle_smoothed"),
                }
                for frame in frames
            ],
        }

    def _reset_visibility_interrupted_attempt(self) -> None:
        self._reset_silent_attempt()
        self.last_prompt = ""

    def _has_enough_rep_data(self) -> bool:
        values = [_as_float(frame.get("target_angle_smoothed")) for frame in self.rep_frames]
        values = [value for value in values if value is not None]
        if len(values) < 2:
            return False
        times = [_as_float(frame.get("relative_time")) for frame in self.rep_frames]
        times = [value for value in times if value is not None]
        if len(times) >= 2:
            min_seconds = float(self.config.get("min_rep_seconds", 1.0))
            if max(times) - min(times) < min_seconds:
                return False
        return True

    def _output(
        self,
        *,
        visible: bool,
        action_keypoints_valid: bool = True,
        angle: float | None,
        rep_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tut_status = self._current_tut_status()
        tut_status["in_tut_zone"] = self._angle_in_tut_zone(angle)
        return {
            "state": self.state.value,
            "segment_mode": self.segment_mode,
            "visible": visible,
            "action_keypoints_valid": action_keypoints_valid,
            "angle": angle,
            "baseline_angle": self.baseline_angle,
            "rest_anchor": self.rest_anchor,
            "rest_noise": self.rest_noise,
            "velocity": self.last_velocity,
            "peak_value": self.rep_peak_angle,
            "stable_return_seconds": self._current_stable_return_seconds(),
            "target_range": list(self.targets.target_range),
            **tut_status,
            "prompt": self.last_prompt,
            "rep_result": rep_result,
            "rep_started_at": self.rep_started_at,
            "rep_state_started_at": self.rep_state_started_at,
            "rep_start_signal": self.rep_start_angle,
            "rep_peak_signal": self.rep_peak_angle,
            "rep_lowest_after_peak": self.rep_lowest_after_peak,
            "rep_last_signal": self.rep_last_angle,
            "return_close_to_start": self._is_returned(angle) if angle is not None else False,
            "rep_audio_suppressed": self.state in ACTIVE_REP_STATES,
        }

    def _current_tut_status(self) -> dict[str, Any]:
        tut_required = self.targets.tut_target * self.targets.tut_ratio_min
        tut_count_range = self._tut_count_range()
        if not self.rep_frames:
            return {
                "tut_seconds": 0.0,
                "tut_target": self.targets.tut_target,
                "missing_seconds": max(0.0, tut_required),
                "tut_count_mode": self._tut_count_mode(),
                "tut_count_range": self._public_tut_count_range(tut_count_range),
            }
        tut = compute_tut(self.rep_frames, tut_count_range, "target_angle_smoothed")
        tut_seconds = float(tut.get("tut_seconds", 0.0))
        return {
            "tut_seconds": tut_seconds,
            "tut_target": self.targets.tut_target,
            "missing_seconds": max(0.0, tut_required - tut_seconds),
            "tut_count_mode": self._tut_count_mode(),
            "tut_count_range": self._public_tut_count_range(tut_count_range),
        }

    def _rest_window_seconds(self) -> float:
        return float(self.config.get("rest_window_seconds", 0.8))

    def _rest_noise_max(self) -> float:
        return float(self.config.get("rest_noise_max", self.config.get("return_delta", 6.0) * 0.5))

    def _rest_velocity_max(self) -> float:
        return float(self.config.get("rest_velocity_max", max(self._attempt_start_delta(), 1e-6) * 0.75))

    def _motion_start_velocity_min(self) -> float:
        return float(self.config.get("motion_start_velocity_min", max(self._attempt_start_delta(), 1e-6) * 0.25))

    def _return_velocity_max(self) -> float:
        return float(self.config.get("return_velocity_max", -max(self._motion_start_velocity_min() * 0.25, 1e-6)))

    def _start_delta_from_rest(self) -> float:
        return float(self.config.get("start_delta_from_rest", self.config.get("attempt_start_delta", self.config.get("start_delta", 10.0))))

    def _return_delta_from_rest(self) -> float:
        return float(self.config.get("return_delta_from_rest", self.config.get("return_delta", 6.0)))

    def _return_to_start_delta(self) -> float:
        return float(self.config.get("return_to_start_delta", self._return_delta_from_rest()))

    def _stable_return_seconds(self) -> float:
        if "stable_return_seconds" in self.config:
            return float(self.config.get("stable_return_seconds") or 0.0)
        return 0.0

    def _current_stable_return_seconds(self) -> float:
        if self.stable_return_started_at is None or self.last_sample_at is None:
            return 0.0
        return max(0.0, self.last_sample_at - self.stable_return_started_at)

    def _attempt_start_delta(self) -> float:
        return float(self.config.get("attempt_start_delta", self.config.get("start_delta", 10.0)))

    def _hold_done_prompt(self) -> str:
        return str(self.config.get("hold_done_prompt") or "可以慢慢放下")

    def _returning_prompt(self) -> str:
        return str(self.config.get("returning_prompt") or "慢慢放下")

    def _tut_count_mode(self) -> str:
        return str(self.config.get("tut_count_mode") or "target_range")

    def _tut_count_range(self) -> tuple[float, float]:
        if self._tut_count_mode() == "at_or_above_target":
            return self.target_low, float("inf")
        return self.targets.target_range

    def _public_tut_count_range(self, target_range: tuple[float, float]) -> list[float | None]:
        low, high = target_range
        return [low, high if math.isfinite(high) else None]

    def _angle_in_tut_zone(self, angle: float | None) -> bool:
        if angle is None:
            return False
        low, high = self._tut_count_range()
        return float(angle) >= low and (not math.isfinite(high) or float(angle) <= high)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 1.0
    return numerator / denominator


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None




