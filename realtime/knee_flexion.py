"""Realtime knee-flexion state machine and per-rep metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from evaluate.core.speed_check import check_speed
from evaluate.core.tut import compute_tut

from realtime.state_machine import ConsecutiveConfirm, MotionState


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
    def __init__(self, config: dict[str, Any], targets: KneeFlexionTargets) -> None:
        self.config = config
        self.targets = targets
        self.state = MotionState.BASELINE
        self.baseline_angle: float | None = None
        self.baseline_started_at: float | None = None
        self.baseline_samples: list[float] = []
        self.rep_started_at: float | None = None
        self.rep_frames: list[dict[str, Any]] = []
        self.reached_target = False
        self.lost_visibility_frames = 0
        self.last_prompt = "请保持静止，正在校准"
        self.last_error: str | None = None
        confirm_frames = int(config.get("confirm_frames", 3))
        self._start_confirm = ConsecutiveConfirm(confirm_frames)
        self._hold_confirm = ConsecutiveConfirm(confirm_frames)
        self._return_confirm = ConsecutiveConfirm(confirm_frames)
        self._finish_confirm = ConsecutiveConfirm(confirm_frames)

    @property
    def target_low(self) -> float:
        return self.targets.target_range[0]

    @property
    def target_high(self) -> float:
        return self.targets.target_range[1]

    def process(self, frame: dict[str, Any]) -> dict[str, Any]:
        angle = _as_float(frame.get("target_angle_smoothed"))
        visibility = _as_float(frame.get("visibility_min") or frame.get("visibility"))
        now = _as_float(frame.get("relative_time")) or 0.0
        visibility_threshold = float(self.config.get("visibility_threshold", 0.55))

        if angle is None or visibility is None or visibility < visibility_threshold:
            self.last_prompt = "请站到摄像头前"
            if self.state in {MotionState.RISING, MotionState.HOLDING, MotionState.RETURNING}:
                self.lost_visibility_frames += 1
                max_lost = int(self.config.get("max_lost_visibility_frames", 8))
                if self.lost_visibility_frames >= max_lost:
                    rep_result = self._finish_rep(countable=False, forced_error="VISIBILITY_LOW")
                    return self._output(visible=False, angle=angle, rep_result=rep_result)
            return self._output(visible=False, angle=angle)
        self.lost_visibility_frames = 0

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
            self._process_holding(angle)
        elif self.state == MotionState.RETURNING:
            self._append_rep_frame(frame)
            rep_result = self._process_returning(angle)

        return self._output(visible=True, angle=angle, rep_result=rep_result)

    def _process_baseline(self, now: float, angle: float) -> dict[str, Any]:
        if self.baseline_started_at is None:
            self.baseline_started_at = now
        self.baseline_samples.append(angle)
        baseline_seconds = float(self.config.get("baseline_seconds", 2.0))
        if now - self.baseline_started_at >= baseline_seconds and self.baseline_samples:
            self.baseline_angle = mean(self.baseline_samples)
            self.state = MotionState.IDLE
            self.last_prompt = "准备开始第一遍"
        else:
            self.last_prompt = "请保持静止，正在校准"
        return self._output(visible=True, angle=angle)

    def _process_idle(self, now: float, angle: float, frame: dict[str, Any]) -> None:
        baseline = self.baseline_angle if self.baseline_angle is not None else angle
        start_delta = float(self.config.get("start_delta", 10.0))
        if self._start_confirm.update(angle - baseline >= start_delta):
            self.state = MotionState.RISING
            self.rep_started_at = now
            self.rep_frames = []
            self.reached_target = False
            self.lost_visibility_frames = 0
            self._append_rep_frame(frame)
            self.last_prompt = str(self.config.get("raise_prompt") or "再抬高一点")
            self._hold_confirm.reset()
            self._return_confirm.reset()
            self._finish_confirm.reset()
        else:
            self.last_prompt = "准备开始下一遍"

    def _process_rising(self, now: float, angle: float) -> dict[str, Any] | None:
        if self._hold_confirm.update(angle >= self.target_low):
            self.reached_target = True
            self.state = MotionState.HOLDING
            self.last_prompt = "保持住"
            return None
        baseline = self.baseline_angle if self.baseline_angle is not None else angle
        return_delta = float(self.config.get("return_delta", 6.0))
        if self.rep_started_at is not None and now - self.rep_started_at >= float(self.config.get("min_rep_seconds", 1.0)):
            if self._finish_confirm.update(angle <= baseline + return_delta):
                if self._is_noise_attempt():
                    self._reset_silent_attempt()
                    return None
                return self._finish_rep(countable=False)
        self.last_prompt = str(self.config.get("raise_prompt") or "再抬高一点")
        return None

    def _process_holding(self, angle: float) -> None:
        if self._return_confirm.update(angle < self.target_low):
            self.state = MotionState.RETURNING
            self.last_prompt = "慢慢放下"
        else:
            self.last_prompt = "保持住"

    def _process_returning(self, angle: float) -> dict[str, Any] | None:
        baseline = self.baseline_angle if self.baseline_angle is not None else angle
        return_delta = float(self.config.get("return_delta", 6.0))
        if self._finish_confirm.update(angle <= baseline + return_delta):
            return self._finish_rep(countable=self.reached_target)
        self.last_prompt = "慢慢放下"
        return None

    def _append_rep_frame(self, frame: dict[str, Any]) -> None:
        if not self.rep_frames or frame.get("frame_index") != self.rep_frames[-1].get("frame_index"):
            self.rep_frames.append(dict(frame))

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
            min_attempt_delta = float(self.config.get("start_delta", 10.0)) * 1.5
        return max(values) - baseline < float(min_attempt_delta)

    def _reset_silent_attempt(self) -> None:
        self.state = MotionState.IDLE
        self.rep_started_at = None
        self.rep_frames = []
        self.reached_target = False
        self.lost_visibility_frames = 0
        self._start_confirm.reset()
        self._hold_confirm.reset()
        self._return_confirm.reset()
        self._finish_confirm.reset()
        self.last_prompt = "准备开始下一遍"

    def _finish_rep(self, *, countable: bool, forced_error: str | None = None) -> dict[str, Any]:
        result = self._evaluate_rep(self.rep_frames, forced_error=forced_error)
        countable = result.get("primary_error") == "OK"
        result["countable"] = countable
        self.state = MotionState.IDLE
        self.rep_started_at = None
        self.rep_frames = []
        self.reached_target = False
        self.lost_visibility_frames = 0
        self._start_confirm.reset()
        self._hold_confirm.reset()
        self._return_confirm.reset()
        self._finish_confirm.reset()
        self.last_prompt = "很好，准备下一遍" if countable else str(self.config.get("raise_prompt") or "再抬高一点")
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
        tut = compute_tut(frames, self.targets.target_range, "target_angle_smoothed")
        speed = check_speed(frames, "target_angle_smoothed")
        tut_actual = float(tut.get("tut_seconds", 0.0))
        peak_speed = float(speed.get("peak_angular_velocity", 0.0))
        rom_diff = max(0.0, self.targets.rom_target - rom)
        tut_ratio = _safe_ratio(tut_actual, self.targets.tut_target)
        speed_ratio = _safe_ratio(peak_speed, self.targets.template_peak_speed)

        strict_quality_errors = bool(self.config.get("strict_quality_errors", False))

        if forced_error:
            primary_error = forced_error
        elif not reached_target:
            primary_error = "ROM_LOW"
        elif rom_diff > self.targets.rom_diff_max:
            primary_error = "ROM_LOW"
        elif strict_quality_errors and tut_ratio < self.targets.tut_ratio_min:
            primary_error = "TUT_LOW"
        elif strict_quality_errors and (
            duration < float(self.config.get("min_rep_seconds", 1.0)) or speed_ratio > self.targets.speed_ratio_max
        ):
            primary_error = "TOO_FAST"
        else:
            primary_error = "OK"

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
            "missing_seconds": max(0.0, self.targets.tut_target - tut_actual),
            "tut_ratio": tut_ratio,
            "peak_speed": peak_speed,
            "speed_ratio": speed_ratio,
            "primary_error": primary_error,
            "countable": primary_error == "OK",
            "reached_target": reached_target,
            "angle_curve": [
                {
                    "relative_time": frame.get("relative_time"),
                    "target_angle_smoothed": frame.get("target_angle_smoothed"),
                }
                for frame in frames
            ],
        }

    def _output(self, *, visible: bool, angle: float | None, rep_result: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "visible": visible,
            "angle": angle,
            "baseline_angle": self.baseline_angle,
            "target_range": list(self.targets.target_range),
            "prompt": self.last_prompt,
            "rep_result": rep_result,
        }


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
