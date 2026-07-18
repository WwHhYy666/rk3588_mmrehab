from __future__ import annotations

from pathlib import Path
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feedback_runtime import load_rules, rep_feedback
from training.action_state_machine import KneeFlexionRealtimeMachine, KneeFlexionTargets
from training.training_session import RealtimeTrainingSession
from training.state_machine import MotionState


def _targets(
    *,
    rom_target: float,
    tut_target: float,
    target_range: tuple[float, float],
    rom_diff_max: float,
    tut_ratio_min: float,
    min_rom_ratio: float = 0.0,
    min_tut_seconds: float = 0.0,
    min_rom_absolute: float = 0.0,
) -> KneeFlexionTargets:
    return KneeFlexionTargets(
        rom_target=rom_target,
        tut_target=tut_target,
        target_range=target_range,
        template_peak_speed=0.0,
        rom_diff_max=rom_diff_max,
        tut_ratio_min=tut_ratio_min,
        speed_ratio_max=1.5,
        min_rom_ratio=min_rom_ratio,
        min_tut_seconds=min_tut_seconds,
        min_rom_absolute=min_rom_absolute,
    )


def test_npu_dynamic_target_uses_absolute_floor_above_rest_pose() -> None:
    machine = _machine(
        {"dynamic_target_from_baseline": True},
        _targets(
            rom_target=0.135,
            tut_target=0.5,
            target_range=(-0.01, 0.13),
            rom_diff_max=0.05,
            tut_ratio_min=0.8,
            min_rom_ratio=0.90,
            min_rom_absolute=0.35,
        ),
    )
    machine.prime_baseline(0.10)

    assert abs(machine.required_rom - 0.35) < 1e-9
    assert abs(machine.target_low - 0.45) < 1e-9


def test_npu_slow_low_raise_waits_for_real_reversal_before_closing() -> None:
    machine = _machine(
        {
            "segment_mode": "raise_lower",
            "dynamic_target_from_baseline": True,
            "require_return_reversal": True,
            "return_reversal_fraction": 0.35,
            "return_reversal_confirm_frames": 2,
            "start_delta": 0.05,
            "start_delta_from_rest": 0.05,
            "attempt_start_delta": 0.05,
            "min_attempt_delta": 0.05,
            "return_delta": 0.10,
            "return_delta_from_rest": 0.06,
            "finish_confirm_frames": 2,
            "stable_return_seconds": 0.15,
            "return_pose_fallback_enabled": True,
            "min_rep_seconds": 0.2,
        },
        _targets(
            rom_target=0.60,
            tut_target=0.5,
            target_range=(0.50, 0.65),
            rom_diff_max=0.05,
            tut_ratio_min=0.8,
            min_rom_ratio=0.85,
            min_rom_absolute=0.25,
        ),
    )
    machine.prime_baseline(0.0)

    for index, value in enumerate([0.06, 0.12, 0.20, 0.18, 0.16], start=1):
        frame = _frame(index * 0.1, value)
        frame["return_pose_stable"] = True
        output = machine.process(frame)
        assert output.get("rep_result") is None

    assert machine.state == MotionState.RISING
    assert machine.return_reversal_confirmed is False

    result = None
    for relative_time, value in [(0.6, 0.08), (0.7, 0.04), (0.8, 0.01), (0.9, 0.0)]:
        frame = _frame(relative_time, value)
        frame["return_pose_stable"] = True
        output = machine.process(frame)
        if output.get("rep_result"):
            result = output["rep_result"]
            break

    assert result is not None
    assert result["primary_error"] == "ROM_LOW"
    assert result["completion_trigger"] == "return_pose_fallback"
    assert result["watchdog_used"] is False


def _machine(config: dict[str, float | bool], targets: KneeFlexionTargets) -> KneeFlexionRealtimeMachine:
    merged = {
        "baseline_seconds": 0.05,
        "confirm_frames": 1,
        "visibility_threshold": 0.55,
        "min_rep_seconds": 1.0,
        "strict_quality_errors": True,
        **config,
    }
    return KneeFlexionRealtimeMachine(merged, targets)


def _npu_seated_knee_raise_machine() -> KneeFlexionRealtimeMachine:
    base = yaml.safe_load(
        (PROJECT_ROOT / "training" / "configs" / "training_defaults_npu.yaml").read_text(encoding="utf-8")
    )
    action = yaml.safe_load(
        (PROJECT_ROOT / "evaluation" / "configs" / "npu" / "seated_knee_raise.yaml").read_text(encoding="utf-8")
    )
    realtime = {**base, **action["realtime"]}
    machine = KneeFlexionRealtimeMachine(
        realtime,
        _targets(
            rom_target=0.50,
            tut_target=0.6,
            target_range=(0.50, 0.70),
            rom_diff_max=0.10,
            tut_ratio_min=0.80,
            min_rom_ratio=0.85,
        ),
    )
    machine.prime_baseline(0.20)
    return machine


def _frame(relative_time: float, value: float) -> dict[str, float | bool]:
    return {
        "frame_index": int(relative_time * 100),
        "relative_time": relative_time,
        "target_angle_smoothed": value,
        "visibility_min": 1.0,
        "person_visible": True,
    }


def _sit_to_stand_metric_frame(hip_y: float) -> dict[str, object]:
    return {
        "selected_side": "left",
        "visibility_threshold": 0.18,
        "rehab_keypoints": {
            "left_shoulder": {"x": 0.4, "y": 0.2, "visibility": 0.9},
            "left_hip": {"x": 0.4, "y": hip_y, "visibility": 0.9},
        },
    }


def _prime_baseline(machine: KneeFlexionRealtimeMachine, baseline: float) -> None:
    machine.process(_frame(0.0, baseline))
    machine.process(_frame(0.1, baseline))
    machine.process(_frame(0.2, baseline))


def _run_values(machine: KneeFlexionRealtimeMachine, values: list[tuple[float, float]]) -> dict:
    result = None
    for relative_time, value in values:
        output = machine.process(_frame(relative_time, value))
        if isinstance(output, dict) and isinstance(output.get("rep_result"), dict):
            result = output["rep_result"]
    if result is None and values:
        last_time, last_value = values[-1]
        for step in range(1, 4):
            output = machine.process(_frame(last_time + step * 0.05, last_value))
            if isinstance(output, dict) and isinstance(output.get("rep_result"), dict):
                result = output["rep_result"]
                break
    assert result is not None, "expected one rep_result"
    return result


def test_extension_short_rom_speaks_rom_low() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    result = _run_values(machine, [(0.3, 111.0), (0.6, 116.0), (1.45, 104.0)])

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False
    rules = load_rules(PROJECT_ROOT / "action_feedback" / "rules" / "seated_knee_extension_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_extension")
    assert feedback["tts_text"] == "腿再伸直一点"


def test_extension_low_amplitude_attempt_speaks_rom_low() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    result = _run_values(machine, [(0.3, 105.0), (0.6, 108.0), (1.45, 103.0)])

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False
    rules = load_rules(PROJECT_ROOT / "action_feedback" / "rules" / "seated_knee_extension_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_extension")
    assert feedback["tts_text"] == "腿再伸直一点"



def test_dynamic_rest_anchor_recalibrates_after_slow_drift() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=0.5,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "segment_mode": "sit_to_stand",
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.06,
            "rest_window_seconds": 0.5,
            "rest_noise_max": 0.02,
            "rest_velocity_max": 0.08,
            "motion_start_velocity_min": 0.10,
            "min_attempt_delta": 0.16,
            "tut_count_mode": "at_or_above_target",
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    for relative_time, value in [(0.30, 0.015), (0.45, 0.02), (0.60, 0.02), (0.75, 0.02), (0.90, 0.02)]:
        output = machine.process(_frame(relative_time, value))

    assert output["state"] == "IDLE"
    assert output["rest_anchor"] is not None
    assert abs(output["rest_anchor"] - 0.02) < 0.01

    result = _run_values(machine, [(1.10, 0.14), (1.20, 0.72), (1.80, 0.72), (2.00, 0.04)])

    assert result["primary_error"] == "OK"
    assert result["countable"] is True

def test_extension_tiny_motion_stays_silent() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    outputs = [machine.process(_frame(time, value)) for time, value in [(0.3, 102.0), (0.6, 103.0), (1.45, 101.0)]]

    assert [output.get("state") for output in outputs if isinstance(output, dict)] == ["IDLE", "IDLE", "IDLE"]
    assert all(isinstance(output, dict) and output.get("rep_result") is None for output in outputs)


def test_extension_short_hold_surfaces_tut_low() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    result = _run_values(machine, [(0.3, 111.0), (0.4, 140.0), (0.7, 130.0), (1.5, 104.0)])

    assert result["primary_error"] == "TUT_LOW"
    assert result["all_errors"] == ["TUT_LOW"]
    rules = load_rules(PROJECT_ROOT / "action_feedback" / "rules" / "seated_knee_extension_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_extension")
    assert feedback["tts_text"].startswith("再坚持 ")


def test_raise_peak_target_counts_even_when_rom_diff_is_large() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=0.5,
        target_range=(0.40, 0.55),
        rom_diff_max=0.25,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "return_delta": 0.04,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    result = _run_values(machine, [(0.3, 0.28), (0.4, 0.62), (1.1, 0.62), (1.4, 0.22), (1.5, 0.22)])

    assert result["primary_error"] == "OK"
    assert result["countable"] is True
    assert result["peak_ok"] is True
    assert result["rom_ok"] is False
    assert result["rom_blocks_count"] is False
    assert result["tut_count_mode"] == "at_or_above_target"
    assert result["tut_count_range"] == [0.40, None]


def test_raise_above_target_high_counts_tut_without_lower_prompt() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.40, 0.55),
        rom_diff_max=0.25,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "return_delta": 0.04,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
            "hold_done_prompt": "保持住",
            "returning_prompt": "保持住",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    outputs = [
        machine.process(_frame(0.3, 0.28)),
        machine.process(_frame(0.4, 0.62)),
        machine.process(_frame(1.1, 0.62)),
    ]

    assert outputs[-1]["state"] == "HOLDING"
    assert outputs[-1]["in_tut_zone"] is True
    assert outputs[-1]["missing_seconds"] == 0.0
    assert outputs[-1]["prompt"] == "保持住"
    returning = machine.process(_frame(1.2, 0.30))
    assert returning["prompt"] == "保持住"


def test_raise_peak_target_short_hold_reports_tut_low_even_when_rom_delta_is_small() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.40, 0.55),
        rom_diff_max=0.25,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "return_delta": 0.04,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    result = _run_values(machine, [(0.3, 0.28), (0.4, 0.42), (0.55, 0.22), (1.5, 0.22)])

    assert result["primary_error"] == "TUT_LOW"
    assert result["countable"] is False
    assert "ROM_LOW" not in result["all_errors"]
    assert result["peak_ok"] is True
    assert result["rom_ok"] is False
    assert result["rom_blocks_count"] is False


def test_raise_below_peak_target_still_reports_rom_low() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.40, 0.55),
        rom_diff_max=0.25,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "return_delta": 0.04,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    result = _run_values(machine, [(0.3, 0.28), (0.4, 0.34), (0.55, 0.22), (1.5, 0.22)])

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False
    assert result["peak_ok"] is False
    assert result["rom_ok"] is False
    assert result["rom_blocks_count"] is False

def test_raise_rom_ok_short_hold_reports_tut_low() -> None:
    targets = _targets(
        rom_target=0.50,
        tut_target=1.0,
        target_range=(0.40, 0.55),
        rom_diff_max=0.25,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "return_delta": 0.04,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    result = _run_values(machine, [(0.3, 0.28), (0.4, 0.60), (0.5, 0.22), (1.5, 0.22)])

    assert result["primary_error"] == "TUT_LOW"
    assert "ROM_LOW" not in result["all_errors"]
    rules = load_rules(PROJECT_ROOT / "action_feedback" / "rules" / "seated_knee_raise_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_raise")
    assert feedback["tts_text"].startswith("再坚持 ")


def test_low_rom_attempt_waits_for_stable_return_before_result() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "return_stable_frames": 2,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    for time_value, angle in [(0.3, 105.0), (0.6, 108.0)]:
        output = machine.process(_frame(time_value, angle))
        assert output["rep_result"] is None

    first_return = machine.process(_frame(1.45, 103.0))
    assert first_return["rep_result"] is None
    assert first_return["state"] == "RISING"

    second_return = machine.process(_frame(1.50, 103.0))
    assert second_return["rep_result"] is None
    third_return = machine.process(_frame(1.55, 103.0))
    assert third_return["rep_result"]["primary_error"] == "ROM_LOW"
    assert third_return["rep_result"]["countable"] is False


def test_short_hold_waits_for_stable_return_before_tut_low() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "return_stable_frames": 2,
            "min_attempt_delta": 3.0,
            "count_by_peak_target": True,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    for time_value, angle in [(0.3, 111.0), (0.4, 140.0), (0.7, 130.0)]:
        output = machine.process(_frame(time_value, angle))
        assert output["rep_result"] is None

    first_return = machine.process(_frame(1.5, 104.0))
    assert first_return["rep_result"] is None
    assert first_return["state"] == "RETURNING"

    second_return = machine.process(_frame(1.55, 104.0))
    assert second_return["rep_result"] is None
    third_return = machine.process(_frame(1.60, 104.0))
    assert third_return["rep_result"]["primary_error"] == "TUT_LOW"
    assert third_return["rep_result"]["countable"] is False
def test_visibility_loss_resets_without_error_result() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "max_lost_visibility_frames": 2,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)
    machine.process(_frame(0.3, 111.0))

    bad_frame = _frame(0.4, 112.0)
    bad_frame["visibility_min"] = 0.1
    first = machine.process(bad_frame)
    second = machine.process({**bad_frame, "frame_index": 41, "relative_time": 0.41})

    assert first["rep_result"] is None
    assert second["rep_result"] is None
    assert second["action_keypoints_valid"] is False
    assert second["prompt"] == ""
    assert second["state"] == "IDLE"

def test_idle_prompt_does_not_show_prepare_next_rep() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "raise_prompt": "再伸直一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    output = machine.process(_frame(0.3, 100.5))

    assert output["state"] == "IDLE"
    assert output["prompt"] == ""




def test_hamstring_low_amplitude_attempt_reports_rom_low() -> None:
    targets = _targets(
        rom_target=55.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=15.0,
        tut_ratio_min=0.65,
    )
    machine = _machine(
        {
            "start_delta": 12.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "raise_prompt": "小腿再往后弯一点",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    result = _run_values(
        machine,
        [
            (0.3, 104.5),
            (1.5, 110.0),
            (1.7, 104.0),
        ],
    )

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False




def test_sit_to_stand_above_target_high_counts_tut() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_stable_frames": 2,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "tut_count_mode": "at_or_above_target",
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    result = _run_values(machine, [(0.3, 0.12), (0.4, 0.82), (1.25, 0.84), (1.45, 0.04), (1.5, 0.04), (1.55, 0.04)])

    assert result["primary_error"] == "OK"
    assert result["countable"] is True
    assert result["tut_count_mode"] == "at_or_above_target"
    assert result["tut_count_range"] == [0.58, None]
def test_sit_to_stand_low_height_reports_rom_low_after_sitting_back() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_stable_frames": 2,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    result = _run_values(machine, [(0.3, 0.12), (1.5, 0.30), (1.7, 0.04), (1.75, 0.04)])

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False


def test_sit_to_stand_short_hold_reports_tut_low_after_sitting_back() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_stable_frames": 2,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    result = _run_values(machine, [(0.3, 0.12), (0.4, 0.70), (0.6, 0.70), (1.4, 0.04), (1.45, 0.04), (1.5, 0.04)])

    assert result["primary_error"] == "TUT_LOW"
    assert result["countable"] is False

def test_sit_to_stand_bad_attempt_unlocks_next_rep() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_to_start_delta": 0.08,
            "return_stable_frames": 2,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "max_rep_seconds": 6.0,
            "return_timeout_seconds": 2.0,
            "tut_count_mode": "at_or_above_target",
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    ok = _run_values(machine, [(0.3, 0.12), (0.4, 0.82), (1.25, 0.84), (1.45, 0.04), (1.50, 0.04), (1.55, 0.04)])
    assert ok["primary_error"] == "OK"
    bad = _run_values(machine, [(2.0, 0.12), (3.3, 0.30), (3.55, 0.06), (3.60, 0.06), (3.65, 0.06)])
    assert bad["primary_error"] == "ROM_LOW"
    assert bad["countable"] is False
    next_ok = _run_values(machine, [(4.0, 0.12), (4.1, 0.82), (4.9, 0.82), (5.2, 0.04), (5.25, 0.04), (5.30, 0.04)])
    assert next_ok["primary_error"] == "OK"


def test_sit_to_stand_returns_near_attempt_start_even_above_old_baseline() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_to_start_delta": 0.08,
            "return_stable_frames": 2,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "tut_count_mode": "at_or_above_target",
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    result = _run_values(machine, [(0.3, 0.12), (0.4, 0.70), (0.6, 0.70), (1.4, 0.17), (1.45, 0.17), (1.50, 0.17)])

    assert result["primary_error"] == "TUT_LOW"
    assert result["countable"] is False


def test_sit_to_stand_return_timeout_finishes_instead_of_staying_returning() -> None:
    targets = _targets(
        rom_target=0.70,
        tut_target=1.0,
        target_range=(0.58, 0.78),
        rom_diff_max=0.18,
        tut_ratio_min=0.75,
    )
    machine = _machine(
        {
            "start_delta": 0.10,
            "attempt_start_delta": 0.10,
            "return_delta": 0.05,
            "return_to_start_delta": 0.08,
            "return_stable_frames": 3,
            "min_attempt_delta": 0.16,
            "min_rep_seconds": 1.2,
            "max_rep_seconds": 3.0,
            "return_timeout_seconds": 0.5,
            "tut_count_mode": "at_or_above_target",
            "raise_prompt": "再站起来一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    result = _run_values(machine, [(0.3, 0.12), (0.4, 0.70), (0.5, 0.70), (1.2, 0.30), (1.9, 0.18)])

    assert result["primary_error"] == "TUT_LOW"
    assert machine.state.value == "IDLE"


def test_sit_to_stand_config_has_deadlock_recovery_thresholds() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / "sit_to_stand.yaml").read_text(encoding="utf-8"))
    realtime = config["realtime"]

    assert realtime["return_to_start_delta"] == 0.08
    assert realtime["max_rep_seconds"] == 6.0
    assert realtime["return_timeout_seconds"] == 2.0
    assert realtime["reset_on_offscreen"] is True

def test_train_page_renders_care_dialog_controls() -> None:
    script = (PROJECT_ROOT / "rehab_app" / "server" / "static" / "train.js").read_text(encoding="utf-8")

    assert "id=\"care-dialog\"" in script
    assert "function renderCareDialog" in script
    assert "awaiting_care_response" in script
    assert "/api/realtime/care_response" in script
    assert "submitCareResponse(true)" in script
    assert "submitCareResponse(false)" in script

def test_demo_plan_requires_side_view_and_delayed_offscreen_prompt() -> None:
    plan = yaml.safe_load((PROJECT_ROOT / "training" / "configs" / "rehab_demo_plan.yaml").read_text(encoding="utf-8"))
    actions = plan["actions"]

    assert plan["rest_seconds"] == 6
    assert plan["offscreen_timeout_seconds"] == 5.0
    assert plan["rest_music_fade_seconds"] == 1.0
    assert plan["front_orientation_confirm_frames"] == 2
    assert plan["rknn_front_orientation_confirm_frames"] == 2
    assert plan["rknn_orientation_confirm_frames"] == 4
    assert plan["return_confirm_frames"] == 2
    assert plan["cpu_reentry_v2"] is True
    assert plan["cpu_return_presence_enter_frames"] == 2
    assert plan["cpu_return_presence_grace_frames"] == 6
    assert plan["cpu_return_core_points_min"] == 3
    assert plan["cpu_reentry_max_wait_seconds"] == 2.5
    assert plan["rknn_return_confirm_frames"] == 2
    assert plan["return_orientation_required"] is False
    assert [action["action_id"] for action in actions] == ["sit_to_stand", "standing_hamstring_curl", "seated_knee_raise"]
    assert all(action["require_side_view"] is True for action in actions)




def test_sit_to_stand_config_counts_tut_at_or_above_target() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / "sit_to_stand.yaml").read_text(encoding="utf-8"))
    realtime = config["realtime"]

    assert config["primary_metric"] == "hip_rise_height_ratio"
    assert realtime["tut_count_mode"] == "at_or_above_target"
def test_seated_knee_raise_return_prompts_allow_lowering() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / "seated_knee_raise.yaml").read_text(encoding="utf-8"))
    realtime = config["realtime"]

    assert config["thresholds"]["rom_diff_max"] == 0.10
    assert realtime["return_delta"] == 0.08
    assert realtime["return_delta_from_rest"] == 0.08
    assert realtime["stable_return_seconds"] == 0.10
    assert realtime["tut_range_padding"] == 0.08
    assert realtime["correction_tts_interval_seconds"] == 4.0
    assert realtime["hold_done_prompt"] == "可以慢慢放下"
    assert realtime["returning_prompt"] == "慢慢放下"
def test_standing_hamstring_curl_thresholds_allow_small_wrong_attempts() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / "standing_hamstring_curl.yaml").read_text(encoding="utf-8"))
    realtime = config["realtime"]

    assert config["thresholds"]["rom_diff_max"] == 25.0
    assert config["thresholds"]["dtw_normalized_max"] == 0.25
    assert realtime["start_delta"] == 12.0
    assert realtime["return_delta"] == 5.0
    assert realtime["attempt_start_delta"] == 4.0
    assert realtime["min_attempt_delta"] == 3.0
    assert realtime["tut_range_padding"] == 15.0
    assert realtime["tut_ratio_min"] == 0.65
    assert "tut_count_mode" not in realtime
    assert "count_by_peak_target" not in realtime


if __name__ == "__main__":
    test_extension_short_rom_speaks_rom_low()
    test_extension_low_amplitude_attempt_speaks_rom_low()
    test_extension_tiny_motion_stays_silent()
    test_extension_short_hold_surfaces_tut_low()
    test_raise_peak_target_counts_even_when_rom_diff_is_large()
    test_raise_above_target_high_counts_tut_without_lower_prompt()
    test_raise_peak_target_short_hold_reports_tut_low_even_when_rom_delta_is_small()
    test_raise_below_peak_target_still_reports_rom_low()
    test_raise_rom_ok_short_hold_reports_tut_low()
    test_low_rom_attempt_waits_for_stable_return_before_result()
    test_short_hold_waits_for_stable_return_before_tut_low()
    test_visibility_loss_resets_without_error_result()
    test_idle_prompt_does_not_show_prepare_next_rep()
    test_hamstring_low_amplitude_attempt_reports_rom_low()
    test_sit_to_stand_above_target_high_counts_tut()
    test_sit_to_stand_low_height_reports_rom_low_after_sitting_back()
    test_sit_to_stand_short_hold_reports_tut_low_after_sitting_back()
    test_train_page_renders_care_dialog_controls()
    test_demo_plan_requires_side_view_and_delayed_offscreen_prompt()
    test_sit_to_stand_config_counts_tut_at_or_above_target()
    test_seated_knee_raise_return_prompts_allow_lowering()
    test_standing_hamstring_curl_thresholds_allow_small_wrong_attempts()
    print("knee_flexion realtime tests passed")




def test_training_session_rep_audio_gate_blocks_active_rep() -> None:
    session = RealtimeTrainingSession()
    session.status = "running"
    session.last_machine_output = {"state": "RISING"}

    assert session._rep_audio_allowed() is False
    assert session._rep_audio_allowed(rep_result={"primary_error": "OK"}) is True

    session.last_machine_output = {"state": "IDLE"}
    assert session._rep_audio_allowed() is True


def test_runtime_thresholds_include_cycle_diagnostics() -> None:
    session = RealtimeTrainingSession()
    session.current_realtime_config = {"segment_mode": "raise_lower", "start_delta": 0.1}
    snapshot = session._runtime_threshold_snapshot(
        {
            "state": "RISING",
            "segment_mode": "raise_lower",
            "angle": 0.3,
            "rest_anchor": 0.2,
            "rest_noise": 0.01,
            "velocity": 0.12,
            "peak_value": 0.5,
            "stable_return_seconds": 0.0,
            "rep_audio_suppressed": True,
        }
    )

    assert snapshot["segment_mode"] == "raise_lower"
    assert snapshot["rest_anchor"] == 0.2
    assert snapshot["velocity"] == 0.12
    assert snapshot["peak_value"] == 0.5
    assert snapshot["rep_audio_suppressed"] is True



class _FakeTTSWorker:
    def __init__(self) -> None:
        self.calls = []

    def is_busy(self, *args, **kwargs) -> bool:
        return False

    def speak(self, *args, **kwargs) -> bool:
        self.calls.append((args, kwargs))
        return True


class _BusyFakeTTSWorker(_FakeTTSWorker):
    def is_busy(self, *args, **kwargs) -> bool:
        return True


def _reentry_targets() -> KneeFlexionTargets:
    return _targets(
        rom_target=0.5,
        tut_target=0.5,
        target_range=(0.45, 0.8),
        rom_diff_max=0.2,
        tut_ratio_min=0.5,
    )


def _session_with_reentry_machine(config: dict | None = None) -> RealtimeTrainingSession:
    realtime_config = {
        "baseline_seconds": 0.0,
        "confirm_frames": 1,
        "rest_noise_max": 0.02,
        "rest_velocity_max": 0.05,
        "reentry_min_stable_samples": 3,
        "start_delta_from_rest": 0.08,
        "min_attempt_delta": 0.08,
        "motion_start_velocity_min": 0.0,
        **(config or {}),
    }
    targets = _reentry_targets()
    session = RealtimeTrainingSession()
    session.current_targets = targets
    session.current_realtime_config = realtime_config
    session.machine = KneeFlexionRealtimeMachine(realtime_config, targets)
    session.start_time = time.time()
    return session


def test_offscreen_wait_always_discards_active_motion_state() -> None:
    session = _session_with_reentry_machine()
    fake_tts = _FakeTTSWorker()
    session.tts_worker = fake_tts
    session.status = "running"
    session.current_realtime_config = {**session.current_realtime_config, "reset_on_offscreen": False}
    session.machine.state = MotionState.RETURNING
    session.last_machine_output = {"state": "RETURNING"}
    session.pending_action_start = {"reset_timing": True}
    session.pending_feedback_resume = True
    session.offscreen_since = time.time() - 5.2
    session.offscreen_seconds = 5.2

    session._enter_offscreen_wait()

    assert session.status == "awaiting_return"
    assert session.pause_reason == "offscreen"
    assert session.last_machine_output is None
    assert session.machine is not None
    assert session.machine.state == MotionState.BASELINE
    assert session.pending_action_start is None
    assert session.pending_feedback_resume is False
    assert session.reentry_state == "awaiting_return"
    assert session.reentry_ready is False
    assert session.offscreen_seconds == 5.2


def test_return_gate_ignores_busy_tts_when_pose_is_valid() -> None:
    session = _session_with_reentry_machine()
    session.tts_worker = _BusyFakeTTSWorker()
    session.status = "awaiting_return"
    session.orientation_required = True
    session.return_confirm_frames = 2
    frame = {
        "person_visible": True,
        "pose_detected": True,
        "target_angle_smoothed": 0.0,
        "action_keypoints_valid": True,
        "side_view_ok": True,
        "orientation_ok": True,
    }

    session._process_return_gate(True, True, frame)
    assert session.status == "awaiting_return"
    assert session.return_confirm_count == 1

    session._process_return_gate(True, True, frame)
    assert session.status == "running"
    assert session.reentry_state == "reentry_calibrating"
    assert session.reentry_ready is False


def test_return_gate_resumes_when_side_view_temporarily_misclassified() -> None:
    session = _session_with_reentry_machine()
    session.status = "awaiting_return"
    session.orientation_required = True
    session.return_confirm_frames = 2
    session.orientation_prompt = "请侧身站好。"
    frame = {
        "person_visible": True,
        "pose_detected": True,
        "target_angle_smoothed": 0.0,
        "action_keypoints_valid": True,
        "side_view_ok": False,
        "orientation_ok": False,
    }

    session._process_return_gate(True, False, frame)
    assert session.status == "awaiting_return"
    assert session.return_confirm_count == 1
    assert session.last_prompt == "请保持稳定，马上继续。"

    session._process_return_gate(True, False, frame)
    assert session.status == "running"
    assert session.pause_reason is None
    assert session.last_prompt == "请回到画面中，我们继续"


def test_return_gate_does_not_hard_block_on_side_view_even_if_config_is_stale() -> None:
    session = _session_with_reentry_machine()
    session.status = "awaiting_return"
    session.orientation_required = True
    session.return_orientation_required = True
    session.return_confirm_frames = 1
    session.orientation_prompt = "请侧身站好。"
    frame = {
        "person_visible": True,
        "pose_detected": True,
        "target_angle_smoothed": 0.0,
        "action_keypoints_valid": True,
        "side_view_ok": False,
        "orientation_ok": False,
    }

    session._process_return_gate(True, False, frame)

    assert session.status == "running"
    assert session.pause_reason is None
    assert session.last_prompt == "请回到画面中，我们继续"


def _cpu_return_frame(*, person_visible: bool = True, core_visibility: float = 1.0) -> dict:
    rehab_keypoints = {
        name: {"x": 0.4, "y": 0.5, "visibility": core_visibility}
        for name in (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        )
    }
    return {
        "person_visible": person_visible,
        "pose_detected": person_visible,
        "actual_backend": "mediapipe",
        "action_keypoints_valid": False,
        "target_angle_smoothed": None,
        "rehab_keypoints": rehab_keypoints,
    }


def test_cpu_reentry_v2_resumes_after_two_presence_frames_without_action_angle() -> None:
    session = _session_with_reentry_machine()
    session.status = "awaiting_return"
    session.pose_backend = "mediapipe"
    session.cpu_reentry_v2 = True
    session.return_confirm_frames = 2
    session.cpu_return_presence_enter_frames = 2

    frame = _cpu_return_frame()
    session._process_return_gate(True, False, frame)
    assert session.status == "awaiting_return"
    assert session.return_presence_hits == 1

    session._process_return_gate(True, False, frame)
    assert session.status == "running"
    assert session.reentry_state == "reentry_calibrating"


def test_cpu_reentry_v2_keeps_presence_progress_across_short_dropout() -> None:
    session = _session_with_reentry_machine()
    session.status = "awaiting_return"
    session.pose_backend = "mediapipe"
    session.cpu_reentry_v2 = True
    session.return_confirm_frames = 2
    session.cpu_return_presence_enter_frames = 2
    session.cpu_return_presence_grace_frames = 2

    frame = _cpu_return_frame()
    session._process_return_gate(True, False, frame)
    session._process_return_gate(False, False, _cpu_return_frame(person_visible=False))
    assert session.return_presence_hits == 1
    assert session.status == "awaiting_return"

    session._process_return_gate(True, False, frame)
    assert session.status == "running"


def test_cpu_reentry_v2_does_not_change_rknn_return_gate() -> None:
    session = _session_with_reentry_machine()
    session.status = "awaiting_return"
    session.pose_backend = "rknn"
    session.cpu_reentry_v2 = True
    session.rknn_return_confirm_frames = 1
    frame = _cpu_return_frame()
    frame["actual_backend"] = "rknn"

    session._process_return_gate(True, True, frame)

    assert session.status == "awaiting_return"
    assert session.last_prompt == "请完整入画并站稳。"


def test_cpu_reentry_v2_accepts_stable_anchor_after_max_wait() -> None:
    session = _session_with_reentry_machine()
    session.pose_backend = "mediapipe"
    session.cpu_reentry_v2 = True
    session.cpu_reentry_max_wait_seconds = 2.5
    now = time.time()
    session.last_offscreen_resume_at = now - 3.0
    session._reentry_samples = [(now - 0.4, 0.44), (now - 0.3, 0.44), (now - 0.2, 0.44)]
    session.last_machine_output = {"state": "IDLE", "baseline_angle": 0.0}

    assert session._reentry_start_pose_ready([0.44, 0.44, 0.44]) is False
    assert session._reentry_baseline_ready(now) is True

def test_offscreen_return_during_playlist_rest_advances_to_next_action() -> None:
    session = _session_with_reentry_machine()
    called = []

    def fake_start_playlist_action(index: int) -> None:
        called.append(index)
        session.status = "running"
        session.playlist_index = index
        session.rest_context = None
        session.rest_until = None

    session._start_playlist_action = fake_start_playlist_action  # type: ignore[method-assign]
    session.status = "awaiting_return"
    session.pause_reason = "offscreen"
    session.rest_context = "playlist_transition"
    session.rest_until = time.time() - 0.1
    session.playlist_mode = True
    session.playlist_index = 0
    session.playlist_actions = [{"action_id": "first"}, {"action_id": "second"}]
    session.return_confirm_frames = 1

    session._process_return_gate(
        True,
        True,
        {
            "person_visible": True,
            "pose_detected": True,
            "target_angle_smoothed": 0.0,
            "action_keypoints_valid": True,
            "visibility_min": 1.0,
        },
    )

    assert called == [1]
    assert session.status == "running"
    assert session.playlist_index == 1
def test_reentry_calibration_waits_for_start_pose_before_ready() -> None:
    session = _session_with_reentry_machine()
    session.offscreen_reentry_guard_seconds = 0.5
    session._resume_running_after_offscreen()
    now = time.time()
    session.last_offscreen_resume_at = now - 1.0
    session.start_time = now - 1.0

    for offset, value in [(0.0, 0.55), (0.1, 0.56), (0.2, 0.55)]:
        session._warm_reentry_calibration(
            {"person_visible": True, "pose_detected": True, "target_angle_smoothed": value, "action_keypoints_valid": True, "visibility_min": 1.0},
            now + offset,
        )

    assert session.reentry_ready is False
    assert session.reentry_state == "reentry_calibrating"
    session.inscreen_prompt_until = now - 1.0
    session._warm_reentry_calibration(
        {"person_visible": True, "pose_detected": True, "target_angle_smoothed": 0.55, "action_keypoints_valid": True, "visibility_min": 1.0},
        now + 0.3,
    )
    assert session.last_prompt == "请先回到起始姿势站稳"

    session._resume_running_after_offscreen()
    now = time.time()
    session.last_offscreen_resume_at = now - 1.0
    session.start_time = now - 1.0
    for offset in [0.0, 0.1, 0.2]:
        session._warm_reentry_calibration(
            {"person_visible": True, "pose_detected": True, "target_angle_smoothed": 0.1, "action_keypoints_valid": True, "visibility_min": 1.0},
            now + offset,
        )

    assert session.reentry_ready is True
    assert session.reentry_state == "reentry_ready"
    session.inscreen_prompt_until = now - 1.0
    session._warm_reentry_calibration(
        {"person_visible": True, "pose_detected": True, "target_angle_smoothed": 0.1, "action_keypoints_valid": True, "visibility_min": 1.0},
        now + 0.3,
    )
    assert session.last_prompt == "可以开始动作"
    assert session.machine is not None
    assert session.machine.state == MotionState.IDLE

def test_visibility_loss_uses_interrupt_reset_seconds() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
            "max_lost_visibility_frames": 100,
            "interrupt_reset_seconds": 0.2,
            "raise_prompt": "raise more",
        },
        targets,
    )
    _prime_baseline(machine, 100.0)
    machine.process(_frame(0.3, 111.0))

    bad_frame = _frame(0.35, 112.0)
    bad_frame["visibility_min"] = 0.1
    first = machine.process(bad_frame)
    second = machine.process({**bad_frame, "frame_index": 45, "relative_time": 0.45})
    third = machine.process({**bad_frame, "frame_index": 56, "relative_time": 0.56})

    assert first["state"] == "RISING"
    assert second["state"] == "RISING"
    assert third["state"] == "IDLE"
    assert third["rep_result"] is None


def test_offscreen_wait_suppresses_audio_when_active_rep_interrupted() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _FakeTTSWorker()
    session.tts_worker = fake_tts
    session.status = "running"
    session.current_realtime_config = {"reset_on_offscreen": True}
    session.last_machine_output = {"state": "RISING"}

    session._enter_offscreen_wait()

    assert session.status == "awaiting_return"
    assert session.last_prompt
    assert session.offscreen_prompt_pending is False
    assert fake_tts.calls
    assert fake_tts.calls[0][1]["event_type"] == "offscreen"

def test_visibility_min_zero_does_not_fallback_to_overall_visibility() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "min_attempt_delta": 3.0,
        },
        targets,
    )
    _prime_baseline(machine, 100.0)

    frame = _frame(0.3, 120.0)
    frame["visibility_min"] = 0.0
    frame["visibility"] = 1.0
    output = machine.process(frame)

    assert output["action_keypoints_valid"] is False
    assert output["state"] == "IDLE"
    assert output["rep_result"] is None


def test_return_delta_from_rest_does_not_fallback_to_return_to_start_delta() -> None:
    targets = _targets(
        rom_target=50.0,
        tut_target=1.0,
        target_range=(135.0, 150.0),
        rom_diff_max=18.0,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "start_delta": 10.0,
            "attempt_start_delta": 4.0,
            "return_delta": 5.0,
            "return_to_start_delta": 99.0,
        },
        targets,
    )

    assert machine._return_delta_from_rest() == 5.0
    assert machine._return_to_start_delta() == 99.0


def test_care_dialog_audio_allowed_after_status_change() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _FakeTTSWorker()
    session.tts_worker = fake_tts
    session.status = "running"

    session._show_care_dialog()

    assert session.status == "awaiting_care_response"
    assert fake_tts.calls
    args, kwargs = fake_tts.calls[-1]
    assert kwargs["event_type"] == "care"

def test_offscreen_timeout_runs_during_action_audio_wait() -> None:
    session = RealtimeTrainingSession()
    session.status = "awaiting_action_audio"
    session.offscreen_timeout_seconds = 5.0
    session.offscreen_since = time.time() - 5.2

    session.process_frame({"person_visible": False, "pose_detected": False})

    assert session.status == "awaiting_return"
    assert session.pause_reason == "offscreen"
    assert session.offscreen_seconds >= 5.0


def test_offscreen_timeout_runs_during_rep_feedback_wait() -> None:
    session = RealtimeTrainingSession()
    session.status = "awaiting_rep_feedback"
    session.offscreen_timeout_seconds = 5.0
    session.offscreen_since = time.time() - 5.2

    session.process_frame({"person_visible": False, "pose_detected": False})

    assert session.status == "awaiting_return"
    assert session.pause_reason == "offscreen"
    assert session.offscreen_seconds >= 5.0


def test_front_orientation_uses_fast_confirm_frames() -> None:
    session = RealtimeTrainingSession()
    session.orientation_required = True
    session.orientation_prompt = "请侧身对准镜头。"
    session.front_orientation_confirm_frames = 2
    session.orientation_confirm_frames = 8
    session._enter_orientation_wait(speak=False)

    session._process_orientation_gate(True, False, {"front_view_ok": True, "side_view_ok": False})
    assert session.orientation_phase == "awaiting_front"
    assert session.front_orientation_confirm_count == 1

    session._process_orientation_gate(True, False, {"front_view_ok": True, "side_view_ok": False})
    assert session.orientation_phase == "awaiting_side"
    assert session.orientation_state == "waiting_side_view"
    assert session.last_prompt == "请侧身对准镜头。"


def test_side_orientation_keeps_original_confirm_frames() -> None:
    session = RealtimeTrainingSession()
    session.orientation_required = True
    session.orientation_prompt = "请侧身对准镜头。"
    session.front_orientation_confirm_frames = 1
    session.orientation_confirm_frames = 3
    session._enter_orientation_wait(speak=False)
    session.orientation_phase = "awaiting_side"

    frame = {"front_view_ok": True, "side_view_ok": True, "orientation_ok": True}
    session._process_orientation_gate(True, True, frame)
    assert session.orientation_state == "side_view_confirming"
    assert session.orientation_phase == "awaiting_side"

    session._process_orientation_gate(True, True, frame)
    assert session.orientation_state == "side_view_confirming"
    assert session.orientation_phase == "awaiting_side"

    session._process_orientation_gate(True, True, frame)
    assert session.orientation_state == "side_view_ok"
    assert session.orientation_phase == "ready"

def test_seated_knee_raise_relaxed_target_enters_holding_below_old_line() -> None:
    targets = _targets(
        rom_target=0.50,
        tut_target=1.0,
        target_range=(0.30, 0.55),
        rom_diff_max=0.30,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "segment_mode": "raise_lower",
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "start_delta_from_rest": 0.03,
            "return_delta": 0.08,
            "return_delta_from_rest": 0.08,
            "stable_return_seconds": 0.10,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    machine.process(_frame(0.30, 0.25))
    output = machine.process(_frame(0.40, 0.34))

    assert 0.30 <= output["angle"] < 0.40
    assert output["state"] == "HOLDING"
    assert output["prompt"] == "保持住"


def test_seated_knee_raise_relaxed_return_closes_single_rep() -> None:
    targets = _targets(
        rom_target=0.50,
        tut_target=1.0,
        target_range=(0.30, 0.55),
        rom_diff_max=0.30,
        tut_ratio_min=0.55,
    )
    machine = _machine(
        {
            "segment_mode": "raise_lower",
            "start_delta": 0.07,
            "attempt_start_delta": 0.03,
            "start_delta_from_rest": 0.03,
            "return_delta": 0.08,
            "return_delta_from_rest": 0.08,
            "stable_return_seconds": 0.10,
            "min_attempt_delta": 0.02,
            "tut_count_mode": "at_or_above_target",
            "count_by_peak_target": True,
            "raise_prompt": "膝盖再抬高一点",
        },
        targets,
    )
    _prime_baseline(machine, 0.20)

    result = _run_values(
        machine,
        [
            (0.30, 0.25),
            (0.40, 0.50),
            (0.70, 0.50),
            (1.20, 0.27),
            (1.30, 0.27),
            (1.40, 0.27),
        ],
    )

    assert result["primary_error"] in {"OK", "TUT_LOW"}
    assert result["max_signal"] >= 0.30
    assert machine.state.value == "IDLE"


def test_npu_seated_knee_raise_held_leg_ignores_short_low_value_jitter() -> None:
    machine = _npu_seated_knee_raise_machine()

    values = [
        (0.10, 0.24),
        (0.20, 0.28),
        (0.30, 0.32),
        (0.45, 0.38),
        (0.60, 0.44),
        (0.90, 0.44),
        (1.30, 0.44),
        (1.45, 0.25),
        (1.51, 0.25),
        (1.57, 0.25),
        (1.63, 0.25),
        (1.70, 0.44),
    ]

    for relative_time, value in values:
        output = machine.process(_frame(relative_time, value))
        assert output["rep_result"] is None

    assert machine.state == MotionState.RISING


def test_npu_seated_knee_raise_real_return_still_reports_low_rom() -> None:
    machine = _npu_seated_knee_raise_machine()

    result = None
    values = [
        (0.10, 0.24),
        (0.20, 0.28),
        (0.30, 0.32),
        (0.45, 0.38),
        (0.60, 0.44),
        (0.90, 0.44),
        (1.30, 0.44),
        (1.45, 0.34),
        (1.55, 0.28),
        (1.65, 0.23),
        (1.75, 0.22),
        (1.85, 0.22),
        (1.95, 0.22),
        (2.05, 0.22),
        (2.20, 0.22),
    ]
    for relative_time, value in values:
        output = machine.process(_frame(relative_time, value))
        if output["rep_result"] is not None:
            result = output["rep_result"]
            break

    assert result is not None
    assert result["primary_error"] == "ROM_LOW"
    assert result["completion_trigger"] == "metric_return"
    assert machine.state == MotionState.IDLE


def test_npu_sit_to_stand_metric_filter_rejects_single_frame_spike() -> None:
    session = RealtimeTrainingSession()
    session.pose_backend = "rknn"
    session.action_id = "sit_to_stand"
    session.current_realtime_config = {"metric_median_window": 5}
    session.eval_config = yaml.safe_load(
        (PROJECT_ROOT / "evaluation" / "configs" / "npu" / "sit_to_stand.yaml").read_text(encoding="utf-8")
    )

    baseline = session._apply_action_metric(_sit_to_stand_metric_frame(0.5))
    session._apply_action_metric(_sit_to_stand_metric_frame(0.5))
    session._apply_action_metric(_sit_to_stand_metric_frame(0.5))
    session._apply_action_metric(_sit_to_stand_metric_frame(0.5))
    spike = session._apply_action_metric(_sit_to_stand_metric_frame(0.2))
    recovered = session._apply_action_metric(_sit_to_stand_metric_frame(0.5))

    assert baseline["target_angle_smoothed"] == 0.0
    assert spike["target_angle_raw"] == 1.0
    assert spike["target_angle_smoothed"] == 0.0
    assert recovered["target_angle_smoothed"] == 0.0

    sustained = [
        session._apply_action_metric(_sit_to_stand_metric_frame(0.25))["target_angle_smoothed"]
        for _ in range(3)
    ]
    assert sustained[-1] > 0.80


def test_training_overlay_hides_backend_and_debug_metric_row() -> None:
    train_js = (PROJECT_ROOT / "rehab_app" / "server" / "static" / "train.js").read_text(encoding="utf-8")
    main_render = train_js.split('document.getElementById("live-grid").innerHTML = `', 1)[1].split('document.getElementById("timeline")', 1)[0]

    assert 'document.getElementById("train-pill-row")' not in train_js
    assert "actual_backend" not in main_render
    assert 'row("Metric"' not in main_render
    assert 'row("Backend"' not in main_render
    assert 'row("Pose FPS"' not in main_render
    assert 'id="live-grid"' in train_js


def test_feedback_resume_does_not_reapply_action_guard() -> None:
    session = RealtimeTrainingSession()
    session.status = "awaiting_rep_feedback"
    session.pending_feedback_resume = True
    session.action_start_guard_seconds = 2.0
    session.action_guard_until = None

    session._maybe_resume_after_feedback()

    assert session.status == "running"
    assert session.pending_feedback_resume is False
    assert session.action_guard_until is None


def test_action_guard_prompt_stays_in_calibration_not_start_now() -> None:
    session = RealtimeTrainingSession()
    session.status = "running"
    session.start_time = time.time()
    session.machine = object()
    session.action_guard_until = time.time() + 1.0

    session.process_frame({"person_visible": True, "pose_detected": True, "orientation_ok": True})


def test_action_level_correction_tts_cooldown_overrides_global() -> None:
    session = RealtimeTrainingSession()
    session.correction_tts_interval_seconds = 4.0
    session.current_realtime_config = {"correction_tts_interval_seconds": 1.0, "min_rep_seconds": 1.0}
    session.last_correction_tts_at = time.time() - 1.2

    assert session._should_speak_correction("ROM_LOW", {"duration_seconds": 1.2}) is True


def test_global_correction_tts_cooldown_still_applies_without_action_override() -> None:
    session = RealtimeTrainingSession()
    session.correction_tts_interval_seconds = 4.0
    session.current_realtime_config = {"min_rep_seconds": 1.0}
    session.last_correction_tts_at = time.time() - 1.2

    assert session._should_speak_correction("ROM_LOW", {"duration_seconds": 1.2}) is False

def test_offscreen_resume_restarts_static_calibration_guard() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _FakeTTSWorker()
    session.tts_worker = fake_tts
    session.status = "awaiting_return"
    session.offscreen_reentry_guard_seconds = 1.2

    before = time.time()
    session._resume_running_after_offscreen()

    assert session.status == "running"
    assert session.action_guard_until is not None
    assert session.action_guard_until >= before + 1.0
    assert session.last_offscreen_resume_at is not None
    assert session.offscreen_reentry_until is not None
    assert session.offscreen_reentry_until > session.action_guard_until
    assert fake_tts.calls
    assert session.last_prompt == "请回到画面中，我们继续"


def test_reentry_pose_jump_attempt_is_filtered_before_feedback_and_scoring() -> None:
    session = RealtimeTrainingSession()
    session.status = "running"
    session.offscreen_reentry_until = time.time() + 10.0

    session._handle_rep_done(
        {
            "primary_error": "ROM_LOW",
            "countable": False,
            "duration_seconds": 1.7,
            "tut_ratio": 0.0,
            "rom": 0.14,
            "rom_target": 0.67,
            "start_time": 0.0,
            "end_time": 1.7,
        }
    )

    assert session._filtered_reentry_attempts == 1
    assert session.invalid_attempts == []
    assert session.quality_attempt_segments == []
    assert session._feedback_attempt_sequence == 0


def test_rom_low_after_reentry_window_is_still_recorded() -> None:
    session = RealtimeTrainingSession()
    session.status = "running"
    session.offscreen_reentry_until = time.time() - 1.0

    session._handle_rep_done(
        {
            "primary_error": "ROM_LOW",
            "countable": False,
            "duration_seconds": 1.7,
            "tut_ratio": 0.0,
            "rom": 0.14,
            "rom_target": 0.67,
            "start_time": 0.0,
            "end_time": 1.7,
        }
    )

    assert session._filtered_reentry_attempts == 0
    assert len(session.invalid_attempts) == 1
    assert session._feedback_attempt_sequence == 1


def test_reentry_strict_start_requires_extra_confirm_frames() -> None:
    targets = _targets(
        rom_target=0.5,
        tut_target=0.5,
        target_range=(0.45, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.5,
    )
    machine = _machine(
        {
            "segment_mode": "raise_lower",
            "start_delta_from_rest": 0.03,
            "min_attempt_delta": 0.05,
            "motion_start_velocity_min": 0.0,
            "reentry_start_delta_multiplier": 1.0,
            "reentry_min_attempt_delta": 0.05,
            "reentry_start_confirm_frames": 3,
        },
        targets,
    )
    _prime_baseline(machine, 0.0)

    for index, (relative_time, value) in enumerate([(0.3, 0.08), (0.4, 0.12)], start=1):
        frame = _frame(relative_time, value)
        frame["reentry_strict_start"] = True
        output = machine.process(frame)
        assert output["start_ready"] is True
        assert machine.state == MotionState.IDLE, f"started too early on frame {index}"

    frame = _frame(0.5, 0.16)
    frame["reentry_strict_start"] = True
    output = machine.process(frame)
    assert output["start_ready"] is True
    assert machine.state == MotionState.RISING


def _npu_start_pose_frame(*, included_angle: float, shift: float = 0.0, valid: bool = True) -> dict:
    return {
        "person_visible": True,
        "pose_detected": True,
        "person_count": 1,
        "actual_backend": "rknn",
        "pose_backend": "rknn",
        "selected_side": "left",
        "selected_included_angle": included_angle,
        "action_keypoints_valid": valid,
        "rehab_keypoints": {
            "left_shoulder": {"x": 0.45 + shift, "y": 0.20, "visibility": 0.9},
            "right_shoulder": {"x": 0.55 + shift, "y": 0.20, "visibility": 0.9},
            "left_hip": {"x": 0.46 + shift, "y": 0.48, "visibility": 0.9},
            "right_hip": {"x": 0.54 + shift, "y": 0.48, "visibility": 0.9},
            "left_knee": {"x": 0.62 + shift, "y": 0.55, "visibility": 0.9},
            "right_knee": {"x": 0.66 + shift, "y": 0.55, "visibility": 0.9},
            "left_ankle": {"x": 0.62 + shift, "y": 0.78, "visibility": 0.9},
            "right_ankle": {"x": 0.66 + shift, "y": 0.78, "visibility": 0.9},
        },
    }


def _npu_start_pose_session() -> RealtimeTrainingSession:
    session = RealtimeTrainingSession()
    session.pose_backend = "rknn"
    session.action_id = "sit_to_stand"
    session.current_realtime_config = {
        "start_pose_gate_enabled": True,
        "start_pose_confirm_frames": 3,
        "start_pose_visibility_min": 0.18,
        "start_pose_geometry_min": 55.0,
        "start_pose_geometry_max": 140.0,
        "start_pose_max_joint_motion": 0.035,
        "start_pose_prompt": "请先坐回椅子并保持坐稳",
    }
    session.current_targets = _targets(
        rom_target=0.6,
        tut_target=0.5,
        target_range=(0.5, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.5,
    )
    session._reset_active_motion_state()
    session.status = "running"
    session.start_time = time.time()
    return session


def test_npu_start_pose_gate_rejects_standing_and_walking_before_baseline() -> None:
    session = _npu_start_pose_session()

    for index in range(6):
        session.process_frame(_npu_start_pose_frame(included_angle=170.0, shift=index * 0.02))

    assert session.start_pose_ready is False
    assert session.start_pose_reason in {"wrong_start_geometry", "start_pose_moving"}
    assert session.machine is not None
    assert session.machine.state == MotionState.BASELINE
    assert session.metric_baseline_hip_y is None
    assert session.last_machine_output is None
    assert session.invalid_attempts == []
    assert session.last_prompt == "请先坐回椅子并保持坐稳"


def test_npu_start_pose_gate_accepts_stable_seated_pose_then_starts_clean_baseline() -> None:
    session = _npu_start_pose_session()

    for _ in range(3):
        session.process_frame(_npu_start_pose_frame(included_angle=92.0))

    assert session.start_pose_ready is True
    assert session.start_pose_reason == "ready"
    assert session.metric_baseline_hip_y is None
    assert session.machine is not None
    assert session.machine.state == MotionState.BASELINE
    assert session.last_prompt == "请保持静止，正在校准"


def test_npu_start_pose_gate_config_does_not_change_mediapipe_lane() -> None:
    session = RealtimeTrainingSession()
    session.pose_backend = "mediapipe"
    session.current_realtime_config = {"start_pose_gate_enabled": True}

    session._reset_start_pose_gate()

    assert session.start_pose_ready is True
    assert session.start_pose_reason == "disabled"


def test_npu_presence_hysteresis_keeps_visible_person_during_short_keypoint_dropout() -> None:
    session = RealtimeTrainingSession()
    session.pose_backend = "rknn"
    session.npu_presence_v2 = True
    session.npu_presence_enter_frames = 2
    session.npu_presence_grace_frames = 4
    session.status = "running"

    visible = _npu_start_pose_frame(included_angle=92.0)
    session.process_frame(visible)
    session.process_frame(visible)
    assert session.presence_stable is True

    dropout = {
        "person_visible": False,
        "pose_detected": False,
        "person_count": 0,
        "actual_backend": "rknn",
        "pose_backend": "rknn",
        "action_keypoints_valid": False,
        "rehab_keypoints": {},
    }
    for _ in range(4):
        session.process_frame(dropout)

    assert session.presence_raw is False
    assert session.presence_stable is True
    assert session.status == "running"
    assert session.offscreen_seconds == 0.0


def test_npu_return_gate_uses_person_and_core_points_not_action_metric() -> None:
    session = RealtimeTrainingSession()
    session.pose_backend = "rknn"
    session.npu_presence_v2 = True
    session.status = "awaiting_return"
    session.rknn_return_confirm_frames = 2
    frame = _npu_start_pose_frame(included_angle=92.0, valid=False)
    frame["target_angle_smoothed"] = None

    session._process_return_gate(True, False, frame)
    assert session.status == "awaiting_return"
    assert session.return_confirm_count == 1

    session._process_return_gate(True, False, frame)
    assert session.status == "running"
    assert session.reentry_state == "reentry_calibrating"


def test_return_pose_fallback_finishes_rep_when_metric_has_residual_drift() -> None:
    targets = _targets(
        rom_target=0.6,
        tut_target=0.0,
        target_range=(0.5, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.0,
    )
    machine = _machine(
        {
            "segment_mode": "sit_to_stand",
            "return_pose_fallback_enabled": True,
            "return_delta_from_rest": 0.08,
            "return_to_start_delta": 0.08,
            "strict_quality_errors": False,
        },
        targets,
    )
    machine.prime_baseline(0.0)
    machine.state = MotionState.RETURNING
    machine.rep_started_at = 0.2
    machine.rep_state_started_at = 1.0
    machine.rep_start_angle = 0.0
    machine.rep_peak_angle = 0.62
    machine.rep_lowest_after_peak = 0.20
    machine.reached_target = True
    machine.rep_frames = [_frame(0.2, 0.0), _frame(0.7, 0.62), _frame(1.0, 0.62)]
    frame = _frame(1.4, 0.20)
    frame["return_pose_ok"] = True
    frame["return_pose_stable"] = True

    output = machine.process(frame)

    assert output["rep_result"] is not None
    assert output["rep_result"]["completion_trigger"] == "return_pose_fallback"
    assert machine.state == MotionState.IDLE


def test_rising_watchdog_settles_meaningful_stuck_attempt_as_rom_low() -> None:
    targets = _targets(
        rom_target=0.6,
        tut_target=0.5,
        target_range=(0.5, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.5,
    )
    machine = _machine(
        {
            "segment_mode": "sit_to_stand",
            "rising_watchdog_seconds": 8.0,
            "return_pose_fallback_enabled": True,
            "start_delta_from_rest": 0.1,
            "return_delta_from_rest": 0.08,
            "return_to_start_delta": 0.08,
            "min_attempt_delta": 0.22,
        },
        targets,
    )
    machine.prime_baseline(0.0)
    machine.state = MotionState.RISING
    machine.rep_started_at = 0.0
    machine.rep_state_started_at = 0.0
    machine.rep_start_angle = 0.0
    machine.rep_peak_angle = 0.3
    machine.rep_frames = [_frame(0.0, 0.0), _frame(1.0, 0.3)]

    output = machine.process(_frame(8.1, 0.3))

    assert output["rep_result"] is not None
    assert output["rep_result"]["primary_error"] == "ROM_LOW"
    assert output["rep_result"]["completion_trigger"] == "watchdog_rising_settle"
    assert output["recovery_event"] is None
    assert machine.state == MotionState.IDLE


def test_returning_watchdog_uses_stable_start_pose_to_settle_attempt() -> None:
    targets = _targets(
        rom_target=0.6,
        tut_target=0.0,
        target_range=(0.5, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.0,
    )
    machine = _machine(
        {
            "segment_mode": "sit_to_stand",
            "return_pose_fallback_enabled": True,
            "returning_watchdog_seconds": 4.0,
            "strict_quality_errors": False,
        },
        targets,
    )
    machine.prime_baseline(0.0)
    machine.state = MotionState.RETURNING
    machine.rep_started_at = 0.0
    machine.rep_state_started_at = 1.0
    machine.rep_start_angle = 0.0
    machine.rep_peak_angle = 0.62
    machine.rep_lowest_after_peak = 0.2
    machine.reached_target = True
    machine.rep_frames = [_frame(0.0, 0.0), _frame(0.7, 0.62)]
    frame = _frame(5.1, 0.2)
    frame["return_pose_stable"] = True

    output = machine.process(frame)

    assert output["rep_result"] is not None
    assert output["recovery_event"] is None
    assert machine.state == MotionState.IDLE


def test_npu_return_pose_requires_geometry_match_and_stable_time() -> None:
    session = _npu_start_pose_session()
    session.current_realtime_config.update(
        {
            "return_pose_fallback_enabled": True,
            "return_pose_confirm_frames": 3,
            "return_pose_stable_seconds": 0.35,
            "return_pose_geometry_tolerance": 20.0,
        }
    )
    session.start_pose_ready = True
    session.start_pose_anchor_geometry = 92.0
    assert session.machine is not None
    session.machine.prime_baseline(0.0)
    session.machine.state = MotionState.RETURNING
    session.machine.rep_state_started_at = 0.0

    base = time.time()
    for offset in (0.0, 0.15, 0.30):
        frame = _npu_start_pose_frame(included_angle=94.0)
        annotated = session._annotate_npu_return_pose(frame, base + offset)
        assert annotated["return_pose_stable"] is False
    annotated = session._annotate_npu_return_pose(
        _npu_start_pose_frame(included_angle=94.0),
        base + 0.45,
    )
    assert annotated["return_pose_stable"] is True
    assert session.return_pose_confirm_count == 4


def test_npu_post_rep_rebaseline_primes_next_cycle_without_full_baseline_wait() -> None:
    session = _npu_start_pose_session()
    session.eval_config = {
        "primary_metric": "hip_rise_height_ratio",
        "metric_unit": "body_ratio",
    }
    session.current_realtime_config.update(
        {
            "rebaseline_each_rep": True,
            "post_rep_start_pose_confirm_frames": 3,
        }
    )
    session.start_pose_ready = True
    session._arm_npu_rebaseline("post_rep")

    for _ in range(3):
        session.process_frame(_npu_start_pose_frame(included_angle=92.0))

    assert session.rebaseline_pending is False
    assert session.rebaseline_state == "ready"
    assert session.rebaseline_cycle_count == 1
    assert session.start_pose_ready is True
    assert session.machine is not None
    assert session.machine.state == MotionState.IDLE
    assert session.machine.baseline_angle == 0.0
    assert session.last_prompt == "可以开始下一次动作"


def test_npu_actions_reuse_cpu_logic_except_seated_raise_return_guard() -> None:
    base = yaml.safe_load((PROJECT_ROOT / "training" / "configs" / "training_defaults_npu.yaml").read_text(encoding="utf-8"))
    cpu_base = yaml.safe_load((PROJECT_ROOT / "training" / "configs" / "training_defaults.yaml").read_text(encoding="utf-8-sig"))
    assert {key: value for key, value in base.items() if key not in {"visibility_threshold", "target_reps"}} == {
        key: value for key, value in cpu_base.items() if key not in {"visibility_threshold", "target_reps"}
    }
    assert base["visibility_threshold"] == 0.18
    assert "rebaseline_each_rep" not in base
    plan = yaml.safe_load((PROJECT_ROOT / "training" / "configs" / "rehab_demo_plan_npu.yaml").read_text(encoding="utf-8"))
    cpu_plan = yaml.safe_load((PROJECT_ROOT / "training" / "configs" / "rehab_demo_plan.yaml").read_text(encoding="utf-8-sig"))
    assert plan["training_fixed_audio_only"] is True
    assert plan["return_confirm_frames"] == cpu_plan["return_confirm_frames"]
    assert plan["rknn_return_confirm_frames"] == cpu_plan["rknn_return_confirm_frames"]
    for action_id in ("sit_to_stand", "standing_hamstring_curl", "seated_knee_raise"):
        npu_config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / "npu" / f"{action_id}.yaml").read_text(encoding="utf-8"))
        cpu_config = yaml.safe_load((PROJECT_ROOT / "evaluation" / "configs" / f"{action_id}.yaml").read_text(encoding="utf-8-sig"))
        assert npu_config["thresholds"] == cpu_config["thresholds"]
        if action_id == "seated_knee_raise":
            expected_realtime = {
                **cpu_config["realtime"],
                "return_delta_from_rest": 0.05,
                "return_to_start_delta": 0.0,
                "stable_return_seconds": 0.30,
                "require_return_reversal": True,
                "return_reversal_confirm_frames": 4,
                "finish_confirm_frames": 4,
            }
            assert npu_config["realtime"] == expected_realtime
            assert cpu_config["realtime"]["return_delta_from_rest"] == 0.08
            assert cpu_config["realtime"]["stable_return_seconds"] == 0.10
            assert "require_return_reversal" not in cpu_config["realtime"]
        elif action_id == "sit_to_stand":
            expected_realtime = {
                **cpu_config["realtime"],
                "start_delta_from_rest": 0.25,
                "start_delta": 0.25,
                "tut_range_padding": 0.05,
                "metric_median_window": 5,
                "start_pose_gate_enabled": True,
                "start_pose_confirm_frames": 4,
                "start_pose_visibility_min": 0.18,
                "start_pose_geometry_min": 55.0,
                "start_pose_geometry_max": 140.0,
                "start_pose_max_joint_motion": 0.035,
                "start_pose_prompt": "请先坐回椅子并保持坐稳",
                "rebaseline_each_rep": True,
                "post_rep_start_pose_confirm_frames": 4,
                "post_rep_ready_prompt": "可以开始下一次坐站",
            }
            assert npu_config["realtime"] == expected_realtime
        else:
            assert npu_config["realtime"] == cpu_config["realtime"]
        assert npu_config["template_validation"]["min_valid_frames"] == 20


def test_npu_active_rep_tolerates_short_visibility_dropout_before_reset() -> None:
    targets = _targets(
        rom_target=0.6,
        tut_target=0.5,
        target_range=(0.5, 0.8),
        rom_diff_max=0.1,
        tut_ratio_min=0.5,
    )
    machine = _machine(
        {
            "visibility_threshold": 0.18,
            "interrupt_reset_seconds": 1.5,
            "max_lost_visibility_frames": 12,
        },
        targets,
    )
    machine.prime_baseline(0.0)
    machine.state = MotionState.RETURNING
    machine.rep_started_at = 0.0
    machine.rep_state_started_at = 0.0
    invalid = {
        "frame_index": 1,
        "target_angle_smoothed": None,
        "visibility_min": 0.0,
        "person_visible": True,
    }

    for relative_time in (0.2, 0.8, 1.4):
        output = machine.process({**invalid, "relative_time": relative_time})
        assert output["recovery_event"] is None
        assert machine.state == MotionState.RETURNING

    output = machine.process({**invalid, "relative_time": 1.8})
    assert output["recovery_event"]["reason"] == "visibility_interrupted"
    assert machine.state == MotionState.IDLE


def test_npu_watchdog_recovery_prompt_is_screen_only_when_tts_disabled() -> None:
    session = _npu_start_pose_session()
    fake_tts = _FakeTTSWorker()
    session.tts_worker = fake_tts
    session.current_realtime_config.update(
        {
            "rebaseline_each_rep": True,
            "watchdog_recovery_tts": False,
            "watchdog_recovery_prompt": "请先回到起始姿势站稳",
        }
    )

    session._handle_machine_recovery_event(
        {"reason": "visibility_interrupted", "state": "RETURNING", "state_age_seconds": 1.6}
    )

    assert session.last_prompt == "请先回到起始姿势站稳"
    assert session.last_watchdog_reason == "visibility_interrupted"
    assert fake_tts.calls == []


def test_return_pose_fallback_closes_all_npu_actions_from_holding() -> None:
    cases = (
        ("sit_to_stand", "sit_to_stand", 0.60, (0.50, 0.80), 0.62, 0.02, 0.22),
        ("standing_hamstring_curl", "angle_curl", 60.0, (50.0, 75.0), 62.0, 2.0, 8.0),
        ("seated_knee_raise", "raise_lower", 0.50, (0.40, 0.70), 0.52, 0.01, 0.10),
    )
    for action_id, segment_mode, rom_target, target_range, peak, returned, min_attempt_delta in cases:
        targets = _targets(
            rom_target=rom_target,
            tut_target=0.2,
            target_range=target_range,
            rom_diff_max=rom_target * 0.12,
            tut_ratio_min=0.5,
            min_rom_ratio=0.82,
            min_tut_seconds=0.10,
        )
        machine = _machine(
            {
                "segment_mode": segment_mode,
                "return_pose_fallback_enabled": True,
                "return_transition_accept_stable_below_target": True,
                "min_attempt_delta": min_attempt_delta,
                "min_rep_seconds": 0.5,
            },
            targets,
        )
        machine.prime_baseline(0.0)
        machine.state = MotionState.HOLDING
        machine.rep_started_at = 0.2
        machine.rep_state_started_at = 0.8
        machine.rep_start_angle = 0.0
        machine.rep_peak_angle = peak
        machine.rep_lowest_after_peak = peak
        machine.reached_target = True
        machine.rep_frames = [_frame(0.2, 0.0), _frame(0.8, peak), _frame(1.2, peak)]
        frame = _frame(1.5, returned)
        frame["return_pose_stable"] = True

        output = machine.process(frame)

        assert output["rep_result"] is not None, action_id
        assert output["rep_result"]["completion_trigger"] == "return_pose_fallback", action_id
        assert output["rep_result"]["primary_error"] == "OK", action_id
        assert machine.state == MotionState.IDLE, action_id


def test_npu_template_relative_rom_gate_rejects_low_amplitude_even_when_absolute_diff_passes() -> None:
    targets = _targets(
        rom_target=0.60,
        tut_target=0.0,
        target_range=(0.45, 0.80),
        rom_diff_max=0.15,
        tut_ratio_min=0.0,
        min_rom_ratio=0.90,
    )
    machine = _machine({}, targets)
    result = machine._evaluate_rep(
        [_frame(0.0, 0.0), _frame(0.8, 0.50), _frame(1.2, 0.50), _frame(1.8, 0.02)]
    )

    assert result["rom_diff"] <= targets.rom_diff_max
    assert result["rom_ratio"] < targets.min_rom_ratio
    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False


def test_npu_minimum_hold_floor_rejects_touch_and_go_attempt() -> None:
    targets = _targets(
        rom_target=0.60,
        tut_target=0.1,
        target_range=(0.50, 0.80),
        rom_diff_max=0.05,
        tut_ratio_min=0.5,
        min_rom_ratio=0.90,
        min_tut_seconds=0.60,
    )
    machine = _machine({}, targets)
    result = machine._evaluate_rep(
        [_frame(0.0, 0.0), _frame(0.8, 0.62), _frame(0.95, 0.62), _frame(1.2, 0.0)]
    )

    assert result["rom_ok"] is True
    assert result["tut_seconds"] < targets.min_tut_seconds
    assert result["primary_error"] == "TUT_LOW"
    assert result["countable"] is False


def test_meaningful_visibility_interruption_is_recorded_instead_of_silently_discarded() -> None:
    targets = _targets(
        rom_target=0.60,
        tut_target=0.5,
        target_range=(0.50, 0.80),
        rom_diff_max=0.1,
        tut_ratio_min=0.5,
    )
    machine = _machine(
        {"interrupt_reset_seconds": 0.5, "max_lost_visibility_frames": 30, "min_attempt_delta": 0.22},
        targets,
    )
    machine.prime_baseline(0.0)
    machine.state = MotionState.RETURNING
    machine.rep_started_at = 0.0
    machine.rep_state_started_at = 0.8
    machine.rep_start_angle = 0.0
    machine.rep_peak_angle = 0.62
    machine.rep_lowest_after_peak = 0.30
    machine.reached_target = True
    machine.rep_frames = [_frame(0.0, 0.0), _frame(0.7, 0.62)]
    invalid = {
        "frame_index": 10,
        "target_angle_smoothed": None,
        "visibility_min": 0.0,
        "person_visible": True,
    }

    first = machine.process({**invalid, "relative_time": 1.0})
    output = machine.process({**invalid, "relative_time": 1.6})

    assert first["rep_result"] is None
    assert output["rep_result"] is not None
    assert output["rep_result"]["primary_error"] == "VISIBILITY_LOW"
    assert output["rep_result"]["completion_trigger"] == "visibility_interrupted"
    assert output["recovery_event"] is None
