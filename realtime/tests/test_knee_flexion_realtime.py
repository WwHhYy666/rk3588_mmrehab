from __future__ import annotations

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.feedback_runtime import load_rules, rep_feedback
from realtime.knee_flexion import KneeFlexionRealtimeMachine, KneeFlexionTargets


def _targets(
    *,
    rom_target: float,
    tut_target: float,
    target_range: tuple[float, float],
    rom_diff_max: float,
    tut_ratio_min: float,
) -> KneeFlexionTargets:
    return KneeFlexionTargets(
        rom_target=rom_target,
        tut_target=tut_target,
        target_range=target_range,
        template_peak_speed=0.0,
        rom_diff_max=rom_diff_max,
        tut_ratio_min=tut_ratio_min,
        speed_ratio_max=1.5,
    )


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


def _frame(relative_time: float, value: float) -> dict[str, float | bool]:
    return {
        "frame_index": int(relative_time * 100),
        "relative_time": relative_time,
        "target_angle_smoothed": value,
        "visibility_min": 1.0,
        "person_visible": True,
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
    rules = load_rules(PROJECT_ROOT / "feedback" / "rules" / "seated_knee_extension_feedback.yaml")
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
    rules = load_rules(PROJECT_ROOT / "feedback" / "rules" / "seated_knee_extension_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_extension")
    assert feedback["tts_text"] == "腿再伸直一点"


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
    rules = load_rules(PROJECT_ROOT / "feedback" / "rules" / "seated_knee_extension_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_extension")
    assert feedback["tts_text"].startswith("再坚持 ")


def test_raise_peak_target_no_longer_counts_when_rom_diff_is_large() -> None:
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

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False
    assert result["peak_ok"] is True
    assert result["rom_ok"] is False
    assert result["rom_blocks_count"] is True
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


def test_raise_peak_target_short_hold_reports_rom_low_first() -> None:
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

    assert result["primary_error"] == "ROM_LOW"
    assert result["countable"] is False
    assert "TUT_LOW" not in result["all_errors"]


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
    rules = load_rules(PROJECT_ROOT / "feedback" / "rules" / "seated_knee_raise_feedback.yaml")
    feedback = rep_feedback({**result, "attempt_index": 1}, rules, action_id="seated_knee_raise")
    assert feedback["tts_text"].startswith("再坚持 ")

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
def test_train_page_renders_care_dialog_controls() -> None:
    script = (PROJECT_ROOT / "prescription" / "banzi" / "static" / "train.js").read_text(encoding="utf-8")

    assert "id=\"care-dialog\"" in script
    assert "function renderCareDialog" in script
    assert "awaiting_care_response" in script
    assert "/api/realtime/care_response" in script
    assert "submitCareResponse(true)" in script
    assert "submitCareResponse(false)" in script
def test_standing_hamstring_curl_thresholds_allow_small_wrong_attempts() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "evaluate" / "configs" / "standing_hamstring_curl.yaml").read_text(encoding="utf-8"))
    realtime = config["realtime"]

    assert config["thresholds"]["rom_diff_max"] == 15.0
    assert config["thresholds"]["dtw_normalized_max"] == 0.25
    assert realtime["start_delta"] == 12.0
    assert realtime["return_delta"] == 5.0
    assert realtime["attempt_start_delta"] == 4.0
    assert realtime["min_attempt_delta"] == 3.0
    assert realtime["tut_range_padding"] == 5.0
    assert realtime["tut_ratio_min"] == 0.65
    assert "tut_count_mode" not in realtime
    assert "count_by_peak_target" not in realtime


if __name__ == "__main__":
    test_extension_short_rom_speaks_rom_low()
    test_extension_low_amplitude_attempt_speaks_rom_low()
    test_extension_tiny_motion_stays_silent()
    test_extension_short_hold_surfaces_tut_low()
    test_raise_peak_target_no_longer_counts_when_rom_diff_is_large()
    test_raise_above_target_high_counts_tut_without_lower_prompt()
    test_raise_peak_target_short_hold_reports_rom_low_first()
    test_raise_rom_ok_short_hold_reports_tut_low()
    test_visibility_loss_resets_without_error_result()
    test_idle_prompt_does_not_show_prepare_next_rep()
    test_hamstring_low_amplitude_attempt_reports_rom_low()
    test_train_page_renders_care_dialog_controls()
    test_standing_hamstring_curl_thresholds_allow_small_wrong_attempts()
    print("knee_flexion realtime tests passed")



