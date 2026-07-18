from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from action_scoring.features import build_input_tensor, mirror_rep_payload
from action_scoring.completion_calibrator import (
    calibrated_completion_details,
    calibrated_completion_percent,
    report_error_cap,
    should_filter_reentry_attempt,
)


def _rep_payload() -> dict:
    return {
        "rep_index": 1,
        "skeleton_sequence": [
            {
                "frame_index": 0,
                "relative_time": 0.0,
                "keypoints": {
                    "nose": {"x": 0.5, "y": 0.1, "visibility": 0.9},
                    "left_shoulder": {"x": 0.4, "y": 0.2, "visibility": 0.9},
                    "right_shoulder": {"x": 0.6, "y": 0.2, "visibility": 0.9},
                    "left_hip": {"x": 0.45, "y": 0.5, "visibility": 0.95},
                    "right_hip": {"x": 0.55, "y": 0.5, "visibility": 0.95},
                    "left_knee": {"x": 0.45, "y": 0.7, "visibility": 0.95},
                    "right_knee": {"x": 0.55, "y": 0.7, "visibility": 0.95},
                    "left_ankle": {"x": 0.45, "y": 0.9, "visibility": 0.95},
                    "right_ankle": {"x": 0.55, "y": 0.9, "visibility": 0.95},
                },
            },
            {
                "frame_index": 1,
                "relative_time": 0.1,
                "keypoints": {
                    "nose": {"x": 0.5, "y": 0.11, "visibility": 0.9},
                    "left_shoulder": {"x": 0.4, "y": 0.21, "visibility": 0.9},
                    "right_shoulder": {"x": 0.6, "y": 0.21, "visibility": 0.9},
                    "left_hip": {"x": 0.45, "y": 0.51, "visibility": 0.95},
                    "right_hip": {"x": 0.55, "y": 0.51, "visibility": 0.95},
                    "left_knee": {"x": 0.45, "y": 0.69, "visibility": 0.95},
                    "right_knee": {"x": 0.55, "y": 0.69, "visibility": 0.95},
                    "left_ankle": {"x": 0.45, "y": 0.89, "visibility": 0.95},
                    "right_ankle": {"x": 0.55, "y": 0.89, "visibility": 0.95},
                },
            },
        ],
    }


def test_build_input_tensor_resamples_to_fixed_shape() -> None:
    tensor, meta = build_input_tensor(_rep_payload(), target_frames=30)
    assert tensor is not None
    assert tensor.shape == (1, 51, 30)
    assert meta.valid_frames == 2
    assert "left_hip" in meta.used_keypoint_names


def test_mirror_rep_payload_swaps_left_and_right() -> None:
    mirrored = mirror_rep_payload(_rep_payload())
    first = mirrored["skeleton_sequence"][0]["keypoints"]
    assert abs(first["right_shoulder"]["x"] - 0.6) < 1e-6
    assert abs(first["left_shoulder"]["x"] - 0.4) < 1e-6


def test_missing_sequence_returns_none() -> None:
    tensor, meta = build_input_tensor({}, target_frames=30)
    assert tensor is None
    assert meta.valid_frames == 0

def test_completion_calibration_keeps_ok_high_and_caps_rom_low_by_severity() -> None:
    ok = calibrated_completion_percent("standing_hamstring_curl", {"primary_error": "OK"}, 83.0)
    severe = calibrated_completion_percent(
        "standing_hamstring_curl",
        {"primary_error": "ROM_LOW", "rom": 55.0, "rom_target": 105.0},
        21.0,
    )
    mild = calibrated_completion_percent(
        "sit_to_stand",
        {"primary_error": "ROM_LOW", "rom": 0.69, "rom_target": 0.77},
        74.0,
    )

    assert ok == 88.0
    assert 20.0 <= severe <= 45.0
    assert 60.0 <= mild <= 82.0


def test_reentry_filter_detects_short_zero_tut_pose_jump() -> None:
    assert should_filter_reentry_attempt(
        {
            "primary_error": "ROM_LOW",
            "countable": False,
            "duration_seconds": 1.7,
            "tut_ratio": 0.0,
            "rom": 0.14,
            "rom_target": 0.67,
        }
    )


def test_npu_ok_score_is_rule_first_and_never_exceeds_96() -> None:
    details = calibrated_completion_details(
        "sit_to_stand",
        {
            "pose_backend": "rknn",
            "training_logic_version": "npu_training_v4",
            "primary_error": "OK",
            "rom": 0.80,
            "required_rom": 0.63,
            "tut_seconds": 0.8,
            "tut_required_seconds": 0.6,
            "speed_ratio": 1.0,
        },
        99.0,
    )

    assert details["calibration_mode"] == "npu_rule_first_v4"
    assert details["rule_score"] == 96.0
    assert details["completion_percent"] == 96.0


def test_npu_error_scores_ignore_untrusted_high_model_score_and_obey_caps() -> None:
    rom_low = calibrated_completion_details(
        "standing_hamstring_curl",
        {
            "pose_backend": "rknn",
            "primary_error": "ROM_LOW",
            "rom": 30.0,
            "required_rom": 45.0,
        },
        99.0,
    )
    visibility = calibrated_completion_details(
        "seated_knee_raise",
        {"pose_backend": "rknn", "primary_error": "VISIBILITY_LOW"},
        99.0,
    )

    assert rom_low["completion_percent"] <= 70.0
    assert visibility["completion_percent"] <= 50.0
    assert report_error_cap([{"primary_error": "TUT_LOW"}, {"primary_error": "ROM_LOW"}]) == 70.0
    assert not should_filter_reentry_attempt(
        {
            "primary_error": "ROM_LOW",
            "countable": False,
            "duration_seconds": 3.2,
            "tut_ratio": 0.0,
            "rom": 0.14,
            "rom_target": 0.67,
        }
    )
