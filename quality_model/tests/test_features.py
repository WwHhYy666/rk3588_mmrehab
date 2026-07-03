from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality_model.features import build_input_tensor, mirror_rep_payload
from quality_model.completion_calibrator import calibrated_completion_percent, should_filter_reentry_attempt


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

