from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from evaluation.core.template_health import validate_template_payload
from training.training_session import RealtimeTrainingSession


def _cycle(rom: float) -> list[float]:
    up = [rom * index / 14 for index in range(15)]
    return up + list(reversed(up))


def _validate(action_id: str, rom: float, geometry: float) -> dict:
    values = _cycle(rom)
    frames = [
        {"relative_time": index * 0.1, "selected_included_angle": geometry}
        for index in range(len(values))
    ]
    config = {"action_id": action_id, "primary_metric": "test_metric"}
    with patch(
        "evaluation.core.template_health.extract_metric_sequence",
        return_value={"values": values, "frame_times": [frame["relative_time"] for frame in frames]},
    ):
        return validate_template_payload(
            {"action_id": action_id, "template_frames": frames},
            config,
            pose_backend="rknn",
        )


def test_current_bad_sit_to_stand_template_is_rejected() -> None:
    health = _validate("sit_to_stand", 0.135, 90.0)
    assert health["ok"] is False
    assert health["rom"] == 0.135
    assert "peak" in health["missing_stages"]


def test_current_bad_hamstring_template_is_rejected() -> None:
    health = _validate("standing_hamstring_curl", 26.8, 170.0)
    assert health["ok"] is False
    assert health["rom"] == 26.8
    assert health["limits"]["min_rom"] == 45.0


def test_current_seated_knee_raise_template_is_accepted() -> None:
    health = _validate("seated_knee_raise", 0.615, 90.0)
    assert health["ok"] is True
    assert health["rom"] == 0.615


def test_training_preflight_blocks_invalid_rknn_active_template(tmp_path: Path, monkeypatch) -> None:
    realtime_config = tmp_path / "realtime.yaml"
    realtime_config.write_text("target_reps: 3\n", encoding="utf-8")
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("action_id: sit_to_stand\nprimary_metric: hip_rise_height_ratio\n", encoding="utf-8")
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"action_id": "sit_to_stand", "template_frames": []}), encoding="utf-8")

    session = RealtimeTrainingSession(realtime_config)
    monkeypatch.setattr(
        session,
        "_load_demo_plan",
        lambda: {
            "actions": [
                {
                    "action_id": "sit_to_stand",
                    "realtime_config_file": str(realtime_config),
                    "config_file": str(eval_config),
                }
            ]
        },
    )
    monkeypatch.setattr(
        session,
        "_get_active_template",
        lambda action_id: {
            "action_id": action_id,
            "template_file": str(template),
            "config_file": str(eval_config),
        },
    )
    monkeypatch.setattr(
        "training.training_session.validate_template_file",
        lambda *args, **kwargs: {
            "ok": False,
            "required": True,
            "reason": "rom_too_low",
            "message": "动作幅度不足：当前 0.135，至少需要 0.350",
            "rom": 0.135,
        },
    )

    result = session.start(action_id="sit_to_stand", pose_backend="rknn")

    assert result["ok"] is False
    assert result["template_health"]["rom"] == 0.135
