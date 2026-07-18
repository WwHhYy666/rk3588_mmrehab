from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_SCORING_DIR = PROJECT_ROOT / "action_scoring"
CONFIGS_DIR = ACTION_SCORING_DIR / "configs"
MODELS_DIR = PROJECT_ROOT / "models" / "action_quality"

SUPPORTED_ACTIONS = (
    "knee_flexion",
    "seated_knee_extension",
    "seated_knee_raise",
    "standing_hamstring_curl",
    "sit_to_stand",
)

CPU_QUALITY_ACTIONS = (
    "sit_to_stand",
    "standing_hamstring_curl",
    "seated_knee_raise",
)


@dataclass(frozen=True)
class ActionModelSpec:
    action_id: str
    config_path: Path
    model_dir: Path
    torch_path: Path
    onnx_path: Path
    rknn_path: Path
    input_frames: int = 30
    input_channels: int = 51


def get_action_model_spec(action_id: str) -> ActionModelSpec:
    normalized = str(action_id or "").strip()
    if not normalized:
        normalized = "knee_flexion"
    return ActionModelSpec(
        action_id=normalized,
        config_path=CONFIGS_DIR / f"{normalized}.yaml",
        model_dir=MODELS_DIR / normalized,
        torch_path=MODELS_DIR / normalized / "best.pt",
        onnx_path=MODELS_DIR / normalized / "model.onnx",
        rknn_path=MODELS_DIR / normalized / "model.rknn",
    )


def list_action_model_specs() -> list[ActionModelSpec]:
    return [get_action_model_spec(action_id) for action_id in SUPPORTED_ACTIONS]
