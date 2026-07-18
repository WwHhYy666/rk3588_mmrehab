from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityModelConfig:
    input_channels: int = 51
    input_frames: int = 30
    conv1_channels: int = 32
    conv2_channels: int = 64
    hidden_dim: int = 64


def build_torch_model(config: QualityModelConfig | None = None):
    """Build the tiny Conv1d model used for ONNX/RKNN export."""

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("PyTorch is required to build the quality model.") from exc

    cfg = config or QualityModelConfig()

    class TinyQualityNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(cfg.input_channels, cfg.conv1_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(cfg.conv1_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(cfg.conv1_channels, cfg.conv2_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            pooled_frames = max(1, cfg.input_frames // 2)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cfg.conv2_channels * pooled_frames, cfg.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return TinyQualityNet()
