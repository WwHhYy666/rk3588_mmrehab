from __future__ import annotations

import argparse

from quality_model.model import QualityModelConfig, build_torch_model
from quality_model.registry import get_action_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained quality model to ONNX.")
    parser.add_argument("--action-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_action_model_spec(args.action_id)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"PyTorch is required for ONNX export: {exc}") from exc
    checkpoint = torch.load(spec.torch_path, map_location="cpu")
    model = build_torch_model(QualityModelConfig(input_channels=spec.input_channels, input_frames=spec.input_frames))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, spec.input_channels, spec.input_frames, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        spec.onnx_path,
        input_names=["input"],
        output_names=["score"],
        opset_version=12,
        dynamic_axes=None,
    )
    print(spec.onnx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
