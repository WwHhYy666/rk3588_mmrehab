from __future__ import annotations

import argparse
from pathlib import Path

from quality_model.registry import get_action_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an ONNX quality model to RKNN.")
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--target-platform", default="rk3588")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = get_action_model_spec(args.action_id)
    try:
        from rknn.api import RKNN
    except Exception as exc:  # pragma: no cover - depends on conversion host
        raise SystemExit(f"rknn-toolkit2 is required for RKNN export: {exc}") from exc
    rknn = RKNN(verbose=False)
    ret = rknn.config(target_platform=args.target_platform)
    if ret != 0:
        raise SystemExit(f"RKNN config failed: {ret}")
    ret = rknn.load_onnx(model=str(spec.onnx_path))
    if ret != 0:
        raise SystemExit(f"RKNN load_onnx failed: {ret}")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise SystemExit(f"RKNN build failed: {ret}")
    spec.model_dir.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(spec.rknn_path))
    if ret != 0:
        raise SystemExit(f"RKNN export failed: {ret}")
    print(spec.rknn_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
