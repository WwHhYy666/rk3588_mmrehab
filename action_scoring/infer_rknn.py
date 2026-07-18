from __future__ import annotations

from pathlib import Path

import numpy as np


class RknnQualityInfer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.rknn = None

    def load(self) -> None:
        if self.rknn is not None:
            return
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:  # pragma: no cover - depends on RK3588 runtime
            raise RuntimeError("RKNNLite is not available.") from exc
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite.load_rknn failed with code {ret}: {self.model_path}")
        runtime_kwargs = {}
        if hasattr(RKNNLite, "NPU_CORE_2"):
            runtime_kwargs["core_mask"] = RKNNLite.NPU_CORE_2
        ret = rknn.init_runtime(**runtime_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNNLite.init_runtime failed with code {ret}: {self.model_path}")
        self.rknn = rknn

    def infer(self, inputs: np.ndarray) -> float:
        self.load()
        outputs = self.rknn.inference(inputs=[inputs.astype(np.float32)])
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs.")
        value = outputs[0]
        return float(np.asarray(value).reshape(-1)[0])
