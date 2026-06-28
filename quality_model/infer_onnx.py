from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OnnxQualityInfer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = ""

    def load(self) -> None:
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("onnxruntime is not available.") from exc
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, inputs: np.ndarray) -> float:
        self.load()
        outputs = self.session.run(None, {self.input_name: inputs.astype(np.float32)})
        if not outputs:
            raise RuntimeError("ONNX inference returned no outputs.")
        value = outputs[0]
        return float(np.asarray(value).reshape(-1)[0])
