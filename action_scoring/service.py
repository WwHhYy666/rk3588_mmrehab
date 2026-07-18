from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from action_scoring.features import build_input_tensor
from action_scoring.infer_onnx import OnnxQualityInfer
from action_scoring.labels import quality_grade
from action_scoring.registry import CPU_QUALITY_ACTIONS, ActionModelSpec, get_action_model_spec


@dataclass
class RuntimeState:
    backend: str | None = None
    model_path: str | None = None
    last_error: str | None = None
    last_score_time_ms: float | None = None


_RUNTIME_STATE: dict[str, RuntimeState] = {}
_BACKEND_CACHE: dict[tuple[str, str], Any] = {}


def get_quality_model_status(action_id: str) -> dict[str, Any]:
    spec = get_action_model_spec(action_id)
    state = _RUNTIME_STATE.setdefault(spec.action_id, RuntimeState())
    backend = _detect_backend(spec)
    if state.backend is None:
        state.backend = backend
        state.model_path = _model_path_for_backend(spec, backend)
    available = backend is not None
    return {
        "available": available,
        "backend": backend,
        "model_path": _model_path_for_backend(spec, backend),
        "action_id": spec.action_id,
        "input_frames": spec.input_frames,
        "score_range": [0, 100],
        "last_score_time_ms": round(state.last_score_time_ms, 2) if state.last_score_time_ms is not None else None,
        "last_error": state.last_error,
    }


def score_rep(action_id: str, rep_payload: dict[str, Any]) -> dict[str, Any] | None:
    spec = get_action_model_spec(action_id)
    state = _RUNTIME_STATE.setdefault(spec.action_id, RuntimeState())
    if spec.action_id not in CPU_QUALITY_ACTIONS:
        state.last_error = "unsupported_quality_action"
        return None
    tensor, meta = build_input_tensor(rep_payload, target_frames=spec.input_frames)
    if tensor is None or meta.valid_frames <= 0:
        state.last_error = "missing_skeleton_sequence"
        return None
    started = time.perf_counter()
    for backend_name in ("onnx_cpu",):
        model_path = _model_path_for_backend(spec, backend_name)
        if not model_path:
            continue
        try:
            backend = _load_backend(spec, backend_name)
            value = float(backend.infer(tensor))
        except Exception as exc:
            state.last_error = f"{backend_name}_infer_failed: {exc}"
            continue
        score = max(0.0, min(100.0, value * 100.0))
        state.backend = backend_name
        state.model_path = model_path
        state.last_error = None
        state.last_score_time_ms = (time.perf_counter() - started) * 1000.0
        return {
            "score": round(score, 2),
            "grade": quality_grade(score),
            "backend": backend_name,
            "model_path": model_path,
            "action_id": spec.action_id,
            "input_frames": spec.input_frames,
            "used_keypoint_names": list(meta.used_keypoint_names),
            "valid_frames": meta.valid_frames,
        }
    return None


def _load_backend(spec: ActionModelSpec, backend_name: str):
    key = (spec.action_id, backend_name)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    if backend_name == "onnx_cpu":
        backend = OnnxQualityInfer(spec.onnx_path)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unsupported backend: {backend_name}")
    backend.load()
    _BACKEND_CACHE[key] = backend
    return backend


def _detect_backend(spec: ActionModelSpec) -> str | None:
    if spec.action_id not in CPU_QUALITY_ACTIONS:
        return None
    if spec.onnx_path.exists():
        return "onnx_cpu"
    return None


def _model_path_for_backend(spec: ActionModelSpec, backend_name: str | None) -> str | None:
    if backend_name == "onnx_cpu" and spec.onnx_path.exists():
        return str(spec.onnx_path)
    return None
