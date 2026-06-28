from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from quality_model.features import build_input_tensor
from quality_model.labels import pseudo_label_from_report_rep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "prescription" / "docs" / "results"
REPORTS_DIR = PROJECT_ROOT / "evaluate" / "reports"


def build_dataset(action_id: str, config_path: str | Path) -> list[dict[str, Any]]:
    config = _load_yaml(config_path)
    dataset: list[dict[str, Any]] = []
    for payload_path in sorted(RESULTS_DIR.glob("*.json")):
        payload = _load_json(payload_path)
        runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
        if str(payload.get("action_id") or runtime_meta.get("action_id") or "").strip() != action_id:
            continue
        record_role = str(runtime_meta.get("record_role") or "")
        rep_segments = runtime_meta.get("rep_segments") if isinstance(runtime_meta.get("rep_segments"), list) else []
        rep_results = runtime_meta.get("rep_results") if isinstance(runtime_meta.get("rep_results"), list) else []
        if rep_segments:
            for index, rep_segment in enumerate(rep_segments, start=1):
                if not isinstance(rep_segment, dict):
                    continue
                rep_result = _find_rep_result(rep_results, rep_segment.get("rep_index") or index)
                tensor, meta = build_input_tensor(rep_segment)
                if tensor is None:
                    continue
                label = 1.0 if record_role == "doctor_template" else pseudo_label_from_report_rep(rep_result or rep_segment, None, config)
                if label is None:
                    continue
                dataset.append(
                    {
                        "sample_id": f"{payload_path.stem}:rep{index}",
                        "action_id": action_id,
                        "source_path": str(payload_path),
                        "record_role": record_role,
                        "input": tensor,
                        "label": float(label),
                        "valid_frames": meta.valid_frames,
                    }
                )
            continue
        fallback_segment = _fallback_full_sequence_segment(payload)
        if not fallback_segment:
            continue
        tensor, meta = build_input_tensor(fallback_segment)
        if tensor is None:
            continue
        label = 1.0 if record_role == "doctor_template" else None
        if label is None:
            continue
        dataset.append(
            {
                "sample_id": f"{payload_path.stem}:full",
                "action_id": action_id,
                "source_path": str(payload_path),
                "record_role": record_role,
                "input": tensor,
                "label": float(label),
                "valid_frames": meta.valid_frames,
            }
        )
    return dataset


def _fallback_full_sequence_segment(payload: dict[str, Any]) -> dict[str, Any] | None:
    frames = payload.get("template_frames")
    if not isinstance(frames, list) or not frames:
        return None
    return {
        "rep_index": 1,
        "start_time": frames[0].get("relative_time"),
        "end_time": frames[-1].get("relative_time"),
        "start_frame_index": frames[0].get("frame_index"),
        "end_frame_index": frames[-1].get("frame_index"),
        "frame_count": len(frames),
        "primary_error": "OK",
        "skeleton_sequence": frames,
    }


def _find_rep_result(rep_results: list[Any], rep_index: Any) -> dict[str, Any] | None:
    for item in rep_results:
        if isinstance(item, dict) and item.get("rep_index") == rep_index:
            return item
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}
