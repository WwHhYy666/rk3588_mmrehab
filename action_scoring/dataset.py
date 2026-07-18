from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from action_scoring.features import build_input_tensor
from action_scoring.labels import label_from_attempt_segment
from action_scoring.registry import CPU_QUALITY_ACTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESCRIPTION_DOCS_DIR = PROJECT_ROOT / "data" / "npu"
RESULTS_DIR = PRESCRIPTION_DOCS_DIR / "results"
PATIENT_ATTEMPTS_DIR = PRESCRIPTION_DOCS_DIR / "patient_attempts"
TRAINING_SOURCE_DIRS = (PATIENT_ATTEMPTS_DIR, RESULTS_DIR)
RELABEL_REVIEWS_DIR = PROJECT_ROOT / "action_scoring" / "label_reviews"


def build_dataset(action_id: str, config_path: str | Path) -> list[dict[str, Any]]:
    if action_id not in CPU_QUALITY_ACTIONS:
        return []
    config = _load_yaml(config_path)
    relabels = _load_relabel_manifest(action_id)
    dataset: list[dict[str, Any]] = []
    for payload_path in _iter_payload_paths():
        payload = _load_json(payload_path)
        runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
        if str(payload.get("action_id") or runtime_meta.get("action_id") or "").strip() != action_id:
            continue
        record_role = str(runtime_meta.get("record_role") or payload.get("record_role") or "")
        segments, source_field = _extract_training_segments(runtime_meta)
        for index, segment in enumerate(segments, start=1):
            tensor, meta = build_input_tensor(segment)
            if tensor is None:
                continue
            training_segment = dict(segment)
            relabel_error = _relabel_for_segment(relabels, payload_path, training_segment)
            if relabel_error:
                training_segment["primary_error"] = relabel_error
            label = label_from_attempt_segment(training_segment, config)
            if label is None:
                continue
            dataset.append(
                {
                    "sample_id": f"{payload_path.parent.name}/{payload_path.stem}:attempt{segment.get('attempt_index') or index}",
                    "action_id": action_id,
                    "source_path": str(payload_path),
                    "source_field": source_field,
                    "record_role": record_role,
                    "primary_error": str(training_segment.get("primary_error") or "OK"),
                    "input": tensor,
                    "label": float(label),
                    "valid_frames": meta.valid_frames,
                }
            )
    return dataset


def _extract_training_segments(runtime_meta: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    for field in ("quality_attempt_segments", "rep_segments"):
        raw_segments = runtime_meta.get(field)
        if isinstance(raw_segments, list):
            return [segment for segment in raw_segments if isinstance(segment, dict)], field
    return [], ""


def _iter_payload_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in TRAINING_SOURCE_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths, key=lambda item: (item.parent.name, item.name))


def _load_relabel_manifest(action_id: str) -> dict[tuple[str, int], str]:
    relabels: dict[tuple[str, int], str] = {}
    if not RELABEL_REVIEWS_DIR.exists():
        return relabels
    for path in sorted(RELABEL_REVIEWS_DIR.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("action_id") or "").strip() != action_id:
                    continue
                if str(row.get("keep_for_training") or "true").strip().lower() in {"0", "false", "no"}:
                    continue
                relabel = str(row.get("relabel_error") or row.get("primary_error") or "").strip()
                attempt_index = _as_int(row.get("attempt_index"))
                source_path = str(row.get("source_path") or row.get("path") or "").strip()
                if relabel and source_path and attempt_index is not None:
                    relabels[(Path(source_path).name, attempt_index)] = relabel
    return relabels


def _relabel_for_segment(relabels: dict[tuple[str, int], str], payload_path: Path, segment: dict[str, Any]) -> str | None:
    attempt_index = _as_int(segment.get("attempt_index"))
    if attempt_index is None:
        return None
    return relabels.get((payload_path.name, attempt_index))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
