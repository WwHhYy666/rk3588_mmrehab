from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_TEMPLATES_PATH = PROJECT_ROOT / "runtime" / "active_templates.json"
SUPPORTED_POSE_BACKENDS = ("mediapipe", "rknn")
DEFAULT_POSE_BACKEND = "mediapipe"


def normalize_pose_backend(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in SUPPORTED_POSE_BACKENDS else DEFAULT_POSE_BACKEND


def load_active_templates(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else ACTIVE_TEMPLATES_PATH
    if not registry_path.exists():
        return _empty_registry()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_registry()
    return normalize_active_template_registry(payload)


def save_active_templates(payload: dict[str, Any], path: str | Path | None = None) -> None:
    registry_path = Path(path) if path is not None else ACTIVE_TEMPLATES_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = normalize_active_template_registry(payload)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def get_active_template(action_id: str, pose_backend: object = DEFAULT_POSE_BACKEND, path: str | Path | None = None) -> dict[str, Any] | None:
    registry = load_active_templates(path)
    backend = normalize_pose_backend(pose_backend)
    by_backend = registry.get("by_backend", {})
    if not isinstance(by_backend, dict):
        return None
    entries = by_backend.get(backend, {})
    if not isinstance(entries, dict):
        return None
    entry = entries.get(str(action_id or "").strip())
    return dict(entry) if isinstance(entry, dict) else None


def set_active_template(
    action_id: str,
    template_file: str | Path,
    *,
    config_file: str,
    pose_backend: object,
    pose_meta: dict[str, Any] | None = None,
    path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    backend = normalize_pose_backend(pose_backend)
    registry = load_active_templates(path)
    by_backend = registry.setdefault("by_backend", {})
    if not isinstance(by_backend, dict):
        by_backend = {}
        registry["by_backend"] = by_backend
    entries = by_backend.setdefault(backend, {})
    if not isinstance(entries, dict):
        entries = {}
        by_backend[backend] = entries

    action = str(action_id or "").strip()
    entry: dict[str, Any] = {
        "action_id": action,
        "template_file": project_relative(template_file, project_root),
        "config_file": str(config_file),
        "pose_backend": backend,
        "actual_backend": backend,
        "backend_key": backend,
        "updated_at": datetime.now().replace(microsecond=0).isoformat(),
    }
    if pose_meta:
        entry.update(dict(pose_meta))
        entry["pose_backend"] = backend
        entry["actual_backend"] = backend
        entry["backend_key"] = backend
    entries[action] = entry
    save_active_templates(registry, path)
    return dict(entry)


def normalize_active_template_registry(payload: object) -> dict[str, Any]:
    registry = _empty_registry()
    if not isinstance(payload, dict):
        return registry

    if isinstance(payload.get("by_backend"), dict):
        raw_by_backend = payload.get("by_backend") or {}
        for backend_name, entries in raw_by_backend.items():
            backend = normalize_pose_backend(backend_name)
            if not isinstance(entries, dict):
                continue
            for action_id, entry in entries.items():
                normalized = _normalize_entry(action_id, entry, backend)
                if normalized is not None:
                    registry["by_backend"][backend][normalized["action_id"]] = normalized
        return registry

    for action_id, entry in payload.items():
        if action_id in {"schema_version", "by_backend"}:
            continue
        if not isinstance(entry, dict):
            continue
        backend = normalize_pose_backend(entry.get("actual_backend") or entry.get("pose_backend") or DEFAULT_POSE_BACKEND)
        normalized = _normalize_entry(action_id, entry, backend)
        if normalized is not None:
            registry["by_backend"][backend][normalized["action_id"]] = normalized
    return registry


def project_relative(path: str | Path, project_root: str | Path | None = None) -> str:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "by_backend": {backend: {} for backend in SUPPORTED_POSE_BACKENDS},
    }


def _normalize_entry(action_id: object, entry: object, backend: str) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    action = str(entry.get("action_id") or action_id or "").strip()
    if not action:
        return None
    normalized = dict(entry)
    normalized["action_id"] = action
    normalized["pose_backend"] = backend
    normalized["actual_backend"] = backend
    normalized["backend_key"] = backend
    return normalized
