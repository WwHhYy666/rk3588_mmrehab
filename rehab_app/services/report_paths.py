from __future__ import annotations

from pathlib import Path


KEYFRAME_PREFIXES = (
    "data/reports/keyframes/",
    "data/reports/npu/keyframes/",
)
SAFE_IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def normalize_keyframe_path(value: object) -> str:
    return str(value or "").replace("\\", "/").strip()


def keyframe_prefix(value: object) -> str | None:
    text = normalize_keyframe_path(value)
    if not text or text.startswith("/") or text.startswith("\\"):
        return None
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in SAFE_IMAGE_SUFFIXES:
        return None
    normalized = path.as_posix()
    return next((prefix for prefix in KEYFRAME_PREFIXES if normalized.startswith(prefix)), None)


def is_safe_keyframe_path(value: object) -> bool:
    return keyframe_prefix(value) is not None


def resolve_keyframe_path(project_root: Path, value: object) -> Path | None:
    text = normalize_keyframe_path(value)
    prefix = keyframe_prefix(text)
    if prefix is None:
        return None
    resolved = (project_root / text).resolve()
    base = (project_root / prefix).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved
