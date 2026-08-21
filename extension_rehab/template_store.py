from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import ACTIONS


class ExtensionTemplateStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else Path("prescription/docs/npu/extension_templates")

    def path_for(self, action_id: str) -> Path:
        config = ACTIONS[action_id]
        return self.root / str(config["group"]) / f"{action_id}.json"

    def load(self, action_id: str) -> dict[str, Any] | None:
        path = self.path_for(action_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def save(self, action_id: str, payload: dict[str, Any]) -> Path:
        if action_id not in ACTIONS:
            raise KeyError(action_id)
        path = self.path_for(action_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            **payload,
            "action_id": action_id,
            "group": ACTIONS[action_id]["group"],
            "label": ACTIONS[action_id]["label"],
            "pose_backend": "rknn",
            "keypoint_schema": "coco17_extension_v1",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def health(self, action_id: str) -> dict[str, Any]:
        if action_id not in ACTIONS:
            return {"ok": False, "reason": "unknown_action", "action_id": action_id}
        payload = self.load(action_id)
        if payload is None:
            return {"ok": False, "reason": "missing", "action_id": action_id, "template_file": str(self.path_for(action_id))}
        config = ACTIONS[action_id]
        reasons: list[str] = []
        valid_frames = int(payload.get("valid_frames") or 0)
        duration = float(payload.get("duration_seconds") or 0.0)
        rom = float(payload.get("rom") or 0.0)
        if valid_frames < 20:
            reasons.append("valid_frames")
        if duration < 2.0:
            reasons.append("duration")
        if rom < float(config["min_valid_rom"]):
            reasons.append("rom")
        if str(payload.get("direction") or "") != str(config.get("direction") or "increase"):
            reasons.append("direction")
        if not bool(payload.get("returned")):
            reasons.append("return")
        return {
            "ok": not reasons,
            "reason": ",".join(reasons) if reasons else "ok",
            "action_id": action_id,
            "template_file": str(self.path_for(action_id)),
            "valid_frames": valid_frames,
            "duration_seconds": duration,
            "rom": rom,
            "target_rom": float(payload.get("target_rom") or rom),
            "direction": payload.get("direction"),
            "returned": bool(payload.get("returned")),
        }
