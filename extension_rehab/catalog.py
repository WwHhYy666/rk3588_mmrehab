from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
PLAN_ROOT = PACKAGE_ROOT / "plans"
GROUPS = ("upper", "full")


def _load_actions() -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, ...]]]:
    actions: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {group: [] for group in GROUPS}
    for group in GROUPS:
        for path in sorted((CONFIG_ROOT / group).glob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            action_id = str(payload.get("action_id") or path.stem)
            if action_id in actions:
                raise ValueError(f"duplicate extension action: {action_id}")
            payload = dict(payload)
            payload["action_id"] = action_id
            payload["group"] = group
            payload["config_file"] = str(path.relative_to(PACKAGE_ROOT.parent)).replace("\\", "/")
            actions[action_id] = payload
            groups[group].append(action_id)
    if len(actions) != 6 or any(len(groups[group]) != 3 for group in GROUPS):
        raise RuntimeError("extension action catalog requires exactly three upper and three full-body actions")
    return actions, {group: tuple(values) for group, values in groups.items()}


ACTIONS, _DISCOVERED_GROUP_ACTIONS = _load_actions()


def _load_group_plans() -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        path = PLAN_ROOT / f"{group}.yaml"
        payload = dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        action_ids = tuple(str(value) for value in payload.get("actions") or [])
        if len(action_ids) != 3 or set(action_ids) != set(_DISCOVERED_GROUP_ACTIONS[group]):
            raise RuntimeError(f"extension plan {group} must contain its three configured actions exactly once")
        payload["group"] = group
        payload["actions"] = action_ids
        payload["default_reps"] = max(1, int(payload.get("default_reps") or 5))
        payload["rest_seconds"] = max(0.0, float(payload.get("rest_seconds") or 0.0))
        payload["rest_music_file"] = str(payload.get("rest_music_file") or "/assets/rest_music.wav")
        payload["rest_music_playback"] = str(payload.get("rest_music_playback") or "backend")
        payload["rest_music_fade_seconds"] = max(0.0, float(payload.get("rest_music_fade_seconds") or 0.0))
        plans[group] = payload
    return plans


GROUP_PLANS = _load_group_plans()
GROUP_ACTIONS = {group: tuple(GROUP_PLANS[group]["actions"]) for group in GROUPS}


def action_catalog(group: str | None = None) -> list[dict[str, Any]]:
    action_ids = GROUP_ACTIONS.get(str(group or "").strip().lower(), tuple(ACTIONS)) if group else tuple(ACTIONS)
    return [
        {
            "action_id": action_id,
            "group": ACTIONS[action_id]["group"],
            "label": ACTIONS[action_id]["label"],
            "description": ACTIONS[action_id].get("description", ""),
            "view": ACTIONS[action_id]["view"],
            "metric_unit": ACTIONS[action_id]["metric_unit"],
            "seed_rom": ACTIONS[action_id]["seed_rom"],
            "instructions": list(ACTIONS[action_id].get("instructions") or []),
        }
        for action_id in action_ids
        if action_id in ACTIONS
    ]
