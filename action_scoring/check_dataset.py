from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from action_scoring.features import build_input_tensor
from action_scoring.registry import CPU_QUALITY_ACTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "npu"
SOURCE_DIRS = (DATA_ROOT / "patient_attempts", DATA_ROOT / "results")
EXPECTED_LABELS = ("OK", "ROM_LOW", "TUT_LOW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check quality_attempt_segments coverage before training.")
    parser.add_argument("--actions", nargs="+", default=list(CPU_QUALITY_ACTIONS))
    parser.add_argument("--min-per-label", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    actions = [str(action).strip() for action in args.actions if str(action).strip()]
    unsupported = sorted(set(actions) - set(CPU_QUALITY_ACTIONS))
    if unsupported:
        raise SystemExit(f"Unsupported CPU quality actions: {', '.join(unsupported)}")
    summary = check_actions(actions, min_per_label=max(0, int(args.min_per_label)))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_summary(summary)
    return 1 if summary["warnings"] else 0


def check_actions(actions: list[str], min_per_label: int = 5) -> dict[str, Any]:
    by_action: dict[str, Counter[str]] = {action: Counter() for action in actions}
    source_counts: dict[str, Counter[str]] = {action: Counter() for action in actions}
    valid_skeleton_counts: dict[str, int] = defaultdict(int)
    file_counts: dict[str, int] = defaultdict(int)
    warnings: list[str] = []

    for path in _iter_payload_paths():
        payload = _load_json(path)
        runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
        action_id = str(payload.get("action_id") or runtime_meta.get("action_id") or "").strip()
        if action_id not in by_action:
            continue
        segments, source_field = _extract_training_segments(runtime_meta)
        if not segments:
            continue
        file_counts[action_id] += 1
        source_counts[action_id][source_field] += len(segments)
        for segment in segments:
            label = str(segment.get("primary_error") or "OK").strip() or "OK"
            by_action[action_id][label] += 1
            tensor, _ = build_input_tensor(segment)
            if tensor is not None:
                valid_skeleton_counts[action_id] += 1

    actions_summary: dict[str, Any] = {}
    for action_id in actions:
        counts = dict(sorted(by_action[action_id].items()))
        missing = [label for label in EXPECTED_LABELS if by_action[action_id][label] <= 0]
        low = [label for label in EXPECTED_LABELS if 0 < by_action[action_id][label] < min_per_label]
        if missing:
            warnings.append(f"{action_id}: missing labels {', '.join(missing)}")
        if low:
            warnings.append(f"{action_id}: low sample labels {', '.join(low)} (< {min_per_label})")
        if valid_skeleton_counts[action_id] <= 0:
            warnings.append(f"{action_id}: no trainable skeleton_sequence samples")
        actions_summary[action_id] = {
            "files": file_counts[action_id],
            "labels": counts,
            "sources": dict(sorted(source_counts[action_id].items())),
            "valid_skeleton_segments": valid_skeleton_counts[action_id],
            "missing_labels": missing,
            "low_labels": low,
        }

    return {
        "data_root": str(DATA_ROOT),
        "min_per_label": min_per_label,
        "expected_labels": list(EXPECTED_LABELS),
        "actions": actions_summary,
        "warnings": warnings,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"data_root: {summary['data_root']}")
    print(f"min_per_label: {summary['min_per_label']}")
    for action_id, row in summary["actions"].items():
        labels = ", ".join(f"{label}={count}" for label, count in row["labels"].items()) or "none"
        sources = ", ".join(f"{name}={count}" for name, count in row["sources"].items()) or "none"
        print(f"{action_id}: files={row['files']} skeleton={row['valid_skeleton_segments']} labels=[{labels}] sources=[{sources}]")
    if summary["warnings"]:
        print("WARNINGS:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    else:
        print("OK: all expected labels are present for each action.")


def _iter_payload_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in SOURCE_DIRS:
        if directory.exists():
            paths.extend(sorted(directory.glob("*.json")))
    return sorted(paths, key=lambda path: (path.parent.name, path.name))


def _extract_training_segments(runtime_meta: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    for field in ("quality_attempt_segments", "rep_segments"):
        raw_segments = runtime_meta.get(field)
        if isinstance(raw_segments, list):
            return [segment for segment in raw_segments if isinstance(segment, dict)], field
    return [], ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
