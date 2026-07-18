from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.report_generator import make_report  # noqa: E402


ACTIONS = ("sit_to_stand", "standing_hamstring_curl", "seated_knee_raise")
ATTEMPTS_DIR = Path("data/npu/patient_attempts")
REPORTS_DIR = Path("data/reports/npu")
TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})$")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def latest_attempt(project_root: Path, action_id: str) -> tuple[Path, dict[str, Any]]:
    attempts_dir = project_root / ATTEMPTS_DIR
    candidates = sorted(attempts_dir.glob("patient_attempt_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = load_json(path)
        runtime_meta = payload.get("runtime_meta") if isinstance(payload.get("runtime_meta"), dict) else {}
        if str(payload.get("action_id") or runtime_meta.get("action_id") or "") == action_id:
            return path, payload
    raise FileNotFoundError(f"no NPU attempt found for {action_id} under {attempts_dir}")


def attempt_timestamp(path: Path) -> str:
    match = TIMESTAMP_RE.search(path.stem)
    if match is None:
        raise ValueError(f"attempt filename has no timestamp: {path.name}")
    return match.group(1)


def artifact_paths(project_root: Path, action_id: str, attempt_path: Path, attempt: dict[str, Any]) -> tuple[Path, Path, Path]:
    runtime_meta = attempt.get("runtime_meta") if isinstance(attempt.get("runtime_meta"), dict) else {}
    active_template = runtime_meta.get("active_template") if isinstance(runtime_meta.get("active_template"), dict) else {}
    template_value = str(active_template.get("template_file") or "").strip()
    config_value = str(active_template.get("config_file") or f"evaluation/configs/npu/{action_id}.yaml").strip()
    if not template_value:
        raise ValueError(f"attempt does not record its active template: {attempt_path}")
    report_path = project_root / REPORTS_DIR / f"report_{action_id}_{attempt_timestamp(attempt_path)}.json"
    return project_root / template_value, project_root / config_value, report_path


def regenerate(project_root: Path, action_id: str) -> Path:
    attempt_path, attempt = latest_attempt(project_root, action_id)
    template_path, config_path, report_path = artifact_paths(project_root, action_id, attempt_path, attempt)
    for path in (attempt_path, template_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    report = make_report(template_path, attempt_path, config_path)
    keyframes = report.get("keyframes") if isinstance(report.get("keyframes"), list) else []
    if not keyframes:
        raise RuntimeError(f"{action_id} report still has no keyframes; inspect runtime_meta.keyframes in {attempt_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {action_id}: {report_path.relative_to(project_root)} ({len(keyframes)} keyframes)")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the latest NPU report for each rehab action.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--action", choices=ACTIONS, action="append")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    actions = tuple(args.action or ACTIONS)
    failed = False
    for action_id in actions:
        try:
            regenerate(project_root, action_id)
        except Exception as exc:
            failed = True
            print(f"[FAIL] {action_id}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
