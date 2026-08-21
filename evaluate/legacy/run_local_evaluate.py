from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any

try:
    from .engine import (
        DEFAULT_RESULTS_DIR,
        evaluate_pair,
        evaluate_single,
        extract_time_series,
        find_latest_json,
        load_actions,
        load_json,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from evaluate.engine import (  # type: ignore
        DEFAULT_RESULTS_DIR,
        evaluate_pair,
        evaluate_single,
        extract_time_series,
        find_latest_json,
        load_actions,
        load_json,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="个性化标准动作录入与本地评估工具。")
    parser.add_argument("--input", default="latest", help="单文件评估输入；默认 latest。")
    parser.add_argument("--template", help="模板 JSON，用于 DTW 对比。")
    parser.add_argument("--attempt", help="本次动作 JSON，用于 DTW 对比。")
    parser.add_argument("--actions", default=str(Path(__file__).with_name("actions.json")), help="动作配置 JSON。")
    parser.add_argument("--mock-motor", action="store_true", help="按评估结果触发马达 mock。")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON。")
    return parser.parse_args()


def resolve_input(value: str) -> Path:
    if value == "latest":
        return find_latest_json(DEFAULT_RESULTS_DIR)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def run_motor_mock(command: str | None) -> None:
    if not command:
        return
    sys.path.append(str(PROJECT_ROOT / "hardware" / "motro_control"))
    from motor_controller import gpio_worker

    motor_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(target=gpio_worker, args=(motor_queue, stop_event, True), daemon=True)
    worker.start()
    motor_queue.put(command)
    motor_queue.put("stop")
    motor_queue.join()
    worker.join(timeout=2.0)


def load_series(path: Path, actions: dict[str, dict[str, Any]]):
    payload = load_json(path)
    return extract_time_series(payload, actions=actions, source_path=path)


def main() -> int:
    args = parse_args()
    actions = load_actions(Path(args.actions))

    try:
        if args.template or args.attempt:
            if not args.template or not args.attempt:
                raise ValueError("--template 和 --attempt 需要同时提供。")
            template_path = resolve_input(args.template)
            attempt_path = resolve_input(args.attempt)
            result = evaluate_pair(load_series(template_path, actions), load_series(attempt_path, actions))
            result["template_path"] = str(template_path)
            result["attempt_path"] = str(attempt_path)
        else:
            input_path = resolve_input(args.input)
            result = evaluate_single(load_series(input_path, actions))
            result["input_path"] = str(input_path)
    except Exception as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 1

    if args.mock_motor:
        run_motor_mock(result.get("motor_command"))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("========== 本地动作评估 ==========")
        print(f"动作：{result.get('action_name')} ({result.get('action_id')})")
        print(f"质量等级：{result.get('quality_level')}")
        print(f"ROM：{_fmt(result.get('rom'))} 度")
        print(f"TUT：{_fmt(result.get('tut_seconds'))} 秒 / 目标 {_fmt(result.get('target_tut_seconds'))} 秒")
        if result.get("dtw_distance") is not None:
            print(f"DTW：{_fmt(result.get('dtw_distance'))} ({result.get('dtw_level')})")
        print(f"马达命令：{result.get('motor_command') or '无'}")
        print("反馈：")
        print(result.get("feedback_text"))
    return 0


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return "未知"


if __name__ == "__main__":
    raise SystemExit(main())
