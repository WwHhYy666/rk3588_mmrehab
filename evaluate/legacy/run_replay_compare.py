from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from .engine import extract_time_series, load_actions, load_json, replay_compare
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from evaluate.engine import extract_time_series, load_actions, load_json, replay_compare  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="个性化标准动作录入与实时比对 replay。")
    parser.add_argument("--template", required=True, help="标准动作 template.json。")
    parser.add_argument("--attempt", required=True, help="患者动作 attempt.json。")
    parser.add_argument("--actions", default=str(Path(__file__).with_name("actions.json")), help="动作配置 JSON。")
    parser.add_argument("--tolerance", type=float, default=8.0, help="逐帧角度容差，单位度。")
    parser.add_argument("--realtime", action="store_true", help="按 attempt 时间间隔模拟实时 replay。")
    parser.add_argument("--max-rows", type=int, default=30, help="最多打印多少行逐帧结果；0 表示全部。")
    parser.add_argument("--mock-motor", action="store_true", help="按最终反馈触发马达 mock。")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON。")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_series(path: Path, actions: dict[str, dict[str, Any]]):
    return extract_time_series(load_json(path), actions=actions, source_path=path)


def run_motor_mock(command: str | None) -> None:
    if not command:
        return
    sys.path.append(str(PROJECT_ROOT / "hardware" / "motro_control"))
    from hardware.motro_control.motor_controller import gpio_worker

    motor_queue: queue.Queue[str] = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(target=gpio_worker, args=(motor_queue, stop_event, True), daemon=True)
    worker.start()
    motor_queue.put(command)
    motor_queue.put("stop")
    motor_queue.join()
    worker.join(timeout=2.0)


def main() -> int:
    args = parse_args()
    actions = load_actions(Path(args.actions))
    template_path = resolve_path(args.template)
    attempt_path = resolve_path(args.attempt)

    try:
        result = replay_compare(
            load_series(template_path, actions),
            load_series(attempt_path, actions),
            tolerance_degrees=args.tolerance,
        )
    except Exception as exc:
        print(f"replay 对比失败：{exc}", file=sys.stderr)
        return 1

    result["template_path"] = str(template_path)
    result["attempt_path"] = str(attempt_path)

    if args.mock_motor:
        run_motor_mock(result.get("motor_command"))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("========== 个性化标准动作录入与实时比对 ==========")
    print(f"标准动作：{template_path}")
    print(f"患者动作：{attempt_path}")
    print(f"动作：{result.get('action_name')} ({result.get('action_id')})")
    print(f"逐帧容差：{result.get('replay_tolerance_degrees'):.1f} 度")
    print()
    print("frame | t(s) | template | attempt | diff | status | hint")
    print("----- | ---- | -------- | ------- | ---- | ------ | ----")

    rows = result["replay"]
    limit = len(rows) if args.max_rows == 0 else min(args.max_rows, len(rows))
    previous_time = rows[0]["attempt_time"] if rows else 0.0
    for row in rows[:limit]:
        if args.realtime:
            delay = max(0.0, float(row["attempt_time"]) - float(previous_time))
            time.sleep(min(delay, 0.25))
            previous_time = row["attempt_time"]
        print(
            f"{row['frame_index']:>5} | "
            f"{row['attempt_time']:>4.2f} | "
            f"{row['template_angle']:>8.2f} | "
            f"{row['attempt_angle']:>7.2f} | "
            f"{row['diff']:>5.2f} | "
            f"{row['status']:<6} | "
            f"{row['hint']}"
        )

    if limit < len(rows):
        print(f"... 已省略 {len(rows) - limit} 行，可用 --max-rows 0 查看全部。")

    print()
    print("========== 汇总反馈 ==========")
    print(f"平均逐帧误差：{result['mean_abs_error']:.2f} 度")
    print(f"最大逐帧误差：{result['max_abs_error']:.2f} 度")
    print(f"超出容差帧：{result['off_frame_count']}/{result['replay_frame_count']}")
    if result.get("dtw_distance") is not None:
        print(f"DTW：{result['dtw_distance']:.2f} ({result.get('dtw_level')})")
    print(f"ROM：{result.get('rom'):.2f} 度")
    print(f"TUT：{result.get('tut_seconds'):.2f} 秒 / 目标 {result.get('target_tut_seconds'):.2f} 秒")
    print(f"质量等级：{result.get('quality_level')}")
    print(f"马达命令：{result.get('motor_command') or '无'}")
    print("反馈：")
    print(result.get("feedback_text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
