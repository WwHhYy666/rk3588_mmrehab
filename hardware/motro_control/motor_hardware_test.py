"""震动马达真实硬件联调脚本。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from motor_controller import GPIO_CHIP_PATH, GPIO_PIN, MotorController


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="ELF 2 震动马达硬件联调脚本，默认只做 Mock 预演。"
    )
    parser.add_argument(
        "mode",
        choices=("short", "long", "rapid", "interval", "all"),
        help="选择要测试的震动模式。",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="真正访问 GPIO 并驱动硬件；不加该参数时只运行 Mock 模式。",
    )
    return parser.parse_args()


def run_mode(controller: MotorController, mode: str) -> None:
    """按指定模式触发震动。"""
    actions: dict[str, Callable[[], None]] = {
        "short": controller.short_buzz,
        "long": controller.long_buzz,
        "rapid": controller.rapid_buzz,
        "interval": controller.interval_buzz,
    }

    if mode == "all":
        for name in ("short", "long", "interval"):
            print(f"[MOTOR TEST] run {name}")
            actions[name]()
            time.sleep(0.5)
        return

    print(f"[MOTOR TEST] run {mode}")
    actions[mode]()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    mock_mode = not args.real

    print(f"[MOTOR TEST] GPIO_PIN={GPIO_PIN}")
    print(f"[MOTOR TEST] GPIO_CHIP_PATH={GPIO_CHIP_PATH}")
    print(f"[MOTOR TEST] mock_mode={mock_mode}")

    if args.real:
        print("[MOTOR TEST] 将访问真实 GPIO，请确认接线、共地和 MOS 驱动模块已检查。")
    else:
        print("[MOTOR TEST] 当前为 Mock 预演；加 --real 才会驱动真实硬件。")

    controller = MotorController(mock_mode=mock_mode)
    try:
        run_mode(controller, args.mode)
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
