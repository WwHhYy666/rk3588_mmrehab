"""震动马达 GPIO 控制模块。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

# python-periphery 使用 gpiochip 内部 line offset，不使用 sysfs 全局 GPIO 编号。
# 当前接线为 GPIO3_B3：sysfs 全局编号是 107，periphery 对应 /dev/gpiochip3 line 11。
GPIO_PIN = 11
GPIO_CHIP_PATH = "/dev/gpiochip3"
SHORT_BUZZ_SECONDS = 0.2
LONG_BUZZ_SECONDS = 0.8
RAPID_PULSE_SECONDS = 0.1

try:
    from periphery import GPIO as PeripheryGPIO
except ImportError:
    PeripheryGPIO = None


class MotorController:
    """用于控制震动马达的线程安全 GPIO 控制器。"""

    def __init__(self, mock_mode: bool = False) -> None:
        """初始化震动马达控制器。

        当 ``mock_mode`` 为 ``True`` 时，所有控制操作都会退化为日志输出；
        当为 ``False`` 时，控制器会尝试通过 ``python-periphery`` 打开真实 GPIO。
        如果库未安装或 GPIO 初始化失败，会自动回退到模拟模式。
        """
        self._lock = threading.Lock()
        self._mock_mode = mock_mode
        self._gpio: Any | None = None
        self._closed = False

        if self._mock_mode:
            print("[MOTOR MOCK] 已启用模拟模式。")
            return

        if PeripheryGPIO is None:
            print("[MOTOR WARNING] 未安装 periphery，已自动回退到模拟模式。")
            self._mock_mode = True
            return

        try:
            self._gpio = PeripheryGPIO(GPIO_CHIP_PATH, GPIO_PIN, "out")
            self._gpio.write(False)
        except Exception as exc:
            print(f"[MOTOR WARNING] GPIO 初始化失败，已自动回退到模拟模式：{exc}")
            self._gpio = None
            self._mock_mode = True

    def short_buzz(self) -> None:
        """触发一次短震反馈，高电平持续 200 毫秒。"""
        self._run_buzz("short_buzz", SHORT_BUZZ_SECONDS)

    def long_buzz(self) -> None:
        """触发一次长震反馈，高电平持续 800 毫秒。"""
        self._run_buzz("long_buzz", LONG_BUZZ_SECONDS)

    def rapid_buzz(self) -> None:
        """触发三次急促震动脉冲，用于提示动作速度过快。"""
        self._run_interval_buzz("rapid_buzz")

    def interval_buzz(self) -> None:
        """触发三次间隔震动脉冲，行为与 rapid_buzz 兼容。"""
        self._run_interval_buzz("interval_buzz")

    def _run_interval_buzz(self, action_name: str) -> None:
        """执行三次间隔震动脉冲。"""
        with self._lock:
            if self._closed:
                print(f"[MOTOR WARNING] 控制器已清理，忽略 {action_name} 调用。")
                return

            if self._mock_mode:
                print(f"[MOTOR MOCK] {action_name} 已触发")
                return

            for _ in range(3):
                self._write_gpio(True)
                time.sleep(RAPID_PULSE_SECONDS)
                self._write_gpio(False)
                time.sleep(RAPID_PULSE_SECONDS)

    def cleanup(self) -> None:
        """释放 GPIO 资源并确保输出拉低，重复调用也安全。"""
        with self._lock:
            if self._closed:
                return

            self._closed = True

            if self._mock_mode:
                print("[MOTOR MOCK] cleanup 已执行")
                return

            if self._gpio is None:
                return

            try:
                self._write_gpio(False)
            except Exception as exc:
                print(f"[MOTOR WARNING] cleanup 拉低 GPIO 失败：{exc}")

            try:
                self._gpio.close()
            except Exception as exc:
                print(f"[MOTOR WARNING] cleanup 关闭 GPIO 失败：{exc}")
            finally:
                self._gpio = None

    def _run_buzz(self, action_name: str, duration: float) -> None:
        """执行单次震动脉冲，模拟模式下退化为日志输出。"""
        with self._lock:
            if self._closed:
                print(f"[MOTOR WARNING] 控制器已清理，忽略 {action_name} 调用。")
                return

            if self._mock_mode:
                print(f"[MOTOR MOCK] {action_name} 已触发")
                return

            self._pulse(duration)

    def _pulse(self, duration: float) -> None:
        """输出一次高电平脉冲并在结束后恢复低电平。"""
        self._write_gpio(True)
        time.sleep(duration)
        self._write_gpio(False)

    def _write_gpio(self, value: bool) -> None:
        """安全写入 GPIO 电平。"""
        if self._gpio is None:
            raise RuntimeError("GPIO 尚未初始化。")
        self._gpio.write(value)


def gpio_worker(
    motor_queue: queue.Queue[str],
    stop_event: threading.Event,
    mock_mode: bool = False,
) -> None:
    """GPIO 工作线程函数。

    该函数设计为在守护线程中运行，阻塞式监听 ``motor_queue`` 中的命令字符串，
    并根据命令触发对应的震动反馈。支持的命令为 ``short``、``long``、``rapid``、``interval`` 和 ``stop``。
    收到 ``stop`` 后会设置 ``stop_event`` 并退出循环。
    """
    controller = MotorController(mock_mode=mock_mode)

    try:
        while not stop_event.is_set():
            try:
                command = motor_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if command == "short":
                    controller.short_buzz()
                elif command == "long":
                    controller.long_buzz()
                elif command == "rapid":
                    controller.rapid_buzz()
                elif command == "interval":
                    controller.interval_buzz()
                elif command == "stop":
                    stop_event.set()
                    break
                else:
                    print(f"[MOTOR WARNING] 收到未知命令：{command}")
            finally:
                motor_queue.task_done()
    finally:
        controller.cleanup()


if __name__ == "__main__":
    motor = MotorController(mock_mode=True)
    motor.short_buzz()
    motor.long_buzz()
    motor.interval_buzz()
    motor.cleanup()
