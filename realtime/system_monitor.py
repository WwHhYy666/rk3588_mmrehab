"""Lightweight RK3588 system resource monitor.

The monitor only reads Linux proc/sysfs files. Missing files are reported as
unavailable so the demo can keep running on Windows or RK3588 images without
NPU debug nodes.
"""

from __future__ import annotations

import glob
import re
import subprocess
import time
from pathlib import Path
from typing import Any


_previous_cpu_sample: tuple[int, int] | None = None
NPU_LOAD_PATH = Path("/sys/kernel/debug/rknpu/load")


def get_system_status(pose_fps: float | None = None) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "cpu": read_cpu_usage(),
        "memory": read_memory_usage(),
        "temperature": read_temperature(),
        "npu": _safe_read_npu_load(),
        "pose_fps": {
            "available": pose_fps is not None,
            "fps": round(float(pose_fps), 2) if pose_fps is not None else None,
            "note": "由摄像头/姿态推理 worker 统计" if pose_fps is not None else "等待首帧画面",
        },
    }


def read_cpu_usage() -> dict[str, Any]:
    global _previous_cpu_sample
    path = Path("/proc/stat")
    if not path.exists():
        return {"available": False, "percent": None, "note": "未检测到 /proc/stat"}

    try:
        line = path.read_text(encoding="utf-8").splitlines()[0]
        parts = [int(value) for value in line.split()[1:]]
    except (OSError, ValueError, IndexError):
        return {"available": False, "percent": None, "note": "读取 /proc/stat 失败"}

    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    total = sum(parts)
    current = (total, idle)
    if _previous_cpu_sample is None:
        _previous_cpu_sample = current
        return {"available": True, "percent": None, "note": "CPU 采样初始化中"}

    previous_total, previous_idle = _previous_cpu_sample
    _previous_cpu_sample = current
    total_delta = total - previous_total
    idle_delta = idle - previous_idle
    if total_delta <= 0:
        return {"available": True, "percent": None, "note": "CPU 采样间隔过短"}

    usage = 100.0 * (1.0 - idle_delta / total_delta)
    return {"available": True, "percent": round(max(0.0, min(100.0, usage)), 1), "note": "ok"}


def read_memory_usage() -> dict[str, Any]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return {"available": False, "percent": None, "note": "未检测到 /proc/meminfo"}

    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        return {"available": False, "percent": None, "note": "读取 /proc/meminfo 失败"}

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return {"available": False, "percent": None, "note": "缺少 MemTotal/MemAvailable"}

    used = max(0, total - available)
    return {
        "available": True,
        "percent": round(100.0 * used / total, 1),
        "used_mb": round(used / 1024, 1),
        "total_mb": round(total / 1024, 1),
        "note": "ok",
    }


def read_temperature() -> dict[str, Any]:
    paths = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
    readings: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            raw = path.read_text(encoding="utf-8").strip()
            value = float(raw)
        except (OSError, ValueError):
            continue
        celsius = value / 1000.0 if value > 200 else value
        zone_type = _read_text(path.parent / "type") or path.parent.name
        readings.append({"zone": zone_type, "celsius": round(celsius, 1)})

    if not readings:
        return {"available": False, "max_celsius": None, "zones": [], "note": "未检测到 thermal_zone 温度接口"}

    return {
        "available": True,
        "max_celsius": max(item["celsius"] for item in readings),
        "zones": readings,
        "note": "ok",
    }


def read_npu_load() -> dict[str, Any]:
    path = NPU_LOAD_PATH
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except FileNotFoundError:
        return _npu_unavailable(source=source, message="NPU load 接口不可用：未检测到 /sys/kernel/debug/rknpu/load")
    except PermissionError:
        text = _read_npu_load_with_sudo(path)
        if text is None:
            return _npu_unavailable(source=source, message="NPU load 接口权限不足，无法读取；请执行 sudo chmod a+r /sys/kernel/debug/rknpu/load")
    except OSError as exc:
        return _npu_unavailable(source=source, message=f"NPU load 接口读取失败：{exc}")

    parsed = _parse_rknpu_debug_load(text)
    if parsed is None:
        return _npu_unavailable(
            source=source,
            raw=text,
            message="已读取到 NPU load，但未能解析 CoreX 百分比",
        )

    average = parsed["average_percent"]
    return {
        "available": True,
        "percent": average,
        "average_percent": average,
        "cores": parsed["cores"],
        "raw": text,
        "source": source,
        "message": "ok",
        "note": "ok",
    }


def _safe_read_npu_load() -> dict[str, Any]:
    try:
        return read_npu_load()
    except Exception as exc:
        return _npu_unavailable(source=str(NPU_LOAD_PATH), message=f"NPU load 监控异常：{exc}")


def _read_npu_load_with_sudo(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["sudo", "-n", "cat", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def _parse_rknpu_debug_load(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"\bCore\s*(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text or "", flags=re.IGNORECASE)
    if not matches:
        return None

    indexed: list[tuple[int, float]] = []
    for index_text, value_text in matches:
        value = float(value_text)
        if value < 0.0 or value > 100.0:
            return None
        indexed.append((int(index_text), value))

    if not indexed:
        return None
    indexed.sort(key=lambda item: item[0])
    cores = {f"Core{index}": round(value, 1) for index, value in indexed}
    average = round(sum(cores.values()) / len(cores), 1)
    return {"cores": cores, "average_percent": average}


def _npu_unavailable(*, source: str, message: str, raw: str | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "percent": None,
        "average_percent": None,
        "cores": {},
        "raw": raw,
        "source": source,
        "message": message,
        "note": message,
    }


def _read_text(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
