"""Lightweight RK3588 system resource monitor.

The monitor only reads Linux proc/sysfs files. Missing files are reported as
unavailable so the demo can keep running on Windows or RK3588 images without
NPU debug nodes.
"""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Any


_previous_cpu_sample: tuple[int, int] | None = None


def get_system_status(pose_fps: float | None = None) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "cpu": read_cpu_usage(),
        "memory": read_memory_usage(),
        "temperature": read_temperature(),
        "npu": read_npu_load(),
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
    candidate_paths = [Path("/sys/kernel/debug/rknpu/load")]
    candidate_paths.extend(Path(path) for path in glob.glob("/sys/class/devfreq/*npu*/load"))
    candidate_paths.extend(Path(path) for path in glob.glob("/sys/class/devfreq/*rknpu*/load"))

    checked: list[str] = []
    for path in candidate_paths:
        checked.append(str(path))
        text = _read_text(path)
        if not text:
            continue
        parsed = _parse_load_percent(text)
        return {
            "available": True,
            "percent": parsed,
            "raw": text,
            "source": str(path),
            "note": "ok" if parsed is not None else "已读取到 NPU load，但未能解析百分比",
        }

    return {
        "available": False,
        "percent": None,
        "raw": None,
        "source": None,
        "checked": checked,
        "note": "未检测到 NPU load 接口；可检查 debugfs/rknpu 驱动是否暴露该节点",
    }


def _parse_load_percent(text: str) -> float | None:
    numbers: list[float] = []
    token = ""
    for char in text:
        if char.isdigit() or char == ".":
            token += char
        elif token:
            try:
                numbers.append(float(token))
            except ValueError:
                pass
            token = ""
    if token:
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    if not numbers:
        return None
    bounded = [value for value in numbers if 0.0 <= value <= 100.0]
    return round(max(bounded or numbers), 1)


def _read_text(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None

