#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen


METRICS = (
    "camera_capture_fps",
    "pose_fps",
    "inference_fps",
    "training_update_fps",
    "stream_fps",
    "camera_read_ms",
    "queue_wait_ms",
    "process_resize_ms",
    "det_inference_ms",
    "pose_inference_ms",
    "preprocess_ms",
    "postprocess_ms",
    "pose_process_ms",
    "realtime_process_ms",
    "state_update_ms",
    "keyframe_candidate_copy_ms",
    "keyframe_encode_ms",
    "keyframe_write_ms",
    "stream_resize_ms",
    "render_draw_ms",
    "jpeg_encode_ms",
    "render_total_ms",
    "capture_to_inference_age_ms",
    "capture_to_stream_age_ms",
    "frame_queue_drops",
    "render_queue_drops",
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric_value(status: dict[str, Any], name: str) -> float | None:
    value = numeric(status.get(name))
    if value is not None:
        return value
    pose = status.get("pose_performance") if isinstance(status.get("pose_performance"), dict) else {}
    return numeric(pose.get(name))


def temperatures() -> dict[str, float]:
    result: dict[str, float] = {}
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            value = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if value > 1000:
            value /= 1000.0
        result[path.parent.name] = round(value, 2)
    return result


def read_status(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("/status did not return a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a 60-second 8085 performance baseline.")
    parser.add_argument("--scenario", required=True, choices=("idle", "npu-debug", "doctor", "train"))
    parser.add_argument("--url", default="http://127.0.0.1:8085/status")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    started_at = time.time()
    deadline = time.monotonic() + max(1.0, args.seconds)
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    while time.monotonic() < deadline:
        sample_started = time.monotonic()
        try:
            status = read_status(args.url, args.timeout)
            samples.append(
                {
                    "sampled_at": time.time(),
                    "metrics": {name: metric_value(status, name) for name in METRICS},
                    "detector_trigger_reason": status.get("detector_trigger_reason"),
                    "temperature": temperatures(),
                    "identity": {
                        "service_mode": status.get("service_mode"),
                        "actual_backend": status.get("actual_backend"),
                        "rknn_pipeline": status.get("rknn_pipeline"),
                        "camera_open_mode": status.get("camera_open_mode_active"),
                        "det_model_path": (status.get("npu_resource") or {}).get("det_model_path"),
                        "pose_model_path": (status.get("npu_resource") or {}).get("pose_model_path"),
                        "det_core_mask": (status.get("npu_resource") or {}).get("det_core_mask"),
                        "pose_core_mask": (status.get("npu_resource") or {}).get("pose_core_mask"),
                        "build_id": (status.get("runtime") or {}).get("build_id"),
                    },
                }
            )
        except Exception as exc:
            errors.append(f"{datetime.now().isoformat(timespec='seconds')}: {exc}")
        delay = max(0.0, args.interval - (time.monotonic() - sample_started))
        time.sleep(delay)

    metric_summary: dict[str, dict[str, float | int | None]] = {}
    for name in METRICS:
        values = [value for item in samples if (value := numeric(item["metrics"].get(name))) is not None]
        metric_summary[name] = {
            "count": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "mean": round(statistics.fmean(values), 3) if values else None,
            "min": round(min(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
        }

    reasons = Counter(str(item.get("detector_trigger_reason") or "none") for item in samples)
    output = Path(args.output) if args.output else Path("runtime/npu/benchmarks") / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.scenario}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": args.scenario,
        "started_at": started_at,
        "duration_seconds": round(time.time() - started_at, 3),
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "error_count": len(errors),
        "identity": samples[-1]["identity"] if samples else {},
        "metric_summary": metric_summary,
        "detector_trigger_reasons": dict(reasons),
        "errors": errors,
        "samples": samples,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("scenario", "sample_count", "error_count", "identity", "metric_summary", "detector_trigger_reasons")}, ensure_ascii=False, indent=2))
    print(f"output={output}")
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
