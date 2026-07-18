from __future__ import annotations

import random

from training.system_monitor import NpuDisplayEstimator


def test_npu_display_estimator_uses_training_bands_and_keeps_actual_sample() -> None:
    estimator = NpuDisplayEstimator(random.Random(7))
    actual = {
        "available": True,
        "percent": 0.0,
        "average_percent": 0.0,
        "cores": {"Core0": 0.0},
        "source": "/sys/kernel/debug/rknpu/load",
    }

    visible = estimator.apply(actual, enabled=True, person_visible=True, now=10.0)
    assert 40.0 <= visible["percent"] <= 60.0
    assert visible["source"] == "simulated_cpu_training"
    assert visible["estimated"] is True
    assert visible["actual_percent"] == 0.0
    assert visible["actual_cores"] == {"Core0": 0.0}

    unchanged = estimator.apply(actual, enabled=True, person_visible=True, now=10.5)
    assert unchanged["percent"] == visible["percent"]

    moved = estimator.apply(actual, enabled=True, person_visible=True, now=11.5)
    assert 40.0 <= moved["percent"] <= 60.0
    assert abs(moved["percent"] - visible["percent"]) <= 3.0

    absent = estimator.apply(actual, enabled=True, person_visible=False, now=12.5)
    assert 18.0 <= absent["percent"] <= 24.0


def test_npu_display_estimator_returns_real_sample_when_disabled() -> None:
    estimator = NpuDisplayEstimator(random.Random(11))
    actual = {"available": True, "percent": 12.5, "source": "real"}

    assert estimator.apply(actual, enabled=False, person_visible=True, now=1.0) is actual
