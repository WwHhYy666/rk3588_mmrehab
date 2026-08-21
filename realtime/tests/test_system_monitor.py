from __future__ import annotations

import random

from realtime.system_monitor import NpuDisplayEstimator, NpuPeakHold, apply_npu_display_offset


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


def test_npu_training_display_offset_adds_50_and_retains_actual_sample() -> None:
    actual = {
        "available": True,
        "percent": 5.0,
        "average_percent": 5.0,
        "cores": {"Core0": 4.0, "Core1": 6.0},
        "source": "/sys/kernel/debug/rknpu/load",
    }

    displayed = apply_npu_display_offset(actual, enabled=True, offset_percent=50.0)

    assert displayed["percent"] == 55.0
    assert displayed["average_percent"] == 55.0
    assert displayed["cores"] == {"Core0": 54.0, "Core1": 56.0}
    assert displayed["actual_percent"] == 5.0
    assert displayed["actual_cores"] == actual["cores"]
    assert displayed["display_offset_percent"] == 50.0
    assert displayed["estimated"] is True


def test_npu_training_display_offset_caps_at_100_and_is_noop_when_disabled() -> None:
    actual = {"available": True, "percent": 72.0, "cores": {"Core0": 80.0}}

    assert apply_npu_display_offset(actual, enabled=False) is actual
    displayed = apply_npu_display_offset(actual, enabled=True)

    assert displayed["percent"] == 100.0
    assert displayed["cores"] == {"Core0": 100.0}


def test_qwen_peak_hold_keeps_only_real_peak_and_expires() -> None:
    hold = NpuPeakHold(hold_seconds=15.0)
    idle = {"available": True, "percent": 0.0, "cores": {"Core0": 0.0}, "source": "real"}

    hold.start()
    hold.observe(idle)
    hold.observe({"available": True, "percent": 43.0, "cores": {"Core0": 43.0}, "source": "real"})
    hold.observe({"available": True, "percent": 18.0, "cores": {"Core0": 18.0}, "source": "real"})

    active = hold.apply(idle, enabled=True, now=10.0)
    assert active["percent"] == 43.0
    assert active["sample_mode"] == "qwen_actual_peak_active"
    assert active["estimated"] is False

    hold.finish(retain=True, now=10.0)
    retained = hold.apply(idle, enabled=True, now=20.0)
    assert retained["percent"] == 43.0
    assert retained["sample_mode"] == "qwen_actual_peak_hold"
    assert retained["hold_remaining_seconds"] == 5.0
    assert hold.apply(idle, enabled=True, now=26.0) is idle


def test_qwen_peak_hold_discards_structured_rule_answer() -> None:
    hold = NpuPeakHold(hold_seconds=15.0)
    idle = {"available": True, "percent": 0.0, "source": "real"}
    hold.start()
    hold.observe({"available": True, "percent": 51.0, "source": "real"})
    hold.finish(retain=False, now=10.0)

    assert hold.apply(idle, enabled=True, now=11.0) is idle
