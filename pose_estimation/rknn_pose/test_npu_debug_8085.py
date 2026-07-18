from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_npu_entry(monkeypatch):
    fake_app = types.ModuleType("rehab_app.server.rehab_http_server")
    fake_app.PrescriptionHTTPHandler = object
    fake_result_storage = types.ModuleType("rehab_app.services.result_storage")
    fake_result_storage.save_prescription_artifacts = lambda *args, **kwargs: None
    fake_training = types.ModuleType("training.training_session")

    class FakeTrainingSession:
        pass

    fake_training.RealtimeTrainingSession = FakeTrainingSession
    monkeypatch.setitem(sys.modules, "rehab_app.server.rehab_http_server", fake_app)
    monkeypatch.setitem(sys.modules, "rehab_app.services.result_storage", fake_result_storage)
    monkeypatch.setitem(sys.modules, "training.training_session", fake_training)

    path = PROJECT_ROOT / "rehab_app/server/npu_rehab_server.py"
    spec = importlib.util.spec_from_file_location("test_npu_rehab_8085_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_npu_debug_page_is_button_activated(monkeypatch) -> None:
    module = load_npu_entry(monkeypatch)
    html = module.build_npu_debug_page().decode("utf-8")

    assert "开始 NPU 检测" in html
    assert "/api/npu/debug/start" in html
    assert "/api/npu/debug/heartbeat" in html
    assert "/api/npu/debug/stop" in html
    assert "页面打开不会自动占用 NPU" in html


def test_npu_debug_lease_expires_and_release_is_called(monkeypatch) -> None:
    module = load_npu_entry(monkeypatch)

    class FakeRuntime:
        def __init__(self) -> None:
            self.release_count = 0

        def release(self) -> None:
            self.release_count += 1

    runtime = FakeRuntime()
    module.app.state = types.SimpleNamespace(is_recording=False)
    module.app.realtime_session = types.SimpleNamespace(snapshot=lambda: {"status": "idle"})
    module.app.ACTIVE_REALTIME_STATUSES = set()
    module.app.rknn_backend = runtime
    monkeypatch.setattr(module, "qwen_busy", lambda: False)

    started = module.start_npu_debug()
    assert started["ok"] is True
    assert module.npu_debug_active() is True

    module._npu_debug_last_heartbeat_monotonic -= module.NPU_DEBUG_LEASE_SECONDS + 1.0
    assert module.npu_debug_active() is False

    module.start_npu_debug()
    stopped = module.stop_npu_debug()
    assert stopped["ok"] is True
    assert runtime.release_count == 1


def test_npu_debug_rejects_qwen_busy(monkeypatch) -> None:
    module = load_npu_entry(monkeypatch)
    module.app.state = types.SimpleNamespace(is_recording=False)
    module.app.realtime_session = types.SimpleNamespace(snapshot=lambda: {"status": "idle"})
    module.app.ACTIVE_REALTIME_STATUSES = set()
    monkeypatch.setattr(module, "qwen_busy", lambda: True)

    result = module.start_npu_debug()

    assert result["ok"] is False
    assert "小爱正在生成回答" in result["error"]
