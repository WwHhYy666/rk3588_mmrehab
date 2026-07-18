from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import time

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.training_session import RealtimeTrainingSession
from training.tts_worker import TTSWorker


def test_new_correction_drops_old_correction() -> None:
    worker = TTSWorker(use_real_tts=False)

    assert worker.speak("再坚持 2 秒", priority="high", event_type="correction") is True
    assert worker.speak("腿再伸直一点", priority="high", event_type="correction") is True

    assert worker._queue.qsize() == 1
    item = worker._queue.get_nowait()
    assert item.event_type == "correction"
    assert item.text == "腿再伸直一点"


def test_count_drops_pending_correction() -> None:
    worker = TTSWorker(use_real_tts=False)

    assert worker.speak("腿再伸直一点", priority="high", event_type="correction") is True
    assert worker.speak("一", priority="high", event_type="rep_count", phrase_key="count_1") is True

    assert worker._queue.qsize() == 1
    item = worker._queue.get_nowait()
    assert item.event_type == "rep_count"
    assert item.text == "一"


def test_fixed_audio_file_is_preferred_when_present() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        audio_dir = root / "audio"
        audio_dir.mkdir()
        wav = audio_dir / "welcome.wav"
        wav.write_bytes(b"RIFF0000WAVE")
        config = root / "tts_phrases.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "audio_root": str(audio_dir),
                    "phrases": {"welcome": {"text": "请坐稳，准备开始。", "file": "welcome.wav"}},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        worker = TTSWorker(use_real_tts=False, phrase_config_path=config, project_root=root)
        worker._load_phrase_catalog()

        assert worker.speak("请坐稳，准备开始。", priority="high", event_type="welcome", phrase_key="welcome") is True
        item = worker._queue.get_nowait()
        assert item.audio_path == wav
        with patch("training.audio_output.shutil.which", return_value="/usr/bin/aplay"), patch(
            "training.audio_output.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")
        ):
            assert worker._play_audio_file(item.audio_path) is True
        assert worker.fixed_audio_hits == 1


def test_missing_fixed_audio_falls_back_to_tts_payload() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config = root / "tts_phrases.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "audio_root": str(root / "audio"),
                    "phrases": {"welcome": {"text": "请坐稳，准备开始。", "file": "missing.wav"}},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        worker = TTSWorker(use_real_tts=False, phrase_config_path=config, project_root=root)
        worker._load_phrase_catalog()

        assert worker.speak("请坐稳，准备开始。", priority="high", event_type="welcome", phrase_key="welcome") is True
        item = worker._queue.get_nowait()
        assert item.audio_path is None
    assert worker.fixed_audio_misses == 1


def test_fixed_audio_only_rejects_unrecorded_phrase_without_tts_fallback() -> None:
    worker = TTSWorker(
        use_real_tts=True,
        lazy_real_tts_init=True,
        fixed_audio_only=True,
        phrase_config_path=PROJECT_ROOT / "training" / "configs" / "tts_phrases.yaml",
        project_root=PROJECT_ROOT,
    )
    worker.start()
    try:
        assert worker.speak("请回到起始位置站稳", priority="high", event_type="resume") is False
        snapshot = worker.snapshot()
        assert snapshot["fixed_audio_only"] is True
        assert snapshot["queued"] == 0
        assert snapshot["real_tts_initialized"] is False
    finally:
        worker.stop()


def test_tts_worker_reports_queue_to_start_timing() -> None:
    worker = TTSWorker(
        use_real_tts=False,
        phrase_config_path=PROJECT_ROOT / "training" / "configs" / "tts_phrases.yaml",
        project_root=PROJECT_ROOT,
    )
    worker.start()
    try:
        assert worker.speak("一", priority="high", event_type="rep_count", phrase_key="count_1") is True
        deadline = time.time() + 1.0
        snapshot = worker.snapshot()
        while snapshot.get("last_started_event_type") != "rep_count" and time.time() < deadline:
            time.sleep(0.01)
            snapshot = worker.snapshot()
        assert snapshot["last_queued_event_type"] == "rep_count"
        assert snapshot["last_started_event_type"] == "rep_count"
        assert snapshot["last_queue_delay_seconds"] is not None
        assert snapshot["last_queue_delay_seconds"] < 0.4
    finally:
        worker.stop()


def test_stopped_worker_still_reports_an_inflight_audio_process_as_busy() -> None:
    worker = TTSWorker(use_real_tts=False)
    worker._running = False
    worker._speaking = True

    assert worker.is_busy(extra_guard_seconds=0.3) is True



def test_clear_pending_removes_selected_event_types() -> None:
    worker = TTSWorker(use_real_tts=False)

    assert worker.speak("请侧身对准镜头。", priority="high", event_type="orientation", phrase_key="orientation") is True
    assert worker.speak("开始坐姿伸膝。", priority="high", event_type="action_start", phrase_key="start_seated_knee_extension") is True
    assert worker.speak("一", priority="high", event_type="rep_count", phrase_key="count_1") is True

    assert worker.clear_pending({"orientation", "action_start"}) == 2
    assert worker._queue.qsize() == 1
    item = worker._queue.get_nowait()
    assert item.event_type == "rep_count"


def test_training_phrase_catalog_has_return_and_angle_audio() -> None:
    worker = TTSWorker(use_real_tts=False, phrase_config_path=PROJECT_ROOT / "training" / "configs" / "tts_phrases.yaml", project_root=PROJECT_ROOT)
    worker._load_phrase_catalog()

    assert worker.speak("角度正确，我们继续训练吧", priority="high", event_type="orientation", phrase_key="angle_right") is True
    assert worker._queue.get_nowait().audio_path.name == "angle_right.wav"
    assert worker.speak("回到画面了，我们继续", priority="high", event_type="action_start", phrase_key="inscreen") is True
    assert worker._queue.get_nowait().audio_path.name == "inscreen.wav"
def test_training_session_suppresses_immediate_second_correction() -> None:
    session = RealtimeTrainingSession()
    session.current_realtime_config = {"min_rep_seconds": 1.0}
    session.correction_tts_interval_seconds = 4.0

    assert session._should_speak_correction("TUT_LOW", {"duration_seconds": 2.0}) is True
    assert session._should_speak_correction("ROM_LOW", {"duration_seconds": 2.0}) is False


def test_training_session_does_not_hide_correction_for_pose_jitter() -> None:
    session = RealtimeTrainingSession()
    session.current_realtime_config = {"min_rep_seconds": 1.0}

    assert session._should_speak_correction("ROM_LOW", {"duration_seconds": 2.0}) is True


def test_training_session_filters_process_prompts_from_main_ui() -> None:
    session = RealtimeTrainingSession()

    assert session._display_prompt_from_machine({"prompt": "准备开始下一遍"}) == ""
    assert session._display_prompt_from_machine({"prompt": "请保持目标腿关键点可见"}) == ""
    assert session._display_prompt_from_machine({"prompt": "腿再伸直一点"}) == "腿再伸直一点"



class _BusyOnceTTS:
    def __init__(self) -> None:
        self.busy = True
        self.calls = 0
        self.spoken: list[tuple[str, str | None, str | None]] = []

    def is_busy(self, *, extra_guard_seconds: float = 0.0) -> bool:
        self.calls += 1
        return self.busy

    def speak(self, text: str, priority: str = "low", event_type: str | None = None, phrase_key: str | None = None) -> bool:
        self.spoken.append((text, event_type, phrase_key))
        return True


def test_training_session_waits_for_rep_feedback_audio_before_resuming() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _BusyOnceTTS()
    session.tts_worker = fake_tts
    session.status = "running"
    session.start_time = 1.0
    session.action_start_guard_seconds = 0.0

    session._enter_feedback_wait()
    assert session.status == "awaiting_rep_feedback"
    assert session.pending_feedback_resume is True

    session._maybe_resume_after_feedback()
    assert session.status == "awaiting_rep_feedback"
    assert session.pending_feedback_resume is True
    assert fake_tts.calls == 1

    fake_tts.busy = False
    session._maybe_resume_after_feedback()
    assert session.status == "running"
    assert session.pending_feedback_resume is False
    assert session.pause_reason is None



def test_training_session_forces_left_side_and_start_only_orientation() -> None:
    session = RealtimeTrainingSession()

    session.side_mode = "right"
    session.playlist_mode = True
    session.playlist_index = 0
    session.orientation_required = True
    session.initial_orientation_done = False
    assert session._needs_initial_orientation_gate() is True

    session.side_mode = "auto"
    session.initial_orientation_done = True
    session.playlist_index = 1
    assert session._needs_initial_orientation_gate() is False
    assert session._selected_side_snapshot() == "left"
def test_training_session_orientation_requires_front_then_side() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _BusyOnceTTS()
    fake_tts.busy = False
    session.tts_worker = fake_tts
    session.orientation_required = True
    session.orientation_confirm_frames = 2
    session.rknn_orientation_confirm_frames = 2
    session._enter_orientation_wait(speak=False)

    front = {"pose_detected": True, "person_visible": True, "front_view_ok": True, "side_view_ok": False, "orientation_ok": False}
    session._process_orientation_gate(True, False, front)
    assert session.status == "awaiting_orientation"
    assert session.orientation_phase == "awaiting_front"
    session._process_orientation_gate(True, False, front)
    assert session.orientation_phase == "awaiting_side"
    assert fake_tts.spoken[-1] == ("请侧身对准镜头。", "orientation", "orientation")

    side = {"pose_detected": True, "person_visible": True, "front_view_ok": False, "side_view_ok": True, "orientation_ok": True}
    session._process_orientation_gate(True, True, side)
    assert session.status == "awaiting_orientation"
    session._process_orientation_gate(True, True, side)
    assert session.status == "running"


def test_rknn_orientation_uses_two_front_frames_then_four_side_frames() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _BusyOnceTTS()
    fake_tts.busy = False
    session.tts_worker = fake_tts
    session.orientation_required = True
    session.rknn_front_orientation_confirm_frames = 2
    session.rknn_orientation_confirm_frames = 4
    session._enter_orientation_wait(speak=False)

    front = {
        "actual_backend": "rknn",
        "pose_detected": True,
        "person_visible": True,
        "front_view_ok": True,
        "side_view_ok": False,
        "orientation_ok": False,
    }
    for _ in range(2):
        session._process_orientation_gate(True, False, front)

    assert session.orientation_phase == "awaiting_side"
    orientation_prompts = [item for item in fake_tts.spoken if item[1] == "orientation" and item[2] == "orientation"]
    assert orientation_prompts == [("请侧身对准镜头。", "orientation", "orientation")]

    side = {
        "actual_backend": "rknn",
        "pose_detected": True,
        "person_visible": True,
        "front_view_ok": False,
        "side_view_ok": True,
        "orientation_ok": True,
    }
    for _ in range(3):
        session._process_orientation_gate(True, True, side)
        assert session.status == "awaiting_orientation"
    session._process_orientation_gate(True, True, side)

    assert session.status == "running"
    assert session.initial_orientation_done is True
    orientation_prompts = [item for item in fake_tts.spoken if item[1] == "orientation" and item[2] == "orientation"]
    assert len(orientation_prompts) == 1


def test_training_session_offscreen_waits_for_audio_idle_and_stable_return() -> None:
    session = RealtimeTrainingSession()
    fake_tts = _BusyOnceTTS()
    fake_tts.busy = True
    session.tts_worker = fake_tts
    session.orientation_required = True
    session.orientation_confirm_frames = 2
    session._enter_offscreen_wait()
    assert session.status == "awaiting_return"
    assert session.offscreen_prompt_pending is True
    assert fake_tts.spoken == []

    frame = {"pose_detected": True, "person_visible": True, "side_view_ok": True, "orientation_ok": True, "action_keypoints_valid": True, "target_angle_smoothed": 0.2}
    session._process_return_gate(True, True, frame)
    assert session.status == "awaiting_return"
    fake_tts.busy = False
    session._maybe_speak_pending_offscreen_prompt()
    assert fake_tts.spoken[-1] == ("请回到画面中。", "offscreen", "offscreen")
    session._process_return_gate(True, True, frame)
    assert session.status == "awaiting_return"
    session._process_return_gate(True, True, frame)
    assert session.status == "running"


def test_recording_pipeline_keeps_real_camera_coordinates_and_holds_jump_frames() -> None:
    source = (PROJECT_ROOT / "rehab_app" / "server" / "rehab_http_server.py").read_text(encoding="utf-8")
    assert "frame = cv2.flip(frame, 1)" not in source
    assert "mirror only at display layer" in source
    assert "jump_rejected_hold" in source
    jump_branch = source.split('if selected_result.get("target_leg_jump_rejected"):', 1)[1].split('    else:', 1)[0]
    assert "state.smoother.values[-1]" in jump_branch
    assert "state.smoother.update(raw_flexion)" not in jump_branch


def test_recording_pipeline_uses_stricter_left_leg_hold_for_overlay() -> None:
    source = (PROJECT_ROOT / "rehab_app" / "server" / "rehab_http_server.py").read_text(encoding="utf-8")
    assert 'MEDIAPIPE_TARGET_LEG_MAX_JUMP", "0.10"' in source
    assert "def hold_target_leg_rehab_keypoints" in source
    assert "hold_target_leg_rehab_keypoints(rehab_keypoints, selected_rule, held_points)" in source
    assert "draw_threshold = max(0.05, float(visibility_threshold))" in source

if __name__ == "__main__":
    test_new_correction_drops_old_correction()
    test_count_drops_pending_correction()
    test_fixed_audio_file_is_preferred_when_present()
    test_missing_fixed_audio_falls_back_to_tts_payload()
    test_clear_pending_removes_selected_event_types()
    test_training_phrase_catalog_has_return_and_angle_audio()
    test_training_session_suppresses_immediate_second_correction()
    test_training_session_does_not_hide_correction_for_pose_jitter()
    test_training_session_filters_process_prompts_from_main_ui()
    test_training_session_waits_for_rep_feedback_audio_before_resuming()
    test_training_session_forces_left_side_and_start_only_orientation()
    test_training_session_orientation_requires_front_then_side()
    test_training_session_offscreen_waits_for_audio_idle_and_stable_return()
    test_recording_pipeline_keeps_real_camera_coordinates_and_holds_jump_frames()
    test_recording_pipeline_uses_stricter_left_leg_hold_for_overlay()
    print("tts worker tests passed")
