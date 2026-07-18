from __future__ import annotations

import threading

from training.training_session import RealtimeTrainingSession


def make_session_stub() -> RealtimeTrainingSession:
    session = RealtimeTrainingSession.__new__(RealtimeTrainingSession)
    session.lock = threading.RLock()
    session.session_id = "session-a"
    session.action_id = "sit_to_stand"
    session.action_generation = 3
    session.machine = None
    session.eval_config = {"metric_direction": "increase"}
    session.metric_info = {"metric_name": "hip_height", "metric_unit": "ratio"}
    session._keyframe_candidate = None
    session._keyframe_job_sink = None
    session._pending_keyframe_jobs = {}
    session.keyframes = []
    session.keyframe_errors = []
    session.current_action_meta = {"action_name": "坐站训练"}
    return session


def test_async_keyframe_rejects_stale_action_identity() -> None:
    session = make_session_stub()

    accepted = session.accept_async_keyframe_candidate(
        {"target_angle_smoothed": 0.7},
        b"jpeg",
        session_id="session-a",
        action_id="standing_hamstring_curl",
        machine_state="RISING",
    )

    assert accepted is False
    assert session._keyframe_candidate is None


def test_async_keyframe_keeps_best_metric_for_current_action() -> None:
    session = make_session_stub()

    first = session.accept_async_keyframe_candidate(
        {"target_angle_smoothed": 0.7, "frame_index": 10},
        b"jpeg-a",
        session_id="session-a",
        action_id="sit_to_stand",
        machine_state="RISING",
    )
    lower = session.accept_async_keyframe_candidate(
        {"target_angle_smoothed": 0.6, "frame_index": 11},
        b"jpeg-b",
        session_id="session-a",
        action_id="sit_to_stand",
        machine_state="HOLDING",
    )
    higher = session.accept_async_keyframe_candidate(
        {"target_angle_smoothed": 0.8, "frame_index": 12},
        b"jpeg-c",
        session_id="session-a",
        action_id="sit_to_stand",
        machine_state="RETURNING",
    )

    assert first is True and lower is True and higher is True
    assert session._keyframe_candidate["image_jpeg"] == b"jpeg-c"
    assert session._keyframe_candidate["frame_index"] == 12


def test_bgr_candidate_is_copied_only_when_metric_improves() -> None:
    session = make_session_stub()
    factory_calls: list[str] = []

    first = session.offer_keyframe_candidate(
        {"target_angle_smoothed": 0.7, "frame_index": 10},
        lambda: factory_calls.append("first") or "frame-a",
        session_id="session-a",
        action_id="sit_to_stand",
        action_generation=3,
        machine_state="RISING",
    )
    lower = session.offer_keyframe_candidate(
        {"target_angle_smoothed": 0.6, "frame_index": 11},
        lambda: factory_calls.append("lower") or "frame-b",
        session_id="session-a",
        action_id="sit_to_stand",
        action_generation=3,
        machine_state="HOLDING",
    )
    higher = session.offer_keyframe_candidate(
        {"target_angle_smoothed": 0.8, "frame_index": 12},
        lambda: factory_calls.append("higher") or "frame-c",
        session_id="session-a",
        action_id="sit_to_stand",
        action_generation=3,
        machine_state="RETURNING",
    )

    assert first is True and lower is False and higher is True
    assert factory_calls == ["first", "higher"]
    assert session._keyframe_candidate["image_frame"] == "frame-c"


def test_rep_close_creates_one_generation_scoped_write_job(tmp_path, monkeypatch) -> None:
    session = make_session_stub()
    captured: list[dict] = []
    session.set_keyframe_job_sink(lambda job: captured.append(job) or True)
    session._keyframe_candidate = {
        "image_frame": "frame-a",
        "signal_value": 0.8,
        "selected_side": "left",
        "rehab_keypoints": {},
    }
    monkeypatch.setattr("training.training_session.KEYFRAMES_DIR", tmp_path)

    keyframe = session._save_keyframe_candidate(1)

    assert keyframe is not None and keyframe["write_status"] == "pending"
    assert len(captured) == 1
    assert captured[0]["token"] == "session-a:sit_to_stand:3:1"
    captured[0]["keyframe"].update({"write_status": "complete"})
    captured[0]["event"].set()
    session._wait_for_keyframe_jobs(max_seconds=0.01)
    assert session._pending_keyframe_jobs == {}


def test_keyframe_timeout_cancels_late_worker_result(tmp_path, monkeypatch) -> None:
    session = make_session_stub()
    captured: list[dict] = []
    session.set_keyframe_job_sink(lambda job: captured.append(job) or True)
    session._keyframe_candidate = {
        "image_frame": "frame-a",
        "signal_value": 0.8,
        "selected_side": "left",
        "rehab_keypoints": {},
    }
    monkeypatch.setattr("training.training_session.KEYFRAMES_DIR", tmp_path)

    keyframe = session._save_keyframe_candidate(1)
    session._wait_for_keyframe_jobs(max_seconds=0.0)

    assert keyframe is not None and keyframe["write_status"] == "timeout"
    assert captured[0]["accept_result"] is False


def test_current_action_keyframes_are_generation_scoped() -> None:
    session = make_session_stub()
    session.keyframes = [
        {"action_id": "sit_to_stand", "action_generation": 2, "rep_index": 1},
        {"action_id": "sit_to_stand", "action_generation": 3, "rep_index": 1},
        {"action_id": "seated_knee_raise", "action_generation": 3, "rep_index": 1},
    ]

    assert session._current_action_keyframes() == [
        {"action_id": "sit_to_stand", "action_generation": 3, "rep_index": 1}
    ]
