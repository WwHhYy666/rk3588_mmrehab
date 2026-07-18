from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import pose_estimation.rknn_pose.pose_frame_adapter as pose_adapter_module

from pose_estimation.rknn_pose.pose_frame_adapter import (
    Coco17DisplayStabilizer,
    RknnPoseStabilizer,
    StablePersonSelector,
    adapt_rknn_pose_frame,
    compute_coco17_orientation_metrics,
)
from pose_estimation.rknn_pose.rknn_backend import RKNNPoseBackend
from pose_estimation.rknn_pose.yolov5n_rtmpose_backend import YOLOv5nRTMPoseBackend


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _detection(points: dict[int, tuple[float, float, float]]) -> dict[str, object]:
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, value in points.items():
        keypoints[index] = [float(value[0]), float(value[1]), float(value[2])]
    return {
        "bbox": [80.0, 20.0, 560.0, 460.0],
        "score": 0.95,
        "keypoints": keypoints,
    }


def _orientation_points(points: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {"x": float(x), "y": float(y), "visibility": float(visibility)}
        for name, (x, y, visibility) in points.items()
    }


def test_coco17_orientation_detects_front_view() -> None:
    metrics = compute_coco17_orientation_metrics(
        _orientation_points(
            {
                "left_shoulder": (0.35, 0.25, 0.9),
                "right_shoulder": (0.65, 0.25, 0.9),
                "left_hip": (0.42, 0.55, 0.9),
                "right_hip": (0.58, 0.55, 0.9),
            }
        ),
        {"person_count": 1, "side": "left"},
        0.18,
    )

    assert metrics["front_view_ok"] is True
    assert metrics["side_view_ok"] is False
    assert metrics["rknn_orientation_mode"] == "torso_ratio"
    assert float(metrics["orientation_ratio"]) >= 0.55


def test_coco17_orientation_detects_side_view() -> None:
    metrics = compute_coco17_orientation_metrics(
        _orientation_points(
            {
                "left_shoulder": (0.49, 0.25, 0.9),
                "right_shoulder": (0.53, 0.25, 0.9),
                "left_hip": (0.50, 0.55, 0.9),
                "right_hip": (0.52, 0.55, 0.9),
            }
        ),
        {"person_count": 1, "side": "left"},
        0.18,
    )

    assert metrics["front_view_ok"] is False
    assert metrics["side_view_ok"] is True
    assert metrics["orientation_ok"] is True
    assert float(metrics["orientation_ratio"]) <= 0.32


def test_coco17_orientation_rejects_intermediate_angle() -> None:
    metrics = compute_coco17_orientation_metrics(
        _orientation_points(
            {
                "left_shoulder": (0.43, 0.25, 0.9),
                "right_shoulder": (0.57, 0.25, 0.9),
                "left_hip": (0.45, 0.55, 0.9),
                "right_hip": (0.55, 0.55, 0.9),
            }
        ),
        {"person_count": 1, "side": "left"},
        0.18,
    )

    assert metrics["front_view_ok"] is False
    assert metrics["side_view_ok"] is False
    assert metrics["orientation_message"] == "adjust camera angle"


def test_coco17_orientation_uses_complete_near_side_chain_fallback() -> None:
    metrics = compute_coco17_orientation_metrics(
        _orientation_points(
            {
                "left_shoulder": (0.50, 0.20, 0.9),
                "left_hip": (0.51, 0.45, 0.9),
                "left_knee": (0.52, 0.68, 0.9),
                "right_shoulder": (0.54, 0.20, 0.02),
                "right_hip": (0.55, 0.45, 0.02),
            }
        ),
        {"person_count": 1, "side": "left"},
        0.18,
        {"side": "left"},
    )

    assert metrics["front_view_ok"] is False
    assert metrics["side_view_ok"] is True
    assert metrics["rknn_orientation_mode"] == "side_chain_fallback"


def test_orientation_uses_raw_points_when_action_stabilizer_holds_previous_pose() -> None:
    action_config = {
        "point_names": ["hip", "knee", "ankle"],
        "metric_kind": "angle",
        "angle_kind": "flexion",
        "target_joint": "knee",
    }
    selector = StablePersonSelector()
    stabilizer = RknnPoseStabilizer(alpha=0.35, low_conf_alpha=0.18, jump_scale=0.20)
    front = _detection(
        {
            5: (180, 100, 0.9),
            6: (460, 100, 0.9),
            11: (220, 240, 0.9),
            12: (420, 240, 0.9),
            13: (220, 360, 0.9),
            14: (420, 360, 0.9),
            15: (220, 450, 0.9),
            16: (420, 450, 0.9),
        }
    )
    side = _detection(
        {
            5: (310, 100, 0.9),
            6: (330, 100, 0.9),
            11: (312, 240, 0.9),
            12: (328, 240, 0.9),
            13: (312, 360, 0.9),
            14: (328, 360, 0.9),
            15: (312, 450, 0.9),
            16: (328, 450, 0.9),
        }
    )

    adapt_rknn_pose_frame(
        [front],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )
    adapted = adapt_rknn_pose_frame(
        [side],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )

    raw_x = float(adapted["orientation_rehab_keypoints"]["left_shoulder"]["x"])
    stabilized_x = float(adapted["rehab_keypoints"]["left_shoulder"]["x"])
    assert raw_x == 310 / 640
    assert stabilized_x != raw_x

    metrics = compute_coco17_orientation_metrics(
        adapted["orientation_rehab_keypoints"],
        adapted["selected_result"],
        0.18,
        adapted["selected_rule"],
    )
    assert metrics["side_view_ok"] is True
    assert metrics["front_view_ok"] is False


def test_idle_backend_does_not_load_rknn_models() -> None:
    backend = YOLOv5nRTMPoseBackend()
    backend.set_enabled_fn(lambda: False)
    result = backend.infer(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result.meta["npu_idle"] is True
    assert result.meta["npu_resource"]["state"] == "qwen_available"
    assert result.meta["npu_resource"]["models_loaded"] is False
    assert backend.det_model.rknn is None
    assert backend.pose_model.rknn is None


def test_low_latency_backend_settings_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("RKNN_DET_INTERVAL", "2")
    monkeypatch.setenv("RKNN_DET_CACHE_SECONDS", "0.6")
    monkeypatch.setenv("RKNN_DET_RETRY_SECONDS", "0.3")
    monkeypatch.setenv("RKNN_YOLOV5_BACKEND_DRAW", "0")
    monkeypatch.setenv("RKNN_YOLOV5_PERSON_ONLY_FAST", "1")
    monkeypatch.setenv("RKNN_CORE_MASK", "NPU_CORE_0_1_2")
    monkeypatch.setenv("RKNN_DET_CORE_MASK", "NPU_CORE_0")
    monkeypatch.setenv("RKNN_POSE_CORE_MASK", "NPU_CORE_1_2")

    backend = YOLOv5nRTMPoseBackend()

    assert backend.det_interval == 2
    assert backend.det_cache_seconds == 0.6
    assert backend.det_retry_seconds == 0.3
    assert backend.draw_enabled is False
    assert backend.person_only_fast is True
    assert backend.det_model.core_mask == "NPU_CORE_0"
    assert backend.pose_model.core_mask == "NPU_CORE_1_2"


def test_resource_snapshot_rejects_renamed_legacy_large_detector(tmp_path) -> None:
    detector = tmp_path / "renamed_detector.rknn"
    with detector.open("wb") as handle:
        handle.truncate(101 * 1024 * 1024)

    backend = YOLOv5nRTMPoseBackend(det_model_path=detector)
    resource = backend.resource_snapshot()

    assert resource["det_model_size_bytes"] == 101 * 1024 * 1024
    assert resource["det_model_deprecated"] is True


def test_8085_launch_defaults_use_stabilized_overlay_and_longer_grace() -> None:
    entrypoint = (PROJECT_ROOT / "rehab_app/server/npu_rehab_server.py").read_text(encoding="utf-8")
    launcher = (PROJECT_ROOT / "scripts/start_npu_rehab_8085.sh").read_text(encoding="utf-8")

    assert 'RKNN_STABILIZER_MAX_HOLD_FRAMES", "8"' in entrypoint
    assert 'RKNN_DET_INTERVAL", "3"' in entrypoint
    assert 'RKNN_DET_CACHE_SECONDS", "1.5"' in entrypoint
    assert 'RKNN_DET_RETRY_SECONDS", "0.25"' in entrypoint
    assert 'RKNN_YOLOV5_BACKEND_DRAW", "0"' in entrypoint
    assert 'RK_CAMERA_OPEN_MODE", "auto"' in entrypoint
    assert 'RKNN_PROCESS_WIDTH", "1280"' in entrypoint
    assert 'RKNN_PROCESS_HEIGHT", "720"' in entrypoint
    assert 'RKNN_STREAM_WIDTH", "960"' in entrypoint
    assert 'RKNN_STREAM_HEIGHT", "540"' in entrypoint
    assert 'RK_JPEG_QUALITY", "72"' in entrypoint
    assert 'RKNN_DISPLAY_MAX_HOLD_FRAMES", "4"' in entrypoint
    assert 'RKNN_DISPLAY_BBOX_HOLD_FRAMES", "6"' in entrypoint
    assert 'RKNN_STABILIZER_MAX_HOLD_FRAMES:-8' in launcher
    assert 'RKNN_DET_INTERVAL:-3' in launcher
    assert 'RKNN_DET_CACHE_SECONDS:-1.5' in launcher
    assert 'RKNN_DET_RETRY_SECONDS:-0.25' in launcher
    assert 'RKNN_DET_SCORE_THRES:-0.80' in launcher
    assert 'REHAB_ASSISTANT_TTS_GAIN:-1.35' in launcher
    assert 'RKNN_YOLOV5_BACKEND_DRAW:-0' in launcher
    assert 'RK_CAMERA_OPEN_MODE:-auto' in launcher
    assert 'RKNN_PROCESS_WIDTH:-1280' in launcher
    assert 'RKNN_PROCESS_HEIGHT:-720' in launcher
    assert 'RKNN_STREAM_WIDTH:-960' in launcher
    assert 'RKNN_STREAM_HEIGHT:-540' in launcher
    assert 'RK_JPEG_QUALITY:-72' in launcher


def test_npu_session_reports_cpu_aligned_logic_version_while_idle() -> None:
    source = (PROJECT_ROOT / "rehab_app" / "server" / "npu_rehab_server.py").read_text(encoding="utf-8")

    class_source = source.split("class NpuRealtimeTrainingSession", 1)[1].split("def configure_isolated_runtime", 1)[0]
    assert 'self.pose_backend = "rknn"' in class_source
    assert 'return "npu_training_v8_stage2_pipeline"' in class_source


def test_npu_session_resets_tracking_on_offscreen_reentry_and_action_transition() -> None:
    source = (PROJECT_ROOT / "rehab_app" / "server" / "npu_rehab_server.py").read_text(encoding="utf-8")
    class_source = source.split("class NpuRealtimeTrainingSession", 1)[1].split("def configure_isolated_runtime", 1)[0]

    assert "def _enter_offscreen_wait" in class_source
    assert 'reset_npu_tracking_state_for("offscreen_wait")' in class_source
    assert "def _resume_running_after_offscreen" in class_source
    assert 'reset_npu_tracking_state_for("offscreen_reentry")' in class_source
    assert "def _start_playlist_action" in class_source
    assert 'reset_npu_tracking_state_for(f"action_transition:{self.action_id}")' in class_source


def test_npu_return_gate_requires_stable_full_body_reentry() -> None:
    plan = yaml.safe_load((PROJECT_ROOT / "training/configs/rehab_demo_plan_npu.yaml").read_text(encoding="utf-8"))

    assert plan["rknn_return_confirm_frames"] == 2
    assert plan["npu_return_core_points_min"] == 5
    assert plan["npu_return_core_visibility_min"] == 0.12


def test_release_clears_both_runtime_handles() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    backend = YOLOv5nRTMPoseBackend()
    det_runtime = FakeRuntime()
    pose_runtime = FakeRuntime()
    backend.det_model.rknn = det_runtime
    backend.pose_model.rknn = pose_runtime

    backend.release()

    assert det_runtime.released is True
    assert pose_runtime.released is True
    assert backend.det_model.rknn is None
    assert backend.pose_model.rknn is None
    assert backend.resource_snapshot()["state"] == "qwen_available"


def test_rknn_dispatcher_selects_yolov5n_rtmpose(monkeypatch) -> None:
    monkeypatch.setenv("RKNN_POSE_PIPELINE", "yolov5n_rtmpose")
    monkeypatch.setenv("RKNN_DET_MODEL", "models/vision/yolov5n_raw_fp.rknn")
    monkeypatch.setenv("RKNN_RTMPOSE_MODEL", "models/vision/rtmpose_m_256x192_fp.rknn")

    backend = RKNNPoseBackend(keypoint_thres=0.18)

    assert backend.pipeline == "yolov5n_rtmpose"
    assert backend.backend is not None
    assert "yolov5n_raw_fp.rknn" in backend.model_path
    assert "rtmpose_m_256x192_fp.rknn" in backend.model_path


def test_coco17_hamstring_curl_uses_2d_hip_knee_ankle_angle() -> None:
    detection = _detection(
        {
            5: (180, 70, 0.9),
            11: (200, 120, 0.9),
            13: (200, 260, 0.9),
            15: (320, 260, 0.9),
        }
    )
    action_config = {
        "point_names": ["hip", "knee", "ankle"],
        "metric_kind": "angle",
        "angle_kind": "flexion",
        "target_joint": "knee",
    }

    adapted = adapt_rknn_pose_frame(
        [detection],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=StablePersonSelector(),
        visibility_threshold=0.18,
    )

    assert adapted["selected_result"]["valid"] is True
    assert adapted["selected_result"]["selected_source"] == "rknn_2d_image"
    assert 89.0 <= float(adapted["selected_result"]["selected_target_angle"]) <= 91.0


def test_coco17_seated_raise_uses_shoulder_hip_knee_ratio() -> None:
    detection = _detection(
        {
            5: (200, 80, 0.9),
            11: (200, 240, 0.9),
            13: (200, 160, 0.9),
        }
    )
    action_config = {
        "point_names": ["shoulder", "hip", "knee"],
        "metric_kind": "knee_raise_height_ratio",
        "angle_kind": "included",
        "target_joint": "hip",
    }

    adapted = adapt_rknn_pose_frame(
        [detection],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=StablePersonSelector(),
        visibility_threshold=0.18,
    )

    assert adapted["selected_result"]["valid"] is True
    assert 0.49 <= float(adapted["selected_result"]["selected_target_angle"]) <= 0.51


def test_npu_yaml_keeps_coco17_schema_and_scopes_seated_raise_return_guard() -> None:
    sit = yaml.safe_load((PROJECT_ROOT / "evaluation/configs/npu/sit_to_stand.yaml").read_text(encoding="utf-8"))
    curl = yaml.safe_load((PROJECT_ROOT / "evaluation/configs/npu/standing_hamstring_curl.yaml").read_text(encoding="utf-8"))
    raise_config = yaml.safe_load((PROJECT_ROOT / "evaluation/configs/npu/seated_knee_raise.yaml").read_text(encoding="utf-8"))

    assert sit["keypoint_rule"] == {"hip_index": 11, "knee_index": 13, "ankle_index": 15, "target_joint": "left_knee"}
    assert raise_config["keypoint_rule"]["shoulder_index"] == 5
    for action_id, npu_config in {
        "sit_to_stand": sit,
        "standing_hamstring_curl": curl,
        "seated_knee_raise": raise_config,
    }.items():
        cpu_config = yaml.safe_load((PROJECT_ROOT / f"evaluation/configs/{action_id}.yaml").read_text(encoding="utf-8-sig"))
        assert npu_config["thresholds"] == cpu_config["thresholds"]
        if action_id == "seated_knee_raise":
            expected_realtime = {
                **cpu_config["realtime"],
                "return_delta_from_rest": 0.05,
                "return_to_start_delta": 0.0,
                "stable_return_seconds": 0.30,
                "require_return_reversal": True,
                "return_reversal_confirm_frames": 4,
                "finish_confirm_frames": 4,
            }
            assert npu_config["realtime"] == expected_realtime
        elif action_id == "sit_to_stand":
            expected_realtime = {
                **cpu_config["realtime"],
                "start_delta_from_rest": 0.25,
                "start_delta": 0.25,
                "tut_range_padding": 0.05,
                "metric_median_window": 5,
                "start_pose_gate_enabled": True,
                "start_pose_confirm_frames": 4,
                "start_pose_visibility_min": 0.18,
                "start_pose_geometry_min": 55.0,
                "start_pose_geometry_max": 140.0,
                "start_pose_max_joint_motion": 0.035,
                "start_pose_prompt": "请先坐回椅子并保持坐稳",
                "rebaseline_each_rep": True,
                "post_rep_start_pose_confirm_frames": 4,
                "post_rep_ready_prompt": "可以开始下一次坐站",
            }
            assert npu_config["realtime"] == expected_realtime
        else:
            assert npu_config["realtime"] == cpu_config["realtime"]


def test_adapter_holds_complete_pose_dropout_for_stable_overlay() -> None:
    action_config = {
        "point_names": ["hip", "knee", "ankle"],
        "metric_kind": "angle",
        "angle_kind": "included",
        "target_joint": "knee",
    }
    selector = StablePersonSelector()
    stabilizer = RknnPoseStabilizer(max_hold_frames=2, lock_confirm_frames=1)
    detected = adapt_rknn_pose_frame(
        [_detection({11: (220, 180, 0.9), 13: (230, 300, 0.9), 15: (240, 420, 0.9)})],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )
    held = adapt_rknn_pose_frame(
        [],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )
    adapt_rknn_pose_frame(
        [],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )
    expired = adapt_rknn_pose_frame(
        [],
        frame_width=640,
        frame_height=480,
        action_config=action_config,
        side_mode="left",
        selector=selector,
        visibility_threshold=0.18,
        stabilizer=stabilizer,
    )

    assert detected["selected_result"]["valid"] is True
    assert held["selected_result"]["valid"] is True
    assert held["selected_result"]["pose_dropout_held"] is True
    assert held["selected_result"]["selected_source"] == "rknn_2d_held"
    assert held["rehab_keypoints"]["left_knee"]["visibility"] >= 0.18
    assert expired["selected_result"]["valid"] is False


def test_display_stabilizer_keeps_full_coco17_without_changing_training_points() -> None:
    points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    detection = _detection(points)
    display_stabilizer = Coco17DisplayStabilizer(max_hold_frames=2, jump_confirm_frames=2)
    adapted = adapt_rknn_pose_frame(
        [detection],
        frame_width=640,
        frame_height=480,
        action_config={
            "point_names": ["hip", "knee", "ankle"],
            "metric_kind": "angle",
            "angle_kind": "included",
            "target_joint": "knee",
        },
        side_mode="left",
        selector=StablePersonSelector(),
        visibility_threshold=0.18,
        stabilizer=RknnPoseStabilizer(lock_confirm_frames=1),
        display_stabilizer=display_stabilizer,
    )

    assert len(adapted["display_keypoints"]) == 17
    assert adapted["selected_result"]["display_keypoint_count"] == 17
    assert "nose" in adapted["display_keypoints"]
    assert "left_wrist" in adapted["display_keypoints"]
    assert "left_elbow" in adapted["display_keypoints"]
    assert len(adapted["rehab_keypoints"]) == 8
    assert "nose" not in adapted["rehab_keypoints"]
    assert adapted["selected_result"]["valid"] is True


def test_display_stabilizer_holds_weak_face_and_wrist_then_expires() -> None:
    clear_points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    weak_points = dict(clear_points)
    weak_points[0] = (130.0, 85.0, 0.04)
    weak_points[9] = (200.0, 190.0, 0.04)
    stabilizer = Coco17DisplayStabilizer(max_hold_frames=2, jump_confirm_frames=2)

    stabilizer.stabilize(_detection(clear_points), frame_width=640, frame_height=480)
    held, _, diagnostics = stabilizer.stabilize(_detection(weak_points), frame_width=640, frame_height=480)
    stabilizer.stabilize(None, frame_width=640, frame_height=480)
    stabilizer.stabilize(None, frame_width=640, frame_height=480)
    expired, _, expired_diagnostics = stabilizer.stabilize(None, frame_width=640, frame_height=480)

    assert {"nose", "left_wrist"}.issubset(set(diagnostics["display_held_keypoints"]))
    assert held["nose"]["visibility"] >= 0.05
    assert held["left_wrist"]["visibility"] >= 0.05
    assert expired["nose"]["visibility"] == 0.0
    assert expired["left_wrist"]["visibility"] == 0.0
    assert expired_diagnostics["display_keypoint_count"] == 0


def test_display_bbox_is_smoothed_held_and_then_released() -> None:
    stabilizer = Coco17DisplayStabilizer(bbox_alpha=0.5, bbox_hold_frames=2)
    first_detection = _detection({11: (220, 180, 0.9), 13: (230, 300, 0.9), 15: (240, 420, 0.9)})
    second_detection = dict(first_detection)
    second_detection["bbox"] = [100.0, 40.0, 580.0, 470.0]

    _, first_bbox, _ = stabilizer.stabilize(first_detection, frame_width=640, frame_height=480)
    _, second_bbox, second_diag = stabilizer.stabilize(second_detection, frame_width=640, frame_height=480)
    _, held_bbox, held_diag = stabilizer.stabilize(None, frame_width=640, frame_height=480)
    stabilizer.stabilize(None, frame_width=640, frame_height=480)
    _, expired_bbox, expired_diag = stabilizer.stabilize(None, frame_width=640, frame_height=480)

    assert first_bbox == [80.0, 20.0, 560.0, 460.0]
    assert second_bbox is not None and second_bbox[0] == 90.0
    assert second_diag["display_bbox_jump_pending"] is False
    assert held_bbox == second_bbox
    assert held_diag["display_bbox_held"] is True
    assert expired_bbox is None
    assert expired_diag["display_bbox_held"] is False


def test_display_stabilizer_accepts_second_consistent_fast_motion() -> None:
    base_points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    first_jump = dict(base_points)
    second_jump = dict(base_points)
    first_jump[9] = (520.0, 180.0, 0.9)
    second_jump[9] = (524.0, 184.0, 0.9)
    stabilizer = Coco17DisplayStabilizer(jump_scale=0.20, jump_confirm_frames=2)

    stabilizer.stabilize(_detection(base_points), frame_width=640, frame_height=480)
    held, _, first_diag = stabilizer.stabilize(_detection(first_jump), frame_width=640, frame_height=480)
    recovered, _, second_diag = stabilizer.stabilize(_detection(second_jump), frame_width=640, frame_height=480)

    assert "left_wrist" in first_diag["display_jump_pending"]
    assert held["left_wrist"]["x"] != first_jump[9][0] / 640.0
    assert "left_wrist" in second_diag["display_jump_recovered"]
    assert recovered["left_wrist"]["x"] == second_jump[9][0] / 640.0


def test_display_time_windows_hold_points_and_bbox_then_expire(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(pose_adapter_module.time, "monotonic", lambda: clock[0])
    points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    stabilizer = Coco17DisplayStabilizer(hold_seconds=0.25, bbox_hold_seconds=0.50)

    stabilizer.stabilize(_detection(points), frame_width=640, frame_height=480)
    clock[0] = 10.10
    held_points, held_bbox, held_diag = stabilizer.stabilize(None, frame_width=640, frame_height=480)
    clock[0] = 10.36
    expired_points, still_held_bbox, _ = stabilizer.stabilize(None, frame_width=640, frame_height=480)
    clock[0] = 10.61
    _, expired_bbox, expired_diag = stabilizer.stabilize(None, frame_width=640, frame_height=480)

    assert held_points["nose"]["visibility"] >= 0.05
    assert held_bbox is not None and held_diag["display_bbox_held"] is True
    assert expired_points["nose"]["visibility"] == 0.0
    assert still_held_bbox is not None
    assert expired_bbox is None and expired_diag["display_bbox_held"] is False


def test_display_reset_clears_time_based_hold_state(monkeypatch) -> None:
    clock = [20.0]
    monkeypatch.setattr(pose_adapter_module.time, "monotonic", lambda: clock[0])
    points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    stabilizer = Coco17DisplayStabilizer(hold_seconds=0.25, bbox_hold_seconds=0.50)

    stabilizer.stabilize(_detection(points), frame_width=640, frame_height=480)
    stabilizer.reset()
    held_points, held_bbox, diagnostics = stabilizer.stabilize(None, frame_width=640, frame_height=480)

    assert diagnostics["display_keypoint_count"] == 0
    assert held_bbox is None
    assert all(point["visibility"] == 0.0 for point in held_points.values())


def test_display_confidence_hysteresis_avoids_threshold_flicker(monkeypatch) -> None:
    clock = [30.0]
    monkeypatch.setattr(pose_adapter_module.time, "monotonic", lambda: clock[0])
    clear = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    marginal = dict(clear)
    marginal[0] = (124.0, 82.0, 0.08)

    fresh = Coco17DisplayStabilizer(disappear_threshold_ratio=0.65)
    fresh_points, _, _ = fresh.stabilize(_detection(marginal), frame_width=640, frame_height=480)
    assert fresh_points["nose"]["visibility"] == 0.0

    tracked = Coco17DisplayStabilizer(disappear_threshold_ratio=0.65)
    tracked.stabilize(_detection(clear), frame_width=640, frame_height=480)
    clock[0] = 30.05
    tracked_points, _, diagnostics = tracked.stabilize(_detection(marginal), frame_width=640, frame_height=480)
    assert tracked_points["nose"]["visibility"] == 0.08
    assert "nose" not in diagnostics["display_held_keypoints"]


def test_display_clears_all_stale_pose_after_500ms(monkeypatch) -> None:
    clock = [40.0]
    monkeypatch.setattr(pose_adapter_module.time, "monotonic", lambda: clock[0])
    points = {index: (120.0 + index * 8.0, 80.0 + index * 12.0, 0.9) for index in range(17)}
    stabilizer = Coco17DisplayStabilizer(hold_seconds=1.0, bbox_hold_seconds=1.0, max_stale_seconds=0.50)

    stabilizer.stabilize(_detection(points), frame_width=640, frame_height=480)
    clock[0] = 40.51
    stale_points, stale_bbox, diagnostics = stabilizer.stabilize(None, frame_width=640, frame_height=480)

    assert diagnostics["display_pose_stale"] is True
    assert diagnostics["display_keypoint_count"] == 0
    assert stale_bbox is None


def test_stabilizer_holds_weak_keypoint_briefly_to_avoid_display_flash() -> None:
    stabilizer = RknnPoseStabilizer(
        alpha=0.55,
        low_conf_alpha=0.32,
        jump_scale=0.55,
        max_hold_frames=3,
        lock_confirm_frames=1,
    )
    action_config = {
        "point_names": ["hip", "knee", "ankle"],
        "metric_kind": "angle",
        "angle_kind": "included",
        "target_joint": "knee",
    }
    clear = {
        "left_hip": {"x": 0.40, "y": 0.35, "visibility": 0.9},
        "left_knee": {"x": 0.42, "y": 0.60, "visibility": 0.9},
        "left_ankle": {"x": 0.44, "y": 0.85, "visibility": 0.9},
    }
    stabilizer.stabilize(
        clear,
        side_mode="left",
        action_config=action_config,
        visibility_threshold=0.18,
    )
    weak = {name: dict(point) for name, point in clear.items()}
    weak["left_knee"] = {"x": 0.46, "y": 0.61, "visibility": 0.10}

    stabilized, diagnostics = stabilizer.stabilize(
        weak,
        side_mode="left",
        action_config=action_config,
        visibility_threshold=0.18,
    )

    assert "left_knee" in diagnostics["held_keypoints"]
    assert stabilized["left_knee"]["visibility"] >= 0.18
    assert 0.42 < stabilized["left_knee"]["x"] < 0.46


def test_stabilizer_accepts_second_consistent_large_jump_instead_of_freezing() -> None:
    stabilizer = RknnPoseStabilizer(
        alpha=0.55,
        low_conf_alpha=0.32,
        jump_scale=0.20,
        max_hold_frames=3,
        lock_confirm_frames=1,
    )
    action_config = {
        "point_names": ["hip", "knee", "ankle"],
        "metric_kind": "angle",
        "angle_kind": "included",
        "target_joint": "knee",
    }
    clear = {
        "left_hip": {"x": 0.30, "y": 0.30, "visibility": 0.9},
        "left_knee": {"x": 0.32, "y": 0.55, "visibility": 0.9},
        "left_ankle": {"x": 0.34, "y": 0.82, "visibility": 0.9},
    }
    stabilizer.stabilize(
        clear,
        side_mode="left",
        action_config=action_config,
        visibility_threshold=0.18,
    )
    first_jump = {name: dict(point) for name, point in clear.items()}
    first_jump["left_hip"] = {"x": 0.30, "y": 0.52, "visibility": 0.9}
    held, first_diag = stabilizer.stabilize(
        first_jump,
        side_mode="left",
        action_config=action_config,
        visibility_threshold=0.18,
    )
    assert held["left_hip"]["y"] == clear["left_hip"]["y"]
    assert "left_hip" in first_diag["jump_pending"]

    second_jump = {name: dict(point) for name, point in clear.items()}
    second_jump["left_hip"] = {"x": 0.30, "y": 0.54, "visibility": 0.9}
    recovered, second_diag = stabilizer.stabilize(
        second_jump,
        side_mode="left",
        action_config=action_config,
        visibility_threshold=0.18,
    )
    assert recovered["left_hip"]["y"] == 0.54
    assert "left_hip" in second_diag["jump_recovery_accepted"]
    assert "left_hip" not in second_diag["jump_pending"]
