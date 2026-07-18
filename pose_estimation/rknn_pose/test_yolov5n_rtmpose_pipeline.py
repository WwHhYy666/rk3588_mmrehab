from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pose_estimation.rknn_pose.yolov5n_rtmpose_backend import (
    LetterboxMeta,
    RKNNModel,
    RestoreInfo,
    YOLOv5nRTMPoseBackend,
    adaptive_detector_decision,
    adaptive_pose_x_scale,
    aspect_bbox,
    bound_pose_bbox_for_frame,
    decode_simcc,
    expand_bbox_for_pose,
    postprocess_yolov5_person,
    tracker_bbox_from_keypoints,
)


def test_pose_bbox_keeps_base_expansion_for_normal_upright_person() -> None:
    bbox = np.asarray([100, 100, 300, 500], dtype=np.float32)

    expanded = expand_bbox_for_pose(
        bbox,
        (720, 1280, 3),
        scale=1.25,
        top_expand=0.0,
        wide_ratio=0.65,
        wide_scale=1.50,
    )

    np.testing.assert_allclose(expanded, [75, 50, 325, 550], atol=1e-5)
    assert adaptive_pose_x_scale(bbox, base_scale=1.25, wide_ratio=0.65, wide_scale=1.50) == pytest.approx(1.25)


def test_pose_bbox_adds_horizontal_context_smoothly_for_arms_out() -> None:
    bbox = np.asarray([100, 100, 500, 400], dtype=np.float32)
    transition_bbox = np.asarray([100, 100, 425, 500], dtype=np.float32)

    expanded = expand_bbox_for_pose(
        bbox,
        (720, 1280, 3),
        scale=1.25,
        top_expand=0.0,
        wide_ratio=0.65,
        wide_scale=1.50,
    )

    np.testing.assert_allclose(expanded, [0, 62.5, 600, 437.5], atol=1e-5)
    assert adaptive_pose_x_scale(bbox, base_scale=1.25, wide_ratio=0.65, wide_scale=1.50) == pytest.approx(1.50)
    assert adaptive_pose_x_scale(transition_bbox, base_scale=1.25, wide_ratio=0.65, wide_scale=1.50) == pytest.approx(1.375)


def test_pose_crop_preserves_padding_and_rtmpose_aspect_ratio_at_frame_edge() -> None:
    bbox = np.asarray([0, 100, 400, 400], dtype=np.float32)
    expanded = expand_bbox_for_pose(
        bbox,
        (720, 1280, 3),
        scale=1.25,
        top_expand=0.0,
        wide_ratio=0.65,
        wide_scale=1.50,
    )
    crop = aspect_bbox(expanded, (720, 1280, 3), (192, 256))

    assert expanded[0] < 0
    assert crop[0] <= expanded[0] and crop[1] <= expanded[1]
    assert crop[2] >= expanded[2] and crop[3] >= expanded[3]
    assert (crop[2] - crop[0]) / (crop[3] - crop[1]) == pytest.approx(192 / 256)


def make_raw_outputs(*, layout: str = "nchw") -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for grid in (80, 40, 20):
        cls = np.full((1, 240, grid, grid), -20.0, dtype=np.float32)
        bbox = np.zeros((1, 12, grid, grid), dtype=np.float32)
        obj = np.full((1, 3, grid, grid), -20.0, dtype=np.float32)
        if layout == "nhwc":
            cls = cls.transpose(0, 2, 3, 1)
            bbox = bbox.transpose(0, 2, 3, 1)
            obj = obj.transpose(0, 2, 3, 1)
        outputs.extend((cls, bbox, obj))
    return outputs


def set_raw_logit(output: np.ndarray, channel: int, y: int, x: int, value: float) -> None:
    if output.shape[1] in (3, 12, 240):
        output[0, channel, y, x] = value
    else:
        output[0, y, x, channel] = value


def make_combined_outputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for grid in (80, 40, 20):
        outputs.append(np.full((1, grid, grid, 255), -20.0, dtype=np.float32))
    return outputs


def test_yolov5_raw_head_nchw_decodes_anchor_grid_and_letterbox() -> None:
    outputs = make_raw_outputs(layout="nchw")
    set_raw_logit(outputs[0], 0, 20, 10, 10.0)
    set_raw_logit(outputs[2], 0, 20, 10, 10.0)
    meta = LetterboxMeta(640, 640, 1280, 720, 0.5, 0.0, 140.0)

    dets, diagnostics = postprocess_yolov5_person(outputs, meta, score_thr=0.25, iou_thr=0.65, topk=20)

    assert dets.shape == (1, 5)
    np.testing.assert_allclose(dets[0, :4], [158, 35, 178, 61], atol=1)
    np.testing.assert_allclose(dets[0, 4], 0.999909, atol=1e-5)
    assert diagnostics["rknn_decoder"] == "yolov5_raw_head"
    assert diagnostics["raw_count"] == 25200
    assert [item["stride"] for item in diagnostics["feature_map_shapes"]] == [8, 16, 32]


def test_yolov5_raw_head_nhwc_is_order_independent() -> None:
    outputs = make_raw_outputs(layout="nhwc")
    set_raw_logit(outputs[0], 0, 20, 10, 10.0)
    set_raw_logit(outputs[2], 0, 20, 10, 10.0)
    outputs = list(reversed(outputs))
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)

    dets, diagnostics = postprocess_yolov5_person(outputs, meta, score_thr=0.25, iou_thr=0.65, topk=20)

    assert dets.shape == (1, 5)
    assert diagnostics["rknn_decoder"] == "yolov5_raw_head"
    assert diagnostics["output_contract"].startswith("9 split")


def test_yolov5_combined_raw_heads_decode() -> None:
    outputs = make_combined_outputs()
    # anchor 0 at stride 8: [tx,ty,tw,th,obj,person,...]
    outputs[0][0, 20, 10, 0:4] = 0.0
    outputs[0][0, 20, 10, 4] = 10.0
    outputs[0][0, 20, 10, 5] = 10.0
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)

    dets, diagnostics = postprocess_yolov5_person(list(reversed(outputs)), meta, score_thr=0.25, iou_thr=0.65, topk=20)

    assert dets.shape == (1, 5)
    assert diagnostics["rknn_decoder"] == "yolov5_combined_raw_head"
    assert diagnostics["output_contract"].startswith("3 combined")


def test_yolov5_person_only_fast_path_matches_person_detection() -> None:
    outputs = make_combined_outputs()
    outputs[0][0, 20, 10, 0:4] = 0.0
    outputs[0][0, 20, 10, 4] = 10.0
    outputs[0][0, 20, 10, 5] = 10.0
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)

    normal, _ = postprocess_yolov5_person(
        outputs,
        meta,
        score_thr=0.25,
        iou_thr=0.65,
        topk=20,
    )
    fast, diagnostics = postprocess_yolov5_person(
        outputs,
        meta,
        score_thr=0.25,
        iou_thr=0.65,
        topk=20,
        person_only_fast=True,
    )

    np.testing.assert_allclose(fast, normal, atol=1e-6)
    assert diagnostics["person_only_fast"] is True


def test_yolov5_split_person_only_fast_path_matches_normal_detection() -> None:
    outputs = make_raw_outputs()
    set_raw_logit(outputs[0], 0, 20, 10, 10.0)
    set_raw_logit(outputs[2], 0, 20, 10, 10.0)
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)

    normal, _ = postprocess_yolov5_person(outputs, meta, score_thr=0.25, iou_thr=0.65, topk=20)
    fast, diagnostics = postprocess_yolov5_person(
        outputs, meta, score_thr=0.25, iou_thr=0.65, topk=20, person_only_fast=True
    )

    np.testing.assert_allclose(fast, normal, atol=1e-6)
    assert diagnostics["person_only_fast"] is True
    assert diagnostics["raw_count"] == 25200


def test_adaptive_detector_throttles_lost_and_weak_pose_retries() -> None:
    assert adaptive_detector_decision(
        has_tracker=False,
        detector_age_seconds=None,
        refresh_seconds=0.75,
        retry_seconds=0.25,
        bad_pose_frames=0,
        bad_pose_limit=2,
    ) == (True, "initial_or_lost")
    assert adaptive_detector_decision(
        has_tracker=False,
        detector_age_seconds=0.10,
        refresh_seconds=0.75,
        retry_seconds=0.25,
        bad_pose_frames=0,
        bad_pose_limit=2,
    ) == (False, "lost_retry_cooldown")
    assert adaptive_detector_decision(
        has_tracker=False,
        detector_age_seconds=0.25,
        refresh_seconds=0.75,
        retry_seconds=0.25,
        bad_pose_frames=0,
        bad_pose_limit=2,
    ) == (True, "lost_retry")
    assert adaptive_detector_decision(
        has_tracker=True,
        detector_age_seconds=0.10,
        refresh_seconds=0.75,
        retry_seconds=0.25,
        bad_pose_frames=2,
        bad_pose_limit=2,
    ) == (False, "pose_quality_retry_cooldown")
    assert adaptive_detector_decision(
        has_tracker=True,
        detector_age_seconds=0.25,
        refresh_seconds=0.75,
        retry_seconds=0.25,
        bad_pose_frames=2,
        bad_pose_limit=2,
    ) == (True, "pose_quality_drop")


def test_tracking_reset_clears_roi_and_forces_detector_reacquisition() -> None:
    backend = YOLOv5nRTMPoseBackend()
    backend._cached_dets = np.asarray([[10.0, 20.0, 300.0, 600.0, 0.9]], dtype=np.float32)
    backend._cached_frame_shape = (720, 1280, 3)
    backend._cached_at = 12.0
    backend._last_detector_at = 12.0
    backend._bad_pose_frames = 4
    backend._tracker_quality = "pose_keypoints"

    backend.reset_tracking_state("offscreen_reentry")

    assert len(backend._cached_dets) == 0
    assert backend._cached_frame_shape is None
    assert backend._last_detector_at == 0.0
    assert backend._bad_pose_frames == 0
    assert backend._tracker_quality == "empty"
    assert adaptive_detector_decision(
        has_tracker=False,
        detector_age_seconds=None,
        refresh_seconds=backend.det_refresh_seconds,
        retry_seconds=backend.det_retry_seconds,
        bad_pose_frames=backend._bad_pose_frames,
        bad_pose_limit=backend.det_bad_pose_frames,
    ) == (True, "initial_or_lost")
    diagnostics = backend.diagnostics_snapshot()
    assert diagnostics["last_tracking_reset_reason"] == "offscreen_reentry"
    assert diagnostics["tracking_reset_count"] == 1


def test_pose_tracker_updates_bbox_without_extending_real_yolo_cache_lifetime() -> None:
    backend = YOLOv5nRTMPoseBackend()
    frame_shape = (720, 1280, 3)
    original = np.asarray([[10.0, 20.0, 300.0, 600.0, 0.91]], dtype=np.float32)
    tracked = np.asarray([[20.0, 25.0, 310.0, 605.0, 0.91]], dtype=np.float32)

    backend.det_cache_seconds = 1.5
    backend._update_cache(original, frame_shape, 10.0, [], {"source": "yolo"})
    backend._update_tracked_cache_bbox(tracked, frame_shape)

    assert backend._cached_at == 10.0
    assert np.allclose(backend._cached_dets, tracked)
    assert backend._cache_valid(frame_shape, 11.49) is True
    assert backend._cache_valid(frame_shape, 11.51) is False


def test_adaptive_detector_never_uses_or_renews_an_expired_yolo_cache() -> None:
    source = (Path(__file__).resolve().parent / "yolov5n_rtmpose_backend.py").read_text(encoding="utf-8")

    assert "has_tracker=self._cache_valid(frame_shape, now)" in source
    assert "self._update_tracked_cache_bbox(tracked[None, :], frame_shape)" in source
    assert "if self._cache_valid(frame_shape, now):\n                dets, det_output_shapes, det_meta = self._cached_detection()" in source
    diagnostics_block = source.split("self._last_diagnostics =", 1)[1].split('result.meta["npu_resource"]', 1)[0]
    assert '"det_cache_valid",' in diagnostics_block


def test_tracker_roi_uses_visible_keypoints_and_margin() -> None:
    points = np.asarray([[100.0 + index * 10.0, 80.0 + index * 12.0] for index in range(17)], dtype=np.float32)
    scores = np.full((17,), 0.9, dtype=np.float32)
    scores[:3] = 0.05

    bbox, visible_count = tracker_bbox_from_keypoints(
        points,
        scores,
        (720, 1280, 3),
        score_threshold=0.16,
        margin=0.20,
    )

    assert visible_count == 14
    assert bbox is not None
    assert bbox[0] < points[3, 0] and bbox[1] < points[3, 1]
    assert bbox[2] > points[-1, 0] and bbox[3] > points[-1, 1]


def test_abnormal_pose_roi_falls_back_to_bounded_detection_crop() -> None:
    safe, used_fallback, padding_ratio = bound_pose_bbox_for_frame(
        np.asarray([-900.0, -700.0, 2100.0, 1700.0], dtype=np.float32),
        np.asarray([300.0, 80.0, 900.0, 700.0], dtype=np.float32),
        (720, 1280, 3),
        (192, 256),
        max_padding_ratio=0.55,
    )

    assert used_fallback is True
    assert safe[2] - safe[0] < 1280
    assert safe[3] - safe[1] <= 720 * 1.10 + 1
    assert padding_ratio <= 0.55


def test_yolov5_raw_head_uses_person_class_zero_only() -> None:
    outputs = make_raw_outputs()
    set_raw_logit(outputs[0], 1, 20, 10, 10.0)  # bicycle, not person
    set_raw_logit(outputs[2], 0, 20, 10, 10.0)
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)

    dets, diagnostics = postprocess_yolov5_person(outputs, meta, score_thr=0.25, iou_thr=0.65, topk=20)

    assert dets.shape == (0, 5)
    assert diagnostics["best_class_id"] == 1


def test_legacy_detector_is_strict_xyxy_rollback_only() -> None:
    boxes = np.zeros((1, 5000, 4), dtype=np.float32)
    scores = np.zeros((1, 80, 5000), dtype=np.float32)
    boxes[0, 0] = [100, 170, 300, 570]
    boxes[0, 1] = [105, 175, 305, 575]
    scores[0, 0, 0] = 0.9
    scores[0, 0, 1] = 0.8
    meta = LetterboxMeta(640, 640, 1280, 720, 0.5, 0.0, 140.0)

    dets, diagnostics = postprocess_yolov5_person(
        [boxes, scores], meta, score_thr=0.25, iou_thr=0.65, topk=20, score_mode="raw"
    )

    assert dets.shape == (1, 5)
    np.testing.assert_allclose(dets[0, :4], [200, 60, 600, 719], atol=1)
    assert diagnostics["rknn_decoder"] == "legacy_decoded_xyxy"
    assert "deprecated" in diagnostics["model_contract_warning"]


def test_legacy_detector_does_not_guess_invalid_box_format() -> None:
    boxes = np.zeros((1, 5000, 4), dtype=np.float32)
    scores = np.zeros((1, 80, 5000), dtype=np.float32)
    boxes[0, 0] = [200, 370, 200, 400]
    scores[0, 0, 0] = 0.9
    meta = LetterboxMeta(640, 640, 1280, 720, 0.5, 0.0, 140.0)

    dets, diagnostics = postprocess_yolov5_person(
        [boxes, scores], meta, score_thr=0.25, iou_thr=0.65, topk=20, score_mode="raw"
    )

    assert dets.shape == (0, 5)
    assert diagnostics["valid_bbox_count"] == 0


def test_yolov5_rejects_unknown_output_contract() -> None:
    meta = LetterboxMeta(640, 640, 640, 640, 1.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="expected 3 combined or 9 split"):
        postprocess_yolov5_person([np.zeros((1, 10), dtype=np.float32)], meta, score_thr=0.25, iou_thr=0.65, topk=20)


def test_simcc_decode_restores_points_to_camera_coordinates() -> None:
    simcc_x = np.zeros((1, 17, 384), dtype=np.float32)
    simcc_y = np.zeros((1, 17, 512), dtype=np.float32)
    simcc_x[:, :, 192] = 0.81
    simcc_y[:, :, 256] = 0.64
    restore = RestoreInfo(np.asarray([100, 50, 292, 306], dtype=np.float32), (192, 256))

    points, scores, crop_points, meta = decode_simcc(
        [simcc_x, simcc_y], restore, input_size=(192, 256), split_ratio=2.0, score_mode="sqrt"
    )

    np.testing.assert_allclose(crop_points[0], [96, 128])
    np.testing.assert_allclose(points[0], [196, 178])
    np.testing.assert_allclose(scores[0], 0.72, atol=1e-6)
    assert meta["simcc_x_shape"] == [17, 384]
    assert meta["simcc_y_shape"] == [17, 512]


def test_rknn_inference_passes_explicit_nhwc_data_format() -> None:
    class FakeRKNN:
        def __init__(self) -> None:
            self.kwargs = None

        def inference(self, **kwargs):
            self.kwargs = kwargs
            return [np.zeros((1,), dtype=np.float32)]

    model = RKNNModel("unused.rknn")
    fake = FakeRKNN()
    model.rknn = fake
    tensor = np.zeros((1, 640, 640, 3), dtype=np.float32)

    model.inference([tensor], data_format="nhwc")

    assert fake.kwargs is not None
    assert fake.kwargs["data_format"] == "nhwc"
    assert fake.kwargs["inputs"][0].shape == (1, 640, 640, 3)
