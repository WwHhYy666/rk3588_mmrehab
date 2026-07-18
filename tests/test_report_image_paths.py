from pathlib import Path

from rehab_app.services.report_paths import is_safe_keyframe_path, resolve_keyframe_path
from rehab_app.services.report_visuals import attach_keyframe_urls, build_metric_cards, render_report_images


def test_cpu_and_npu_keyframe_paths_are_allowed(tmp_path: Path) -> None:
    cpu = "data/reports/keyframes/session/cpu.jpg"
    npu = "data/reports/npu/keyframes/session/npu.jpeg"

    assert is_safe_keyframe_path(cpu)
    assert is_safe_keyframe_path(npu)
    assert resolve_keyframe_path(tmp_path, cpu) == (tmp_path / cpu).resolve()
    assert resolve_keyframe_path(tmp_path, npu) == (tmp_path / npu).resolve()


def test_keyframe_path_policy_rejects_traversal_and_other_report_folders(tmp_path: Path) -> None:
    rejected = [
        "../keyframes/secret.jpg",
        "/data/reports/keyframes/secret.jpg",
        "data/reports/npu/secret.jpg",
        "data/reports/npu/keyframes/session/secret.png",
    ]

    for value in rejected:
        assert not is_safe_keyframe_path(value)
        assert resolve_keyframe_path(tmp_path, value) is None


def test_npu_keyframe_can_render_report_image_and_url(tmp_path: Path) -> None:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image_path = tmp_path / "data/reports/npu/keyframes/session/npu.jpg"
    image_path.parent.mkdir(parents=True)
    image_module.new("RGB", (320, 240), (30, 40, 50)).save(image_path, "JPEG")
    relative = image_path.relative_to(tmp_path).as_posix()
    report = {
        "action_id": "sit_to_stand",
        "action_name": "坐站训练",
        "errors": {"primary_error": "OK"},
        "report_card_metrics": {
            "source": "best_correct",
            "rep_index": 1,
            "rom": {"actual": 1.0, "target": 0.9},
            "tut": {"actual": 4.7, "target": 4.4},
            "speed": {"ratio": 1.0},
            "dtw": {"normalized_distance": 0.3},
        },
        "keyframes": [
            {
                "action_id": "sit_to_stand",
                "rep_index": 1,
                "image_path": relative,
                "selected_side": "left",
                "rehab_keypoints": {
                    "left_hip": {"x": 0.45, "y": 0.35, "visibility": 0.9},
                    "left_knee": {"x": 0.5, "y": 0.58, "visibility": 0.9},
                    "left_ankle": {"x": 0.55, "y": 0.82, "visibility": 0.9},
                },
            }
        ],
    }

    urls = attach_keyframe_urls(report)
    rendered = render_report_images(tmp_path, report, build_metric_cards(report))

    assert urls[0]["url"] == f"/report-images/{relative}"
    assert rendered["items"][0]["comparison_image"]["url"].startswith(
        "/report-images/data/reports/npu/keyframes/session/"
    )
    comparison_path = tmp_path / rendered["items"][0]["comparison_image"]["path"]
    assert comparison_path.exists()
