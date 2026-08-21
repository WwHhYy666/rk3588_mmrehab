from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "prescription" / "banzi" / "record_prescription_http.py"


def test_mediapipe_display_mapping_has_exactly_coco17_points() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8-sig"))
    mapping = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MEDIAPIPE_COCO17_INDICES" for target in node.targets):
            mapping = ast.literal_eval(node.value)
            break

    assert mapping is not None
    assert len(mapping) == 17
    assert set(mapping) == {
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }


def test_mediapipe_frame_uses_only_cpu_coco17_overlay() -> None:
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    process_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_mediapipe_frame")
    process_source = ast.get_source_segment(source, process_node) or ""

    assert "draw_mediapipe_coco17_overlay" in process_source
    assert "mp_drawing.draw_landmarks" not in process_source
    assert "build_rehab_keypoints" in process_source
