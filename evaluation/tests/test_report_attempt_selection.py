from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.report_generator import (  # noqa: E402
    build_report_card_metrics,
    build_quality_attempt_reports,
    build_completion_summary,
    build_selected_attempts,
    enrich_quality_segments_with_runtime_metrics,
    extract_report_keyframes,
)
from rehab_app.services import llm_assistant  # noqa: E402


def test_report_card_uses_single_rep_tut_not_full_session_sum() -> None:
    metrics = {
        "rom": {"actual": 67.2, "target": 18.5},
        "tut": {"actual": 12.5, "target": 4.4},
        "speed": {"ratio": 7.11},
        "dtw": {"normalized_distance": 4.60},
    }
    attempts = [
        {
            "attempt_index": 3,
            "rep_index": 1,
            "countable": True,
            "primary_error": "OK",
            "rom": 23.95,
            "rom_target": 18.46,
            "rom_diff": 0.0,
            "tut_seconds": 3.13,
            "tut_target": 4.4,
            "tut_ratio": 0.71,
        },
        {
            "attempt_index": 7,
            "rep_index": 2,
            "countable": True,
            "primary_error": "OK",
            "rom": 23.35,
            "rom_target": 18.46,
            "rom_diff": 0.0,
            "tut_seconds": 4.50,
            "tut_target": 4.4,
            "tut_ratio": 1.02,
        },
        {
            "attempt_index": 9,
            "rep_index": 3,
            "countable": True,
            "primary_error": "OK",
            "rom": 23.61,
            "rom_target": 18.46,
            "rom_diff": 0.0,
            "tut_seconds": 4.93,
            "tut_target": 4.4,
            "tut_ratio": 1.12,
        },
    ]

    selected = build_selected_attempts(attempts)
    card_metrics = build_report_card_metrics(metrics, selected)

    assert card_metrics["source"] == "best_correct"
    assert card_metrics["attempt_index"] == 9
    assert card_metrics["tut"]["actual"] == 4.93
    assert card_metrics["tut"]["actual"] != metrics["tut"]["actual"]


def test_old_patient_attempt_segments_are_enriched_from_runtime_results() -> None:
    payload = {
        "runtime_meta": {
            "rep_results": [
                {
                    "attempt_index": 2,
                    "rep_index": 1,
                    "rom": 20.0,
                    "rom_target": 18.0,
                    "rom_diff": 0.0,
                    "tut_seconds": 3.2,
                    "tut_target": 3.0,
                    "tut_ratio": 1.06,
                }
            ],
            "invalid_attempts": [
                {
                    "attempt_index": 1,
                    "primary_error": "ROM_LOW",
                    "rom": 10.0,
                    "rom_target": 18.0,
                    "rom_diff": 8.0,
                    "tut_seconds": 0.0,
                    "tut_target": 3.0,
                    "missing_seconds": 3.0,
                    "screen_prompt": "腿再伸直一点",
                }
            ],
        }
    }
    segments = [
        {"attempt_index": 1, "countable": False, "primary_error": "ROM_LOW"},
        {"attempt_index": 2, "rep_index": 1, "countable": True, "primary_error": "OK"},
    ]

    enriched = enrich_quality_segments_with_runtime_metrics(payload, segments)
    reports, _, _ = build_quality_attempt_reports("seated_knee_extension", enriched)

    assert reports[0]["rom_diff"] == 8.0
    assert reports[0]["reason"] == "腿再伸直一点"
    assert reports[1]["tut_seconds"] == 3.2


def test_llm_compact_report_contains_selected_attempts_and_card_metrics() -> None:
    report = {
        "action_name": "坐姿伸膝",
        "report_card_metrics": {
            "source": "best_correct",
            "attempt_index": 2,
            "rep_index": 1,
            "primary_error": "OK",
            "rom": {"actual": 20.0, "target": 18.0},
            "tut": {"actual": 3.2, "target": 3.0},
        },
        "metrics": {"tut": {"actual": 12.5, "target": 3.0}},
        "selected_attempts": {
            "best_correct": {"attempt_index": 2, "rep_index": 1, "countable": True, "primary_error": "OK", "tut_seconds": 3.2},
            "representative_wrong": {"attempt_index": 1, "countable": False, "primary_error": "ROM_LOW", "rom_diff": 8.0},
        },
    }

    compact = llm_assistant._compact_report(report)

    assert compact["metrics"]["tut"]["actual"] == 3.2
    assert compact["full_session_metrics"]["tut"]["actual"] == 12.5
    assert compact["selected_attempts"]["best_correct"]["attempt_index"] == 2
    assert compact["selected_attempts"]["representative_wrong"]["primary_error"] == "ROM_LOW"


def test_npu_keyframes_are_preserved_in_report_payload() -> None:
    attempt = {
        "runtime_meta": {
            "keyframes": [
                {
                    "session_id": "session-npu",
                    "action_id": "sit_to_stand",
                    "rep_index": 1,
                    "image_path": "data/reports/npu/keyframes/session-npu/sit_to_stand_rep1_best.jpg",
                    "write_status": "complete",
                    "rehab_keypoints": {"left_knee": {"x": 0.4, "y": 0.6, "visibility": 0.9}},
                }
            ]
        }
    }

    keyframes = extract_report_keyframes(attempt)

    assert len(keyframes) == 1
    assert keyframes[0]["rep_index"] == 1
    assert keyframes[0]["image_path"].startswith("data/reports/npu/keyframes/")
    assert keyframes[0]["write_status"] == "complete"
    assert keyframes[0]["rehab_keypoints"]["left_knee"]["visibility"] == 0.9


def test_report_keyframes_reject_paths_outside_report_roots() -> None:
    attempt = {
        "runtime_meta": {
            "keyframes": [
                {"image_path": "../secret.jpg"},
                {"image_path": "data/reports/npu/other/not-a-keyframe.jpg"},
                {"image_path": "data/reports/npu/keyframes/session/payload.png"},
            ]
        }
    }

    assert extract_report_keyframes(attempt) == []


if __name__ == "__main__":
    test_report_card_uses_single_rep_tut_not_full_session_sum()
    test_old_patient_attempt_segments_are_enriched_from_runtime_results()
    test_llm_compact_report_contains_selected_attempts_and_card_metrics()
    print("report attempt selection tests passed")

def test_seated_knee_raise_startup_reentry_artifact_is_skipped_in_report() -> None:
    segments = [
        {
            "attempt_index": 1,
            "countable": False,
            "primary_error": "ROM_LOW",
            "duration_seconds": 1.7,
            "tut_ratio": 0.0,
            "rom": 0.14,
            "rom_target": 0.67,
            "quality_score": 2.7,
        },
        {
            "attempt_index": 2,
            "countable": True,
            "primary_error": "OK",
            "duration_seconds": 5.0,
            "tut_ratio": 1.0,
            "rom": 0.68,
            "rom_target": 0.67,
            "quality_score": 96.0,
        },
    ]

    reports, overall, _ = build_quality_attempt_reports("seated_knee_raise", segments)
    summary = build_completion_summary("seated_knee_raise", "坐姿抬膝", reports, overall)

    assert [item["attempt_index"] for item in reports] == [2]
    assert summary["attempts"][0]["completion_percent"] == 96.0
    assert overall == 96.0


def test_npu_report_average_is_capped_when_any_attempt_has_error() -> None:
    segments = [
        {
            "attempt_index": 1,
            "countable": True,
            "primary_error": "OK",
            "pose_backend": "rknn",
            "training_logic_version": "npu_training_v4",
            "rom": 0.70,
            "required_rom": 0.63,
            "tut_ratio": 1.1,
            "raw_quality_score": 99.0,
        },
        {
            "attempt_index": 2,
            "countable": False,
            "primary_error": "ROM_LOW",
            "pose_backend": "rknn",
            "training_logic_version": "npu_training_v4",
            "rom": 0.50,
            "required_rom": 0.63,
            "raw_quality_score": 99.0,
        },
    ]

    reports, overall, status = build_quality_attempt_reports("sit_to_stand", segments)

    assert reports[0]["completion_percent"] <= 96.0
    assert reports[1]["completion_percent"] <= 70.0
    assert reports[1]["quality_grade"] != "优秀"
    assert overall <= 70.0
    assert status["average_error_cap"] == 70.0
