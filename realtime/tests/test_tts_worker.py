from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.training_session import RealtimeTrainingSession
from realtime.tts_worker import TTSWorker


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
        with patch("realtime.tts_worker.shutil.which", return_value="aplay"), patch(
            "realtime.tts_worker.subprocess.run", return_value=SimpleNamespace(returncode=0)
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


if __name__ == "__main__":
    test_new_correction_drops_old_correction()
    test_count_drops_pending_correction()
    test_fixed_audio_file_is_preferred_when_present()
    test_missing_fixed_audio_falls_back_to_tts_payload()
    test_training_session_suppresses_immediate_second_correction()
    test_training_session_does_not_hide_correction_for_pose_jitter()
    test_training_session_filters_process_prompts_from_main_ui()
    print("tts worker tests passed")
