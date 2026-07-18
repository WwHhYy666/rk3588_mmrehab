from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from training.natural_tts import NaturalTTS, _normalize_tts_text, apply_output_gain


def test_assistant_tts_synthesizes_and_plays_complete_answer_once(tmp_path: Path) -> None:
    tts = NaturalTTS(audio_output_device="plughw:1,0")
    wav_path = tmp_path / "answer.wav"
    wav_path.write_bytes(b"RIFF0000WAVE")
    synthesized: list[str] = []

    def fake_synthesize(text: str) -> Path:
        synthesized.append(text)
        return wav_path

    tts.synthesize_to_wav = fake_synthesize  # type: ignore[method-assign]
    with patch("training.audio_output.shutil.which", return_value="/usr/bin/aplay"), patch(
        "training.audio_output.subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")
    ) as run:
        assert tts.speak("第一句，请保持稳定。第二句，请慢慢完成。") is True

    assert synthesized == ["第一句，请保持稳定。第二句，请慢慢完成。"]
    command = run.call_args.args[0]
    assert command == ["/usr/bin/aplay", "-D", "plughw:1,0", str(wav_path)]
    assert tts.last_audio_command == " ".join(command)
    assert tts.last_audio_returncode == 0


def test_pronunciation_override_only_changes_spoken_text() -> None:
    assert _normalize_tts_text("建议重新完成一次") == "建议崇新完成一次。"


def test_assistant_output_gain_boosts_quiet_audio_without_clipping() -> None:
    samples = np.asarray([-0.8, -0.25, 0.0, 0.25, 0.8], dtype=np.float32)

    boosted, stats = apply_output_gain(samples, 1.35)

    assert stats["requested_gain"] == 1.35
    assert 1.0 < stats["applied_gain"] < 1.35
    assert float(np.max(np.abs(boosted))) <= 0.980001
    assert abs(float(boosted[1])) > abs(float(samples[1]))


def test_assistant_tts_gain_defaults_from_environment() -> None:
    with patch.dict("os.environ", {"REHAB_ASSISTANT_TTS_GAIN": "1.35"}):
        tts = NaturalTTS()

    assert tts.output_gain == 1.35
