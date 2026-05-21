from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import warnings
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import speech_recognition as sr
import torch
from transformers import AutoProcessor

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
)

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers 4.x compatibility.
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None


ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "pipeline"
RUNS_DIR = PIPELINE_DIR / "runs"
TEST_TEXT_DIR = PIPELINE_DIR / "test_text"

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MODEL_PATH = os.getenv("QWEN_MODEL_PATH") or (
    str(ROOT_DIR / "qwen2_vl_2b_instruct")
    if (ROOT_DIR / "qwen2_vl_2b_instruct").is_dir()
    else DEFAULT_MODEL_ID
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文优先的语音助手。除非用户明确要求其他语言，否则使用中文回答。"
    "回答要简洁、自然，适合直接进行语音播报。"
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def normalize_device(device: str | None) -> int | str | None:
    if device is None:
        return None
    stripped = device.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def write_text(path: str | Path, text: str) -> Path:
    target = ensure_parent(path)
    target.write_text(text.strip() + "\n", encoding="utf-8")
    return target


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def timestamped_run_dir(prefix: str = "run") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = RUNS_DIR / f"{stamp}_{prefix}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = RUNS_DIR / f"{stamp}_{prefix}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def copy_into(src: str | Path, dst: str | Path) -> Path:
    target = ensure_parent(dst)
    shutil.copy2(src, target)
    return target


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(audio).reshape(-1)
    if mono.dtype != np.int16:
        mono = np.clip(mono, -32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(mono.tobytes())
    return buffer.getvalue()


def wav_file_to_audio(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV playback is supported, got {sample_width * 8}-bit")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate


def play_wav_file(path: str | Path, output_device: int | str | None = None) -> None:
    audio, sample_rate = wav_file_to_audio(path)
    sd.play(audio, samplerate=sample_rate, device=output_device)
    sd.wait()


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    target = ensure_parent(path)
    target.write_bytes(audio_to_wav_bytes(audio, sample_rate))
    return target


def split_for_tts(text: str, max_chars: int = 4500) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts: list[str] = []
    current = ""
    sentences = re.split(r"(?<=[.!?;:。！？；：])\s*", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def list_audio_devices() -> None:
    print(sd.query_devices())


@dataclass
class Recorder:
    sample_rate: int = 16000
    input_device: int | str | None = None

    def record_array(self, seconds: float) -> np.ndarray:
        frames = int(seconds * self.sample_rate)
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.input_device,
        )
        sd.wait()
        return audio

    def record_to_file(self, seconds: float, output: str | Path) -> Path:
        audio = self.record_array(seconds)
        return write_wav(output, audio, self.sample_rate)


def whisper_language(language: str, backend: str) -> str:
    normalized = language.lower()
    if normalized.startswith("zh") or normalized in {"cmn-cn", "chinese"}:
        return "zh" if backend == "faster-whisper" else "chinese"
    if normalized.startswith("en") or normalized == "english":
        return "en" if backend == "faster-whisper" else "english"
    return normalized.split("-")[0] if backend == "faster-whisper" else normalized


class SpeechRecognitionAsr:
    def __init__(
        self,
        backend: str = "faster-whisper",
        language: str = "zh-CN",
        whisper_model: str = "tiny",
    ) -> None:
        self.backend = backend
        self.language = language
        self.whisper_model = whisper_model
        self.recognizer = sr.Recognizer()

    def transcribe_file(self, audio_file: str | Path) -> str:
        with sr.AudioFile(str(audio_file)) as source:
            audio = self.recognizer.record(source)
        return self.transcribe_audio_data(audio)

    def transcribe_wav_bytes(self, wav_data: bytes) -> str:
        with sr.AudioFile(io.BytesIO(wav_data)) as source:
            audio = self.recognizer.record(source)
        return self.transcribe_audio_data(audio)

    def transcribe_audio_data(self, audio: sr.AudioData) -> str:
        try:
            if self.backend == "google":
                return self.recognizer.recognize_google(
                    audio,
                    language=self.language,
                    show_all=False,
                ).strip()
            if self.backend == "sphinx":
                if not self.language.lower().startswith("en"):
                    raise RuntimeError("当前 PocketSphinx 配置只适合英文识别。中文请使用 google、whisper 或 faster-whisper。")
                return self.recognizer.recognize_sphinx(
                    audio,
                    language=self.language,
                    show_all=False,
                ).strip()
            if self.backend == "whisper":
                return self.recognizer.recognize_whisper(
                    audio,
                    model=self.whisper_model,
                    language=whisper_language(self.language, self.backend),
                ).strip()
            if self.backend == "faster-whisper":
                init_options = {}
                if not torch.cuda.is_available():
                    init_options = {"device": "cpu", "compute_type": "int8"}
                return self.recognizer.recognize_faster_whisper(
                    audio,
                    model=self.whisper_model,
                    init_options=init_options,
                    language=whisper_language(self.language, self.backend),
                ).strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            raise RuntimeError(f"ASR request failed: {exc}") from exc
        except ImportError as exc:
            raise RuntimeError(
                f"ASR 后端 '{self.backend}' 缺少额外依赖。"
                "中文/英文离线识别请安装 openai-whisper 或 faster-whisper。"
            ) from exc

        raise ValueError(f"不支持的 ASR 后端：{self.backend}")


class QwenLLM:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
        attn_implementation: str | None = None,
        device_map: str | None = None,
        torch_dtype: str = "auto",
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=Path(self.model_path).is_dir(),
        )
        self.model = self._load_model(attn_implementation, device_map, torch_dtype)
        self.model.eval()
        if self.temperature <= 0 and hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

    def _load_model(
        self,
        attn_implementation: str | None,
        device_map: str | None,
        torch_dtype: str,
    ):
        dtype_map = {
            "auto": "auto" if torch.cuda.is_available() else torch.float32,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        kwargs = {
            "torch_dtype": dtype_map[torch_dtype],
            "local_files_only": Path(self.model_path).is_dir(),
        }
        if device_map == "auto" or (device_map is None and torch.cuda.is_available()):
            kwargs["device_map"] = "auto"
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        if AutoModelForImageTextToText is not None:
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    **kwargs,
                )
            except Exception as exc:
                if Qwen2VLForConditionalGeneration is None:
                    raise exc

        if Qwen2VLForConditionalGeneration is None:
            raise RuntimeError("当前 transformers 环境不可用 Qwen2VLForConditionalGeneration，请检查 transformers 版本。")
        return Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, **kwargs)

    def generate(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=[prompt], padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs.update({"temperature": self.temperature, "top_p": self.top_p})

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        trimmed = output_ids[:, inputs["input_ids"].shape[-1] :]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


def resolve_model_path(model_path: str) -> str:
    candidates = [
        model_path,
        str(ROOT_DIR / model_path),
        str(ROOT_DIR / "qwen2_vl_2b_instruct"),
        str(ROOT_DIR / "Qwen2-VL-2B-Instruct"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return str(Path(candidate).resolve())
    return model_path


def looks_like_local_path(model_path: str) -> bool:
    return os.path.isabs(model_path) or model_path.startswith(".") or "\\" in model_path


class EchoLLM:
    def generate(self, user_text: str) -> str:
        return f"你刚才说：{user_text}"


class WindowsSapiTts:
    def __init__(
        self,
        voice: str | None = None,
        language: str = "zh-CN",
        rate: int = 0,
        volume: float = 1.0,
    ) -> None:
        import win32com.client

        self.win32com = win32com
        self.engine = win32com.client.Dispatch("SAPI.SpVoice")
        self.engine.Rate = max(-10, min(10, int(rate)))
        self.engine.Volume = int(max(0.0, min(1.0, volume)) * 100)
        self._select_voice(voice or self.default_voice(language))

    @staticmethod
    def default_voice(language: str) -> str:
        lowered = language.lower()
        if lowered.startswith("en"):
            return "Zira"
        return "Huihui"

    def _select_voice(self, voice: str) -> None:
        lowered = voice.lower()
        voices = self.engine.GetVoices()
        for index in range(voices.Count):
            candidate = voices.Item(index)
            values = [str(candidate.Id), str(candidate.GetDescription())]
            if any(lowered in value.lower() for value in values):
                self.engine.Voice = candidate
                return
        raise RuntimeError(f"未找到 Windows SAPI 语音：{voice}")

    def speak(self, text: str) -> None:
        self.engine.Speak(text, 0)

    def synthesize_to_file(self, text: str, output: str | Path) -> Path:
        target = ensure_parent(output)
        stream = self.win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(target.resolve()), 3, False)
        old_stream = self.engine.AudioOutputStream
        try:
            self.engine.AudioOutputStream = stream
            for chunk in split_for_tts(text):
                self.engine.Speak(chunk, 0)
            self.engine.WaitUntilDone(-1)
        finally:
            stream.Close()
            self.engine.AudioOutputStream = old_stream
        return target

    @staticmethod
    def list_voices() -> None:
        import win32com.client

        engine = win32com.client.Dispatch("SAPI.SpVoice")
        voices = engine.GetVoices()
        for index in range(voices.Count):
            voice = voices.Item(index)
            print(f"{index}: {voice.GetDescription()}\n  id={voice.Id}")


def pipeline_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "seed_text": run_dir / "00_seed_text.txt",
        "input_audio": run_dir / "01_input_audio.wav",
        "asr_text": run_dir / "02_asr_text.txt",
        "llm_text": run_dir / "03_llm_output.txt",
        "tts_audio": run_dir / "04_tts_output.wav",
        "manifest": run_dir / "manifest.json",
    }
