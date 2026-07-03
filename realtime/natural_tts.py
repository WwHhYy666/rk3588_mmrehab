"""Natural female TTS backend based on the local sherpa-onnx VITS model."""

from __future__ import annotations

import subprocess
import re
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTS_ROOT = PROJECT_ROOT / "tts"
MODEL_DIR = TTS_ROOT / "tts_model_pack"
RUNS_DIR = TTS_ROOT / "runs"


class NaturalTTS:
    def __init__(
        self,
        *,
        model: Path | None = None,
        sid: int = 66,
        speed: float = 0.92,
        silence_scale: float = 0.55,
        provider: str = "cpu",
        num_threads: int = 2,
    ) -> None:
        self.model = model or MODEL_DIR / "vits-aishell3.onnx"
        self.lexicon = MODEL_DIR / "lexicon.txt"
        self.tokens = MODEL_DIR / "tokens.txt"
        self.phone_fst = MODEL_DIR / "phone.fst"
        self.date_fst = MODEL_DIR / "date.fst"
        self.number_fst = MODEL_DIR / "number.fst"
        self.sid = sid
        self.speed = speed
        self.silence_scale = silence_scale
        self.provider = provider
        self.num_threads = num_threads
        self.backend_name = "natural_tts"
        self.last_error: str | None = None
        self._tts: Any | None = None
        self._sherpa_onnx: Any | None = None
        self._soundfile: Any | None = None

    def is_available(self) -> bool:
        try:
            self._check_files()
            self._ensure_loaded()
        except Exception as exc:
            self.last_error = str(exc)
            return False
        return True

    def synthesize_to_wav(self, text: str) -> Path:
        text = str(text or "").strip()
        if not text:
            raise ValueError("TTS 文本不能为空")
        self._ensure_loaded()
        assert self._tts is not None
        assert self._sherpa_onnx is not None
        assert self._soundfile is not None

        if hasattr(self._sherpa_onnx, "GenerationConfig"):
            gen_config = self._sherpa_onnx.GenerationConfig()
            gen_config.sid = self.sid
            gen_config.speed = self.speed
            gen_config.silence_scale = self.silence_scale
            audio = self._tts.generate(text, gen_config)
        else:
            audio = self._tts.generate(text, sid=self.sid, speed=self.speed)

        if len(audio.samples) == 0:
            raise RuntimeError("生成音频为空")

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_wav = RUNS_DIR / f"realtime_tts_{int(time.time() * 1000)}.wav"
        self._soundfile.write(
            str(out_wav),
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )
        return out_wav

    def speak(self, text: str) -> bool:
        try:
            chunks = split_tts_text(text)
            wav_paths = [self.synthesize_to_wav(chunk) for chunk in chunks]
            ok = True
            for out_wav in wav_paths:
                completed = subprocess.run(["aplay", str(out_wav)], check=False)
                ok = ok and completed.returncode == 0
            return ok
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[NaturalTTS] 播报失败: {exc}")
            return False

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return
        self._check_files()
        try:
            import sherpa_onnx
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError(f"缺少自然女声 TTS 依赖: {exc}") from exc

        rule_fsts = f"{self.phone_fst},{self.date_fst},{self.number_fst}"
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(self.model),
                    lexicon=str(self.lexicon),
                    tokens=str(self.tokens),
                    data_dir="",
                ),
                num_threads=self.num_threads,
                debug=False,
                provider=self.provider,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=2,
        )
        if not config.validate():
            raise RuntimeError("NaturalTTS config validate failed")
        self._sherpa_onnx = sherpa_onnx
        self._soundfile = sf
        self._tts = sherpa_onnx.OfflineTts(config)

    def _check_files(self) -> None:
        for path in [self.model, self.lexicon, self.tokens, self.phone_fst, self.date_fst, self.number_fst]:
            if not path.exists():
                raise FileNotFoundError(f"找不到 TTS 文件: {path}")


def split_tts_text(text: str, *, max_chars: int = 40) -> list[str]:
    normalized = _normalize_tts_text(text)
    if not normalized:
        return []
    parts = [part for part in re.split(r"(?<=[。！？；])", normalized) if part.strip()]
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) <= max_chars:
            chunks.append(part)
            continue
        chunks.extend(_split_long_sentence(part, max_chars=max_chars))
    return chunks or [normalized]


def _normalize_tts_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", "，", text)
    text = text.replace(",", "，").replace(";", "；").replace("!", "！").replace("?", "？")
    text = text.replace("。请", "。请").replace("。现在", "。现在")
    if text and text[-1] not in "。！？；":
        text += "。"
    return text


def _split_long_sentence(text: str, *, max_chars: int) -> list[str]:
    pieces = [piece for piece in re.split(r"(?<=[，、])", text) if piece.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) > max_chars:
            chunks.append(_ensure_sentence_end(current))
            current = piece
        else:
            current += piece
    if current:
        chunks.append(_ensure_sentence_end(current))
    return chunks


def _ensure_sentence_end(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in "。！？；":
        return text + "。"
    return text
