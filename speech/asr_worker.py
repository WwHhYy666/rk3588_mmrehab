from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(
    os.getenv("REHAB_ASR_MODEL_DIR", str(PROJECT_ROOT / "models" / "audio" / "asr" / "paraformer"))
)
DEFAULT_PROVIDER = os.getenv("REHAB_ASR_PROVIDER", "sherpa_paraformer").strip() or "sherpa_paraformer"
DEFAULT_NUM_THREADS = int(os.getenv("REHAB_ASR_NUM_THREADS", "2"))
DEFAULT_PREFER_INT8 = os.getenv("REHAB_ASR_PREFER_INT8", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_MIN_RMS = float(os.getenv("REHAB_ASR_MIN_RMS", "0.002"))
DEFAULT_MIN_PEAK = float(os.getenv("REHAB_ASR_MIN_PEAK", "0.015"))
DEFAULT_EDGE_PADDING_MS = int(os.getenv("REHAB_ASR_EDGE_PADDING_MS", "180"))


@dataclass
class AsrJob:
    job_id: str
    audio_path: str
    status: str = "queued"
    text: str = ""
    error: str | None = None
    latency_ms: int | None = None
    audio_stats: dict[str, Any] | None = None
    suppressed_text: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "audio_stats": self.audio_stats,
            "suppressed_text": self.suppressed_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ParaformerAsrWorker:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        provider: str = DEFAULT_PROVIDER,
        num_threads: int = DEFAULT_NUM_THREADS,
        max_queue_size: int = 3,
        max_jobs: int = 20,
        prefer_int8: bool = DEFAULT_PREFER_INT8,
        min_rms: float = DEFAULT_MIN_RMS,
        min_peak: float = DEFAULT_MIN_PEAK,
        edge_padding_ms: int = DEFAULT_EDGE_PADDING_MS,
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.provider = provider
        self.num_threads = num_threads
        self.max_jobs = max_jobs
        self.prefer_int8 = prefer_int8
        self.min_rms = min_rms
        self.min_peak = min_peak
        self.edge_padding_ms = max(0, int(edge_padding_ms))
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._jobs: OrderedDict[str, AsrJob] = OrderedDict()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._recognizer: Any | None = None
        self._sherpa_onnx: Any | None = None
        self._recognizer_api: str | None = None
        self._sherpa_version: str | None = None
        self.last_error: str | None = None
        self.last_text: str = ""
        self.last_latency_ms: int | None = None
        self.last_audio_stats: dict[str, Any] | None = None
        self.last_suppressed_text: str | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="paraformer-asr-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, audio_path: str | Path) -> dict[str, Any]:
        if not self._running:
            self.start()
        job_id = f"asr_{uuid.uuid4().hex[:12]}"
        job = AsrJob(job_id=job_id, audio_path=str(audio_path))
        with self._lock:
            self._jobs[job_id] = job
            self._trim_jobs_locked()
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                job.status = "failed"
                job.error = "ASR queue is full"
                job.updated_at = time.time()
            return {"ok": False, "job_id": job_id, "error": job.error}
        return {"ok": True, "job_id": job_id}

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"ok": False, "error": "ASR job not found", "job_id": job_id}
            return {"ok": job.status != "failed", **job.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            recent = [job.snapshot() for job in list(self._jobs.values())[-5:]]
        model_status = self._model_status()
        sherpa_info = self._sherpa_info()
        return {
            "provider": self.provider,
            "backend": "sherpa_onnx_paraformer",
            "model_dir": str(self.model_dir),
            "model_available": model_status["available"],
            "model_path": str(model_status["model"]) if model_status.get("model") else None,
            "tokens_path": str(model_status["tokens"]) if model_status.get("tokens") else None,
            "missing_files": model_status["missing_files"],
            "prefer_int8": self.prefer_int8,
            "audio_gate": {"min_rms": self.min_rms, "min_peak": self.min_peak},
            "edge_padding_ms": self.edge_padding_ms,
            "sherpa_available": sherpa_info["available"],
            "sherpa_version": sherpa_info["version"],
            "sherpa_import_error": sherpa_info["error"],
            "recognizer_api": self._recognizer_api,
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "last_error": self.last_error,
            "last_text": self.last_text,
            "last_latency_ms": self.last_latency_ms,
            "last_audio_stats": self.last_audio_stats,
            "last_suppressed_text": self.last_suppressed_text,
            "recent_jobs": recent,
        }

    def _run(self) -> None:
        while self._running:
            job_id = self._queue.get()
            if job_id is None:
                break
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                job.status = "running"
                job.updated_at = time.time()
            started = time.monotonic()
            try:
                text = self._transcribe(job.audio_path)
                latency = int((time.monotonic() - started) * 1000)
                with self._lock:
                    job.text = text
                    job.status = "done"
                    job.latency_ms = latency
                    job.audio_stats = self.last_audio_stats
                    job.suppressed_text = self.last_suppressed_text
                    job.updated_at = time.time()
                self.last_text = text
                self.last_latency_ms = latency
                self.last_error = None
            except Exception as exc:
                latency = int((time.monotonic() - started) * 1000)
                error = str(exc)
                with self._lock:
                    job.status = "failed"
                    job.error = error
                    job.latency_ms = latency
                    job.audio_stats = self.last_audio_stats
                    job.suppressed_text = self.last_suppressed_text
                    job.updated_at = time.time()
                self.last_error = error
                self.last_latency_ms = latency

    def _transcribe(self, audio_path: str | Path) -> str:
        if self.provider not in {"sherpa_paraformer", "paraformer"}:
            raise RuntimeError(f"Unsupported ASR provider: {self.provider}")
        self._ensure_recognizer()
        assert self._recognizer is not None
        samples, sample_rate = self._read_wave_mono_float32(audio_path)
        stats = self._audio_stats(samples, sample_rate)
        self.last_audio_stats = stats
        self.last_suppressed_text = None
        if not self._audio_has_speech(stats):
            self.last_suppressed_text = "audio_gate"
            return ""
        samples = _pad_waveform_edges(samples, sample_rate, self.edge_padding_ms)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        text = str(stream.result.text or "").strip()
        if _looks_like_asr_hallucination(text):
            self.last_suppressed_text = text
            return ""
        return text

    def _model_status(self) -> dict[str, Any]:
        model_names = ["model.int8.onnx", "model.onnx", "*.onnx"] if self.prefer_int8 else ["model.onnx", "model.int8.onnx", "*.onnx"]
        model = _find_first(self.model_dir, model_names)
        tokens = _find_first(self.model_dir, ["tokens.txt"])
        missing = []
        if model is None:
            missing.append("model.onnx 或 model.int8.onnx")
        if tokens is None:
            missing.append("tokens.txt")
        return {"available": model is not None and tokens is not None, "model": model, "tokens": tokens, "missing_files": missing}

    def _sherpa_info(self) -> dict[str, Any]:
        if self._sherpa_onnx is not None:
            version = str(getattr(self._sherpa_onnx, "__version__", self._sherpa_version or "unknown"))
            self._sherpa_version = version
            return {"available": True, "version": version, "error": None}
        try:
            import sherpa_onnx
        except Exception as exc:
            return {"available": False, "version": None, "error": str(exc)}
        self._sherpa_onnx = sherpa_onnx
        self._sherpa_version = str(getattr(sherpa_onnx, "__version__", "unknown"))
        return {"available": True, "version": self._sherpa_version, "error": None}

    def _ensure_recognizer(self) -> None:
        if self._recognizer is not None:
            return
        model_status = self._model_status()
        model = model_status.get("model")
        tokens = model_status.get("tokens")
        if not model_status.get("available"):
            missing = ", ".join(model_status.get("missing_files") or [])
            raise FileNotFoundError(f"ASR模型缺失：{self.model_dir} 缺少 {missing}。请复制 sherpa-onnx paraformer 中文模型，或设置 REHAB_ASR_MODEL_DIR。")
        sherpa_info = self._sherpa_info()
        if not sherpa_info["available"]:
            raise RuntimeError(f"sherpa_onnx import failed: {sherpa_info['error']}")
        assert self._sherpa_onnx is not None
        assert isinstance(model, Path)
        assert isinstance(tokens, Path)
        self._recognizer = self._build_recognizer(self._sherpa_onnx, model=model, tokens=tokens)

    def _build_recognizer(self, sherpa_onnx: Any, *, model: Path, tokens: Path) -> Any:
        recognizer_cls = getattr(sherpa_onnx, "OfflineRecognizer", None)
        from_paraformer_error: str | None = None
        if recognizer_cls is not None and hasattr(recognizer_cls, "from_paraformer"):
            try:
                self._recognizer_api = "OfflineRecognizer.from_paraformer"
                return recognizer_cls.from_paraformer(
                    paraformer=str(model),
                    tokens=str(tokens),
                    num_threads=self.num_threads,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False,
                    provider="cpu",
                )
            except TypeError:
                try:
                    return self._build_from_paraformer_legacy_kwargs(recognizer_cls, model=model, tokens=tokens)
                except RuntimeError as exc:
                    from_paraformer_error = str(exc)
            except Exception:
                self._recognizer_api = None
                raise

        required_legacy = [
            "OfflineRecognizerConfig",
            "FeatureConfig",
            "OfflineModelConfig",
            "OfflineParaformerModelConfig",
            "OfflineRecognizer",
        ]
        if all(hasattr(sherpa_onnx, name) for name in required_legacy):
            self._recognizer_api = "OfflineRecognizerConfig"
            config = sherpa_onnx.OfflineRecognizerConfig(
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=16000, feature_dim=80),
                model_config=sherpa_onnx.OfflineModelConfig(
                    paraformer=sherpa_onnx.OfflineParaformerModelConfig(
                        model=str(model),
                        tokens=str(tokens),
                    ),
                    num_threads=self.num_threads,
                    debug=False,
                    provider="cpu",
                ),
            )
            if not config.validate():
                raise RuntimeError("Paraformer ASR config validate failed")
            return sherpa_onnx.OfflineRecognizer(config)

        available = ", ".join(name for name in required_legacy if hasattr(sherpa_onnx, name))
        has_from_paraformer = bool(recognizer_cls is not None and hasattr(recognizer_cls, "from_paraformer"))
        raise RuntimeError(
            "当前 sherpa_onnx 版本无法创建 Paraformer ASR recognizer："
            f"version={self._sherpa_version}, "
            f"has_from_paraformer={has_from_paraformer}, "
            f"from_paraformer_error={from_paraformer_error or '-'}, "
            f"legacy_available=[{available or 'none'}], "
            f"model_dir={self.model_dir}, model={model.name}, tokens={tokens.name}"
        )

    def _build_from_paraformer_legacy_kwargs(self, recognizer_cls: Any, *, model: Path, tokens: Path) -> Any:
        attempts = [
            {
                "paraformer": str(model),
                "tokens": str(tokens),
                "num_threads": self.num_threads,
                "sample_rate": 16000,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "debug": False,
                "provider": "cpu",
            },
            {
                "model": str(model),
                "tokens": str(tokens),
                "num_threads": self.num_threads,
                "sample_rate": 16000,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "debug": False,
                "provider": "cpu",
            },
            {
                "paraformer": str(model),
                "tokens": str(tokens),
                "num_threads": self.num_threads,
                "decoding_method": "greedy_search",
                "debug": False,
                "provider": "cpu",
            },
        ]
        errors = []
        for kwargs in attempts:
            try:
                self._recognizer_api = "OfflineRecognizer.from_paraformer"
                return recognizer_cls.from_paraformer(**kwargs)
            except TypeError as exc:
                errors.append(str(exc))
        self._recognizer_api = None
        raise RuntimeError(
            "sherpa_onnx.OfflineRecognizer.from_paraformer 参数不兼容："
            f"version={self._sherpa_version}, errors={errors[-3:]}"
        )

    def _audio_stats(self, samples: Any, sample_rate: int) -> dict[str, Any]:
        import numpy as np

        if len(samples) == 0:
            return {"duration_seconds": 0.0, "rms": 0.0, "peak": 0.0, "mean_abs": 0.0}
        abs_samples = np.abs(samples)
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(abs_samples))
        mean_abs = float(np.mean(abs_samples))
        return {
            "duration_seconds": round(float(len(samples) / max(1, sample_rate)), 3),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "mean_abs": round(mean_abs, 6),
        }

    def _audio_has_speech(self, stats: dict[str, Any]) -> bool:
        if float(stats.get("duration_seconds") or 0.0) < 0.4:
            return False
        return float(stats.get("rms") or 0.0) >= self.min_rms and float(stats.get("peak") or 0.0) >= self.min_peak

    @staticmethod
    def _read_wave_mono_float32(path: str | Path):
        import numpy as np

        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2:
            raise RuntimeError(f"Only 16-bit PCM WAV is supported, got {sample_width * 8}-bit")
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return audio.astype("float32") / 32768.0, sample_rate

    def _trim_jobs_locked(self) -> None:
        while len(self._jobs) > self.max_jobs:
            self._jobs.popitem(last=False)


def transcribe_wav_file(audio_path: str | Path, *, model_dir: str | Path | None = None) -> dict[str, Any]:
    worker = ParaformerAsrWorker(model_dir=Path(model_dir) if model_dir else None)
    started = time.monotonic()
    text = worker._transcribe(audio_path)
    latency_ms = int((time.monotonic() - started) * 1000)
    status = worker.snapshot()
    return {
        "ok": True,
        "text": text,
        "latency_ms": latency_ms,
        "model_dir": status["model_dir"],
        "model_path": status["model_path"],
        "tokens_path": status["tokens_path"],
        "sherpa_version": status["sherpa_version"],
        "recognizer_api": status["recognizer_api"],
        "audio_stats": status["last_audio_stats"],
        "suppressed_text": status["last_suppressed_text"],
    }


def _looks_like_asr_hallucination(text: str) -> bool:
    value = str(text or "").strip().replace(" ", "")
    if not value:
        return False
    known_bad = {"没有没有有没有没有异议", "没有没有有没有没有有异议"}
    if value in known_bad:
        return True
    if value.count("没有") >= 3 and len(value) <= 16:
        return True
    if value.endswith("异议") and value.count("有") + value.count("没") >= 5:
        return True
    return False


def _pad_waveform_edges(samples: Any, sample_rate: int, padding_ms: int):
    if padding_ms <= 0 or len(samples) == 0:
        return samples
    import numpy as np

    padding_samples = max(1, int(round(sample_rate * padding_ms / 1000.0)))
    return np.pad(samples, (padding_samples, padding_samples), mode="constant")


def _find_first(root: Path, names: list[str]) -> Path | None:
    for name in names:
        if "*" in name:
            matches = sorted(root.glob(name))
            if matches:
                return matches[0]
        else:
            path = root / name
            if path.exists():
                return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test sherpa-onnx Paraformer ASR with an existing WAV file.")
    parser.add_argument("wav", help="16-bit PCM WAV file to transcribe.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Paraformer model directory containing tokens.txt and model.onnx/model.int8.onnx.")
    args = parser.parse_args()
    try:
        result = transcribe_wav_file(args.wav, model_dir=args.model_dir)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "model_dir": args.model_dir}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
