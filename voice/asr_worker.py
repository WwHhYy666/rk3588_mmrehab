from __future__ import annotations

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
DEFAULT_MODEL_DIR = Path(os.getenv("REHAB_ASR_MODEL_DIR", "/home/elf/models/sherpa-onnx-paraformer-zh"))
DEFAULT_PROVIDER = os.getenv("REHAB_ASR_PROVIDER", "sherpa_paraformer").strip() or "sherpa_paraformer"
DEFAULT_NUM_THREADS = int(os.getenv("REHAB_ASR_NUM_THREADS", "2"))


@dataclass
class AsrJob:
    job_id: str
    audio_path: str
    status: str = "queued"
    text: str = ""
    error: str | None = None
    latency_ms: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "text": self.text,
            "error": self.error,
            "latency_ms": self.latency_ms,
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
    ) -> None:
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.provider = provider
        self.num_threads = num_threads
        self.max_jobs = max_jobs
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._jobs: OrderedDict[str, AsrJob] = OrderedDict()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._recognizer: Any | None = None
        self._sherpa_onnx: Any | None = None
        self.last_error: str | None = None
        self.last_text: str = ""
        self.last_latency_ms: int | None = None

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
        return {
            "provider": self.provider,
            "backend": "sherpa_onnx_paraformer",
            "model_dir": str(self.model_dir),
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "last_error": self.last_error,
            "last_text": self.last_text,
            "last_latency_ms": self.last_latency_ms,
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
                    job.updated_at = time.time()
                self.last_error = error
                self.last_latency_ms = latency

    def _transcribe(self, audio_path: str | Path) -> str:
        if self.provider not in {"sherpa_paraformer", "paraformer"}:
            raise RuntimeError(f"Unsupported ASR provider: {self.provider}")
        self._ensure_recognizer()
        assert self._recognizer is not None
        samples, sample_rate = self._read_wave_mono_float32(audio_path)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text or "").strip()

    def _ensure_recognizer(self) -> None:
        if self._recognizer is not None:
            return
        model = _find_first(self.model_dir, ["model.int8.onnx", "model.onnx", "*.onnx"])
        tokens = _find_first(self.model_dir, ["tokens.txt"])
        if model is None or tokens is None:
            raise FileNotFoundError(f"Paraformer model/tokens not found in {self.model_dir}")
        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError(f"sherpa_onnx import failed: {exc}") from exc
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
        self._sherpa_onnx = sherpa_onnx
        self._recognizer = sherpa_onnx.OfflineRecognizer(config)

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

