from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


BLOCKED_TRAINING_STATUSES = {
    "running",
    "paused",
    "awaiting_orientation",
    "awaiting_return",
    "awaiting_care_response",
}


@dataclass
class LLMJob:
    job_id: str
    report: dict[str, Any]
    question: str
    report_payload: dict[str, Any] = field(default_factory=dict)
    speak: bool = False
    status: str = "queued"
    active_provider: str | None = None
    answer: str = ""
    spoken_text: str = ""
    error: str | None = None
    latency_ms: int | None = None
    tts: dict[str, Any] | None = None
    speech_generation: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "active_provider": self.active_provider,
            "answer": self.answer,
            "spoken_text": self.spoken_text,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tts": self.tts,
            "speech_generation": self.speech_generation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **self.report_payload,
        }


class VoiceLLMWorker:
    def __init__(
        self,
        *,
        answer_fn: Callable[[dict[str, Any], str, bool], dict[str, Any]],
        training_status_fn: Callable[[], str],
        speak_fn: Callable[[object, str], dict[str, Any]],
        max_queue_size: int = 2,
        max_jobs: int = 20,
    ) -> None:
        self.answer_fn = answer_fn
        self.training_status_fn = training_status_fn
        self.speak_fn = speak_fn
        self.max_jobs = max_jobs
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._jobs: OrderedDict[str, LLMJob] = OrderedDict()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self.last_error: str | None = None
        self.last_latency_ms: int | None = None
        self.last_active_provider: str | None = None
        self.last_block_reason: str | None = None
        self._speech_generation = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="voice-llm-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def submit(
        self,
        *,
        report: dict[str, Any],
        question: str,
        report_payload: dict[str, Any] | None = None,
        speak: bool = False,
    ) -> dict[str, Any]:
        if not self._running:
            self.start()
        job_id = f"llm_{uuid.uuid4().hex[:12]}"
        job = LLMJob(
            job_id=job_id,
            report=report,
            question=str(question or "").strip(),
            report_payload=dict(report_payload or {}),
            speak=bool(speak),
        )
        with self._lock:
            self._speech_generation += 1
            job.speech_generation = self._speech_generation
        block = self._training_block()
        if block:
            job.status = "blocked_training"
            job.error = block
            job.answer = block
            job.spoken_text = block
            self.last_block_reason = block
            with self._lock:
                self._jobs[job_id] = job
                self._trim_jobs_locked()
            return {"ok": True, "job_id": job_id, "status": job.status}
        with self._lock:
            self._jobs[job_id] = job
            self._trim_jobs_locked()
        try:
            self._queue.put_nowait(job_id)
        except queue.Full:
            with self._lock:
                job.status = "failed"
                job.error = "LLM queue is full"
                job.updated_at = time.time()
            return {"ok": False, "job_id": job_id, "error": job.error}
        return {"ok": True, "job_id": job_id, "status": "queued"}

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"ok": False, "error": "LLM job not found", "job_id": job_id}
            return {"ok": job.status not in {"failed"}, **job.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
            recent = [job.snapshot() for job in jobs[-5:]]
            current = next((job.snapshot() for job in reversed(jobs) if job.status in {"queued", "running"}), None)
            last_completed = next(
                (job.snapshot() for job in reversed(jobs) if job.status in {"done", "failed", "blocked_training"}),
                None,
            )
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
            "last_active_provider": self.last_active_provider,
            "last_block_reason": self.last_block_reason,
            "current_job": current,
            "last_completed_job": last_completed,
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
            block = self._training_block()
            if block:
                with self._lock:
                    job.status = "blocked_training"
                    job.error = block
                    job.answer = block
                    job.spoken_text = block
                    job.updated_at = time.time()
                self.last_block_reason = block
                continue
            started = time.monotonic()
            try:
                result = self.answer_fn(job.report, job.question, True)
                latency = int((time.monotonic() - started) * 1000)
                with self._lock:
                    job.status = "done" if result.get("ok") else "failed"
                    job.active_provider = str(result.get("active_provider") or result.get("provider") or "")
                    job.answer = str(result.get("answer") or result.get("message") or "")
                    job.spoken_text = str(result.get("spoken_text") or job.answer)
                    job.error = None if result.get("ok") else str(result.get("last_error") or result.get("message") or result.get("error") or "")
                    job.latency_ms = int(result.get("latency_ms") or latency)
                    job.report_payload.update({k: v for k, v in result.items() if k.startswith("source_") or k == "report_file" or k.startswith("qwen_") or k.startswith("rkllm_")})
                    should_speak = bool(result.get("ok") and job.speak and job.spoken_text)
                    if should_speak:
                        job.tts = {"pending": True, "queued": False}
                    job.updated_at = time.time()
                self.last_error = job.error
                self.last_latency_ms = job.latency_ms
                self.last_active_provider = job.active_provider
                if should_speak:
                    self._speak_job_async(job_id, job.spoken_text, job.speech_generation)
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

    def _speak_job_async(self, job_id: str, spoken_text: str, speech_generation: int) -> None:
        def run() -> None:
            with self._lock:
                if speech_generation != self._speech_generation:
                    job = self._jobs.get(job_id)
                    if job is not None:
                        job.tts = {"ok": False, "skipped": "stale_speech_generation"}
                        job.updated_at = time.time()
                    return
            try:
                tts_result = self.speak_fn(spoken_text, "llm_qa")
            except Exception as exc:
                tts_result = {"ok": False, "error": str(exc)}
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    if speech_generation == self._speech_generation:
                        job.tts = tts_result
                    else:
                        job.tts = {"ok": False, "skipped": "stale_speech_generation"}
                    job.updated_at = time.time()

        threading.Thread(target=run, name=f"voice-tts-{job_id}", daemon=True).start()

    def _training_block(self) -> str | None:
        status = str(self.training_status_fn() or "")
        if status in BLOCKED_TRAINING_STATUSES:
            return "Voice QA is blocked while realtime training is active. Please ask during rest or after training."
        return None

    def _trim_jobs_locked(self) -> None:
        while len(self._jobs) > self.max_jobs:
            self._jobs.popitem(last=False)

