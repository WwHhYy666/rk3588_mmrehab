"""Asynchronous prioritized TTS worker used by realtime training."""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from action_feedback.tts.board.tts_backend import speak_with_espeak, speak_with_pyttsx3
from training.audio_output import play_wav_files
from training.natural_tts import NaturalTTS


PRIORITY_VALUE = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


@dataclass(order=True)
class TTSItem:
    priority_value: int
    sequence: int
    text: str = field(compare=False)
    priority: str = field(compare=False, default="low")
    event_type: str | None = field(compare=False, default=None)
    phrase_key: str | None = field(compare=False, default=None)
    audio_path: Path | None = field(compare=False, default=None)
    queued_at: float = field(compare=False, default=0.0)


ALLOWED_EVENT_TYPES = {
    "welcome",
    "action_start",
    "rep_count",
    "correction",
    "set_done",
    "training_finished",
    "orientation",
    "offscreen",
    "care",
    "encouragement",
    "resume",
    "llm_summary",
    "llm_qa",
}


class TTSWorker:
    def __init__(
        self,
        global_cooldown: float = 3.0,
        same_text_cooldown: float = 5.0,
        use_real_tts: bool = True,
        lazy_real_tts_init: bool = False,
        natural_tts_options: dict[str, Any] | None = None,
        phrase_config_path: str | Path | None = None,
        project_root: str | Path | None = None,
        fixed_audio_only: bool = False,
    ) -> None:
        self.global_cooldown = max(0.0, float(global_cooldown))
        self.same_text_cooldown = max(0.0, float(same_text_cooldown))
        self.use_real_tts = use_real_tts
        self.lazy_real_tts_init = lazy_real_tts_init
        self.natural_tts_options = dict(natural_tts_options or {})
        self.fixed_audio_only = bool(fixed_audio_only)
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        self.phrase_config_path = Path(phrase_config_path) if phrase_config_path else None
        if self.phrase_config_path is not None and not self.phrase_config_path.is_absolute():
            self.phrase_config_path = self.project_root / self.phrase_config_path
        self._phrase_audio_root: Path | None = None
        self._phrase_by_key: dict[str, dict[str, str]] = {}
        self._phrase_key_by_text: dict[str, str] = {}
        self._queue: queue.PriorityQueue[TTSItem] = queue.PriorityQueue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._sequence = 0
        self._last_spoken_at = 0.0
        self._last_text_at: dict[str, float] = {}
        self.last_text: str | None = None
        self.last_error: str | None = None
        self.last_priority: str | None = None
        self.backend_name = "mock"
        self.mock_fallback = False
        self.fixed_audio_hits = 0
        self.fixed_audio_misses = 0
        self.last_audio_path: str | None = None
        self.audio_output_device = os.environ.get("REHAB_AUDIO_OUTPUT_DEVICE", "").strip() or None
        self.last_audio_command: str | None = None
        self.last_audio_returncode: int | None = None
        self.last_phrase_key: str | None = None
        self.last_missing_phrase_key: str | None = None
        self._natural_tts: NaturalTTS | None = None
        self._real_tts_initialized = False
        self._state_lock = threading.Lock()
        self._speaking = False
        self._current_event_type: str | None = None
        self._current_text: str | None = None
        self._last_finished_at = 0.0
        self.last_queued_at: float | None = None
        self.last_queued_event_type: str | None = None
        self.last_queued_phrase_key: str | None = None
        self.last_started_at: float | None = None
        self.last_started_event_type: str | None = None
        self.last_started_phrase_key: str | None = None
        self.last_finished_at: float | None = None
        self.last_queue_delay_seconds: float | None = None

    def start(self) -> None:
        if self._running:
            return
        self._init_backend()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(
            TTSItem(
                priority_value=-1,
                sequence=self._sequence + 1,
                text="",
                priority="high",
                event_type="__stop__",
                queued_at=time.time(),
            )
        )

    def clear_pending(self, event_types: set[str] | list[str] | tuple[str, ...] | None = None) -> int:
        targets = set(event_types or [])
        removed = 0
        kept: list[TTSItem] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item.event_type == "__stop__":
                kept.append(item)
                continue
            if not targets or item.event_type in targets:
                removed += 1
                continue
            kept.append(item)
        for item in kept:
            self._queue.put(item)
        return removed
    def speak(
        self,
        text: str,
        priority: str = "low",
        event_type: str | None = None,
        phrase_key: str | None = None,
    ) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if event_type not in ALLOWED_EVENT_TYPES:
            return False
        priority = priority if priority in PRIORITY_VALUE else "low"
        now = time.time()
        if priority != "high" and now - self._last_spoken_at < self.global_cooldown:
            return False
        if priority != "high" and now - self._last_text_at.get(text, 0.0) < self.same_text_cooldown:
            return False

        phrase_key = str(phrase_key or "").strip() or self._infer_phrase_key(text, event_type)
        audio_path = self._audio_path_for_phrase(phrase_key)
        if self.fixed_audio_only and audio_path is None:
            return False

        if event_type == "correction":
            self._drop_event_type_items("correction")
        elif event_type in {"rep_count", "training_finished", "care", "offscreen", "orientation", "action_start", "set_done"}:
            self._drop_event_type_items("correction")

        if priority == "high":
            self._drop_lower_priority_items()

        self._last_spoken_at = now
        self._last_text_at[text] = now
        self._sequence += 1
        self.last_queued_at = now
        self.last_queued_event_type = event_type
        self.last_queued_phrase_key = phrase_key
        self._queue.put(
            TTSItem(
                priority_value=PRIORITY_VALUE[priority],
                sequence=self._sequence,
                text=text,
                priority=priority,
                event_type=event_type,
                phrase_key=phrase_key,
                audio_path=audio_path,
                queued_at=now,
            )
        )
        return True

    def snapshot(self) -> dict[str, Any]:
        queued = self._queue.qsize()
        with self._state_lock:
            speaking = self._speaking
            current_event_type = self._current_event_type
            current_text = self._current_text
        return {
            "running": self._running,
            "queued": queued,
            "speaking": speaking,
            "busy": self._running and (speaking or queued > 0),
            "current_event_type": current_event_type,
            "current_text": current_text,
            "last_text": self.last_text,
            "last_error": self.last_error,
            "last_priority": self.last_priority,
            "backend": self.backend_name,
            "mock_fallback": self.mock_fallback,
            "last_audio_path": self.last_audio_path,
            "audio_output_device": self.audio_output_device,
            "last_audio_command": self.last_audio_command,
            "last_audio_returncode": self.last_audio_returncode,
            "last_phrase_key": self.last_phrase_key,
            "last_missing_phrase_key": self.last_missing_phrase_key,
            "fixed_audio_hits": self.fixed_audio_hits,
            "fixed_audio_misses": self.fixed_audio_misses,
            "lazy_real_tts_init": self.lazy_real_tts_init,
            "real_tts_initialized": self._real_tts_initialized,
            "phrase_catalog_size": len(self._phrase_by_key),
            "fixed_audio_only": self.fixed_audio_only,
            "last_queued_at": self.last_queued_at,
            "last_queued_event_type": self.last_queued_event_type,
            "last_queued_phrase_key": self.last_queued_phrase_key,
            "last_started_at": self.last_started_at,
            "last_started_event_type": self.last_started_event_type,
            "last_started_phrase_key": self.last_started_phrase_key,
            "last_finished_at": self.last_finished_at,
            "last_queue_delay_seconds": self.last_queue_delay_seconds,
        }

    def is_busy(self, *, extra_guard_seconds: float = 0.0) -> bool:
        with self._state_lock:
            if self._speaking:
                return True
            last_finished_at = self._last_finished_at
        if self._running and self._queue.qsize() > 0:
            return True
        extra_guard_seconds = max(0.0, float(extra_guard_seconds))
        return extra_guard_seconds > 0.0 and time.time() - last_finished_at < extra_guard_seconds

    def _init_backend(self) -> None:
        self._load_phrase_catalog()
        if not self.use_real_tts:
            self.backend_name = "mock"
            return
        if self.lazy_real_tts_init:
            self.backend_name = "fixed_wav"
            return
        self._init_real_tts_backend()

    def _init_real_tts_backend(self) -> None:
        if self._real_tts_initialized:
            return
        self._real_tts_initialized = True
        natural_options = dict(self.natural_tts_options)
        natural_options.setdefault("audio_output_device", self.audio_output_device)
        natural = NaturalTTS(**natural_options)
        if natural.is_available():
            self._natural_tts = natural
            self.backend_name = natural.backend_name
            print("[TTS] 使用自然女声 sherpa-onnx backend")
            return
        self.last_error = natural.last_error
        self.backend_name = "pyttsx3/espeak"
        print(f"[TTS] 自然女声不可用，降级到 pyttsx3/espeak: {self.last_error}")

    def _run(self) -> None:
        while self._running:
            item = self._queue.get()
            if item.event_type == "__stop__":
                break
            self.last_text = item.text
            self.last_priority = item.priority
            started_at = time.time()
            with self._state_lock:
                self._speaking = True
                self._current_event_type = item.event_type
                self._current_text = item.text
                self.last_started_at = started_at
                self.last_started_event_type = item.event_type
                self.last_started_phrase_key = item.phrase_key
                self.last_queue_delay_seconds = max(0.0, started_at - float(item.queued_at or started_at))
            ok = False
            try:
                if item.audio_path is not None:
                    ok = self._play_audio_file(item.audio_path)
                    if ok:
                        self.backend_name = "fixed_wav"
                        self.last_audio_path = self._project_relative(item.audio_path)
                if self.use_real_tts and not self.fixed_audio_only and not ok:
                    self._init_real_tts_backend()
                    if self._natural_tts is not None:
                        ok = self._natural_tts.speak(item.text)
                        self.last_error = self._natural_tts.last_error
                        self.last_audio_command = self._natural_tts.last_audio_command
                        self.last_audio_returncode = self._natural_tts.last_audio_returncode
                        if ok:
                            self.backend_name = "natural_tts"
                    if not ok:
                        ok = speak_with_pyttsx3(item.text) or speak_with_espeak(item.text)
                        if ok:
                            self.backend_name = "pyttsx3/espeak"
                if not ok:
                    self.mock_fallback = True
                    self.backend_name = "mock"
                    print(f"[TTS MOCK] {item.text}")
            finally:
                with self._state_lock:
                    self._speaking = False
                    self._current_event_type = None
                    self._current_text = None
                    self._last_finished_at = time.time()
                    self.last_finished_at = self._last_finished_at

    def _drop_lower_priority_items(self) -> None:
        kept: list[TTSItem] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item.priority_value <= PRIORITY_VALUE["high"]:
                kept.append(item)
        for item in kept:
            self._queue.put(item)

    def _drop_event_type_items(self, event_type: str) -> None:
        kept: list[TTSItem] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item.event_type != event_type:
                kept.append(item)
        for item in kept:
            self._queue.put(item)

    def _load_phrase_catalog(self) -> None:
        if self.phrase_config_path is None or not self.phrase_config_path.exists():
            return
        try:
            payload = yaml.safe_load(self.phrase_config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            self.last_error = f"phrase config load failed: {exc}"
            return
        if not isinstance(payload, dict):
            return
        audio_root = str(payload.get("audio_root") or "rehab_app/server/static/assets/tts")
        root = Path(audio_root)
        self._phrase_audio_root = root if root.is_absolute() else self.project_root / root
        phrases = payload.get("phrases")
        if not isinstance(phrases, dict):
            return
        for key, value in phrases.items():
            if not isinstance(value, dict):
                continue
            text = str(value.get("text") or "").strip()
            file_name = str(value.get("file") or "").strip()
            if not text or not file_name:
                continue
            phrase_key = str(key).strip()
            self._phrase_by_key[phrase_key] = {"text": text, "file": file_name}
            self._phrase_key_by_text[self._normalize_text(text)] = phrase_key

    def _infer_phrase_key(self, text: str, event_type: str | None) -> str | None:
        normalized = self._normalize_text(text)
        direct = self._phrase_key_by_text.get(normalized)
        if direct:
            return direct
        if event_type == "rep_count":
            return {"一": "count_1", "二": "count_2", "三": "count_3", "四": "count_4", "五": "count_5"}.get(normalized)
        if event_type == "correction":
            match = re.search(r"再坚持\s*([0-9一二三四五])\s*秒", text)
            if match:
                number = match.group(1)
                value = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}.get(number)
                if value is None:
                    try:
                        value = int(number)
                    except ValueError:
                        value = None
                if value is not None:
                    return f"tut_{min(5, max(1, value))}"
        return None

    def _audio_path_for_phrase(self, phrase_key: str | None) -> Path | None:
        self.last_phrase_key = phrase_key
        if not phrase_key:
            return None
        item = self._phrase_by_key.get(phrase_key)
        if not item:
            self.fixed_audio_misses += 1
            self.last_missing_phrase_key = phrase_key
            return None
        file_value = item.get("file") or ""
        path = Path(file_value)
        if not path.is_absolute():
            root = self._phrase_audio_root or (self.project_root / "rehab_app" / "server" / "static" / "assets" / "tts")
            path = root / path
        if path.exists():
            return path
        self.fixed_audio_misses += 1
        self.last_missing_phrase_key = phrase_key
        return None

    def _play_audio_file(self, path: Path) -> bool:
        playback = play_wav_files([path], self.audio_output_device)
        self.last_audio_command = " ".join(str(part) for part in playback.get("command") or []) or None
        self.last_audio_returncode = playback.get("returncode")
        ok = bool(playback.get("ok"))
        if ok:
            self.fixed_audio_hits += 1
            self.last_error = None
        else:
            detail = str(playback.get("error") or "aplay failed")
            self.last_error = f"fixed audio failed: {detail}"
            self.fixed_audio_misses += 1
        return ok

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", "", str(text or "").strip())

    def _project_relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return str(path)
