"""Asynchronous prioritized TTS worker used by realtime training."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from feedback.tts.banzi.tts_board import speak_with_espeak, speak_with_pyttsx3
from realtime.natural_tts import NaturalTTS


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
}


class TTSWorker:
    def __init__(
        self,
        global_cooldown: float = 3.0,
        same_text_cooldown: float = 5.0,
        use_real_tts: bool = True,
        natural_tts_options: dict[str, Any] | None = None,
    ) -> None:
        self.global_cooldown = max(0.0, float(global_cooldown))
        self.same_text_cooldown = max(0.0, float(same_text_cooldown))
        self.use_real_tts = use_real_tts
        self.natural_tts_options = dict(natural_tts_options or {})
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
        self._natural_tts: NaturalTTS | None = None

    def start(self) -> None:
        if self._running:
            return
        self._init_backend()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(TTSItem(-1, self._sequence + 1, "", "high", "__stop__"))

    def speak(self, text: str, priority: str = "low", event_type: str | None = None) -> bool:
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

        if priority == "high":
            self._drop_lower_priority_items()

        self._last_spoken_at = now
        self._last_text_at[text] = now
        self._sequence += 1
        self._queue.put(TTSItem(PRIORITY_VALUE[priority], self._sequence, text, priority, event_type))
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "queued": self._queue.qsize(),
            "last_text": self.last_text,
            "last_error": self.last_error,
            "last_priority": self.last_priority,
            "backend": self.backend_name,
            "mock_fallback": self.mock_fallback,
        }

    def _init_backend(self) -> None:
        if not self.use_real_tts:
            self.backend_name = "mock"
            return
        natural = NaturalTTS(**self.natural_tts_options)
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
            ok = False
            if self.use_real_tts and self._natural_tts is not None:
                ok = self._natural_tts.speak(item.text)
                self.last_error = self._natural_tts.last_error
                if ok:
                    self.backend_name = "natural_tts"
            if self.use_real_tts and not ok:
                ok = speak_with_pyttsx3(item.text) or speak_with_espeak(item.text)
                if ok:
                    self.backend_name = "pyttsx3/espeak"
            if not ok:
                self.mock_fallback = True
                self.backend_name = "mock"
                print(f"[TTS MOCK] {item.text}")

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
