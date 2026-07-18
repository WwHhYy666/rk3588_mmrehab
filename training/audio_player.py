"""Board-side WAV playback helpers for realtime training."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Any


class RestAudioPlayer:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.runtime_dir = project_root / "runtime" / "audio"
        self.lock = threading.RLock()
        self.process: subprocess.Popen[Any] | None = None
        self.launch_token = 0
        self.last_error: str | None = None
        self.last_file: str | None = None
        self.audio_output_device = os.environ.get("REHAB_AUDIO_OUTPUT_DEVICE", "").strip() or None
        self.last_command: str | None = None
        self.last_returncode: int | None = None
        self.retry_count = 0

    def play(
        self,
        file_value: str,
        *,
        duration_seconds: float,
        fade_seconds: float,
        delay_seconds: float = 0.0,
    ) -> None:
        source = self._resolve_file(file_value)
        with self.lock:
            self._stop_locked()
            self.launch_token += 1
            token = self.launch_token
            self.last_error = None
            self.last_file = self._project_relative(source)
            if not source.exists():
                self.last_error = f"audio file not found: {self.last_file}"
                return

        thread = threading.Thread(
            target=self._delayed_play,
            args=(token, source, duration_seconds, fade_seconds, delay_seconds),
            daemon=True,
        )
        thread.start()

    def stop(self) -> None:
        with self.lock:
            self.launch_token += 1
            self._stop_locked()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "running": running,
                "last_file": self.last_file,
                "last_error": self.last_error,
                "audio_output_device": self.audio_output_device,
                "last_command": self.last_command,
                "last_returncode": self.last_returncode,
                "retry_count": self.retry_count,
                "returncode": self.process.poll() if self.process is not None else None,
            }


    def _delayed_play(
        self,
        token: int,
        source: Path,
        duration_seconds: float,
        fade_seconds: float,
        delay_seconds: float,
    ) -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        with self.lock:
            if token != self.launch_token:
                return
        try:
            play_path = self._prepare_wav(source, duration_seconds=duration_seconds, fade_seconds=fade_seconds)
        except Exception as exc:
            play_path = source
            with self.lock:
                self.last_error = f"fade fallback: {exc}"
        retry_until = time.monotonic() + min(4.0, max(0.5, duration_seconds))
        attempt = 0
        while True:
            with self.lock:
                if token != self.launch_token:
                    return
            attempt += 1
            try:
                command = ["aplay"]
                if self.audio_output_device:
                    command.extend(["-D", self.audio_output_device])
                command.append(str(play_path))
                with self.lock:
                    self.last_command = " ".join(command)
                    self.retry_count = max(0, attempt - 1)
                    self.last_returncode = None
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
                return
            with self.lock:
                if token != self.launch_token:
                    process.terminate()
                    return
                self.process = process
            try:
                _, stderr = process.communicate(timeout=0.25)
            except subprocess.TimeoutExpired:
                return
            error_text = ""
            if isinstance(stderr, bytes):
                error_text = stderr.decode("utf-8", errors="replace")
            elif stderr:
                error_text = str(stderr)
            with self.lock:
                if self.process is process:
                    self.process = None
                self.last_returncode = process.returncode
                self.last_error = f"aplay failed ({process.returncode}): {error_text.strip()[-300:]}" if process.returncode else None
            if process.returncode == 0 or time.monotonic() >= retry_until:
                return
            time.sleep(0.35)

    def _prepare_wav(self, source: Path, *, duration_seconds: float, fade_seconds: float) -> Path:
        if duration_seconds <= 0:
            return source
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.runtime_dir / "rest_music_current.wav"
        with wave.open(str(source), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            frame_rate = reader.getframerate()
            frame_count = reader.getnframes()
            params = reader.getparams()
            if sample_width != 2:
                raise ValueError(f"unsupported sample width: {sample_width}")
            raw = reader.readframes(frame_count)
        frame_size = channels * sample_width
        if frame_size <= 0 or not raw:
            raise ValueError("empty wav")
        source_frames = len(raw) // frame_size
        target_frames = max(1, int(round(duration_seconds * frame_rate)))
        needed_bytes = target_frames * frame_size
        repeats = (needed_bytes // len(raw)) + 1
        looped = (raw * repeats)[:needed_bytes]
        faded = self._apply_fade(looped, channels=channels, frame_rate=frame_rate, fade_seconds=fade_seconds)
        with wave.open(str(out_path), "wb") as writer:
            writer.setparams(params)
            writer.setnframes(target_frames)
            writer.writeframes(faded)
        return out_path

    def _apply_fade(self, raw: bytes, *, channels: int, frame_rate: int, fade_seconds: float) -> bytes:
        if fade_seconds <= 0:
            return raw
        samples = array("h")
        samples.frombytes(raw)
        total_frames = len(samples) // channels
        fade_frames = min(total_frames, max(1, int(round(fade_seconds * frame_rate))))
        fade_start = max(0, total_frames - fade_frames)
        for frame_index in range(fade_start, total_frames):
            factor = max(0.0, (total_frames - frame_index) / fade_frames)
            sample_start = frame_index * channels
            for offset in range(channels):
                sample_index = sample_start + offset
                samples[sample_index] = int(samples[sample_index] * factor)
        return samples.tobytes()

    def _resolve_file(self, file_value: str) -> Path:
        value = str(file_value or "").strip()
        if value.startswith("/assets/"):
            return self.project_root / "rehab_app" / "server" / "static" / value.removeprefix("/assets/")
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _project_relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def _stop_locked(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
