from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _output_device_candidates(output_device: str | None) -> list[str | None]:
    """Prefer the named onboard codec, then retain the known card-index fallback."""
    selected = str(output_device or "").strip() or None
    candidates = [selected]
    if selected == "plughw:CARD=rockchipnau8822,DEV=0":
        candidates.append("plughw:1,0")
    return list(dict.fromkeys(candidates))


def play_wav_files(paths: Iterable[str | Path], output_device: str | None = None) -> dict[str, Any]:
    wav_paths = [str(Path(path)) for path in paths]
    if not wav_paths:
        return {"ok": False, "command": [], "returncode": None, "error": "no audio files"}
    aplay = shutil.which("aplay")
    if not aplay:
        return {"ok": False, "command": [], "returncode": None, "error": "aplay not found"}
    last_command: list[str] = [aplay, *wav_paths]
    last_returncode: int | None = None
    last_error = "aplay failed"
    for candidate in _output_device_candidates(output_device):
        command = [aplay]
        if candidate:
            command.extend(["-D", candidate])
        command.extend(wav_paths)
        last_command = command
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            last_error = str(exc)
            last_returncode = None
            continue
        last_returncode = completed.returncode
        detail = (completed.stderr or "").strip()
        if completed.returncode == 0:
            return {
                "ok": True,
                "command": command,
                "returncode": 0,
                "error": "",
            }
        last_error = detail[-300:] or "aplay failed"
    return {
        "ok": False,
        "command": last_command,
        "returncode": last_returncode,
        "error": last_error,
    }
