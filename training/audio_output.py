from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def play_wav_files(paths: Iterable[str | Path], output_device: str | None = None) -> dict[str, Any]:
    wav_paths = [str(Path(path)) for path in paths]
    if not wav_paths:
        return {"ok": False, "command": [], "returncode": None, "error": "no audio files"}
    aplay = shutil.which("aplay")
    if not aplay:
        return {"ok": False, "command": [], "returncode": None, "error": "aplay not found"}
    command = [aplay]
    if output_device:
        command.extend(["-D", output_device])
    command.extend(wav_paths)
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        return {"ok": False, "command": command, "returncode": None, "error": str(exc)}
    detail = (completed.stderr or "").strip()
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "error": "" if completed.returncode == 0 else (detail[-300:] or "aplay failed"),
    }
