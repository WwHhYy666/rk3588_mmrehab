"""Small state-machine helpers for realtime rehabilitation actions."""

from __future__ import annotations

from enum import Enum


class MotionState(str, Enum):
    BASELINE = "BASELINE"
    IDLE = "IDLE"
    RISING = "RISING"
    HOLDING = "HOLDING"
    RETURNING = "RETURNING"
    REP_DONE = "REP_DONE"


class ConsecutiveConfirm:
    """Confirm a condition only after it stays true for N consecutive frames."""

    def __init__(self, frames: int) -> None:
        self.frames = max(1, int(frames))
        self.count = 0

    def update(self, condition: bool) -> bool:
        if condition:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.frames

    def reset(self) -> None:
        self.count = 0
