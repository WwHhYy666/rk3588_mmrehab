from __future__ import annotations

from typing import Any

import numpy as np


try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
except Exception as exc:  # pragma: no cover - depends on the board image
    Gst = None
    GSTREAMER_GI_IMPORT_ERROR = str(exc)
else:
    GSTREAMER_GI_IMPORT_ERROR = None


def gstreamer_gi_available() -> bool:
    return Gst is not None


def bgr_frame_from_bytes(data: object, *, width: int, height: int) -> np.ndarray:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("invalid GStreamer frame dimensions")
    flat = np.frombuffer(data, dtype=np.uint8)
    if flat.size % height != 0:
        raise ValueError("GStreamer frame buffer has an invalid row stride")
    row_stride = flat.size // height
    pixel_bytes = width * 3
    if row_stride < pixel_bytes:
        raise ValueError("GStreamer frame buffer is smaller than BGR caps")
    rows = flat.reshape(height, row_stride)
    return rows[:, :pixel_bytes].reshape(height, width, 3).copy()


class GStreamerGiCapture:
    """Small cv2.VideoCapture-compatible wrapper around a GI appsink."""

    def __init__(
        self,
        pipeline_description: str,
        *,
        width: int,
        height: int,
        fps: float,
        appsink_name: str = "rehab_sink",
        pull_timeout_seconds: float = 1.0,
    ) -> None:
        if Gst is None:
            raise RuntimeError(f"GStreamer GI unavailable: {GSTREAMER_GI_IMPORT_ERROR}")
        self.pipeline_description = str(pipeline_description)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.pull_timeout_ns = max(1, int(float(pull_timeout_seconds) * 1_000_000_000))
        self.last_error: str | None = None
        self._opened = False
        self._pipeline = Gst.parse_launch(self.pipeline_description)
        self._appsink = self._pipeline.get_by_name(appsink_name)
        if self._appsink is None:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"GStreamer appsink not found: {appsink_name}")
        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(self._pipeline_error() or "GStreamer pipeline failed to enter PLAYING")
        state_result, _, _ = self._pipeline.get_state(2 * Gst.SECOND)
        if state_result == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(self._pipeline_error() or "GStreamer pipeline state negotiation failed")
        self._opened = True

    def _pipeline_error(self) -> str | None:
        if Gst is None or self._pipeline is None:
            return None
        bus = self._pipeline.get_bus()
        message = bus.pop_filtered(Gst.MessageType.ERROR) if bus is not None else None
        if message is None:
            return None
        error, debug = message.parse_error()
        return f"{error}: {debug}" if debug else str(error)

    def isOpened(self) -> bool:
        return bool(self._opened)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._appsink is None:
            return False, None
        sample = self._appsink.emit("try-pull-sample", self.pull_timeout_ns)
        if sample is None:
            self.last_error = self._pipeline_error() or "GStreamer appsink timed out"
            return False, None
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None and caps.get_size() else None
        if structure is None:
            self.last_error = "GStreamer sample has no caps"
            return False, None
        width = int(structure.get_value("width") or self.width)
        height = int(structure.get_value("height") or self.height)
        pixel_format = str(structure.get_value("format") or "")
        if pixel_format and pixel_format != "BGR":
            self.last_error = f"unexpected GStreamer format: {pixel_format}"
            return False, None
        buffer = sample.get_buffer()
        mapped, map_info = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            self.last_error = "GStreamer buffer map failed"
            return False, None
        try:
            frame = bgr_frame_from_bytes(map_info.data, width=width, height=height)
        except Exception as exc:
            self.last_error = str(exc)
            return False, None
        finally:
            buffer.unmap(map_info)
        self.width = width
        self.height = height
        self.last_error = None
        return True, frame

    def get(self, property_id: int) -> float:
        property_id = int(property_id)
        if property_id == 3:  # cv2.CAP_PROP_FRAME_WIDTH
            return float(self.width)
        if property_id == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
            return float(self.height)
        if property_id == 5:  # cv2.CAP_PROP_FPS
            return float(self.fps)
        return 0.0

    def set(self, property_id: int, value: Any) -> bool:
        return False

    def release(self) -> None:
        if self._pipeline is not None and Gst is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._opened = False
        self._appsink = None
        self._pipeline = None

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.release()
        except Exception:
            pass
