from __future__ import annotations

import numpy as np
import pytest

from pose_estimation.gstreamer_gi_capture import bgr_frame_from_bytes


def test_bgr_frame_from_bytes_handles_padded_rows() -> None:
    width = 2
    height = 2
    rows = np.array(
        [
            [1, 2, 3, 4, 5, 6, 99, 99],
            [7, 8, 9, 10, 11, 12, 88, 88],
        ],
        dtype=np.uint8,
    )

    frame = bgr_frame_from_bytes(rows.tobytes(), width=width, height=height)

    assert frame.shape == (2, 2, 3)
    assert frame.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
    assert frame.flags["C_CONTIGUOUS"] is True


def test_bgr_frame_from_bytes_rejects_short_rows() -> None:
    with pytest.raises(ValueError, match="smaller than BGR caps"):
        bgr_frame_from_bytes(bytes(10), width=2, height=2)
