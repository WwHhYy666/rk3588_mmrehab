import numpy as np

from speech.asr_worker import _pad_waveform_edges


def test_asr_edge_padding_preserves_audio_and_adds_silence() -> None:
    samples = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)

    padded = _pad_waveform_edges(samples, sample_rate=1000, padding_ms=2)

    assert padded.tolist() == [0.0, 0.0, 0.25, -0.5, 0.75, 0.0, 0.0]


def test_asr_edge_padding_can_be_disabled() -> None:
    samples = np.asarray([0.1, 0.2], dtype=np.float32)

    assert _pad_waveform_edges(samples, sample_rate=16000, padding_ms=0) is samples
