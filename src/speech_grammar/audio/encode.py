"""Encode PCM samples as a metadata-free WAV file (for cloud STT uploads)."""

from __future__ import annotations

import io
import wave

import numpy as np

from speech_grammar.domain.ports import FloatArray


def to_wav_bytes(audio: FloatArray, sample_rate: int) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV.

    Re-encoding (instead of forwarding the original upload) strips container
    metadata such as device names, GPS tags or ID3 fields before audio leaves
    the server.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()
