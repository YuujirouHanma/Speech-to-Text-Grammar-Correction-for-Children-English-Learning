"""In-memory audio decoding to 16 kHz mono float32 (no temporary files).

Decoding goes through PyAV (bundled with faster-whisper), so no system ffmpeg
is required and the child's recording is never written to disk.
"""

from __future__ import annotations

import io

import numpy as np

from speakfix.domain.errors import AudioDurationError, InvalidAudioError
from speakfix.domain.ports import FloatArray

TARGET_SAMPLE_RATE = 16_000


def probe_duration(data: bytes) -> float | None:
    """Read the container duration without decoding the whole file.

    Args:
        data: Encoded audio bytes.

    Returns:
        Duration in seconds, or ``None`` when the container does not declare it.

    Raises:
        InvalidAudioError: The container cannot be opened or has no audio stream.
    """
    import av

    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise InvalidAudioError("The file contains no audio stream.")
            if container.duration:
                return float(container.duration) / float(av.time_base)
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError("The audio container could not be read.") from exc
    return None


def decode_audio(
    data: bytes, sample_rate: int = TARGET_SAMPLE_RATE, *, max_seconds: float | None = None
) -> FloatArray:
    """Decode any supported container to mono float32 PCM at ``sample_rate``.

    Uses PyAV directly (decode + resample), independent of faster-whisper's
    helper, and stops early when the stream exceeds ``max_seconds`` so a
    small-but-long file (a "decompression bomb") cannot exhaust memory.

    Args:
        data: Encoded audio bytes.
        sample_rate: Output sample rate (Whisper expects 16 kHz).
        max_seconds: Abort with :class:`AudioDurationError` beyond this length.

    Returns:
        Samples in [-1, 1].

    Raises:
        InvalidAudioError: Corrupt data, unsupported codec or empty stream.
        AudioDurationError: The stream is longer than ``max_seconds``.
    """
    import av

    limit = int(max_seconds * sample_rate) if max_seconds else None
    chunks: list[np.ndarray] = []
    total = 0
    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise InvalidAudioError("The file contains no audio stream.")
            for frame in container.decode(stream):
                frame.pts = None
                for out in resampler.resample(frame):
                    pcm = out.to_ndarray().reshape(-1)
                    chunks.append(pcm)
                    total += pcm.size
                if limit is not None and total > limit:
                    raise AudioDurationError(f"Audio is longer than the {max_seconds:.0f} s limit.")
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except (InvalidAudioError, AudioDurationError):
        raise
    except Exception as exc:
        raise InvalidAudioError("The audio could not be decoded (corrupt file?).") from exc
    if not chunks:
        raise InvalidAudioError("The audio stream is empty.")
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise InvalidAudioError("The audio stream is empty.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def check_duration(seconds: float, *, minimum: float, maximum: float) -> None:
    """Raise :class:`AudioDurationError` when ``seconds`` is out of range."""
    if seconds > maximum:
        raise AudioDurationError(f"Audio is longer than the {maximum:.0f} s limit.")
    if seconds < minimum:
        raise AudioDurationError(f"Audio is shorter than {minimum:.1f} s.")
