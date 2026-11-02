"""Signal conditioning before speech recognition.

Children's speech is often quiet, recorded with cheap microphones and mixed
with room noise. Whisper normalizes its log-mel input itself, but the Silero VAD
that removes silence is level-sensitive: a soft-spoken child can be filtered
out as "silence". This module therefore:

1. removes DC offset and low-frequency rumble (high-pass; children's F0 is
   ~250-400 Hz, so an 80 Hz cut never touches the voice),
2. optionally applies conservative spectral-gating noise reduction (off by
   default: aggressive denoising tends to *hurt* ASR accuracy),
3. normalizes loudness of the voiced part to a target level with a peak limit.

Everything is pure NumPy (no SciPy/librosa) to keep the runtime image small.
"""

from __future__ import annotations

import numpy as np

from speech_grammar.domain.errors import InvalidAudioError
from speech_grammar.domain.models import AudioStats
from speech_grammar.domain.ports import FloatArray

_EPS = 1e-10
_FRAME_S = 0.02
_MAX_GAIN_DB = 30.0
_MIN_GAIN_DB = -20.0
_PEAK_LIMIT = 0.97


def _dbfs(rms: float) -> float:
    return float(20.0 * np.log10(max(rms, _EPS)))


def frame_rms(x: FloatArray, sample_rate: int, frame_s: float = _FRAME_S) -> FloatArray:
    """RMS of non-overlapping frames (used for speech-activity estimation)."""
    frame = max(1, int(sample_rate * frame_s))
    n = x.size // frame
    if n == 0:
        return np.array([np.sqrt(np.mean(np.square(x)) + _EPS)], dtype=np.float32)
    frames = x[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(np.square(frames), axis=1) + _EPS).astype(np.float32)


def speech_mask(rms: FloatArray) -> FloatArray:
    """Boolean mask of frames that likely contain speech (energy-based)."""
    db = 20.0 * np.log10(np.maximum(rms, _EPS))
    noise_floor = float(np.percentile(db, 10))
    threshold = max(noise_floor + 10.0, -55.0)
    return (db > threshold).astype(np.float32)


def highpass(x: FloatArray, sample_rate: int, cutoff_hz: float) -> FloatArray:
    """Zero-phase FFT high-pass with a raised-cosine transition band.

    Args:
        x: Mono samples.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Frequencies below ``cutoff_hz / 2`` are removed; full pass above.

    Returns:
        Filtered samples (same length).
    """
    if cutoff_hz <= 0 or x.size < 16:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    ramp = np.clip((freqs - 0.5 * cutoff_hz) / (0.5 * cutoff_hz), 0.0, 1.0)
    spectrum *= 0.5 - 0.5 * np.cos(np.pi * ramp)
    return np.fft.irfft(spectrum, n=x.size).astype(np.float32)


def spectral_gate(
    x: FloatArray, *, frame: int = 512, hop: int = 128, strength: float = 1.0, floor: float = 0.15
) -> FloatArray:
    """Conservative stationary-noise reduction by spectral gating.

    The noise profile is the 10th percentile magnitude of each frequency bin over
    time; bins close to that floor are attenuated (never below ``floor``), and
    the mask is smoothed to avoid "musical noise" artefacts.

    Args:
        x: Mono samples.
        frame: STFT frame length.
        hop: STFT hop length.
        strength: Multiplier of the noise profile that is subtracted.
        floor: Minimum gain applied to any time-frequency bin.

    Returns:
        Denoised samples (same length).
    """
    if x.size < frame * 2:
        return x
    pad = frame // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (padded.size - frame) // hop
    window = np.hanning(frame).astype(np.float32)
    index = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(padded[index] * window, axis=1)
    mag = np.abs(spec)
    noise = np.percentile(mag, 10, axis=0, keepdims=True)
    mask = np.clip((mag - strength * noise) / (mag + _EPS), floor, 1.0)
    mask = (np.roll(mask, 1, axis=0) + mask + np.roll(mask, -1, axis=0)) / 3.0
    mask = (np.roll(mask, 1, axis=1) + mask + np.roll(mask, -1, axis=1)) / 3.0
    frames = np.fft.irfft(spec * mask, n=frame, axis=1).astype(np.float32) * window
    out = np.zeros(padded.size, dtype=np.float32)
    norm = np.zeros(padded.size, dtype=np.float32)
    win_sq = window**2
    for i in range(n_frames):
        start = i * hop
        out[start : start + frame] += frames[i]
        norm[start : start + frame] += win_sq
    out /= np.maximum(norm, _EPS)
    return out[pad : pad + x.size]


def normalize_loudness(
    x: FloatArray, sample_rate: int, target_dbfs: float
) -> tuple[FloatArray, float]:
    """Scale the voiced part to ``target_dbfs`` RMS, then peak-limit.

    Returns:
        The scaled samples and the applied gain in dB.
    """
    rms = frame_rms(x, sample_rate)
    voiced = rms[speech_mask(rms) > 0]
    level = float(np.sqrt(np.mean(np.square(voiced)))) if voiced.size else float(np.mean(rms))
    gain_db = float(np.clip(target_dbfs - _dbfs(level), _MIN_GAIN_DB, _MAX_GAIN_DB))
    y = x * np.float32(10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > _PEAK_LIMIT:
        y *= np.float32(_PEAK_LIMIT / peak)
    return y.astype(np.float32), gain_db


def analyze_signal(x: FloatArray, sample_rate: int) -> AudioStats:
    """Compute content-free statistics of a signal."""
    rms = frame_rms(x, sample_rate)
    return AudioStats(
        duration_s=round(x.size / sample_rate, 3),
        sample_rate=sample_rate,
        rms_dbfs=round(_dbfs(float(np.sqrt(np.mean(np.square(x)) + _EPS))), 2),
        peak=round(float(np.max(np.abs(x))) if x.size else 0.0, 4),
        clipped_ratio=round(float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0, 5),
        speech_ratio=round(float(np.mean(speech_mask(rms))), 3),
    )


def preprocess(
    audio: FloatArray,
    sample_rate: int,
    *,
    highpass_hz: float = 80.0,
    normalize: bool = True,
    target_dbfs: float = -20.0,
    denoise: bool = False,
) -> tuple[FloatArray, AudioStats]:
    """Condition a decoded clip for speech recognition.

    Args:
        audio: Mono float32 samples.
        sample_rate: Sample rate in Hz.
        highpass_hz: High-pass cutoff (0 disables).
        normalize: Apply loudness normalization.
        target_dbfs: Target RMS level of voiced frames.
        denoise: Apply spectral-gating noise reduction.

    Returns:
        The processed samples and statistics of the *input* signal.

    Raises:
        InvalidAudioError: Empty or non-finite input.
    """
    if audio.size == 0:
        raise InvalidAudioError("The audio stream is empty.")
    if not np.all(np.isfinite(audio)):
        raise InvalidAudioError("The audio contains invalid samples.")
    stats = analyze_signal(audio, sample_rate)
    x = audio.astype(np.float32, copy=True)
    x -= np.float32(x.mean())
    x = highpass(x, sample_rate, highpass_hz)
    if denoise:
        x = spectral_gate(x)
    if normalize:
        x, _ = normalize_loudness(x, sample_rate, target_dbfs)
    return np.clip(x, -1.0, 1.0).astype(np.float32), stats


def quality_warnings(stats: AudioStats) -> list[str]:
    """Human-readable warnings derived from :class:`AudioStats`."""
    warnings: list[str] = []
    if stats.rms_dbfs < -50.0:
        warnings.append("very_quiet_audio")
    if stats.speech_ratio < 0.05:
        warnings.append("little_or_no_speech_detected")
    if stats.clipped_ratio > 0.01:
        warnings.append("clipping_detected")
    return warnings
