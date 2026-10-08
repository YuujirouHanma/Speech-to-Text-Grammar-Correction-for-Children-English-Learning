import contextlib

import numpy as np
import pytest

from speakfix.audio.decode import check_duration, decode_audio, probe_duration
from speakfix.audio.encode import to_wav_bytes
from speakfix.audio.preprocess import (
    analyze_signal,
    highpass,
    normalize_loudness,
    preprocess,
    quality_warnings,
    spectral_gate,
)
from speakfix.audio.validation import sniff_format, validate_upload
from speakfix.domain.errors import (
    AudioDurationError,
    AudioTooLargeError,
    InvalidAudioError,
    UnsupportedMediaError,
)
from speakfix.services.analysis import AnalysisService, AudioPolicy
from tests.conftest import make_wav

SR = 16_000


@pytest.mark.parametrize(
    ("head", "fmt"),
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x64\x00", "mp3"),
        (b"\xff\xf1\x50\x80\x00", "aac"),
        (b"fLaC\x00\x00\x00\x22", "flac"),
        (b"OggS\x00\x02\x00\x00", "ogg"),
        (b"\x1a\x45\xdf\xa3\x9f\x42", "webm"),
        (b"\x00\x00\x00\x20ftypM4A ", "mp4"),
        (b"%PDF-1.7 hello", None),
        (b"\x89PNG\r\n\x1a\n", None),
    ],
)
def test_sniff_format(head: bytes, fmt: str | None) -> None:
    assert sniff_format(head + b"\x00" * 16) == fmt


def test_validate_upload_errors() -> None:
    with pytest.raises(InvalidAudioError):
        validate_upload(b"", max_bytes=100)
    with pytest.raises(AudioTooLargeError):
        validate_upload(b"x" * 101, max_bytes=100)
    with pytest.raises(UnsupportedMediaError):
        validate_upload(b"%PDF-1.7" + b"\x00" * 50, max_bytes=1000)


def test_decode_and_probe_wav() -> None:
    data = make_wav(2.0)
    assert validate_upload(data, max_bytes=10_000_000) == "wav"
    duration = probe_duration(data)
    assert duration is not None and abs(duration - 2.0) < 0.05
    audio = decode_audio(data)
    assert audio.dtype == np.float32
    assert abs(audio.size - 2 * SR) < SR * 0.05


def test_decode_resamples_44k_to_16k() -> None:
    audio = decode_audio(make_wav(1.0, sample_rate=44_100))
    assert abs(audio.size - SR) < SR * 0.02


def test_corrupt_audio_raises() -> None:
    corrupt = make_wav(1.0)[:44] + b"\x13\x37" * 7  # valid header, broken body
    with pytest.raises(InvalidAudioError):
        decode_audio(b"RIFF\x00\x00\x00\x00WAVEjunk" + b"\x00" * 10)
    # A truncated PCM body is still decodable but almost empty: must not crash.
    with contextlib.suppress(InvalidAudioError):
        decode_audio(corrupt)


def test_check_duration_bounds() -> None:
    check_duration(5.0, minimum=0.3, maximum=60)
    with pytest.raises(AudioDurationError):
        check_duration(61.0, minimum=0.3, maximum=60)
    with pytest.raises(AudioDurationError):
        check_duration(0.1, minimum=0.3, maximum=60)


def test_highpass_removes_rumble_keeps_voice() -> None:
    t = np.arange(SR, dtype=np.float32) / SR
    rumble = 0.5 * np.sin(2 * np.pi * 20 * t)
    voice = 0.5 * np.sin(2 * np.pi * 300 * t)
    filtered = highpass((rumble + voice).astype(np.float32), SR, 80.0)
    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(filtered.size, 1 / SR)
    assert spectrum[np.argmin(np.abs(freqs - 20))] < 0.01 * spectrum[np.argmin(np.abs(freqs - 300))]


def test_loudness_normalization_boosts_quiet_speech() -> None:
    quiet = decode_audio(make_wav(1.5, amplitude=0.01))
    louder, gain_db = normalize_loudness(quiet, SR, -20.0)
    assert gain_db > 10
    assert np.max(np.abs(louder)) <= 0.97 + 1e-6


def test_spectral_gate_reduces_stationary_noise() -> None:
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal(SR * 2)).astype(np.float32)
    speech = decode_audio(make_wav(2.0, amplitude=0.4))
    n = min(noise.size, speech.size)
    mixed = speech[:n] + noise[:n]
    denoised = spectral_gate(mixed)
    assert denoised.shape == mixed.shape
    residual_before = np.mean((mixed - speech[:n]) ** 2)
    residual_after = np.mean((denoised - speech[:n]) ** 2)
    assert residual_after < residual_before


def test_preprocess_output_and_stats() -> None:
    audio = decode_audio(make_wav(1.5))
    processed, stats = preprocess(audio, SR, denoise=True)
    assert processed.dtype == np.float32
    assert np.all(np.isfinite(processed))
    assert np.max(np.abs(processed)) <= 1.0
    assert stats.duration_s == pytest.approx(1.5, abs=0.05)
    assert stats.speech_ratio > 0.3
    assert quality_warnings(stats) == []


def test_preprocess_rejects_invalid_samples() -> None:
    with pytest.raises(InvalidAudioError):
        preprocess(np.array([], dtype=np.float32), SR)
    with pytest.raises(InvalidAudioError):
        preprocess(np.array([0.1, np.nan], dtype=np.float32), SR)


def test_silence_produces_warnings() -> None:
    silent = decode_audio(make_wav(1.0, silence=True))
    warnings = quality_warnings(analyze_signal(silent, SR))
    assert "very_quiet_audio" in warnings


def test_wav_encoding_round_trip() -> None:
    audio = decode_audio(make_wav(1.0))
    again = decode_audio(to_wav_bytes(audio, SR))
    assert abs(again.size - audio.size) <= 1
    assert np.corrcoef(audio[: again.size], again[: audio.size])[0, 1] > 0.99


def test_analysis_service_prepare_audio_limits() -> None:
    service = AnalysisService(
        corrector=lambda: None,
        transcriber=lambda: None,
        policy=AudioPolicy(max_seconds=1.0, min_seconds=0.3),
    )
    with pytest.raises(AudioDurationError):
        service.prepare_audio(make_wav(3.0))
    with pytest.raises(AudioDurationError):
        service.prepare_audio(make_wav(0.1))
    prepared = service.prepare_audio(make_wav(0.8))
    assert prepared.source_format == "wav"
    assert prepared.sample_rate == SR
