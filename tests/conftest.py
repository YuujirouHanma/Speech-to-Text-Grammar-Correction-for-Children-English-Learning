"""Shared fixtures: fake model runtimes, synthetic audio and app factories."""

from __future__ import annotations

import io
import math
import wave
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speakfix.api.app import create_app
from speakfix.config import Settings
from speakfix.domain.errors import ResourceExhaustedError
from speakfix.domain.models import Transcript, TranscriptSegment
from speakfix.services.container import Container
from speakfix.services.corrector import CorrectionConfig, GrammarCorrector
from speakfix.services.transcriber import Transcriber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PREFIX = "grammar: "

DEFAULT_FIXES = {
    "She go to school every day.": "She goes to school every day.",
    "He don't like apples.": "He doesn't like apples.",
    "I am student.": "I am a student.",
    "She have completed her homework.": "She has completed her homework.",
}


class FakeBackend:
    """Deterministic stand-in for a T5 runtime."""

    def __init__(
        self,
        fixes: dict[str, str] | None = None,
        *,
        name: str = "fake:t5",
        fail_times: int = 0,
        error: type[Exception] = ResourceExhaustedError,
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self.fixes = DEFAULT_FIXES if fixes is None else fixes
        self._name = name
        self.fail_times = fail_times
        self.error = error
        self.transform = transform
        self.calls: list[tuple[list[str], int, int]] = []

    @property
    def name(self) -> str:
        return self._name

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 1

    def generate(self, sources: Sequence[str], *, num_beams: int, max_new_tokens: int) -> list[str]:
        self.calls.append((list(sources), num_beams, max_new_tokens))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error("simulated failure")
        outputs = []
        for source in sources:
            text = source.removeprefix(PREFIX)
            outputs.append(self.transform(text) if self.transform else self.fixes.get(text, text))
        return outputs


class FakeSTT:
    """Returns a fixed transcript."""

    def __init__(self, text: str = "He don't like apples.", *, name: str = "fake:stt") -> None:
        self.text = text
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        self.calls += 1
        duration = audio.size / sample_rate
        return Transcript(
            text=self.text,
            duration_s=round(duration, 3),
            engine=self.name,
            language="en",
            language_probability=0.99,
            segments=(TranscriptSegment(start=0.0, end=round(duration, 2), text=self.text),),
        )


def make_wav(
    seconds: float = 1.5,
    *,
    sample_rate: int = 16_000,
    freq: float = 220.0,
    amplitude: float = 0.3,
    silence: bool = False,
) -> bytes:
    """Synthesize a speech-like (amplitude-modulated) tone as WAV bytes."""
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    if silence:
        signal = np.zeros(n)
    else:
        envelope = 0.5 * (1 + np.sin(2 * math.pi * 3.0 * t))
        signal = amplitude * envelope * np.sin(2 * math.pi * freq * t)
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav()


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def corrector(fake_backend: FakeBackend) -> GrammarCorrector:
    return GrammarCorrector(fake_backend, config=CorrectionConfig())


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "environment": "test",
        "log_json": False,
        "log_level": "WARNING",
        "rate_limit_per_minute": 0,
        "models_dir": PROJECT_ROOT / "models",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


def make_container(
    settings: Settings,
    *,
    backend: FakeBackend | None = None,
    stt: FakeSTT | None = None,
    fail_corrector: bool = False,
) -> Container:
    backend = backend or FakeBackend()
    stt = stt or FakeSTT()

    def corrector_loader(s: Settings) -> GrammarCorrector:
        if fail_corrector:
            msg = "model files missing"
            raise FileNotFoundError(msg)
        return GrammarCorrector(backend, config=CorrectionConfig(max_chars=s.max_text_chars))

    return Container(
        settings,
        corrector_loader=corrector_loader,
        transcriber_loader=lambda _s: Transcriber(stt),
    )


@pytest.fixture
def app_factory() -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(
        *,
        backend: FakeBackend | None = None,
        stt: FakeSTT | None = None,
        fail_corrector: bool = False,
        **settings_overrides: object,
    ) -> TestClient:
        settings = make_settings(**settings_overrides)
        container = make_container(
            settings, backend=backend, stt=stt, fail_corrector=fail_corrector
        )
        app = create_app(settings, container=container, background_load=False)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(app_factory: Callable[..., TestClient]) -> TestClient:
    return app_factory()
