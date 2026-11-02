"""Speech-to-text use case with an optional fallback engine."""

from __future__ import annotations

import threading
from collections.abc import Callable

from speakfix.domain.errors import InferenceError, SpeakFixError
from speakfix.domain.models import Transcript
from speakfix.domain.ports import FloatArray, SpeechToText
from speakfix.logs import get_logger
from speakfix.metrics import FALLBACKS, INFERENCE_ERRORS, track_stage

log = get_logger(__name__)


class Transcriber:
    """Run the primary STT engine; on failure, try the (lazily loaded) fallback."""

    def __init__(
        self,
        engine: SpeechToText,
        *,
        fallback_factory: Callable[[], SpeechToText] | None = None,
    ) -> None:
        self._engine = engine
        self._fallback_factory = fallback_factory
        self._fallback: SpeechToText | None = None
        self._lock = threading.Lock()

    @property
    def engine_name(self) -> str:
        """Identifier of the primary engine."""
        return self._engine.name

    def transcribe(self, audio: FloatArray, sample_rate: int) -> Transcript:
        """Transcribe audio, falling back to the secondary engine on errors."""
        try:
            with track_stage("stt"):
                return self._engine.transcribe(audio, sample_rate)
        except SpeakFixError as exc:
            INFERENCE_ERRORS.labels(component="stt", kind=exc.code).inc()
            primary_error: Exception = exc
        except Exception as exc:
            INFERENCE_ERRORS.labels(component="stt", kind="error").inc()
            primary_error = exc
        log.warning(
            "stt_primary_failed", engine=self._engine.name, error=type(primary_error).__name__
        )

        fallback = self._get_fallback()
        if fallback is None:
            if isinstance(primary_error, SpeakFixError):
                raise primary_error  # keeps its meaning, e.g. 503 when out of memory
            raise InferenceError("Transcription failed.") from primary_error
        FALLBACKS.labels(component="stt").inc()
        with track_stage("stt"):
            return fallback.transcribe(audio, sample_rate)

    def _get_fallback(self) -> SpeechToText | None:
        if self._fallback_factory is None:
            return None
        with self._lock:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
        return self._fallback
