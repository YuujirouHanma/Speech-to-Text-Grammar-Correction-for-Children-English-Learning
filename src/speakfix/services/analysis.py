"""Application use cases: correct text, transcribe audio, full speech analysis."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from speakfix.audio.decode import TARGET_SAMPLE_RATE, check_duration, decode_audio, probe_duration
from speakfix.audio.preprocess import preprocess, quality_warnings
from speakfix.audio.validation import validate_upload
from speakfix.domain.errors import InferenceError, ModelNotReadyError
from speakfix.domain.models import Analysis, AudioStats, Correction, Transcript
from speakfix.domain.ports import FloatArray
from speakfix.logs import get_logger
from speakfix.metrics import AUDIO_DURATION, track_stage
from speakfix.services.corrector import GrammarCorrector
from speakfix.services.transcriber import Transcriber

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AudioPolicy:
    """Limits and preprocessing options for uploaded audio."""

    max_bytes: int = 10 * 1024 * 1024
    min_seconds: float = 0.3
    max_seconds: float = 60.0
    highpass_hz: float = 80.0
    normalize: bool = True
    target_dbfs: float = -20.0
    denoise: bool = False


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """Decoded, validated and conditioned audio ready for STT."""

    samples: FloatArray
    sample_rate: int
    stats: AudioStats
    warnings: tuple[str, ...]
    source_format: str


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


class AnalysisService:
    """Orchestrates audio preparation, transcription and correction.

    Model components are obtained through providers so the service can exist
    (and report "not ready") while models are still loading.
    """

    def __init__(
        self,
        *,
        corrector: Callable[[], GrammarCorrector | None],
        transcriber: Callable[[], Transcriber | None],
        policy: AudioPolicy | None = None,
    ) -> None:
        self._corrector = corrector
        self._transcriber = transcriber
        self._policy = policy or AudioPolicy()

    # ---------------------------------------------------------------- helpers
    def _require_corrector(self) -> GrammarCorrector:
        corrector = self._corrector()
        if corrector is None:
            raise ModelNotReadyError("The grammar model is still loading.")
        return corrector

    def _require_transcriber(self) -> Transcriber:
        transcriber = self._transcriber()
        if transcriber is None:
            raise ModelNotReadyError("The speech model is still loading.")
        return transcriber

    # -------------------------------------------------------------- use cases
    def correct_text(self, text: str, *, num_beams: int | None = None) -> Correction:
        """Correct typed or transcribed text."""
        return self._require_corrector().correct(text, num_beams=num_beams)

    def prepare_audio(self, data: bytes) -> PreparedAudio:
        """Validate, decode and condition an uploaded clip (fully in memory)."""
        policy = self._policy
        fmt = validate_upload(data, max_bytes=policy.max_bytes)
        with track_stage("decode"):
            declared = probe_duration(data)
            if declared is not None:
                # Fail fast before decoding an hour-long file.
                check_duration(declared, minimum=0.0, maximum=policy.max_seconds + 1.0)
            samples = decode_audio(data, TARGET_SAMPLE_RATE, max_seconds=policy.max_seconds)
        duration = samples.size / TARGET_SAMPLE_RATE
        check_duration(duration, minimum=policy.min_seconds, maximum=policy.max_seconds)
        with track_stage("preprocess"):
            processed, stats = preprocess(
                samples,
                TARGET_SAMPLE_RATE,
                highpass_hz=policy.highpass_hz,
                normalize=policy.normalize,
                target_dbfs=policy.target_dbfs,
                denoise=policy.denoise,
            )
        AUDIO_DURATION.observe(duration)
        return PreparedAudio(
            samples=processed,
            sample_rate=TARGET_SAMPLE_RATE,
            stats=stats,
            warnings=tuple(quality_warnings(stats)),
            source_format=fmt,
        )

    def transcribe(self, data: bytes) -> tuple[Transcript, PreparedAudio, dict[str, float]]:
        """Transcribe an uploaded clip."""
        transcriber = self._require_transcriber()
        timings: dict[str, float] = {}
        start = time.perf_counter()
        prepared = self.prepare_audio(data)
        timings["audio_ms"] = _ms(start)
        start = time.perf_counter()
        transcript = transcriber.transcribe(prepared.samples, prepared.sample_rate)
        timings["stt_ms"] = _ms(start)
        return transcript, prepared, timings

    def analyze(self, data: bytes, *, num_beams: int | None = None) -> Analysis:
        """Full pipeline: audio -> transcript -> grammar correction.

        The correction step degrades gracefully: if it fails, the transcript is
        still returned with a ``correction_unavailable`` warning.
        """
        corrector = self._require_corrector()
        transcript, prepared, timings = self.transcribe(data)
        warnings = list(prepared.warnings)
        correction: Correction | None = None
        if not transcript.text:
            warnings.append("empty_transcript")
        else:
            start = time.perf_counter()
            try:
                correction = corrector.correct(
                    transcript.text, num_beams=num_beams, from_speech=True
                )
                warnings.extend(correction.warnings)
            except InferenceError:
                log.exception("correction_failed_returning_transcript_only")
                warnings.append("correction_unavailable")
            timings["correct_ms"] = _ms(start)
        timings["total_ms"] = round(sum(timings.values()), 1)
        return Analysis(
            transcript=transcript,
            correction=correction,
            audio=prepared.stats,
            warnings=tuple(dict.fromkeys(warnings)),
            timings_ms=timings,
        )
