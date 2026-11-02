"""Local speech-to-text with faster-whisper (Whisper on CTranslate2).

Running Whisper locally keeps children's voices on the deployment's own
hardware (no third-party processing). Settings tuned for this use case:

* ``language="en"``: skips language detection, which often misfires on
  children's or non-native accents.
* Silero VAD: drops silence, the main trigger of Whisper hallucinations
  ("Thank you for watching").
* ``condition_on_previous_text=False``: prevents a mistake in one segment from
  propagating to the next.
* An optional *verbatim prompt* written with grammatical mistakes nudges the
  decoder to keep learners' errors instead of silently fixing them (otherwise
  the grammar-correction step has nothing left to correct).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from speakfix.domain.errors import InferenceError, ResourceExhaustedError, is_out_of_memory
from speakfix.domain.models import Transcript, TranscriptSegment
from speakfix.domain.ports import FloatArray

VERBATIM_PROMPT = (
    "Umm, I goes to school yesterday and she don't like it. "
    "He have two cat. We was playing football."
)


def _display_name(model: str) -> str:
    """``small.en`` for both size names and HF snapshot paths."""
    path = Path(model)
    for part in path.parts:
        if part.startswith("models--Systran--faster-whisper-"):
            return part.removeprefix("models--Systran--faster-whisper-")
    return path.name


class WhisperSpeechToText:
    """faster-whisper adapter implementing the ``SpeechToText`` port."""

    def __init__(
        self,
        model: str = "small.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
        beam_size: int = 5,
        language: str | None = "en",
        vad: bool = True,
        verbatim_prompt: bool = True,
        cpu_threads: int = 0,
    ) -> None:
        """Load (and download if needed) a Whisper model.

        Args:
            model: Size name (``base.en``, ``small.en`` ...) or a local CT2 model path.
            device: ``cpu`` or ``cuda``.
            compute_type: ``int8`` (CPU), ``float16`` (GPU) ...
            download_root: Cache directory for downloaded models.
            beam_size: Beam width for decoding.
            language: Force a language (``None`` = auto-detect).
            vad: Enable Silero voice-activity detection.
            verbatim_prompt: Use :data:`VERBATIM_PROMPT` as initial prompt.
            cpu_threads: CPU threads (0 = CTranslate2 default).
        """
        from faster_whisper import WhisperModel

        root = str(download_root) if download_root else None
        if download_root:
            download_root.mkdir(parents=True, exist_ok=True)
        self._model: Any = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=root,
            cpu_threads=cpu_threads,
        )
        self._model_name = _display_name(model)
        self.device = device
        self.compute_type = compute_type
        self._beam_size = beam_size
        self._language = language
        self._vad = vad
        self._prompt = VERBATIM_PROMPT if verbatim_prompt else None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return f"whisper:{self._model_name}"

    def probe(self) -> None:
        """Run the encoder/decoder once on noise (VAD off) to surface device errors.

        Transcribing silence is not a valid probe: VAD removes it and the model
        never runs, so a broken GPU setup would only fail on the first real request.
        """
        noise = np.random.default_rng(0).normal(0, 0.05, 16_000).astype(np.float32)
        segments, _ = self._model.transcribe(
            noise, language="en", beam_size=1, vad_filter=False, without_timestamps=True
        )
        list(segments)

    def transcribe(self, audio: FloatArray, sample_rate: int) -> Transcript:
        """Transcribe 16 kHz mono audio."""
        if sample_rate != 16_000:
            msg = "Whisper expects 16 kHz audio."
            raise ValueError(msg)
        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=self._vad,
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 300},
                condition_on_previous_text=False,
                initial_prompt=self._prompt,
            )
            segments = tuple(
                TranscriptSegment(
                    start=round(float(s.start), 2),
                    end=round(float(s.end), 2),
                    text=s.text.strip(),
                    avg_logprob=round(float(s.avg_logprob), 4),
                    no_speech_prob=round(float(s.no_speech_prob), 4),
                )
                for s in segments_iter  # the generator runs the actual decoding
            )
        except Exception as exc:  # onnxruntime (the VAD) has its own exception types
            if is_out_of_memory(exc):
                raise ResourceExhaustedError("Out of memory during transcription.") from exc
            raise InferenceError("Transcription failed.") from exc
        text = " ".join(seg.text for seg in segments if seg.text).strip()
        return Transcript(
            text=text,
            duration_s=round(float(info.duration), 3),
            engine=self.name,
            language=info.language,
            language_probability=round(float(info.language_probability), 3),
            segments=segments,
        )
