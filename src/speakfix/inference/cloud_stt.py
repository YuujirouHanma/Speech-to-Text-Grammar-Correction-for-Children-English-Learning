"""Opt-in cloud speech-to-text adapters (Gemini, Deepgram).

These send a child's voice to a third party, so they are disabled unless
``SPEAKFIX_ALLOW_CLOUD_STT=true`` is set *and* an API key is configured. Before
enabling them in production, make sure a data-processing agreement covers
children's data (COPPA / GDPR-K / UU PDP). The Gemini API terms forbid services
directed at, or likely used by, under-18s, so Gemini is for adult-voice
experiments only.

Privacy measures applied when they are enabled:

* audio is re-encoded to metadata-free WAV (see :mod:`speakfix.audio.encode`);
* Gemini receives the audio inline (no Files API upload that persists 48 h);
* Deepgram requests opt out of the Model Improvement Program (``mip_opt_out``).
"""

from __future__ import annotations

from typing import Any

import httpx

from speakfix.audio.encode import to_wav_bytes
from speakfix.domain.errors import InferenceError, InferenceTimeoutError
from speakfix.domain.models import Transcript
from speakfix.domain.ports import FloatArray

VERBATIM_INSTRUCTION = (
    "Transcribe this audio verbatim in English. Write exactly the words that are "
    "spoken, including grammatical mistakes, repetitions and filler words. Do NOT "
    "correct grammar and do not add words. Output only the transcript text."
)


class GeminiSpeechToText:
    """Google Gemini (``google-genai`` SDK) speech-to-text adapter."""

    def __init__(self, api_key: str, *, model: str, timeout_s: float = 30.0) -> None:
        """Create the client (requires ``pip install speakfix[cloud]``)."""
        from google import genai
        from google.genai import types

        self._types: Any = types
        self._client: Any = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self._model = model

    @property
    def name(self) -> str:
        """Engine identifier."""
        return f"gemini:{self._model}"

    def transcribe(self, audio: FloatArray, sample_rate: int) -> Transcript:
        """Transcribe audio with Gemini using an inline WAV part."""
        part = self._types.Part.from_bytes(
            data=to_wav_bytes(audio, sample_rate), mime_type="audio/wav"
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[VERBATIM_INSTRUCTION, part],
                config=self._types.GenerateContentConfig(temperature=0.0),
            )
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() or "deadline" in str(exc).lower():
                raise InferenceTimeoutError("Gemini transcription timed out.") from exc
            raise InferenceError("Gemini transcription failed.") from exc
        return Transcript(
            text=(response.text or "").strip(),
            duration_s=round(audio.size / sample_rate, 3),
            engine=self.name,
            language="en",
        )


class DeepgramSpeechToText:
    """Deepgram pre-recorded REST API adapter (no SDK needed)."""

    URL = "https://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Create the adapter; ``client`` can be injected for testing."""
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def name(self) -> str:
        """Engine identifier."""
        return f"deepgram:{self._model}"

    def transcribe(self, audio: FloatArray, sample_rate: int) -> Transcript:
        """Transcribe audio with Deepgram."""
        params = {
            "model": self._model,
            "language": "en",
            "punctuate": "true",
            "smart_format": "false",
            "filler_words": "true",
            "mip_opt_out": "true",
        }
        try:
            response = self._client.post(
                self.URL,
                params=params,
                content=to_wav_bytes(audio, sample_rate),
                headers={"Authorization": f"Token {self._api_key}", "Content-Type": "audio/wav"},
            )
            response.raise_for_status()
            payload = response.json()
            alternative = payload["results"]["channels"][0]["alternatives"][0]
        except httpx.TimeoutException as exc:
            raise InferenceTimeoutError("Deepgram transcription timed out.") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise InferenceError("Deepgram transcription failed.") from exc
        duration = payload.get("metadata", {}).get("duration") or audio.size / sample_rate
        confidence = alternative.get("confidence")
        return Transcript(
            text=str(alternative.get("transcript", "")).strip(),
            duration_s=round(float(duration), 3),
            engine=self.name,
            language="en",
            language_probability=round(float(confidence), 3) if confidence is not None else None,
        )
