"""Domain exceptions.

Each error carries a stable machine-readable ``code`` that the API layer maps to
an HTTP status. Messages must never include user content (transcripts, text).
"""

from __future__ import annotations


class SpeakFixError(Exception):
    """Base class for all expected service errors."""

    code = "internal_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.__class__.__doc__ or self.code)
        self.message = message or (self.__class__.__doc__ or self.code).strip()


class InvalidTextError(SpeakFixError):
    """The text input is empty or too long."""

    code = "invalid_text"


class InvalidAudioError(SpeakFixError):
    """The audio could not be decoded (corrupt or not audio)."""

    code = "invalid_audio"


class UnsupportedMediaError(InvalidAudioError):
    """The uploaded file is not a supported audio format."""

    code = "unsupported_media_type"


class AudioTooLargeError(InvalidAudioError):
    """The uploaded file exceeds the size limit."""

    code = "payload_too_large"


class AudioDurationError(InvalidAudioError):
    """The audio is too short or too long."""

    code = "invalid_audio_duration"


class ModelNotReadyError(SpeakFixError):
    """Models are still loading; retry shortly."""

    code = "model_not_ready"


class InferenceError(SpeakFixError):
    """Model inference failed."""

    code = "inference_failed"


class ResourceExhaustedError(InferenceError):
    """Not enough memory to run inference (OOM)."""

    code = "resource_exhausted"


class ServerBusyError(SpeakFixError):
    """Too many requests are waiting for inference."""

    code = "server_busy"


class InferenceTimeoutError(InferenceError):
    """Inference did not finish within the configured timeout."""

    code = "inference_timeout"


class ConsentRequiredError(SpeakFixError):
    """Parental consent must be confirmed before processing a child's voice."""

    code = "consent_required"


class AuthenticationError(SpeakFixError):
    """Missing or invalid API key."""

    code = "unauthorized"


class RateLimitedError(SpeakFixError):
    """Too many requests; slow down."""

    code = "rate_limited"

    def __init__(self, retry_after_s: int, message: str | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class JobNotFoundError(SpeakFixError):
    """The job does not exist or has expired."""

    code = "job_not_found"


class QueueFullError(ServerBusyError):
    """The async job queue is full."""

    code = "queue_full"


class CloudSttDisabledError(SpeakFixError):
    """Cloud speech-to-text is disabled by the privacy policy of this deployment."""

    code = "cloud_stt_disabled"
