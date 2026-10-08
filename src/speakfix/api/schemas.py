"""Pydantic request/response models (the public API contract)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from speakfix.domain.models import Analysis, AudioStats, Correction, Edit, Transcript
from speakfix.text.edits import describe_error_type


class ErrorBody(BaseModel):
    """Machine-readable error."""

    code: str = Field(examples=["invalid_audio"])
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Error envelope used by every non-2xx response."""

    error: ErrorBody


class CorrectTextRequest(BaseModel):
    """Body of ``POST /v1/text/correct``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=10_000, examples=["She don't like apples."])
    num_beams: int | None = Field(default=None, ge=1, le=8, description="Beam width override.")


class EditOut(BaseModel):
    """A single change, with offsets into the original text."""

    op: str = Field(examples=["replace"])
    original: str
    corrected: str
    start: int
    end: int
    error_type: str | None = Field(default=None, examples=["R:VERB:SVA"])
    description: str | None = Field(default=None, examples=["Subject-verb agreement"])

    @classmethod
    def from_domain(cls, edit: Edit) -> EditOut:
        """Convert a domain edit."""
        return cls(
            op=edit.op,
            original=edit.original,
            corrected=edit.corrected,
            start=edit.start,
            end=edit.end,
            error_type=edit.error_type,
            description=describe_error_type(edit.error_type),
        )


class CorrectionOut(BaseModel):
    """Grammar-correction result."""

    original: str
    corrected: str
    changed: bool
    edits: list[EditOut]
    sentences: int
    backend: str
    warnings: list[str] = []

    @classmethod
    def from_domain(cls, c: Correction) -> CorrectionOut:
        """Convert a domain correction."""
        return cls(
            original=c.original,
            corrected=c.corrected,
            changed=c.changed,
            edits=[EditOut.from_domain(e) for e in c.edits],
            sentences=c.sentences,
            backend=c.backend,
            warnings=list(c.warnings),
        )


class SegmentOut(BaseModel):
    """Time-aligned transcript segment."""

    start: float
    end: float
    text: str


class TranscriptOut(BaseModel):
    """Speech-to-text result."""

    text: str
    duration_s: float
    engine: str
    language: str | None = None
    language_probability: float | None = None
    segments: list[SegmentOut] = []

    @classmethod
    def from_domain(cls, t: Transcript) -> TranscriptOut:
        """Convert a domain transcript."""
        return cls(
            text=t.text,
            duration_s=t.duration_s,
            engine=t.engine,
            language=t.language,
            language_probability=t.language_probability,
            segments=[SegmentOut(start=s.start, end=s.end, text=s.text) for s in t.segments],
        )


class AudioStatsOut(BaseModel):
    """Content-free signal statistics of the uploaded clip."""

    duration_s: float
    rms_dbfs: float
    peak: float
    clipped_ratio: float
    speech_ratio: float

    @classmethod
    def from_domain(cls, s: AudioStats) -> AudioStatsOut:
        """Convert domain statistics."""
        return cls(
            duration_s=s.duration_s,
            rms_dbfs=s.rms_dbfs,
            peak=s.peak,
            clipped_ratio=s.clipped_ratio,
            speech_ratio=s.speech_ratio,
        )


class CorrectTextResponse(BaseModel):
    """Response of ``POST /v1/text/correct``."""

    request_id: str
    correction: CorrectionOut
    timings_ms: dict[str, float]


class TranscribeResponse(BaseModel):
    """Response of ``POST /v1/audio/transcribe``."""

    request_id: str
    transcript: TranscriptOut
    audio: AudioStatsOut
    warnings: list[str]
    timings_ms: dict[str, float]


class AnalyzeResponse(BaseModel):
    """Response of ``POST /v1/audio/analyze`` (and async job results)."""

    request_id: str
    transcript: TranscriptOut
    correction: CorrectionOut | None
    audio: AudioStatsOut | None
    warnings: list[str]
    timings_ms: dict[str, float]

    @classmethod
    def from_domain(cls, a: Analysis, request_id: str) -> AnalyzeResponse:
        """Convert a domain analysis."""
        return cls(
            request_id=request_id,
            transcript=TranscriptOut.from_domain(a.transcript),
            correction=CorrectionOut.from_domain(a.correction) if a.correction else None,
            audio=AudioStatsOut.from_domain(a.audio) if a.audio else None,
            warnings=list(a.warnings),
            timings_ms=dict(a.timings_ms),
        )


class JobAcceptedResponse(BaseModel):
    """202 response when a job is queued."""

    job_id: str
    status: str
    status_url: str


class JobStatusResponse(BaseModel):
    """Status (and result, once finished) of an async job."""

    job_id: str
    status: str
    created_at: datetime
    expires_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: AnalyzeResponse | None = None
    error: ErrorBody | None = None


def ts(value: float | None) -> datetime | None:
    """Unix timestamp to aware UTC datetime."""
    return datetime.fromtimestamp(value, tz=timezone.utc) if value is not None else None


class ComponentStatus(BaseModel):
    """Readiness of one model component."""

    ready: bool
    name: str | None = None
    error: str | None = None


class ReadinessResponse(BaseModel):
    """Body of ``GET /readyz``."""

    ready: bool
    loading: bool
    components: dict[str, ComponentStatus]


class InfoResponse(BaseModel):
    """Public service information and limits (``GET /v1/info``)."""

    name: str
    version: str
    corrector: str | None
    stt: str | None
    limits: dict[str, Any]
    privacy: dict[str, Any]
