"""Synchronous audio endpoints: transcription and full analysis."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from speakfix.api.deps import ContainerDep, PrincipalDep, read_upload, require_consent
from speakfix.api.schemas import (
    AnalyzeResponse,
    AudioStatsOut,
    ErrorResponse,
    TranscribeResponse,
    TranscriptOut,
)
from speakfix.domain.errors import ModelNotReadyError

router = APIRouter(prefix="/v1/audio", tags=["audio"])

_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 413, 415, 422, 429, 503, 504)
}

AudioFile = Annotated[UploadFile, File(description="wav, mp3, m4a, ogg/opus, webm or flac")]
Consent = Annotated[
    bool, Form(description="Confirms verifiable parental consent for this child's voice.")
]
Beams = Annotated[int | None, Form(ge=1, le=8, description="Beam width override.")]


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses=_ERRORS,
    summary="Transcribe a speech recording",
)
async def transcribe(
    request: Request,
    file: AudioFile,
    _principal: PrincipalDep,
    container: ContainerDep,
    consent: Consent = False,
) -> TranscribeResponse:
    """Speech-to-text only (no grammar correction)."""
    require_consent(consent, container)
    data = await read_upload(file, container.settings.max_upload_bytes)
    if not container.transcriber_ready:
        raise ModelNotReadyError("The speech model is still loading.")
    transcript, prepared, timings = await container.run_inference(
        lambda: container.analysis.transcribe(data)
    )
    return TranscribeResponse(
        request_id=request.state.request_id,
        transcript=TranscriptOut.from_domain(transcript),
        audio=AudioStatsOut.from_domain(prepared.stats),
        warnings=list(prepared.warnings),
        timings_ms=timings,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses=_ERRORS,
    summary="Transcribe a recording and correct its grammar",
)
async def analyze(
    request: Request,
    file: AudioFile,
    _principal: PrincipalDep,
    container: ContainerDep,
    consent: Consent = False,
    num_beams: Beams = None,
) -> AnalyzeResponse:
    """Full pipeline in one synchronous call (clips up to the configured limit)."""
    require_consent(consent, container)
    data = await read_upload(file, container.settings.max_upload_bytes)
    if not container.ready:
        raise ModelNotReadyError("Models are still loading.")
    analysis = await container.run_inference(
        lambda: container.analysis.analyze(data, num_beams=num_beams)
    )
    return AnalyzeResponse.from_domain(
        analysis, request.state.request_id, container.settings.feedback_language
    )
