"""Asynchronous job endpoints (submit -> poll -> fetch/delete)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from speech_grammar.api.deps import ContainerDep, PrincipalDep, read_upload, require_consent
from speech_grammar.api.schemas import (
    AnalyzeResponse,
    ErrorBody,
    ErrorResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    ts,
)
from speech_grammar.domain.errors import ModelNotReadyError

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404, 413, 415, 422, 429, 503)
}


@router.post(
    "/analyze",
    status_code=202,
    response_model=JobAcceptedResponse,
    responses=_ERRORS,
    summary="Queue a recording for analysis",
)
async def submit_analysis(
    request: Request,
    file: Annotated[UploadFile, File()],
    principal: PrincipalDep,
    container: ContainerDep,
    consent: Annotated[bool, Form()] = False,
    num_beams: Annotated[int | None, Form(ge=1, le=8)] = None,
) -> JSONResponse:
    """Return immediately with a job id; poll ``GET /v1/jobs/{job_id}``."""
    require_consent(consent, container)
    data = await read_upload(file, container.settings.max_upload_bytes)
    await container.ensure_models()
    if not container.ready:
        raise ModelNotReadyError("Models are still loading.")
    request_id: str = request.state.request_id

    def work() -> dict[str, Any]:
        analysis = container.analysis.analyze(data, num_beams=num_beams)
        response = AnalyzeResponse.from_domain(
            analysis, request_id, container.settings.feedback_language
        )
        return response.model_dump(mode="json")

    snapshot = container.jobs.submit(work, owner=principal.id)
    url = f"/v1/jobs/{snapshot.id}"
    body = JobAcceptedResponse(job_id=snapshot.id, status=snapshot.state.value, status_url=url)
    return JSONResponse(status_code=202, content=body.model_dump(), headers={"Location": url})


@router.get(
    "/{job_id}", response_model=JobStatusResponse, responses=_ERRORS, summary="Get job status"
)
async def get_job(
    job_id: str, principal: PrincipalDep, container: ContainerDep
) -> JobStatusResponse:
    """Status of a job; includes the result once it has succeeded."""
    snap = container.jobs.get(job_id, owner=principal.id)
    return JobStatusResponse(
        job_id=snap.id,
        status=snap.state.value,
        created_at=ts(snap.created_at),  # type: ignore[arg-type]
        expires_at=ts(snap.expires_at),  # type: ignore[arg-type]
        started_at=ts(snap.started_at),
        finished_at=ts(snap.finished_at),
        result=AnalyzeResponse.model_validate(snap.result) if snap.result else None,
        error=ErrorBody(**snap.error) if snap.error else None,
    )


@router.delete("/{job_id}", status_code=204, responses=_ERRORS, summary="Delete a job now")
async def delete_job(job_id: str, principal: PrincipalDep, container: ContainerDep) -> Response:
    """Erase a job and its result before its TTL expires."""
    container.jobs.delete(job_id, owner=principal.id)
    return Response(status_code=204)
