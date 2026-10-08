"""Text grammar-correction endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from speakfix.api.deps import ContainerDep, PrincipalDep
from speakfix.api.schemas import (
    CorrectionOut,
    CorrectTextRequest,
    CorrectTextResponse,
    ErrorResponse,
)

router = APIRouter(prefix="/v1/text", tags=["text"])

_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (401, 413, 422, 429, 503, 504)
}


@router.post(
    "/correct",
    response_model=CorrectTextResponse,
    responses=_ERRORS,
    summary="Correct the grammar of English text",
)
async def correct_text(
    body: CorrectTextRequest, request: Request, _principal: PrincipalDep, container: ContainerDep
) -> CorrectTextResponse:
    """Correct one or more English sentences and list every edit."""
    start = time.perf_counter()
    correction = await container.run_inference(
        lambda: container.analysis.correct_text(body.text, num_beams=body.num_beams)
    )
    return CorrectTextResponse(
        request_id=request.state.request_id,
        correction=CorrectionOut.from_domain(correction, container.settings.feedback_language),
        timings_ms={"total_ms": round((time.perf_counter() - start) * 1000, 1)},
    )
