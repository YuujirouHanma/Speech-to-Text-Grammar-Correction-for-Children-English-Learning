"""Operational endpoints: liveness, readiness, metrics, service info."""

from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from speakfix import __version__
from speakfix.api.deps import ContainerDep
from speakfix.api.schemas import ComponentStatus, InfoResponse, ReadinessResponse

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz(container: ContainerDep) -> JSONResponse:
    """200 while the process can serve; 503 when an inference thread is stuck.

    Model loading is *not* checked here (that is ``/readyz``), so a slow start
    never triggers a restart loop.
    """
    if container.stuck:
        body = {
            "status": "stuck",
            "version": __version__,
            "oldest_inference_seconds": round(container.oldest_inference_seconds, 1),
        }
        return JSONResponse(status_code=503, content=body)
    return JSONResponse(content={"status": "ok", "version": __version__})


@router.get(
    "/readyz",
    summary="Readiness probe",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readyz(container: ContainerDep) -> JSONResponse:
    """200 when every model is loaded, otherwise 503 with component details."""

    def status(name: str, ready: bool, engine: str | None) -> ComponentStatus:
        error = container.load_errors.get(name)
        return ComponentStatus(
            ready=ready, name=engine, error=error.split(":", 1)[0] if error else None
        )

    body = ReadinessResponse(
        ready=container.ready,
        loading=container.loading,
        components={
            "corrector": status(
                "corrector",
                container.corrector_ready,
                container.corrector.backend_name if container.corrector else None,
            ),
            "stt": status(
                "stt",
                container.transcriber_ready,
                container.transcriber.engine_name if container.transcriber else None,
            ),
        },
    )
    return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/info", response_model=InfoResponse, summary="Service information and limits")
async def info(container: ContainerDep) -> InfoResponse:
    """Public description of loaded models, limits and privacy settings."""
    s = container.settings
    return InfoResponse(
        name="speakfix",
        version=__version__,
        corrector=container.corrector.backend_name if container.corrector else None,
        stt=container.transcriber.engine_name if container.transcriber else None,
        limits={
            "max_upload_mb": s.max_upload_mb,
            "max_audio_seconds": s.max_audio_seconds,
            "max_text_chars": s.max_text_chars,
            "rate_limit_per_minute": s.rate_limit_per_minute,
        },
        privacy={
            "consent_required": s.require_consent,
            "cloud_stt_enabled": s.allow_cloud_stt,
            "stores_audio": False,
            "job_result_ttl_seconds": s.job_ttl_seconds,
        },
        ui={
            "demo_mode": s.demo_mode,
            "feedback_language": s.feedback_language,
            "api_key_required": s.auth_enabled,
        },
    )
