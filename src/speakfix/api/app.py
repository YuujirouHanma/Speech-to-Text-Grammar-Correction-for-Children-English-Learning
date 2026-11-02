"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from speakfix import __version__
from speakfix.api.errors import register_error_handlers
from speakfix.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from speakfix.api.routes import audio, health, jobs, text
from speakfix.config import Settings, get_settings
from speakfix.logs import configure_logging, get_logger
from speakfix.metrics import BUILD_INFO
from speakfix.services.container import Container

log = get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = """
Speech transcription and grammar correction for monitoring children's English
learning. **Whisper** (local, faster-whisper) transcribes speech and a fine-tuned
**T5** model corrects grammar; every response lists word-level edits that can be
shown to the learner.

Privacy: audio is processed in memory and never stored. Audio endpoints require
`consent=true` (verifiable parental consent obtained by the integrating app).
"""


def _keep_uploads_in_memory(settings: Settings) -> None:
    """Prevent Starlette from spooling uploads to disk (child audio stays in RAM)."""
    if hasattr(MultiPartParser, "spool_max_size"):
        MultiPartParser.spool_max_size = settings.max_upload_bytes + 1024


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
    load_models: bool = True,
    background_load: bool = True,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Configuration (defaults to environment-based settings).
        container: Pre-built container (tests inject fakes here).
        load_models: Load models at startup.
        background_load: Load models in the background (readiness flips later)
            instead of blocking startup.

    Returns:
        The configured FastAPI app.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    BUILD_INFO.info({"version": __version__, "environment": settings.environment})
    _keep_uploads_in_memory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or Container(settings)
        app.state.container = c
        await c.start(load_models=load_models, background=background_load)
        if not settings.auth_enabled and settings.environment == "prod":
            log.warning("api_auth_disabled_in_production")
        log.info(
            "service_started",
            version=__version__,
            consent_required=settings.require_consent,
            cloud_stt=settings.allow_cloud_stt,
        )
        try:
            yield
        finally:
            await c.stop()
            log.info("service_stopped")

    app = FastAPI(
        title="SpeakFix API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(text.router)
    app.include_router(audio.router)
    app.include_router(jobs.router)

    if settings.enable_ui and STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After", "Location"],
            max_age=600,
        )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + 64 * 1024)
    app.add_middleware(RequestContextMiddleware)
    return app
