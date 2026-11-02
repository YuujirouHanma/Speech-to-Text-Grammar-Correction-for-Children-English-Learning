"""Map exceptions to consistent JSON error responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from speakfix.domain.errors import RateLimitedError, SpeakFixError
from speakfix.logs import get_logger

log = get_logger(__name__)

STATUS_BY_CODE: dict[str, int] = {
    "invalid_text": 422,
    "invalid_audio": 422,
    "invalid_audio_duration": 422,
    "unsupported_media_type": 415,
    "payload_too_large": 413,
    "consent_required": 403,
    "cloud_stt_disabled": 403,
    "unauthorized": 401,
    "rate_limited": 429,
    "job_not_found": 404,
    "model_not_ready": 503,
    "server_busy": 503,
    "queue_full": 503,
    "resource_exhausted": 503,
    "inference_timeout": 504,
    "inference_failed": 500,
    "internal_error": 500,
}
_RETRY_AFTER = {
    "model_not_ready": "10",
    "server_busy": "2",
    "queue_full": "5",
    "resource_exhausted": "5",
}


def _request_id(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    return str(rid) if rid else None


def error_response(
    request: Request, status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Build the standard error envelope."""
    body = {"error": {"code": code, "message": message, "request_id": _request_id(request)}}
    return JSONResponse(status_code=status, content=body, headers=headers)


async def _speakfix_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, SpeakFixError)
    status = STATUS_BY_CODE.get(exc.code, 500)
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitedError):
        headers["Retry-After"] = str(exc.retry_after_s)
    elif exc.code in _RETRY_AFTER:
        headers["Retry-After"] = _RETRY_AFTER[exc.code]
    if exc.code == "unauthorized":
        headers["WWW-Authenticate"] = "ApiKey"
    if status >= 500:
        log.warning("request_failed", code=exc.code)
    return error_response(request, status, exc.code, exc.message, headers or None)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Never echo the submitted values back: they may contain a child's text.
    problems = [
        f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', 'invalid')}"
        for err in exc.errors()
    ]
    return error_response(request, 422, "validation_error", "; ".join(problems)[:500])


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(
        exc.status_code, "http_error"
    )
    return error_response(request, exc.status_code, code, str(exc.detail))


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", error=type(exc).__name__)
    return error_response(request, 500, "internal_error", "An unexpected error occurred.")


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to ``app``."""
    app.add_exception_handler(SpeakFixError, _speakfix_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled)
