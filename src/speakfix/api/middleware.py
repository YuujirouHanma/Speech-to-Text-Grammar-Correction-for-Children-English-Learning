"""Pure-ASGI middleware: request context/metrics/security headers and body limits."""

from __future__ import annotations

import json
import re
import time
import uuid

import anyio
import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from speakfix.domain.errors import AudioTooLargeError
from speakfix.logs import get_logger
from speakfix.metrics import HTTP_LATENCY, HTTP_REQUESTS

log = get_logger("speakfix.http")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")
_QUIET_ROUTES = frozenset({"/metrics", "/healthz", "/readyz"})
_UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'self'"
)
# An oversized body is read and dropped before answering, within these bounds.
_DRAIN_MAX_FACTOR = 5
_DRAIN_TIMEOUT_SECONDS = 30.0
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class RequestContextMiddleware:
    """Assign a request id, record metrics/logs and add security headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        path: str = scope.get("path", "")
        is_ui = path == "/" or path.startswith("/static/")
        status = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for key, value in SECURITY_HEADERS.items():
                    headers.setdefault(key, value)
                if is_ui:
                    headers.setdefault("Content-Security-Policy", _UI_CSP)
                else:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            template = getattr(route, "path", None) or ("/static" if is_ui else "unmatched")
            method = scope.get("method", "GET")
            HTTP_REQUESTS.labels(route=template, method=method, status=str(status)).inc()
            HTTP_LATENCY.labels(route=template, method=method).observe(elapsed)
            if template not in _QUIET_ROUTES:
                log.info(
                    "http_request",
                    method=method,
                    route=template,
                    status=status,
                    duration_ms=round(elapsed * 1000, 1),
                )
            structlog.contextvars.clear_contextvars()


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` without buffering them."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI entry point."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._drain(receive, int(declared))
            await self._reject(scope, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise AudioTooLargeError("Request body is too large.")
            return message

        await self.app(scope, limited_receive, send)

    async def _drain(self, receive: Receive, declared: int) -> None:
        """Read and drop an oversized body, so that the client gets to read the 413.

        Answering while a client is still sending makes the OS reset the connection, and
        the client sees a network error instead of the status. Nothing is buffered. A body
        far over the limit, or one sent too slowly, is not waited for.
        """
        if declared > self.max_bytes * _DRAIN_MAX_FACTOR:
            return
        with anyio.move_on_after(_DRAIN_TIMEOUT_SECONDS):
            while True:
                message = await receive()
                if message["type"] != "http.request" or not message.get("more_body", False):
                    return

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id")
        body = json.dumps(
            {
                "error": {
                    "code": "payload_too_large",
                    "message": f"Request body exceeds {self.max_bytes} bytes.",
                    "request_id": request_id,
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
