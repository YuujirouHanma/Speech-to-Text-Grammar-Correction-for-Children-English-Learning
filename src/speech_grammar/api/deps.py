"""FastAPI dependencies: container access, authentication, rate limiting, consent."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, UploadFile

from speech_grammar.domain.errors import (
    AudioTooLargeError,
    AuthenticationError,
    ConsentRequiredError,
    RateLimitedError,
)
from speech_grammar.metrics import RATE_LIMITED
from speech_grammar.services.container import Container


@dataclass(frozen=True, slots=True)
class Principal:
    """The caller: a hashed API key, or anonymous when auth is disabled."""

    id: str
    authenticated: bool


def hash_key(key: str) -> str:
    """SHA-256 hex digest of an API key (keys are never kept in plain text)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def get_container(request: Request) -> Container:
    """Return the application container."""
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def authenticate(request: Request, container: ContainerDep) -> Principal:
    """Validate the API key (when configured) and apply rate limiting.

    Raises:
        AuthenticationError: Missing/invalid key.
        RateLimitedError: The caller exceeded its request budget.
    """
    settings = container.settings
    if settings.auth_enabled:
        presented = _presented_key(request)
        if not presented:
            raise AuthenticationError("Missing API key (send the X-API-Key header).")
        digest = hash_key(presented)
        valid = [hash_key(k.get_secret_value()) for k in settings.api_keys]
        if not any(hmac.compare_digest(digest, candidate) for candidate in valid):
            raise AuthenticationError("Invalid API key.")
        principal = Principal(id=f"key:{digest[:16]}", authenticated=True)
        bucket = principal.id
    else:
        principal = Principal(id="anonymous", authenticated=False)
        bucket = f"ip:{request.client.host if request.client else 'unknown'}"

    allowed, retry_after = container.rate_limiter.acquire(bucket)
    if not allowed:
        RATE_LIMITED.inc()
        raise RateLimitedError(retry_after, "Too many requests, slow down.")
    return principal


PrincipalDep = Annotated[Principal, Depends(authenticate)]


def require_consent(consent: bool, container: Container) -> None:
    """Enforce the parental-consent flag for children's voice data."""
    if container.settings.require_consent and not consent:
        raise ConsentRequiredError(
            "Set consent=true to confirm verifiable parental consent for processing "
            "this child's voice recording."
        )


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload fully into memory and release any spooled temp file."""
    try:
        data = await file.read(max_bytes + 1)
    finally:
        await file.close()
    if len(data) > max_bytes:
        raise AudioTooLargeError(f"Audio file exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return data
