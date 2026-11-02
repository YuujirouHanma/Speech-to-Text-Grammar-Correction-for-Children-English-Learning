"""Structured logging (structlog) with automatic redaction of child data.

Transcripts, corrected sentences, raw audio and credentials must never reach
log files: they are children's personal information. The ``_redact`` processor
replaces any value stored under a sensitive key with a length-only marker, so a
careless ``log.info("done", text=transcript)`` cannot leak data.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.tracebacks import ExceptionDictTransformer
from structlog.types import EventDict, Processor, WrappedLogger

SENSITIVE_KEYS = frozenset(
    {
        "text",
        "transcript",
        "corrected",
        "original",
        "sentence",
        "sentences",
        "prompt",
        "audio",
        "content",
        "body",
        "api_key",
        "x-api-key",
        "authorization",
        "password",
        "secret",
    }
)


def _redact(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace sensitive values by a non-reversible description."""
    for key in list(event_dict):
        if key.lower() in SENSITIVE_KEYS:
            value = event_dict[key]
            if isinstance(value, (str, bytes)):
                event_dict[key] = f"<redacted len={len(value)}>"
            else:
                event_dict[key] = "<redacted>"
    return event_dict


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Route stdlib and structlog records through one structured pipeline.

    Args:
        level: Minimum log level name (``DEBUG``, ``INFO`` ...).
        json_logs: Emit JSON lines (production) instead of colored console output.
    """
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact,
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    renderer: Processor
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
        final: list[Processor] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # show_locals=False: frame locals can hold transcripts or audio bytes.
            structlog.processors.ExceptionRenderer(ExceptionDictTransformer(show_locals=False)),
            renderer,
        ]
    else:
        renderer = structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback)
        final = [structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer]

    formatter = structlog.stdlib.ProcessorFormatter(foreign_pre_chain=shared, processors=final)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lib_logger = logging.getLogger(name)
        lib_logger.handlers = []
        lib_logger.propagate = True
    # Requests are logged once by RequestContextMiddleware (with request_id and latency).
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for noisy in ("faster_whisper", "httpx", "urllib3", "huggingface_hub", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger.bind(**initial) if initial else logger
