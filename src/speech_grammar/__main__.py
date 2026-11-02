"""Run the API server: ``python -m speech_grammar`` (or the ``speech-grammar`` console script)."""

from __future__ import annotations

import os


def main() -> None:
    """Start uvicorn with production-friendly defaults.

    One worker per process/container: every worker would load its own copy of
    the models, so scale out with replicas instead of ``--workers``.
    """
    import uvicorn

    from speech_grammar.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "speech_grammar.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=False,
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
