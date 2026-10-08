"""Privacy controls: private temp directory, auto-cleanup, encryption at rest.

Design rules for children's data (COPPA-aligned, also UU PDP / GDPR-K):

* audio is processed in memory; nothing is persisted by default;
* any spill to disk (e.g. large multipart uploads spooled by Starlette) lands
  in a private directory that is swept periodically and wiped on shutdown;
* results kept for async jobs are encrypted with a per-process key and expire.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from speakfix.logs import get_logger

log = get_logger(__name__)


class PrivateTempDir:
    """A ``0700`` temp directory used as the process-wide ``tempfile.tempdir``."""

    def __init__(self, base: Path | None = None, *, max_age_seconds: int = 300) -> None:
        root = Path(base) if base else Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="speakfix-", dir=root))
        with contextlib.suppress(OSError):  # chmod is a no-op on Windows
            self.path.chmod(0o700)
        self.max_age_seconds = max_age_seconds
        self._previous_tempdir = tempfile.tempdir
        self._task: asyncio.Task[None] | None = None

    def activate(self) -> None:
        """Route ``tempfile`` (and Starlette upload spooling) into this directory."""
        tempfile.tempdir = str(self.path)

    def sweep(self, *, max_age_seconds: int | None = None) -> int:
        """Delete files older than ``max_age_seconds``; return how many were removed."""
        limit = self.max_age_seconds if max_age_seconds is None else max_age_seconds
        cutoff = time.time() - limit
        removed = 0
        for entry in self.path.glob("**/*"):
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            log.info("temp_files_swept", removed=removed)
        return removed

    async def run_sweeper(self, interval_seconds: float = 60.0) -> None:
        """Background loop that sweeps stale files."""
        while True:
            await asyncio.sleep(interval_seconds)
            self.sweep()

    def start(self, interval_seconds: float = 60.0) -> None:
        """Start the background sweeper on the running event loop."""
        self._task = asyncio.get_running_loop().create_task(self.run_sweeper(interval_seconds))

    async def close(self) -> None:
        """Stop the sweeper, restore ``tempfile`` and delete the directory."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        tempfile.tempdir = self._previous_tempdir
        shutil.rmtree(self.path, ignore_errors=True)


class ResultCipher:
    """Symmetric encryption for in-memory job results (ephemeral key).

    The key lives only in this process: a crash or restart makes every stored
    result unreadable, which is the desired behavior for transient child data.
    """

    def __init__(self, *, enabled: bool = True, key: bytes | None = None) -> None:
        self.enabled = enabled
        self._fernet = Fernet(key or Fernet.generate_key()) if enabled else None

    def seal(self, payload: dict[str, Any]) -> bytes:
        """Serialize and (optionally) encrypt a JSON-compatible payload."""
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._fernet.encrypt(raw) if self._fernet else raw

    def open(self, token: bytes) -> dict[str, Any]:
        """Decrypt and deserialize a payload produced by :meth:`seal`."""
        raw = self._fernet.decrypt(token) if self._fernet else token
        data: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return data


def wipe_directory(path: Path) -> None:
    """Remove a directory tree, ignoring errors (used in tests and shutdown)."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def secure_environ_summary() -> dict[str, bool]:
    """Which secrets are configured (booleans only, safe to log)."""
    return {
        name: bool(os.environ.get(name))
        for name in ("SPEAKFIX_API_KEYS", "SPEAKFIX_GEMINI_API_KEY", "SPEAKFIX_DEEPGRAM_API_KEY")
    }
