"""In-process asynchronous job queue for long-running analyses.

Why not Celery/Redis by default? Each worker process would load its own copy of
Whisper and T5 (~1-2 GB) and audio would transit through a broker (an extra
place where children's data is stored). A bounded in-process queue on top of
the inference limiter gives non-blocking APIs with zero extra infrastructure.
For multi-node scale-out, put several replicas behind a load balancer; switch
this module to a Redis-backed implementation only if jobs must survive restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from speech_grammar.domain.errors import JobNotFoundError, QueueFullError, SpeechGrammarError
from speech_grammar.logs import get_logger
from speech_grammar.metrics import JOBS
from speech_grammar.services.privacy import ResultCipher

log = get_logger(__name__)

JobFn = Callable[[], dict[str, Any]]
Runner = Callable[[JobFn], Awaitable[dict[str, Any]]]


class JobState(str, Enum):
    """Lifecycle states of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class _Job:
    id: str
    owner: str
    state: JobState
    created_at: float
    expires_at: float
    started_at: float | None = None
    finished_at: float | None = None
    sealed_result: bytes | None = None
    error: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Read-only view of a job returned to the API layer."""

    id: str
    state: JobState
    created_at: float
    expires_at: float
    started_at: float | None
    finished_at: float | None
    result: dict[str, Any] | None
    error: dict[str, str] | None


class JobManager:
    """Bounded queue + worker tasks + TTL purge + encrypted results."""

    def __init__(
        self,
        *,
        runner: Runner,
        workers: int = 1,
        queue_size: int = 32,
        ttl_seconds: int = 600,
        cipher: ResultCipher | None = None,
        delete_after_read: bool = False,
        purge_interval_seconds: float = 15.0,
    ) -> None:
        self._runner = runner
        self._workers = workers
        self._queue_size = queue_size
        self._ttl = ttl_seconds
        self._cipher = cipher or ResultCipher()
        self._delete_after_read = delete_after_read
        self._purge_interval = purge_interval_seconds
        self._jobs: dict[str, _Job] = {}
        self._pending: dict[str, JobFn] = {}
        self._queue: asyncio.Queue[str] | None = None
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Start worker and purge tasks on the running loop."""
        self._queue = asyncio.Queue(maxsize=self._queue_size)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._worker(i)) for i in range(self._workers)]
        self._tasks.append(loop.create_task(self._purge_loop()))

    async def stop(self) -> None:
        """Cancel workers and drop every job (results are transient by design)."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._jobs.clear()
        self._pending.clear()
        self._update_metrics()

    # ----------------------------------------------------------------- public
    def submit(self, fn: JobFn, *, owner: str) -> JobSnapshot:
        """Queue ``fn`` for execution.

        Raises:
            QueueFullError: The queue is at capacity (client should retry later).
        """
        if self._queue is None:
            msg = "JobManager.start() was not called."
            raise RuntimeError(msg)
        if self._queue.full():
            raise QueueFullError("The job queue is full, retry later.")
        now = time.time()
        job = _Job(
            id=secrets.token_urlsafe(16),
            owner=owner,
            state=JobState.QUEUED,
            created_at=now,
            expires_at=now + self._ttl,
        )
        self._jobs[job.id] = job
        self._pending[job.id] = fn
        self._queue.put_nowait(job.id)
        self._update_metrics()
        return self._snapshot(job, include_result=False)

    def get(self, job_id: str, *, owner: str) -> JobSnapshot:
        """Return a job visible to ``owner``.

        Raises:
            JobNotFoundError: Unknown id, expired job or different owner.
        """
        job = self._lookup(job_id, owner)
        snapshot = self._snapshot(job, include_result=True)
        if self._delete_after_read and job.state in (JobState.SUCCEEDED, JobState.FAILED):
            self._jobs.pop(job.id, None)
            self._update_metrics()
        return snapshot

    def delete(self, job_id: str, *, owner: str) -> None:
        """Delete a job and its result immediately."""
        job = self._lookup(job_id, owner)
        self._jobs.pop(job.id, None)
        self._pending.pop(job.id, None)
        self._update_metrics()

    def purge_expired(self) -> int:
        """Remove expired jobs; return the number removed."""
        now = time.time()
        expired = [jid for jid, job in self._jobs.items() if job.expires_at <= now]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._pending.pop(jid, None)
        if expired:
            self._update_metrics()
        return len(expired)

    # -------------------------------------------------------------- internals
    def _lookup(self, job_id: str, owner: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None or job.owner != owner or job.expires_at <= time.time():
            raise JobNotFoundError("Job not found or expired.")
        return job

    def _snapshot(self, job: _Job, *, include_result: bool) -> JobSnapshot:
        result = None
        if include_result and job.sealed_result is not None:
            result = self._cipher.open(job.sealed_result)
        return JobSnapshot(
            id=job.id,
            state=job.state,
            created_at=job.created_at,
            expires_at=job.expires_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result=result,
            error=job.error,
        )

    async def _worker(self, index: int) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                fn = self._pending.pop(job_id, None)
                job = self._jobs.get(job_id)
                if fn is None or job is None:  # deleted or expired while queued
                    continue
                job.state, job.started_at = JobState.RUNNING, time.time()
                self._update_metrics()
                try:
                    result = await self._runner(fn)
                    job.sealed_result = self._cipher.seal(result)
                    job.state = JobState.SUCCEEDED
                except SpeechGrammarError as exc:
                    job.state, job.error = (
                        JobState.FAILED,
                        {"code": exc.code, "message": exc.message},
                    )
                except Exception:
                    log.exception("job_failed", worker=index)
                    job.state = JobState.FAILED
                    job.error = {"code": "internal_error", "message": "Unexpected error."}
                finally:
                    del fn  # release the audio bytes held by the closure
                    job.finished_at = time.time()
                    self._update_metrics()
            finally:
                self._queue.task_done()

    async def _purge_loop(self) -> None:
        while True:
            await asyncio.sleep(self._purge_interval)
            self.purge_expired()

    def _update_metrics(self) -> None:
        counts = dict.fromkeys(JobState, 0)
        for job in self._jobs.values():
            counts[job.state] += 1
        for state, count in counts.items():
            JOBS.labels(state=state.value).set(count)
