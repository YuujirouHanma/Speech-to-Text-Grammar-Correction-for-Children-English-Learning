"""Composition root: owns every long-lived component of the service."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Callable
from typing import TypeVar

import anyio

from speakfix.config import Settings
from speakfix.domain.errors import InferenceTimeoutError, ServerBusyError
from speakfix.logs import get_logger
from speakfix.metrics import INFLIGHT, MODEL_READY
from speakfix.services.analysis import AnalysisService, AudioPolicy
from speakfix.services.corrector import GrammarCorrector
from speakfix.services.jobs import JobManager
from speakfix.services.privacy import PrivateTempDir, ResultCipher
from speakfix.services.ratelimit import TokenBucketLimiter
from speakfix.services.transcriber import Transcriber

log = get_logger(__name__)
T = TypeVar("T")


class Container:
    """Holds models, limiters, the job queue and privacy helpers.

    Models are loaded in a background thread so the process answers liveness
    probes immediately; readiness flips to true once loading completes.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        corrector_loader: Callable[[Settings], GrammarCorrector] | None = None,
        transcriber_loader: Callable[[Settings], Transcriber] | None = None,
    ) -> None:
        self.settings = settings
        self.corrector: GrammarCorrector | None = None
        self.transcriber: Transcriber | None = None
        self.load_errors: dict[str, str] = {}
        self.loading = False
        self._corrector_loader = corrector_loader
        self._transcriber_loader = transcriber_loader
        self._pending = 0
        self._running: dict[object, float] = {}
        self._running_lock = threading.Lock()
        self._limiter = anyio.CapacityLimiter(settings.inference_workers)
        self._load_task: asyncio.Task[None] | None = None

        self.rate_limiter = TokenBucketLimiter(settings.rate_limit_per_minute)
        self.cipher = ResultCipher(enabled=settings.encrypt_job_results)
        self.temp_dir: PrivateTempDir | None = None
        self.analysis = AnalysisService(
            corrector=lambda: self.corrector,
            transcriber=lambda: self.transcriber,
            policy=AudioPolicy(
                max_bytes=settings.max_upload_bytes,
                min_seconds=settings.min_audio_seconds,
                max_seconds=settings.max_audio_seconds,
                highpass_hz=settings.audio_highpass_hz,
                normalize=settings.audio_normalize,
                target_dbfs=settings.audio_target_dbfs,
                denoise=settings.audio_denoise,
            ),
        )
        self.jobs = JobManager(
            runner=self.run_inference,
            workers=settings.inference_workers,
            queue_size=settings.job_queue_size,
            ttl_seconds=settings.job_ttl_seconds,
            cipher=self.cipher,
            delete_after_read=settings.job_delete_after_read,
        )

    # ---------------------------------------------------------------- status
    @property
    def corrector_ready(self) -> bool:
        """Grammar model loaded."""
        return self.corrector is not None

    @property
    def transcriber_ready(self) -> bool:
        """Speech model loaded."""
        return self.transcriber is not None

    @property
    def ready(self) -> bool:
        """All models loaded and no inference is stuck."""
        return self.corrector_ready and self.transcriber_ready and not self.stuck

    @property
    def oldest_inference_seconds(self) -> float:
        """Age of the longest-running model call (0 when idle)."""
        with self._running_lock:
            starts = list(self._running.values())
        return max((time.monotonic() - s for s in starts), default=0.0)

    @property
    def stuck(self) -> bool:
        """A model call has run for more than twice the request timeout.

        Such a thread (e.g. a hung GPU kernel) keeps its limiter slot forever, so
        every later request would time out. ``/healthz`` then reports 503 so the
        orchestrator can restart the instance.
        """
        return self.oldest_inference_seconds > 2 * self.settings.request_timeout_seconds

    # ------------------------------------------------------------- lifecycle
    async def start(self, *, load_models: bool = True, background: bool = True) -> None:
        """Start helpers and (optionally) load the models."""
        self.temp_dir = PrivateTempDir(
            self.settings.temp_dir, max_age_seconds=self.settings.temp_file_max_age_seconds
        )
        self.temp_dir.activate()
        self.temp_dir.start()
        await self.jobs.start()
        if load_models and not self.settings.load_models_on_demand:
            if background:
                self._load_task = asyncio.get_running_loop().create_task(self.load_models())
            else:
                await self.load_models()

    async def stop(self) -> None:
        """Stop background work and wipe temporary data."""
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_task
        await self.jobs.stop()
        if self.temp_dir is not None:
            await self.temp_dir.close()

    async def ensure_models(self) -> None:
        """Load the models now if they load on demand; a no-op otherwise.

        The waiting request is what keeps a serverless instance running while they load.
        Concurrent callers share one load, and a failed load is tried again by the next caller.
        """
        if not self.settings.load_models_on_demand:
            return
        if self.corrector_ready and self.transcriber_ready:
            return
        if self._load_task is None or self._load_task.done():
            self._load_task = asyncio.get_running_loop().create_task(self.load_models())
        await asyncio.shield(self._load_task)  # a client leaving must not cancel it for the others

    async def load_models(self) -> None:
        """Load (and warm up) the models in a worker thread."""
        self.loading = True
        try:
            await anyio.to_thread.run_sync(self._load_models_sync)
        finally:
            self.loading = False

    def _load_models_sync(self) -> None:
        from speakfix.inference.factory import build_corrector, build_transcriber

        corrector_loader = self._corrector_loader or build_corrector
        transcriber_loader = self._transcriber_loader or build_transcriber
        try:
            corrector = corrector_loader(self.settings)
            corrector.warmup()
            self.corrector = corrector
            self.load_errors.pop("corrector", None)
            MODEL_READY.labels(component="corrector").set(1)
            log.info("corrector_ready", backend=corrector.backend_name)
        except Exception as exc:
            self.load_errors["corrector"] = f"{type(exc).__name__}: {exc}"
            log.exception("corrector_load_failed")
        try:
            transcriber = transcriber_loader(self.settings)
            self.transcriber = transcriber
            self.load_errors.pop("stt", None)
            MODEL_READY.labels(component="stt").set(1)
            log.info("stt_ready", engine=transcriber.engine_name)
        except Exception as exc:
            self.load_errors["stt"] = f"{type(exc).__name__}: {exc}"
            log.exception("stt_load_failed")

    # ------------------------------------------------------------- inference
    async def run_inference(self, fn: Callable[[], T]) -> T:
        """Run a blocking model call in a worker thread with backpressure.

        At most ``inference_workers`` calls run concurrently (protects RAM/VRAM
        from OOM); at most ``max_pending_requests`` may wait. A call exceeding
        ``request_timeout_seconds`` returns 504 to the client (the thread
        finishes in the background, still holding its limiter slot; see
        :attr:`stuck`).

        Raises:
            ServerBusyError: Too many waiting calls.
            InferenceTimeoutError: The call did not finish in time.
        """
        if self._pending >= self.settings.max_pending_requests:
            raise ServerBusyError("Server is busy, retry shortly.")
        self._pending += 1
        INFLIGHT.inc()

        def tracked() -> T:
            # Runs in the worker thread: the entry disappears only when the model
            # call really returns, even if the request was abandoned on timeout.
            token = object()
            with self._running_lock:
                self._running[token] = time.monotonic()
            try:
                return fn()
            finally:
                with self._running_lock:
                    self._running.pop(token, None)

        try:
            with anyio.fail_after(self.settings.request_timeout_seconds):
                return await anyio.to_thread.run_sync(
                    tracked, limiter=self._limiter, abandon_on_cancel=True
                )
        except TimeoutError as exc:
            raise InferenceTimeoutError("Processing took too long.") from exc
        finally:
            self._pending -= 1
            INFLIGHT.dec()
