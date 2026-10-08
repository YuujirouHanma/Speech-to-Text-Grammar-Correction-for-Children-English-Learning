import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from speakfix.domain.errors import (
    InferenceError,
    JobNotFoundError,
    ModelNotReadyError,
    QueueFullError,
    ServerBusyError,
)
from speakfix.services.analysis import AnalysisService
from speakfix.services.corrector import GrammarCorrector
from speakfix.services.jobs import JobManager, JobState
from speakfix.services.privacy import PrivateTempDir, ResultCipher
from speakfix.services.ratelimit import TokenBucketLimiter
from speakfix.services.transcriber import Transcriber
from tests.conftest import FakeBackend, FakeSTT, make_container, make_settings, make_wav


async def _direct_runner(fn: Any) -> Any:
    return await asyncio.to_thread(fn)


async def _wait_for(manager: JobManager, job_id: str, owner: str = "me") -> Any:
    for _ in range(200):
        snap = manager.get(job_id, owner=owner)
        if snap.state in (JobState.SUCCEEDED, JobState.FAILED):
            return snap
        await asyncio.sleep(0.01)
    raise AssertionError("job did not finish")


class TestJobs:
    def test_lifecycle_and_encryption(self) -> None:
        async def scenario() -> None:
            cipher = ResultCipher()
            manager = JobManager(runner=_direct_runner, cipher=cipher)
            await manager.start()
            snap = manager.submit(lambda: {"text": "secret child sentence"}, owner="me")
            assert snap.state is JobState.QUEUED
            done = await _wait_for(manager, snap.id)
            assert done.state is JobState.SUCCEEDED
            assert done.result == {"text": "secret child sentence"}
            stored = manager._jobs[snap.id].sealed_result
            assert stored is not None and b"secret child sentence" not in stored
            with pytest.raises(JobNotFoundError):
                manager.get(snap.id, owner="someone-else")
            manager.delete(snap.id, owner="me")
            with pytest.raises(JobNotFoundError):
                manager.get(snap.id, owner="me")
            await manager.stop()

        asyncio.run(scenario())

    def test_failed_job_reports_error_code(self) -> None:
        async def scenario() -> None:
            manager = JobManager(runner=_direct_runner)
            await manager.start()

            def boom() -> dict[str, Any]:
                raise ModelNotReadyError("loading")

            snap = manager.submit(boom, owner="me")
            done = await _wait_for(manager, snap.id)
            assert done.state is JobState.FAILED
            assert done.error == {"code": "model_not_ready", "message": "loading"}
            await manager.stop()

        asyncio.run(scenario())

    def test_queue_full(self) -> None:
        async def scenario() -> None:
            async def never(_fn: Any) -> Any:
                await asyncio.sleep(10)

            manager = JobManager(runner=never, queue_size=1, workers=1)
            await manager.start()
            manager.submit(lambda: {}, owner="me")  # picked by the worker
            await asyncio.sleep(0.01)
            manager.submit(lambda: {}, owner="me")  # fills the queue
            with pytest.raises(QueueFullError):
                manager.submit(lambda: {}, owner="me")
            await manager.stop()

        asyncio.run(scenario())

    def test_expiry_and_delete_after_read(self) -> None:
        async def scenario() -> None:
            manager = JobManager(runner=_direct_runner, ttl_seconds=10, delete_after_read=True)
            await manager.start()
            snap = manager.submit(lambda: {"ok": True}, owner="me")
            await _wait_for(manager, snap.id)
            with pytest.raises(JobNotFoundError):
                manager.get(snap.id, owner="me")  # deleted after the first read
            other = manager.submit(lambda: {"ok": True}, owner="me")
            manager._jobs[other.id].expires_at = time.time() - 1
            assert manager.purge_expired() == 1
            await manager.stop()

        asyncio.run(scenario())


class TestPrivacy:
    def test_temp_dir_sweep_and_close(self, tmp_path: Path) -> None:
        async def scenario() -> None:
            temp = PrivateTempDir(tmp_path, max_age_seconds=60)
            temp.activate()
            assert tempfile.gettempdir() == str(temp.path)
            old, new = temp.path / "old.bin", temp.path / "new.bin"
            old.write_bytes(b"audio")
            new.write_bytes(b"audio")
            past = time.time() - 3600
            os.utime(old, (past, past))
            assert temp.sweep() == 1
            assert not old.exists() and new.exists()
            temp.start(interval_seconds=0.01)
            await temp.close()
            assert not temp.path.exists()
            assert tempfile.gettempdir() != str(temp.path)

        asyncio.run(scenario())

    def test_cipher_round_trip(self) -> None:
        cipher = ResultCipher()
        token = cipher.seal({"text": "child data"})
        assert b"child data" not in token
        assert cipher.open(token) == {"text": "child data"}
        plain = ResultCipher(enabled=False)
        assert plain.open(plain.seal({"a": 1})) == {"a": 1}


class TestRateLimit:
    def test_burst_then_reject(self) -> None:
        limiter = TokenBucketLimiter(rate_per_minute=60, burst=2)
        assert limiter.acquire("k")[0]
        assert limiter.acquire("k")[0]
        allowed, retry = limiter.acquire("k")
        assert not allowed and retry >= 1
        assert limiter.acquire("other")[0]

    def test_disabled(self) -> None:
        limiter = TokenBucketLimiter(rate_per_minute=0)
        assert all(limiter.acquire("k")[0] for _ in range(1000))


class TestTranscriber:
    def test_fallback_on_failure(self) -> None:
        class Broken(FakeSTT):
            def transcribe(self, audio: np.ndarray, sample_rate: int):  # type: ignore[no-untyped-def]
                raise RuntimeError("cloud down")

        transcriber = Transcriber(
            Broken(name="broken"), fallback_factory=lambda: FakeSTT(name="local")
        )
        out = transcriber.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        assert out.engine == "local"

    def test_no_fallback_raises(self) -> None:
        class Broken(FakeSTT):
            def transcribe(self, audio: np.ndarray, sample_rate: int):  # type: ignore[no-untyped-def]
                raise RuntimeError("down")

        with pytest.raises(InferenceError):
            Transcriber(Broken()).transcribe(np.zeros(16000, dtype=np.float32), 16000)


class TestAnalysisService:
    def test_full_analysis(self) -> None:
        service = AnalysisService(
            corrector=lambda: GrammarCorrector(FakeBackend()),
            transcriber=lambda: Transcriber(FakeSTT("He don't like apples.")),
        )
        analysis = service.analyze(make_wav(1.5))
        assert analysis.transcript.text == "He don't like apples."
        assert analysis.correction is not None
        assert analysis.correction.corrected == "He doesn't like apples."
        assert {"audio_ms", "stt_ms", "correct_ms", "total_ms"} <= set(analysis.timings_ms)

    def test_correction_failure_degrades_gracefully(self) -> None:
        service = AnalysisService(
            corrector=lambda: GrammarCorrector(FakeBackend(fail_times=99, error=InferenceError)),
            transcriber=lambda: Transcriber(FakeSTT()),
        )
        analysis = service.analyze(make_wav(1.0))
        assert analysis.correction is None
        assert "correction_unavailable" in analysis.warnings

    def test_empty_transcript(self) -> None:
        service = AnalysisService(
            corrector=lambda: GrammarCorrector(FakeBackend()),
            transcriber=lambda: Transcriber(FakeSTT("")),
        )
        analysis = service.analyze(make_wav(1.0))
        assert analysis.correction is None
        assert "empty_transcript" in analysis.warnings

    def test_not_ready(self) -> None:
        service = AnalysisService(corrector=lambda: None, transcriber=lambda: None)
        with pytest.raises(ModelNotReadyError):
            service.correct_text("hello")
        with pytest.raises(ModelNotReadyError):
            service.transcribe(make_wav(1.0))


class TestContainerBackpressure:
    def test_server_busy_when_too_many_pending(self) -> None:
        async def scenario() -> None:
            container = make_container(make_settings(max_pending_requests=1))
            container._pending = 1
            with pytest.raises(ServerBusyError):
                await container.run_inference(lambda: 1)

        asyncio.run(scenario())

    def test_timeout(self) -> None:
        from speakfix.domain.errors import InferenceTimeoutError

        async def scenario() -> None:
            container = make_container(make_settings(request_timeout_seconds=0.05))
            with pytest.raises(InferenceTimeoutError):
                await container.run_inference(lambda: time.sleep(0.5))

        asyncio.run(scenario())


class TestWatchdog:
    def test_stuck_inference_flips_health(self) -> None:
        async def scenario() -> None:
            container = make_container(make_settings(request_timeout_seconds=0.05))
            container.corrector = GrammarCorrector(FakeBackend())
            container.transcriber = Transcriber(FakeSTT())
            assert container.ready and not container.stuck
            with pytest.raises(Exception):  # noqa: B017 - InferenceTimeoutError
                await container.run_inference(lambda: time.sleep(0.4))
            await asyncio.sleep(0.15)  # abandoned thread still running > 2x timeout
            assert container.stuck
            assert not container.ready
            await asyncio.sleep(0.4)  # thread finally returns
            assert not container.stuck

        asyncio.run(scenario())
