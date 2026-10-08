import time
from collections.abc import Callable

from fastapi.testclient import TestClient

from tests.conftest import FakeBackend, FakeSTT, make_wav

Factory = Callable[..., TestClient]


def _audio(data: bytes | None = None, name: str = "clip.wav") -> dict[str, tuple[str, bytes, str]]:
    return {"file": (name, data if data is not None else make_wav(1.5), "audio/wav")}


class TestOps:
    def test_health_ready_info(self, client: TestClient) -> None:
        assert client.get("/healthz").json()["status"] == "ok"
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["components"]["corrector"]["name"] == "fake:t5"
        info = client.get("/v1/info").json()
        assert info["privacy"]["stores_audio"] is False
        assert info["privacy"]["consent_required"] is True
        assert info["ui"] == {"demo_mode": False, "feedback_language": "id"}

    def test_not_ready_returns_503(self, app_factory: Factory) -> None:
        client = app_factory(fail_corrector=True)
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["components"]["corrector"]["error"] == "FileNotFoundError"
        response = client.post("/v1/text/correct", json={"text": "He don't like apples."})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_not_ready"
        assert response.headers["Retry-After"] == "10"

    def test_metrics_use_route_templates(self, client: TestClient) -> None:
        client.post("/v1/text/correct", json={"text": "He don't like apples."})
        client.get("/v1/jobs/does-not-exist")
        body = client.get("/metrics").text
        assert 'route="/v1/text/correct"' in body
        assert 'route="/v1/jobs/{job_id}"' in body
        assert "does-not-exist" not in body
        assert "speakfix_stage_duration_seconds" in body

    def test_ui_served_with_csp(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "SpeakFix" in response.text
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert client.get("/static/app.js").status_code == 200


class TestTextEndpoint:
    def test_correct_text(self, client: TestClient) -> None:
        response = client.post("/v1/text/correct", json={"text": "He don't like apples."})
        assert response.status_code == 200
        body = response.json()
        assert body["correction"]["corrected"] == "He doesn't like apples."
        assert body["correction"]["edits"][0]["original"] == "don't"
        assert body["request_id"] == response.headers["X-Request-ID"]
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_request_id_propagation(self, client: TestClient) -> None:
        response = client.post(
            "/v1/text/correct",
            json={"text": "I am student."},
            headers={"X-Request-ID": "trace-12345678"},
        )
        assert response.headers["X-Request-ID"] == "trace-12345678"

    def test_validation_error_does_not_echo_input(self, client: TestClient) -> None:
        response = client.post("/v1/text/correct", json={"text": "", "secret": "child name"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
        assert "child name" not in response.text

    def test_text_limit(self, app_factory: Factory) -> None:
        client = app_factory(max_text_chars=10)
        response = client.post(
            "/v1/text/correct", json={"text": "This is way more than ten characters."}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_text"


class TestSecurity:
    def test_api_key_required(self, app_factory: Factory) -> None:
        client = app_factory(api_keys="s3cret-key")
        payload = {"text": "He don't like apples."}
        missing = client.post("/v1/text/correct", json=payload)
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "unauthorized"
        assert (
            client.post("/v1/text/correct", json=payload, headers={"X-API-Key": "nope"}).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/text/correct", json=payload, headers={"X-API-Key": "s3cret-key"}
            ).status_code
            == 200
        )
        bearer = client.post(
            "/v1/text/correct", json=payload, headers={"Authorization": "Bearer s3cret-key"}
        )
        assert bearer.status_code == 200

    def test_rate_limit(self, app_factory: Factory) -> None:
        client = app_factory(rate_limit_per_minute=2)
        payload = {"text": "He don't like apples."}
        assert client.post("/v1/text/correct", json=payload).status_code == 200
        assert client.post("/v1/text/correct", json=payload).status_code == 200
        limited = client.post("/v1/text/correct", json=payload)
        assert limited.status_code == 429
        assert int(limited.headers["Retry-After"]) >= 1

    def test_consent_required_for_audio(self, client: TestClient) -> None:
        response = client.post("/v1/audio/analyze", files=_audio())
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "consent_required"

    def test_consent_can_be_disabled_by_operator(self, app_factory: Factory) -> None:
        client = app_factory(require_consent=False)
        assert client.post("/v1/audio/analyze", files=_audio()).status_code == 200


class TestAudioEndpoints:
    def test_analyze(self, app_factory: Factory) -> None:
        stt = FakeSTT("He don't like apples.")
        client = app_factory(stt=stt)
        response = client.post("/v1/audio/analyze", files=_audio(), data={"consent": "true"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["transcript"]["text"] == "He don't like apples."
        assert body["correction"]["corrected"] == "He doesn't like apples."
        assert body["audio"]["duration_s"] > 1.0
        assert stt.calls == 1

    def test_transcribe(self, client: TestClient) -> None:
        response = client.post("/v1/audio/transcribe", files=_audio(), data={"consent": "true"})
        assert response.status_code == 200
        assert response.json()["transcript"]["engine"] == "fake:stt"

    def test_unsupported_media(self, client: TestClient) -> None:
        response = client.post(
            "/v1/audio/analyze",
            files=_audio(b"%PDF-1.7 not audio" * 10, "x.pdf"),
            data={"consent": "true"},
        )
        assert response.status_code == 415

    def test_corrupt_audio(self, client: TestClient) -> None:
        corrupt = b"RIFF\x00\x00\x00\x00WAVEjunk" + b"\x00" * 64
        response = client.post("/v1/audio/analyze", files=_audio(corrupt), data={"consent": "true"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_audio"

    def test_too_long_audio(self, app_factory: Factory) -> None:
        client = app_factory(max_audio_seconds=1.0)
        response = client.post(
            "/v1/audio/analyze", files=_audio(make_wav(3.0)), data={"consent": "true"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_audio_duration"

    def test_payload_too_large(self, app_factory: Factory) -> None:
        client = app_factory(max_upload_mb=0.05)
        response = client.post(
            "/v1/audio/analyze", files=_audio(make_wav(5.0)), data={"consent": "true"}
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    def test_correction_failure_still_returns_transcript(self, app_factory: Factory) -> None:
        from speakfix.domain.errors import InferenceError

        backend = FakeBackend(error=InferenceError)
        client = app_factory(backend=backend)
        backend.fail_times = 999  # model loaded fine, then starts failing at runtime
        response = client.post("/v1/audio/analyze", files=_audio(), data={"consent": "true"})
        assert response.status_code == 200
        assert response.json()["correction"] is None
        assert "correction_unavailable" in response.json()["warnings"]


class TestJobs:
    def _wait(self, client: TestClient, url: str, headers: dict[str, str] | None = None) -> dict:
        for _ in range(200):
            body = client.get(url, headers=headers).json()
            if body["status"] in ("succeeded", "failed"):
                return body
            time.sleep(0.02)
        raise AssertionError("job did not finish")

    def test_async_job_flow(self, app_factory: Factory) -> None:
        client = app_factory(api_keys="owner-key,other-key")
        owner = {"X-API-Key": "owner-key"}
        submitted = client.post(
            "/v1/jobs/analyze", files=_audio(), data={"consent": "true"}, headers=owner
        )
        assert submitted.status_code == 202
        url = submitted.headers["Location"]
        body = self._wait(client, url, owner)
        assert body["status"] == "succeeded"
        assert body["result"]["correction"]["corrected"] == "He doesn't like apples."
        assert client.get(url, headers={"X-API-Key": "other-key"}).status_code == 404
        assert client.delete(url, headers=owner).status_code == 204
        assert client.get(url, headers=owner).status_code == 404

    def test_job_requires_consent(self, client: TestClient) -> None:
        assert client.post("/v1/jobs/analyze", files=_audio()).status_code == 403

    def test_unknown_job(self, client: TestClient) -> None:
        response = client.get("/v1/jobs/unknown")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "job_not_found"


def test_healthz_reports_stuck_inference(client: TestClient) -> None:
    container = client.app.state.container  # type: ignore[attr-defined]
    container._running[object()] = time.monotonic() - 10_000
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["status"] == "stuck"
    assert client.get("/readyz").status_code == 503
    container._running.clear()
    assert client.get("/healthz").status_code == 200
