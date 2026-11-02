from pathlib import Path

import pytest

from speech_grammar.config import CorrectorBackend, Settings, SttEngine
from speech_grammar.logs import _redact


def test_csv_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_GRAMMAR_API_KEYS", "alpha, beta")
    monkeypatch.setenv("SPEECH_GRAMMAR_CORS_ORIGINS", "https://a.example,https://b.example")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert [k.get_secret_value() for k in settings.api_keys] == ["alpha", "beta"]
    assert settings.cors_origins == ["https://a.example", "https://b.example"]
    assert settings.auth_enabled


def test_derived_paths_and_fallback_dedup() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        models_dir=Path("m"),
        corrector_backend=CorrectorBackend.TORCH,
        corrector_fallback_backend=CorrectorBackend.TORCH,
        stt_engine=SttEngine.WHISPER,
        stt_fallback_engine=SttEngine.WHISPER,
    )
    assert settings.t5_model_dir == Path("m") / "t5-gec-v2"
    assert settings.t5_ct2_dir == Path("m") / "t5-gec-v2-ct2"
    legacy = Settings(_env_file=None, models_dir=Path("m"), t5_model_name="t5-gec")  # type: ignore[call-arg]
    assert legacy.t5_ct2_dir == Path("m") / "t5-gec-ct2"
    assert settings.corrector_fallback_backend is None
    assert settings.stt_fallback_engine is None
    assert not settings.auth_enabled


def test_privacy_first_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.require_consent
    assert not settings.allow_cloud_stt
    assert settings.encrypt_job_results
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_log_redaction_hides_child_data() -> None:
    event = {
        "event": "done",
        "text": "She don't like apples.",
        "transcript": b"abc",
        "audio": object(),
        "ms": 3,
    }
    redacted = _redact(None, "info", event)
    assert redacted["text"] == "<redacted len=22>"
    assert redacted["transcript"] == "<redacted len=3>"
    assert redacted["audio"] == "<redacted>"
    assert redacted["ms"] == 3


def test_tracebacks_never_include_local_variables(capsys: pytest.CaptureFixture[str]) -> None:
    import logging

    from speech_grammar.logs import configure_logging, get_logger

    configure_logging("INFO", json_logs=True)
    child_sentence = "my name is Budi and I live on Jalan Mawar 5"  # noqa: F841 - must not leak
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test").exception("failed")
    out = capsys.readouterr().out
    logging.getLogger().handlers.clear()
    assert "boom" in out
    assert "Jalan Mawar" not in out
    assert '"locals"' not in out
