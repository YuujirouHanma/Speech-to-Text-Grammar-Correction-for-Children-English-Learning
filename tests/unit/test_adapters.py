import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from speakfix.config import Device, SttEngine
from speakfix.domain.errors import (
    CloudSttDisabledError,
    InferenceError,
    InferenceTimeoutError,
    ResourceExhaustedError,
)
from speakfix.inference.cloud_stt import DeepgramSpeechToText
from speakfix.inference.devices import (
    resolve_device,
    resolve_t5_compute_type,
    resolve_whisper_compute_type,
)
from speakfix.inference.factory import (
    build_stt_engine,
    choose_whisper_model,
    resolve_whisper_model,
)
from speakfix.inference.whisper_stt import _display_name
from tests.conftest import make_settings

SILENCE = np.zeros(16_000, dtype=np.float32)


def _deepgram(handler: httpx.MockTransport) -> DeepgramSpeechToText:
    return DeepgramSpeechToText("dg-key", client=httpx.Client(transport=handler))


def test_deepgram_success_and_privacy_params() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["mip_opt_out"] == "true"
        assert request.url.params["smart_format"] == "false"
        assert request.headers["Authorization"] == "Token dg-key"
        assert request.content[:4] == b"RIFF"  # re-encoded, metadata-free WAV
        body = {
            "metadata": {"duration": 1.0},
            "results": {
                "channels": [{"alternatives": [{"transcript": "she go home", "confidence": 0.91}]}]
            },
        }
        return httpx.Response(200, json=body)

    transcript = _deepgram(httpx.MockTransport(handler)).transcribe(SILENCE, 16_000)
    assert transcript.text == "she go home"
    assert transcript.engine == "deepgram:nova-3"
    assert transcript.language_probability == 0.91


def test_deepgram_http_error() -> None:
    stt = _deepgram(httpx.MockTransport(lambda _r: httpx.Response(500, json={"err": "x"})))
    with pytest.raises(InferenceError):
        stt.transcribe(SILENCE, 16_000)


def test_deepgram_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(InferenceTimeoutError):
        _deepgram(httpx.MockTransport(handler)).transcribe(SILENCE, 16_000)


def test_cloud_stt_requires_opt_in() -> None:
    with pytest.raises(CloudSttDisabledError):
        build_stt_engine(SttEngine.GEMINI, make_settings())
    with pytest.raises(CloudSttDisabledError):
        build_stt_engine(SttEngine.DEEPGRAM, make_settings(allow_cloud_stt=True))  # no key


def test_deepgram_built_when_allowed() -> None:
    settings = make_settings(allow_cloud_stt=True, deepgram_api_key="k")
    assert build_stt_engine(SttEngine.DEEPGRAM, settings).name == "deepgram:nova-3"


def test_device_resolution() -> None:
    assert resolve_device(Device.CPU) == "cpu"
    assert resolve_device(Device.CUDA) == "cuda"
    assert resolve_t5_compute_type("auto", "cpu") == "int8"
    assert resolve_t5_compute_type("float32", "cuda") == "float32"
    assert resolve_whisper_compute_type("auto", "cuda") == "float16"
    assert resolve_whisper_compute_type("auto", "cpu") == "int8"


def test_pip_cuda_libraries_are_put_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from speakfix.inference import devices

    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "cublas64_12.dll").write_bytes(b"")
    monkeypatch.setattr(devices, "_pip_cuda_library_dirs", lambda: [tmp_path, torch_lib])
    monkeypatch.setenv("PATH", "original")

    assert devices._expose_pip_cuda_libraries(("cublas64_12.dll", "cudnn64_9.dll"))
    assert os.environ["PATH"].split(os.pathsep) == [str(torch_lib), "original"]
    assert not devices._expose_pip_cuda_libraries(("libcublas.so.12",))


def test_cuda_libraries_fall_back_to_pip_copies_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from speakfix.inference import devices

    attempts = iter([False, True])  # missing system-wide, loadable after exposing
    monkeypatch.setattr(devices.sys, "platform", "win32")
    monkeypatch.setattr(devices, "_can_load", lambda _names: next(attempts))
    monkeypatch.setattr(devices, "_expose_pip_cuda_libraries", lambda _names: True)
    devices.cuda_libraries_available.cache_clear()
    try:
        assert devices.cuda_libraries_available()
    finally:
        devices.cuda_libraries_available.cache_clear()


def test_resolve_whisper_model(tmp_path: Path) -> None:
    assert resolve_whisper_model("base.en", tmp_path) == "base.en"
    snapshot = tmp_path / "models--Systran--faster-whisper-base.en" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x")
    resolved = resolve_whisper_model("base.en", tmp_path)
    assert Path(resolved) == snapshot
    assert _display_name(resolved) == "base.en"
    assert _display_name("small.en") == "small.en"


def test_auto_device_falls_back_to_cpu_when_cuda_libraries_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import speakfix.inference.factory as factory
    import speakfix.inference.t5_ct2 as t5_ct2
    from speakfix.config import CorrectorBackend

    class FakeCt2:
        def __init__(self, model_dir: Path, *, device: str, compute_type: str) -> None:
            self.device = device
            self.name = f"fake-ct2@{device}"

        def generate(self, sources, *, num_beams, max_new_tokens):  # type: ignore[no-untyped-def]
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return list(sources)

    monkeypatch.setattr(t5_ct2, "Ct2Seq2SeqBackend", FakeCt2)
    monkeypatch.setattr(factory, "resolve_device", lambda _d: "cuda")
    backend = factory.build_seq2seq_backend(
        CorrectorBackend.CT2, make_settings(models_dir=tmp_path)
    )
    assert backend.name == "fake-ct2@cpu"

    # An explicitly requested GPU must fail loudly instead of silently using the CPU.
    with pytest.raises(RuntimeError):
        factory.build_seq2seq_backend(
            CorrectorBackend.CT2, make_settings(models_dir=tmp_path, device=Device.CUDA)
        )


def test_choose_whisper_model_per_device(tmp_path: Path) -> None:
    assert choose_whisper_model("small.en", "cpu", tmp_path) == "small.en"  # explicit wins
    assert choose_whisper_model("auto", "cpu", tmp_path) == "base.en"
    # GPU without a prepared small.en: stay on base.en, an offline server must still start.
    assert choose_whisper_model("auto", "cuda", tmp_path) == "base.en"
    snapshot = tmp_path / "models--Systran--faster-whisper-small.en" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x")
    assert choose_whisper_model("auto", "cuda", tmp_path) == "small.en"
    assert choose_whisper_model("auto", "cpu", tmp_path) == "base.en"


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("CUDA failed: out of memory", ResourceExhaustedError),
        ("mkl_malloc: failed to allocate memory", ResourceExhaustedError),
        ("bad allocation", ResourceExhaustedError),
        ("something else went wrong", InferenceError),
    ],
)
def test_ct2_runtime_errors_are_classified(message: str, error: type[Exception]) -> None:
    from speakfix.inference.t5_ct2 import _inference_error

    assert type(_inference_error(RuntimeError(message))) is error


@pytest.mark.parametrize(
    ("raised", "error"),
    [
        (MemoryError(), ResourceExhaustedError),
        # onnxruntime (the VAD) raises its own types, which are not RuntimeError.
        (Exception("[ONNXRuntimeError] : 1 : FAIL : bad allocation"), ResourceExhaustedError),
        (RuntimeError("something else went wrong"), InferenceError),
    ],
)
def test_whisper_failures_are_classified(raised: Exception, error: type[Exception]) -> None:
    from speakfix.inference.whisper_stt import WhisperSpeechToText

    class Failing:
        def transcribe(self, *_args: object, **_kwargs: object) -> None:
            raise raised

    stt = WhisperSpeechToText.__new__(WhisperSpeechToText)
    stt._model = Failing()
    stt._language, stt._beam_size, stt._vad, stt._prompt = "en", 1, False, None
    with pytest.raises(error) as caught:
        stt.transcribe(SILENCE, 16_000)
    assert type(caught.value) is error
