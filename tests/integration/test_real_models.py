"""End-to-end tests with the real model weights (skipped when models/ is missing)."""

from importlib.util import find_spec
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from speakfix.api.app import create_app
from speakfix.config import CorrectorBackend
from speakfix.services.container import Container
from tests.conftest import MODELS_DIR, PROJECT_ROOT, make_settings

CT2_DIR = MODELS_DIR / "t5-gec-ct2"
HF_DIR = MODELS_DIR / "t5-gec"
WHISPER_ROOT = MODELS_DIR / "whisper"
VOICE_SAMPLE = PROJECT_ROOT.parent / "recorded.wav"  # real child-style recording from the project

needs_ct2 = pytest.mark.skipif(
    not (CT2_DIR / "model.bin").exists(), reason="CT2 model not prepared"
)
needs_hf = pytest.mark.skipif(not (HF_DIR / "config.json").exists(), reason="HF model not prepared")
needs_whisper = pytest.mark.skipif(
    not any(WHISPER_ROOT.glob("models--Systran--faster-whisper-base.en")),
    reason="Whisper not prepared",
)
needs_errant = pytest.mark.skipif(
    find_spec("errant") is None or find_spec("en_core_web_sm") is None,
    reason="errant / spaCy en_core_web_sm not installed",
)

pytestmark = pytest.mark.slow

SVA_CASES = {
    "She have completed her homework.": "She has completed her homework.",
    "I goes to the store everyday.": "I go to the store everyday.",
}


@needs_ct2
def test_ct2_backend_corrects() -> None:
    from speakfix.inference.t5_ct2 import Ct2Seq2SeqBackend

    backend = Ct2Seq2SeqBackend(CT2_DIR, device="cpu", compute_type="int8")
    outputs = backend.generate(["grammar: " + s for s in SVA_CASES], num_beams=4, max_new_tokens=32)
    assert outputs == list(SVA_CASES.values())
    assert backend.count_tokens("grammar: hello") >= 3
    assert backend.name.startswith("ctranslate2:int8") and backend.name.endswith("@cpu")


@needs_ct2
def test_ct2_missing_dir_raises(tmp_path: Path) -> None:
    from speakfix.inference.t5_ct2 import Ct2Seq2SeqBackend

    with pytest.raises(FileNotFoundError):
        Ct2Seq2SeqBackend(tmp_path)


@needs_hf
def test_torch_backend_agrees_with_reference() -> None:
    from speakfix.inference.t5_torch import TorchSeq2SeqBackend

    backend = TorchSeq2SeqBackend(HF_DIR, device="cpu", dtype="float32")
    outputs = backend.generate(["grammar: " + s for s in SVA_CASES], num_beams=4, max_new_tokens=32)
    assert outputs == list(SVA_CASES.values())


@needs_hf
def test_factory_falls_back_to_torch_when_ct2_missing(tmp_path: Path) -> None:
    from speakfix.inference.factory import build_corrector

    settings = make_settings(
        t5_ct2_dir=tmp_path / "missing",
        t5_model_dir=HF_DIR,
        device="cpu",
        corrector_fallback_backend=CorrectorBackend.TORCH,
        enable_error_types=False,
    )
    corrector = build_corrector(settings)
    assert corrector.backend_name.startswith("torch:")


@needs_errant
def test_errant_types_clitics_and_punctuation() -> None:
    from speakfix.text.edits import ErrantClassifier

    original = "He don't like apples. They is happy"
    edits = ErrantClassifier().classify(original, "He doesn't like apples. They are happy.", [])
    assert [(e.original, e.corrected, e.error_type) for e in edits] == [
        ("do", "does", "R:VERB:SVA"),
        ("is", "are", "R:VERB:SVA"),
        ("", ".", "M:PUNCT"),
    ]
    assert all(original[e.start : e.end] == e.original for e in edits)


@needs_ct2
@needs_whisper
@pytest.mark.skipif(not VOICE_SAMPLE.exists(), reason="voice sample missing")
def test_end_to_end_api_with_real_models() -> None:
    settings = make_settings(device="cpu", enable_error_types=False, whisper_model="base.en")
    app = create_app(settings, container=Container(settings), background_load=False)
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        text = client.post("/v1/text/correct", json={"text": "She have completed her homework"})
        assert text.json()["correction"]["corrected"] == "She has completed her homework."
        with VOICE_SAMPLE.open("rb") as fh:
            audio = client.post(
                "/v1/audio/analyze",
                files={"file": ("voice.wav", fh, "audio/wav")},
                data={"consent": "true"},
            )
        assert audio.status_code == 200, audio.text
        body = audio.json()
        assert "sing" in body["transcript"]["text"].lower()
        assert body["transcript"]["engine"] == "whisper:base.en"
        assert body["correction"] is not None
