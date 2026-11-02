"""Build model components from :class:`~speech_grammar.config.Settings`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from speech_grammar.config import CorrectorBackend, Device, Settings, SttEngine
from speech_grammar.domain.errors import CloudSttDisabledError, InferenceError
from speech_grammar.domain.ports import EditClassifier, Seq2SeqBackend, SpeechToText
from speech_grammar.inference.devices import (
    resolve_device,
    resolve_t5_compute_type,
    resolve_whisper_compute_type,
)
from speech_grammar.logs import get_logger
from speech_grammar.metrics import MODEL_INFO
from speech_grammar.services.corrector import CorrectionConfig, GrammarCorrector
from speech_grammar.services.transcriber import Transcriber

log = get_logger(__name__)


def _cuda_fallback_allowed(settings: Settings, device: str) -> bool:
    """Only an *auto*-selected GPU may silently fall back to CPU."""
    return device == "cuda" and settings.device is Device.AUTO


def build_seq2seq_backend(kind: CorrectorBackend, settings: Settings) -> Seq2SeqBackend:
    """Instantiate the requested T5 runtime.

    With ``device=auto`` a GPU is probed with one tiny generation: a visible
    driver does not guarantee that the CUDA libraries (cuBLAS/cuDNN) load, e.g.
    on Windows without the CUDA toolkit. On failure the CPU is used instead.
    """
    device = resolve_device(settings.device)
    if kind is CorrectorBackend.CT2:
        from speech_grammar.inference.t5_ct2 import Ct2Seq2SeqBackend

        assert settings.t5_ct2_dir is not None

        def load(target: str) -> Seq2SeqBackend:
            compute_type = resolve_t5_compute_type(settings.ct2_compute_type, target)
            ct2_dir = settings.t5_ct2_dir
            assert ct2_dir is not None
            loaded = Ct2Seq2SeqBackend(ct2_dir, device=target, compute_type=compute_type)
            if target == "cuda":
                loaded.generate([settings.task_prefix + "Hi."], num_beams=1, max_new_tokens=4)
            MODEL_INFO.labels("corrector", "t5", "ctranslate2", target, compute_type).set(1)
            return loaded

        try:
            return load(device)
        except (InferenceError, RuntimeError) as exc:
            if not _cuda_fallback_allowed(settings, device):
                raise
            log.warning("ct2_cuda_unavailable_using_cpu", reason=str(exc.__cause__ or exc)[:200])
            return load("cpu")

    from speech_grammar.inference.t5_torch import TorchSeq2SeqBackend

    assert settings.t5_model_dir is not None
    if device == "cuda":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_backend = TorchSeq2SeqBackend(
        settings.t5_model_dir, device=device, dtype=settings.torch_dtype
    )
    MODEL_INFO.labels("corrector", "t5", "torch", device, torch_backend.dtype).set(1)
    return torch_backend


def build_edit_classifier(settings: Settings) -> EditClassifier | None:
    """ERRANT classifier when enabled and installed, else ``None``."""
    if not settings.enable_error_types:
        return None
    try:
        from speech_grammar.text.edits import ErrantClassifier

        return ErrantClassifier()
    except Exception as exc:
        log.info("error_type_classifier_unavailable", reason=type(exc).__name__)
        return None


def build_corrector(settings: Settings) -> GrammarCorrector:
    """Build the grammar corrector, falling back at load time if needed."""
    config = CorrectionConfig(
        task_prefix=settings.task_prefix,
        input_style=settings.input_style,
        num_beams=settings.num_beams,
        max_input_tokens=settings.max_input_tokens,
        max_output_tokens=settings.max_output_tokens,
        max_batch_size=settings.max_batch_size,
        min_faithfulness=settings.min_faithfulness,
        min_correction_margin=settings.min_correction_margin,
        punctuation_feedback=settings.punctuation_feedback,
        max_chars=settings.max_text_chars,
    )
    fallback_kind = settings.corrector_fallback_backend
    try:
        primary = build_seq2seq_backend(settings.corrector_backend, settings)
    except Exception:
        if fallback_kind is None:
            raise
        log.exception("primary_corrector_load_failed_using_fallback", fallback=fallback_kind.value)
        primary, fallback_kind = build_seq2seq_backend(fallback_kind, settings), None

    fallback_factory: Callable[[], Seq2SeqBackend] | None = None
    if fallback_kind is not None:
        kind = fallback_kind

        def fallback_factory() -> Seq2SeqBackend:
            return build_seq2seq_backend(kind, settings)

    return GrammarCorrector(
        primary,
        config=config,
        fallback_factory=fallback_factory,
        classifier=build_edit_classifier(settings),
    )


def resolve_whisper_model(name: str, root: Path) -> str:
    """Prefer a pre-downloaded local snapshot (offline, read-only friendly).

    Returns the snapshot directory when ``scripts/prepare_models.py whisper``
    was run, otherwise the size name (faster-whisper then downloads it).
    """
    if Path(name).is_dir():
        return name
    repo = root / f"models--Systran--faster-whisper-{name}" / "snapshots"
    snapshots = (
        sorted(p for p in repo.glob("*") if (p / "model.bin").is_file()) if repo.is_dir() else []
    )
    return str(snapshots[-1]) if snapshots else name


def choose_whisper_model(requested: str, device: str, root: Path) -> str:
    """Resolve ``auto``: ``small.en`` on a GPU, ``base.en`` on a CPU, where it is faster (README).

    A GPU without a prepared ``small.en`` keeps ``base.en``, so that a server without
    internet access still starts.
    """
    if requested != "auto":
        return requested
    if device == "cuda" and resolve_whisper_model("small.en", root) != "small.en":
        return "small.en"
    return "base.en"


def build_stt_engine(engine: SttEngine, settings: Settings) -> SpeechToText:
    """Instantiate one speech-to-text engine (cloud engines need explicit opt-in)."""
    if engine is SttEngine.WHISPER:
        from speech_grammar.inference.whisper_stt import WhisperSpeechToText

        def load_whisper(target: str) -> SpeechToText:
            compute_type = resolve_whisper_compute_type(settings.whisper_compute_type, target)
            root = settings.whisper_download_root
            size = choose_whisper_model(settings.whisper_model, target, root)
            loaded: WhisperSpeechToText = WhisperSpeechToText(
                resolve_whisper_model(size, root),
                device=target,
                compute_type=compute_type,
                download_root=root,
                beam_size=settings.whisper_beam_size,
                language=settings.whisper_language,
                vad=settings.whisper_vad,
                verbatim_prompt=settings.whisper_verbatim_prompt,
                cpu_threads=settings.whisper_cpu_threads,
            )
            if target == "cuda":  # loading succeeds even when cuBLAS is missing
                loaded.probe()
            MODEL_INFO.labels("stt", size, "faster-whisper", target, compute_type).set(1)
            return loaded

        device = resolve_device(settings.device)
        try:
            return load_whisper(device)
        except (InferenceError, RuntimeError) as exc:
            if not _cuda_fallback_allowed(settings, device):
                raise
            log.warning(
                "whisper_cuda_unavailable_using_cpu", reason=str(exc.__cause__ or exc)[:200]
            )
            return load_whisper("cpu")

    if not settings.allow_cloud_stt:
        raise CloudSttDisabledError(
            "Cloud STT is disabled. Set SPEECH_GRAMMAR_ALLOW_CLOUD_STT=true after reviewing "
            "the privacy notes."
        )
    if engine is SttEngine.GEMINI:
        from speech_grammar.inference.cloud_stt import GeminiSpeechToText

        if settings.gemini_api_key is None:
            raise CloudSttDisabledError("SPEECH_GRAMMAR_GEMINI_API_KEY is not set.")
        MODEL_INFO.labels("stt", settings.gemini_model, "gemini", "cloud", "n/a").set(1)
        return GeminiSpeechToText(
            settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            timeout_s=settings.cloud_timeout_seconds,
        )
    from speech_grammar.inference.cloud_stt import DeepgramSpeechToText

    if settings.deepgram_api_key is None:
        raise CloudSttDisabledError("SPEECH_GRAMMAR_DEEPGRAM_API_KEY is not set.")
    MODEL_INFO.labels("stt", settings.deepgram_model, "deepgram", "cloud", "n/a").set(1)
    return DeepgramSpeechToText(
        settings.deepgram_api_key.get_secret_value(),
        model=settings.deepgram_model,
        timeout_s=settings.cloud_timeout_seconds,
    )


def build_transcriber(settings: Settings) -> Transcriber:
    """Build the transcriber with an optional (lazily loaded) fallback engine."""
    fallback_engine = settings.stt_fallback_engine
    try:
        primary = build_stt_engine(settings.stt_engine, settings)
    except Exception:
        if fallback_engine is None:
            raise
        log.exception("primary_stt_load_failed_using_fallback", fallback=fallback_engine.value)
        primary, fallback_engine = build_stt_engine(fallback_engine, settings), None

    fallback_factory: Callable[[], SpeechToText] | None = None
    if fallback_engine is not None:
        engine = fallback_engine

        def fallback_factory() -> SpeechToText:
            return build_stt_engine(engine, settings)

    return Transcriber(primary, fallback_factory=fallback_factory)
