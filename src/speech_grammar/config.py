"""Application settings loaded from environment variables (prefix ``SPEECH_GRAMMAR_``).

Every runtime knob lives here so the service can be reconfigured per deployment
(local CPU laptop, Docker CPU, Docker GPU) without code changes. Defaults are
chosen to be *privacy-first*: no cloud speech-to-text, no persistent storage,
consent required for audio.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Device(str, Enum):
    """Compute device for model inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class CorrectorBackend(str, Enum):
    """Inference runtime for the T5 grammar-correction model."""

    CT2 = "ctranslate2"
    TORCH = "torch"


class SttEngine(str, Enum):
    """Speech-to-text engines."""

    WHISPER = "whisper"
    GEMINI = "gemini"
    DEEPGRAM = "deepgram"


class InputStyle(str, Enum):
    """How text is formatted before being fed to the T5 model.

    ``jfleg`` mimics the tokenized style the model was fine-tuned on
    (``do n't`` / ``word .``); ``raw`` only normalizes whitespace and quotes.
    """

    RAW = "raw"
    JFLEG = "jfleg"


def _split_csv(value: Any) -> Any:
    """Allow comma-separated env values for list settings."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """All configuration for the Speech Grammar service."""

    model_config = SettingsConfigDict(
        env_prefix="SPEECH_GRAMMAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_parse_none_str="none",
        extra="ignore",
    )

    # ------------------------------------------------------------------ general
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True
    models_dir: Path = Path("models")

    # ---------------------------------------------------------------------- api
    host: str = "0.0.0.0"  # noqa: S104 - container default, override for bare-metal
    port: int = 8000
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    enable_docs: bool = True
    enable_ui: bool = True
    # Language of the error descriptions in the API and of the demo UI's hints.
    feedback_language: Literal["en", "id"] = "id"
    # Public demo: the UI asks for adult voices only (no children's recordings).
    demo_mode: bool = False
    require_consent: bool = True
    rate_limit_per_minute: int = Field(default=60, ge=0)
    max_upload_mb: float = Field(default=10.0, gt=0)
    max_audio_seconds: float = Field(default=60.0, gt=0)
    min_audio_seconds: float = Field(default=0.3, ge=0)
    max_text_chars: int = Field(default=2000, gt=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)

    # ---------------------------------------------------------------- corrector
    corrector_backend: CorrectorBackend = CorrectorBackend.CT2
    corrector_fallback_backend: CorrectorBackend | None = CorrectorBackend.TORCH
    # models/<name> (HF checkpoint) and models/<name>-ct2 (CTranslate2). v2 = retrained model.
    t5_model_name: str = "t5-gec-v2"
    t5_model_dir: Path | None = None
    t5_ct2_dir: Path | None = None
    device: Device = Device.AUTO
    ct2_compute_type: str = "auto"
    torch_dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    task_prefix: str = "grammar: "
    input_style: InputStyle = InputStyle.RAW
    num_beams: int = Field(default=5, ge=1, le=16)
    max_input_tokens: int = Field(default=200, ge=16, le=512)
    max_output_tokens: int = Field(default=256, ge=16, le=1024)
    max_batch_size: int = Field(default=16, ge=1)
    min_faithfulness: float = Field(default=0.4, ge=0.0, le=1.0)
    # A sentence correction is shown only if the model prefers it over the unchanged
    # sentence by this log-probability margin (calibrated on the validation split with
    # scripts/calibrate_correction.py). "-inf" turns the check off.
    min_correction_margin: float = 2.5
    # Punctuation/capitalization feedback on typed text. Speech never gets it: a
    # transcript's punctuation comes from the STT engine, not from the child.
    punctuation_feedback: bool = True
    enable_error_types: bool = True

    # ---------------------------------------------------------------------- stt
    stt_engine: SttEngine = SttEngine.WHISPER
    stt_fallback_engine: SttEngine | None = SttEngine.WHISPER
    # "auto": small.en on GPU (best and stable in the STT evaluation, see README), base.en on CPU
    # (3-6x faster, accuracy overlaps). Or a size such as "base.en", or a local model path.
    whisper_model: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = Field(default=5, ge=1, le=10)
    whisper_language: str | None = "en"
    whisper_vad: bool = True
    whisper_verbatim_prompt: bool = True
    whisper_cpu_threads: int = Field(default=0, ge=0)

    # -------------------------------------------------------------------- audio
    audio_highpass_hz: float = Field(default=80.0, ge=0)
    audio_normalize: bool = True
    audio_target_dbfs: float = Field(default=-20.0, le=0)
    audio_denoise: bool = False

    # -------------------------------------------------------- cloud (opt-in only)
    allow_cloud_stt: bool = False
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"
    deepgram_api_key: SecretStr | None = None
    deepgram_model: str = "nova-3"
    cloud_timeout_seconds: float = Field(default=30.0, gt=0)

    # ------------------------------------------------------ concurrency and jobs
    inference_workers: int = Field(default=1, ge=1)
    max_pending_requests: int = Field(default=16, ge=1)
    job_queue_size: int = Field(default=32, ge=1)
    job_ttl_seconds: int = Field(default=600, ge=10)
    job_delete_after_read: bool = False
    # Serverless hosts pause an instance between requests, which would freeze a load left in
    # the background: load during the first request that needs the models instead.
    load_models_on_demand: bool = False

    # ------------------------------------------------------------------ privacy
    temp_dir: Path | None = None
    temp_file_max_age_seconds: int = Field(default=300, ge=10)
    encrypt_job_results: bool = True

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        if self.t5_model_dir is None:
            self.t5_model_dir = self.models_dir / self.t5_model_name
        if self.t5_ct2_dir is None:
            self.t5_ct2_dir = self.models_dir / f"{self.t5_model_name}-ct2"
        if self.corrector_fallback_backend == self.corrector_backend:
            self.corrector_fallback_backend = None
        if self.stt_fallback_engine == self.stt_engine:
            self.stt_fallback_engine = None
        return self

    @property
    def max_upload_bytes(self) -> int:
        """Upload limit in bytes."""
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def auth_enabled(self) -> bool:
        """Whether API-key authentication is enforced."""
        return bool(self.api_keys)

    @property
    def whisper_download_root(self) -> Path:
        """Directory where Whisper models are cached."""
        return self.models_dir / "whisper"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
