"""Framework-free domain objects shared by every layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

EditOp = Literal["replace", "insert", "delete"]


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A time-aligned piece of a transcript."""

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    """Speech-to-text output."""

    text: str
    duration_s: float
    engine: str
    language: str | None = None
    language_probability: float | None = None
    segments: tuple[TranscriptSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class Edit:
    """One word-level change between the original and the corrected text.

    ``start``/``end`` are character offsets into the *original* text, so a UI can
    highlight the learner's mistake in place.
    """

    op: EditOp
    original: str
    corrected: str
    start: int
    end: int
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class Correction:
    """Grammar-correction output for a piece of text."""

    original: str
    corrected: str
    edits: tuple[Edit, ...]
    sentences: int
    backend: str
    warnings: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """True when the model changed anything."""
        return self.original != self.corrected


@dataclass(frozen=True, slots=True)
class AudioStats:
    """Signal statistics computed during preprocessing (no audio content)."""

    duration_s: float
    sample_rate: int
    rms_dbfs: float
    peak: float
    clipped_ratio: float
    speech_ratio: float


@dataclass(frozen=True, slots=True)
class Analysis:
    """Full speech pipeline result: transcript and its correction."""

    transcript: Transcript
    correction: Correction | None
    audio: AudioStats | None = None
    warnings: tuple[str, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
