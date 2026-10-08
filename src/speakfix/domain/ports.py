"""Ports (interfaces) implemented by infrastructure adapters.

The application layer depends only on these protocols, which keeps model
runtimes (CTranslate2, PyTorch, Whisper, cloud APIs) swappable and lets tests
inject lightweight fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from speakfix.domain.models import Edit, Transcript

FloatArray = npt.NDArray[np.float32]


@runtime_checkable
class SpeechToText(Protocol):
    """Converts 16 kHz mono float32 audio into text."""

    @property
    def name(self) -> str:
        """Engine identifier, e.g. ``whisper:small.en``."""
        ...

    def transcribe(self, audio: FloatArray, sample_rate: int) -> Transcript:
        """Transcribe ``audio`` (float32 in [-1, 1])."""
        ...


@runtime_checkable
class Seq2SeqBackend(Protocol):
    """A text-to-text model runtime (T5 on CTranslate2 or PyTorch)."""

    @property
    def name(self) -> str:
        """Backend identifier, e.g. ``ctranslate2:int8@cpu``."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of model tokens for ``text`` (including EOS)."""
        ...

    def generate(self, sources: Sequence[str], *, num_beams: int, max_new_tokens: int) -> list[str]:
        """Generate one output string per source string."""
        ...


@runtime_checkable
class EditClassifier(Protocol):
    """Labels edits with an error type (e.g. ERRANT ``R:VERB:SVA``)."""

    def classify(self, original: str, corrected: str, edits: Sequence[Edit]) -> list[Edit]:
        """Return ``edits`` with ``error_type`` filled in where possible."""
        ...
