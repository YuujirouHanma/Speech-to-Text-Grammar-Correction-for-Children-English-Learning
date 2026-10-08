"""Grammar-correction use case built on a ``Seq2SeqBackend``.

Fixes over the original notebook/script pipeline:

* sentence-level batching instead of one long prompt;
* dynamic ``max_new_tokens`` (the old ``max_length=20`` truncated 57% of
  JFLEG references and the old ``min_length=10`` forced invented text);
* terminal punctuation added before inference (prevents hallucinated
  continuations), model spacing undone afterwards (``test .`` -> ``test.``);
* a faithfulness guard: if the output barely overlaps the input (a rewrite or
  hallucination) the sentence is left unchanged and a warning is returned;
* extra sentences invented by the model are trimmed (replaces the old
  ``split(".")[0]`` hack, which also cut legitimate text such as ``3.5``);
* OOM handling: retry with batch size 1, then switch to the fallback backend.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from speakfix.config import InputStyle
from speakfix.domain.errors import InferenceError, InvalidTextError, ResourceExhaustedError
from speakfix.domain.models import Correction
from speakfix.domain.ports import EditClassifier, Seq2SeqBackend
from speakfix.logs import get_logger
from speakfix.metrics import FALLBACKS, INFERENCE_ERRORS, track_stage
from speakfix.text.edits import extract_edits
from speakfix.text.normalize import (
    detokenize,
    ensure_terminal_punctuation,
    faithfulness,
    match_leading_case,
    normalize_text,
    split_sentences,
    to_jfleg_style,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Decoding and safety parameters."""

    task_prefix: str = "grammar: "
    input_style: InputStyle = InputStyle.RAW
    num_beams: int = 5
    max_input_tokens: int = 200
    max_output_tokens: int = 256
    max_batch_size: int = 16
    min_faithfulness: float = 0.4
    max_chars: int = 2000


@dataclass(frozen=True, slots=True)
class _Unit:
    source: str
    model_input: str


def _chunk_words(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


class GrammarCorrector:
    """Correct the grammar of learner text, sentence by sentence."""

    def __init__(
        self,
        backend: Seq2SeqBackend,
        *,
        config: CorrectionConfig | None = None,
        fallback_factory: Callable[[], Seq2SeqBackend] | None = None,
        classifier: EditClassifier | None = None,
    ) -> None:
        """Create the corrector.

        Args:
            backend: Primary model runtime.
            config: Decoding parameters.
            fallback_factory: Lazily builds a secondary runtime (loaded on first
                failure only, so it costs no memory in the happy path).
            classifier: Optional error-type classifier (ERRANT).
        """
        self._backend = backend
        self._config = config or CorrectionConfig()
        self._fallback_factory = fallback_factory
        self._fallback: Seq2SeqBackend | None = None
        self._fallback_lock = threading.Lock()
        self._classifier = classifier

    @property
    def backend_name(self) -> str:
        """Identifier of the primary backend."""
        return self._backend.name

    @property
    def config(self) -> CorrectionConfig:
        """Active configuration."""
        return self._config

    # ------------------------------------------------------------------ public
    def correct(self, text: str, *, num_beams: int | None = None) -> Correction:
        """Correct ``text``.

        Args:
            text: Learner text (one or more sentences).
            num_beams: Override the configured beam width.

        Returns:
            The correction with word-level edits.

        Raises:
            InvalidTextError: Empty or over-long input.
            InferenceError: Every backend failed.
        """
        original = normalize_text(text)
        if not original:
            raise InvalidTextError("Text is empty.")
        if len(original) > self._config.max_chars:
            raise InvalidTextError(f"Text exceeds {self._config.max_chars} characters.")

        units = self._prepare(split_sentences(original))
        beams = num_beams or self._config.num_beams
        with track_stage("correct"):
            outputs, used_backend = self._generate_with_fallback(
                [u.model_input for u in units], beams
            )

        warnings: list[str] = []
        corrected_parts: list[str] = []
        for unit, raw in zip(units, outputs, strict=True):
            fixed, unit_warnings = self._postprocess(unit, raw)
            corrected_parts.append(fixed)
            warnings.extend(unit_warnings)
        corrected = " ".join(corrected_parts)

        edits = extract_edits(original, corrected)
        if self._classifier is not None and edits:
            edits = self._classifier.classify(original, corrected, edits)
        return Correction(
            original=original,
            corrected=corrected,
            edits=tuple(edits),
            sentences=len(units),
            backend=used_backend,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def warmup(self) -> None:
        """Run one tiny inference so the first real request is not slow.

        Bypasses user-facing limits (``max_chars``) but uses the normal fallback
        chain, so a broken primary backend does not make the service unready.
        """
        prompt = self._config.task_prefix + "She go to school every day."
        _, used = self._generate_with_fallback([prompt], self._config.num_beams)
        if used != self._backend.name:
            log.warning("warmup_used_fallback_backend", backend=used)

    # ---------------------------------------------------------------- internals
    def _prepare(self, sentences: Sequence[str]) -> list[_Unit]:
        units: list[_Unit] = []
        prefix = self._config.task_prefix
        for sentence in sentences:
            pieces = [sentence]
            if self._backend.count_tokens(prefix + sentence) > self._config.max_input_tokens:
                pieces = _chunk_words(
                    sentence, max_words=max(8, self._config.max_input_tokens // 3)
                )
            for piece in pieces:
                model_text = ensure_terminal_punctuation(piece)
                if self._config.input_style is InputStyle.JFLEG:
                    model_text = to_jfleg_style(model_text)
                units.append(_Unit(source=piece, model_input=prefix + model_text))
        return units

    def _postprocess(self, unit: _Unit, raw: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        out = detokenize(raw)
        out_sentences = split_sentences(out)
        if len(out_sentences) > 1:
            out = out_sentences[0]
            warnings.append("trimmed_extra_model_output")
        if not out:
            return unit.source, ["empty_model_output"]
        out = match_leading_case(unit.source, out)
        if faithfulness(unit.source, out) < self._config.min_faithfulness:
            return unit.source, ["low_confidence_correction_skipped"]
        return out, warnings

    def _max_new_tokens(self, batch: Sequence[str], backend: Seq2SeqBackend) -> int:
        longest = max(backend.count_tokens(text) for text in batch)
        return int(min(self._config.max_output_tokens, math.ceil(longest * 1.5) + 10))

    def _run(
        self, backend: Seq2SeqBackend, inputs: Sequence[str], beams: int, batch: int
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(inputs), batch):
            chunk = inputs[start : start + batch]
            outputs.extend(
                backend.generate(
                    chunk, num_beams=beams, max_new_tokens=self._max_new_tokens(chunk, backend)
                )
            )
        return outputs

    def _get_fallback(self) -> Seq2SeqBackend | None:
        if self._fallback_factory is None:
            return None
        with self._fallback_lock:
            if self._fallback is None:
                log.warning("loading_fallback_backend")
                self._fallback = self._fallback_factory()
        return self._fallback

    def _generate_with_fallback(self, inputs: Sequence[str], beams: int) -> tuple[list[str], str]:
        batch = self._config.max_batch_size
        try:
            return self._run(self._backend, inputs, beams, batch), self._backend.name
        except ResourceExhaustedError:
            INFERENCE_ERRORS.labels(component="corrector", kind="oom").inc()
            log.warning("corrector_oom_retrying_unbatched", inputs=len(inputs))
            try:
                return self._run(self._backend, inputs, beams, 1), self._backend.name
            except InferenceError as retry_exc:  # OOM again, or a follow-up CUDA error
                kind = "oom" if isinstance(retry_exc, ResourceExhaustedError) else "error"
                INFERENCE_ERRORS.labels(component="corrector", kind=kind).inc()
        except InferenceError:
            INFERENCE_ERRORS.labels(component="corrector", kind="error").inc()
            log.exception("corrector_primary_failed")

        try:
            fallback = self._get_fallback()
        except Exception as exc:
            log.exception("fallback_backend_unavailable")
            raise InferenceError("Grammar correction is unavailable.") from exc
        if fallback is None:
            raise InferenceError("Grammar correction failed and no fallback is configured.")
        FALLBACKS.labels(component="corrector").inc()
        return self._run(fallback, inputs, beams, 1), fallback.name
