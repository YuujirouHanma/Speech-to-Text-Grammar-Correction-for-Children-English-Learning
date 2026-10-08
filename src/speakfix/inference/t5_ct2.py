"""T5 inference on CTranslate2 (int8 on CPU, int8/bf16 on GPU).

CTranslate2 is an optimized C++ runtime for Transformer models. Compared with
PyTorch it gives several times lower latency on CPU with int8 weights, uses far
less RAM and does not need PyTorch in the production image. Tokenization uses
the HuggingFace ``tokenizers`` library directly on ``tokenizer.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from speakfix.domain.errors import InferenceError, ResourceExhaustedError


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


class Ct2Seq2SeqBackend:
    """Seq2seq generation with a converted CTranslate2 model directory."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        intra_threads: int = 0,
        inter_threads: int = 1,
    ) -> None:
        """Load the model.

        Args:
            model_dir: Output of ``scripts/prepare_models.py`` (contains
                ``model.bin`` and ``tokenizer.json``).
            device: ``cpu`` or ``cuda``.
            compute_type: CTranslate2 compute type (``int8``, ``int8_bfloat16`` ...).
            intra_threads: Threads per batch (0 = automatic).
            inter_threads: Batches processed in parallel.

        Raises:
            FileNotFoundError: The directory is not a converted model.
        """
        import ctranslate2
        from tokenizers import Tokenizer

        tokenizer_file = model_dir / "tokenizer.json"
        if not (model_dir / "model.bin").is_file() or not tokenizer_file.is_file():
            msg = (
                f"No CTranslate2 model in '{model_dir}'. "
                "Run: python scripts/prepare_models.py t5 --source <hf_model_dir>"
            )
            raise FileNotFoundError(msg)
        self._tokenizer: Any = Tokenizer.from_file(str(tokenizer_file))
        self._translator: Any = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )
        self.device = device
        self.compute_type = str(getattr(self._translator, "compute_type", compute_type))

    @property
    def name(self) -> str:
        """Backend identifier."""
        return f"ctranslate2:{self.compute_type}@{self.device}"

    def count_tokens(self, text: str) -> int:
        """Number of SentencePiece tokens (including EOS)."""
        return len(self._tokenizer.encode(text).ids)

    def generate(self, sources: Sequence[str], *, num_beams: int, max_new_tokens: int) -> list[str]:
        """Beam-search generation for a batch of sources."""
        batch = [self._tokenizer.encode(text).tokens for text in sources]
        try:
            results = self._translator.translate_batch(
                batch,
                beam_size=num_beams,
                max_decoding_length=max_new_tokens,
                return_scores=False,
            )
        except RuntimeError as exc:
            if _is_oom(exc):
                raise ResourceExhaustedError("Out of memory during T5 inference.") from exc
            raise InferenceError("T5 inference failed.") from exc
        outputs: list[str] = []
        for result in results:
            ids = [self._tokenizer.token_to_id(tok) for tok in result.hypotheses[0]]
            outputs.append(
                self._tokenizer.decode([i for i in ids if i is not None], skip_special_tokens=True)
            )
        return outputs
