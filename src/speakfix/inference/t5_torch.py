"""T5 inference on PyTorch/transformers (reference implementation and fallback)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from speakfix.domain.errors import InferenceError, ResourceExhaustedError


class TorchSeq2SeqBackend:
    """Seq2seq generation with a HuggingFace checkpoint (e.g. ``saved_model/``)."""

    def __init__(self, model_dir: Path, *, device: str = "cpu", dtype: str = "auto") -> None:
        """Load tokenizer and model.

        Args:
            model_dir: HuggingFace model directory (``config.json``, weights, tokenizer).
            device: ``cpu`` or ``cuda``.
            dtype: ``auto`` (bf16 on capable GPUs, else fp32), ``float32``,
                ``bfloat16`` or ``float16`` (not recommended for T5).
        """
        import torch
        import transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if not (model_dir / "config.json").is_file():
            msg = f"No HuggingFace model in '{model_dir}'."
            raise FileNotFoundError(msg)
        self._torch: Any = torch
        self.device = device
        if dtype == "auto":
            use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
            torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.dtype = str(torch_dtype).replace("torch.", "")
        self._tokenizer: Any = AutoTokenizer.from_pretrained(str(model_dir))  # type: ignore[no-untyped-call]
        major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
        dtype_kwarg = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), **{dtype_kwarg: torch_dtype})
        self._model: Any = model.eval().to(device)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return f"torch:{self.dtype}@{self.device}"

    def count_tokens(self, text: str) -> int:
        """Number of tokens (including EOS)."""
        return len(self._tokenizer(text).input_ids)

    def generate(self, sources: Sequence[str], *, num_beams: int, max_new_tokens: int) -> list[str]:
        """Beam-search generation for a batch of sources."""
        torch = self._torch
        try:
            with torch.inference_mode():
                encoded = self._tokenizer(
                    list(sources),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                output = self._model.generate(
                    **encoded,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    early_stopping=num_beams > 1,
                )
            decoded: list[str] = self._tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except RuntimeError as exc:
            raise self._inference_error(exc) from exc
        return decoded

    def score(self, sources: Sequence[str], targets: Sequence[str]) -> list[float]:
        """Total log-probability of each target given its source (EOS included)."""
        torch = self._torch
        try:
            with torch.inference_mode():
                encoded = self._tokenizer(
                    list(sources),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                labels = self._tokenizer(
                    text_target=list(targets),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).input_ids.to(self.device)
                logits = self._model(**encoded, labels=labels).logits.float()
                token_logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                totals = (token_logp * labels.ne(self._tokenizer.pad_token_id)).sum(-1)
        except RuntimeError as exc:
            raise self._inference_error(exc) from exc
        return [float(x) for x in totals.tolist()]

    def _inference_error(self, exc: RuntimeError) -> InferenceError:
        """Map a runtime failure to the domain error (freeing GPU memory after an OOM)."""
        torch = self._torch
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not oom:
            return InferenceError("T5 inference failed.")
        if device_is_cuda(self.device):
            torch.cuda.empty_cache()
        return ResourceExhaustedError("Out of memory during T5 inference.")


def device_is_cuda(device: str) -> bool:
    """True for CUDA device strings."""
    return device.startswith("cuda")
