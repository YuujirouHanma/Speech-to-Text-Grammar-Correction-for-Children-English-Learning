"""Choose the corrector's confidence margin on the validation split (never the test sets).

The validation data is rebuilt exactly as ``train_t5.py`` built it (same files,
same seed): learner sentences with their corrections, plus the corrected
sentences fed back as input, since a feedback tool must also leave correct
sentences alone. For each margin the script reports ERRANT precision, recall
and F0.5 (punctuation edits ignored, as the service does not give punctuation
feedback), and how often correct sentences are changed.

Model outputs and scores are cached, so the sweep runs the model only once.

Example:
    python scripts/calibrate_correction.py --jfleg-train ../train.csv --jfleg-test ../eval.csv \
        --kaggle "../../FP/Grammar Correction.csv" --out reports/correction_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gec_metrics import GecScorer
from speech_grammar.config import Device
from speech_grammar.inference.devices import resolve_device, resolve_t5_compute_type
from speech_grammar.services.corrector import CorrectionConfig, GrammarCorrector
from train_t5 import build_splits, key

MARGINS = [-math.inf, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
GRAMMAR_ONLY = frozenset({"PUNCT"})


class CachedBackend:
    """Memoizes ``generate`` and ``score`` of a backend (inputs are deterministic)."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._outputs: dict[tuple[str, int], str] = {}
        self._scores: dict[tuple[str, str], float] = {}

    @property
    def name(self) -> str:
        """Name of the wrapped backend."""
        return str(self._backend.name)

    def count_tokens(self, text: str) -> int:
        """Delegates to the wrapped backend."""
        return int(self._backend.count_tokens(text))

    def generate(self, sources: Sequence[str], *, num_beams: int, max_new_tokens: int) -> list[str]:
        """Generate only the sources not seen before."""
        todo = [s for s in dict.fromkeys(sources) if (s, num_beams) not in self._outputs]
        if todo:
            outputs = self._backend.generate(
                todo, num_beams=num_beams, max_new_tokens=max_new_tokens
            )
            self._outputs.update({(s, num_beams): o for s, o in zip(todo, outputs, strict=True)})
        return [self._outputs[(s, num_beams)] for s in sources]

    def score(self, sources: Sequence[str], targets: Sequence[str]) -> list[float]:
        """Score only the (source, target) pairs not seen before."""
        pairs = list(zip(sources, targets, strict=True))
        todo = [p for p in dict.fromkeys(pairs) if p not in self._scores]
        if todo:
            scores = self._backend.score([s for s, _ in todo], [t for _, t in todo])
            self._scores.update(dict(zip(todo, scores, strict=True)))
        return [self._scores[p] for p in pairs]


def validation_items(args: argparse.Namespace) -> tuple[list[str], list[list[str]], list[str]]:
    """(learner sentences, their references, correct sentences) of the validation split."""
    splits = build_splits(args)
    if args.split.is_file():  # the rebuilt split must be the one the model was trained with
        saved = json.loads(args.split.read_text(encoding="utf-8"))
        rebuilt = splits["manifest"]
        if (saved["val_sources"], saved["test_sources"]) != (
            rebuilt["val_sources"],
            rebuilt["test_sources"],
        ):
            raise SystemExit(f"rebuilt split differs from {args.split}: check the data/arguments")
        print(f"split matches {args.split}")
    refs: dict[str, list[str]] = defaultdict(list)
    for pair in splits["val"]:
        refs[pair.source].append(pair.target)
    train_sources = {key(p.source) for p in splits["train"]}
    correct = sorted(
        {t for targets in refs.values() for t in targets if key(t) not in train_sources}
    )
    return list(refs), list(refs.values()), correct


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/t5-gec-v2-ct2"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--jfleg-train", type=Path, required=True)
    parser.add_argument("--jfleg-test", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--extra", nargs="*", help="same --extra files used for training")
    # Must match the values used by train_t5.py (its defaults are repeated here).
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--jfleg-val-ratio", type=float, default=0.05)
    parser.add_argument("--identity-ratio", type=float, default=0.3)
    parser.add_argument("--split", type=Path, default=Path("models/t5-gec-v2/split.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/correction_calibration.json"))
    args = parser.parse_args()

    from speech_grammar.inference.t5_ct2 import Ct2Seq2SeqBackend

    device = resolve_device(Device(args.device))
    backend = CachedBackend(
        Ct2Seq2SeqBackend(
            args.model, device=device, compute_type=resolve_t5_compute_type("auto", device)
        )
    )
    learner, references, correct = validation_items(args)
    print(f"validation: {len(learner)} learner sentences, {len(correct)} correct sentences")
    scorer = GecScorer()
    sources = learner + correct
    all_refs = references + [[c] for c in correct]

    rows = []
    for punctuation_feedback in (True, False):
        for margin in MARGINS:
            corrector = GrammarCorrector(
                backend,
                config=CorrectionConfig(
                    max_chars=10_000,
                    min_correction_margin=margin,
                    punctuation_feedback=punctuation_feedback,
                ),
            )
            outputs = [corrector.correct(s) for s in sources]
            hyps = [o.corrected for o in outputs]
            grammar = scorer.errant(sources, hyps, all_refs, ignore=GRAMMAR_ONLY)
            row = {
                "punctuation_feedback": punctuation_feedback,
                "min_correction_margin": margin,
                **{f"{k}_grammar": v for k, v in grammar.items()},
                "correct_changed": round(
                    sum(o.changed for o in outputs[len(learner) :]) / len(correct), 4
                ),
                "learner_unchanged": round(
                    sum(not o.changed for o in outputs[: len(learner)]) / len(learner), 4
                ),
            }
            rows.append(row)
            print(json.dumps(row))

    best = max(
        (r for r in rows if not r["punctuation_feedback"]), key=lambda r: r["errant_f05_grammar"]
    )
    print("best (punctuation_feedback=False):", json.dumps(best))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": str(args.model),
                "backend": backend.name,
                "validation": {"learner": len(learner), "correct": len(correct)},
                "rows": rows,
                "best": best,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
