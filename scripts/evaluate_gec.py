"""Evaluate grammar-correction systems on JFLEG and the Kaggle Grammar Correction set.

Systems compared:
* ``legacy``   - the original notebook decoding (``max_length=20``, one prompt,
                 no sentence handling): reproduces the report's setup.
* ``speakfix`` - the production pipeline (:class:`GrammarCorrector`).

Metrics (all on detokenized text, so JFLEG's ``word .`` style is not penalised):
* ``bleu_1ref``  corpus BLEU against the first reference (what the report used),
* ``bleu_4ref``  corpus BLEU against all references (standard for JFLEG),
* ``exact_any``  share of outputs identical to *any* reference,
* ``truncated``  outputs cut by the length limit,
* ``unchanged``  outputs identical to the input (under-correction indicator).

With ``--gec-metrics`` (needs the ``errant`` extra and ``en_core_web_sm``), also the
standard GEC metrics on spaCy-tokenised text, matching the official scripts
(see ``gec_metrics.py``):
* ``gleu``                      JFLEG's GLEU (keisks/jfleg ``gleu.py``),
* ``errant_p/_r/_f05``          ERRANT span-based correction (``errant_compare``).

Example:
    python scripts/evaluate_gec.py --jfleg ../eval.csv --kaggle "../../FP/Grammar Correction.csv"
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import sacrebleu

from speakfix.config import InputStyle
from speakfix.domain.ports import Seq2SeqBackend
from speakfix.services.corrector import CorrectionConfig, GrammarCorrector
from speakfix.text.normalize import detokenize, normalize_text


def load_jfleg(path: Path, limit: int | None) -> tuple[list[str], list[list[str]]]:
    df = pd.read_csv(path)
    df["input"] = df["input"].str.replace(r"^grammar: ", "", regex=True)
    grouped = df.groupby("input", sort=False)["target"].apply(list)
    sources = [detokenize(s) for s in grouped.index]
    refs = [[detokenize(r) for r in rs] for rs in grouped.tolist()]
    return (sources[:limit], refs[:limit]) if limit else (sources, refs)


def load_kaggle(
    path: Path, limit: int | None, split_file: Path | None
) -> tuple[list[str], list[list[str]]]:
    df = pd.read_csv(path)
    if "Ungrammatical Statement" in df.columns:
        df = df.rename(columns={"Ungrammatical Statement": "input", "Standard English": "target"})
    if split_file and split_file.exists():
        keep = set(json.loads(split_file.read_text())["test_sources"])
        df = df[df["input"].isin(keep)]
    df = df.dropna(subset=["input", "target"]).drop_duplicates(subset=["input"])
    sources = [normalize_text(s) for s in df["input"]]
    refs = [[normalize_text(t)] for t in df["target"]]
    return (sources[:limit], refs[:limit]) if limit else (sources, refs)


def legacy_system(backend: Seq2SeqBackend) -> Callable[[Sequence[str]], list[str]]:
    def run(sources: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        for i in range(0, len(sources), 16):
            batch = ["grammar: " + s for s in sources[i : i + 16]]
            outputs += backend.generate(batch, num_beams=5, max_new_tokens=19)
        return [detokenize(o) for o in outputs]

    return run


def speakfix_system(
    backend: Seq2SeqBackend, style: InputStyle, beams: int = 5
) -> Callable[[Sequence[str]], list[str]]:
    corrector = GrammarCorrector(
        backend, config=CorrectionConfig(input_style=style, max_chars=10_000, num_beams=beams)
    )

    def run(sources: Sequence[str]) -> list[str]:
        return [corrector.correct(s).corrected for s in sources]

    return run


def score(
    sources: Sequence[str],
    outputs: Sequence[str],
    refs: Sequence[Sequence[str]],
    max_new: int | None,
) -> dict[str, float]:
    n_refs = max(len(r) for r in refs)
    ref_streams = [[r[i] if i < len(r) else r[0] for r in refs] for i in range(n_refs)]
    norm = lambda s: " ".join(s.split())  # noqa: E731
    return {
        "bleu_1ref": round(sacrebleu.corpus_bleu(list(outputs), [ref_streams[0]]).score / 100, 4),
        "bleu_4ref": round(sacrebleu.corpus_bleu(list(outputs), ref_streams).score / 100, 4),
        "exact_any": round(
            sum(norm(o) in {norm(x) for x in r} for o, r in zip(outputs, refs, strict=True))
            / len(outputs),
            4,
        ),
        "unchanged": round(
            sum(norm(o) == norm(s) for o, s in zip(outputs, sources, strict=True)) / len(outputs), 4
        ),
        "truncated": round(sum(len(o.split()) >= 0.9 * max_new for o in outputs) / len(outputs), 4)
        if max_new
        else 0.0,
    }


def build_backend(
    kind: str, model_dir: Path, device: str, compute_type: str = "auto"
) -> Seq2SeqBackend:
    from speakfix.config import Device
    from speakfix.inference.devices import resolve_device, resolve_t5_compute_type

    device = resolve_device(Device(device))  # also exposes pip-installed CUDA libraries
    if kind == "ctranslate2":
        from speakfix.inference.t5_ct2 import Ct2Seq2SeqBackend

        compute_type = resolve_t5_compute_type(compute_type, device)
        return Ct2Seq2SeqBackend(model_dir, device=device, compute_type=compute_type)
    from speakfix.inference.t5_torch import TorchSeq2SeqBackend

    return TorchSeq2SeqBackend(model_dir, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/t5-gec-ct2"))
    parser.add_argument("--backend", choices=["ctranslate2", "torch"], default="ctranslate2")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--compute-type", default="auto", help="CTranslate2 compute type")
    parser.add_argument(
        "--jfleg", type=Path, help="eval.csv (JFLEG test, input/target with 4 refs)"
    )
    parser.add_argument("--kaggle", type=Path, help="Grammar Correction.csv")
    parser.add_argument(
        "--kaggle-split", type=Path, help="split.json from train_t5.py (use only its test part)"
    )
    parser.add_argument("--systems", nargs="+", default=["legacy", "speakfix"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-beams", type=int, default=5, help="Beam width (1 = greedy)")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--gec-metrics", action="store_true", help="add GLEU and ERRANT F0.5")
    parser.add_argument("--out", type=Path, default=Path("reports/gec_eval.json"))
    args = parser.parse_args()

    backend = build_backend(args.backend, args.model, args.device, args.compute_type)
    scorer = None
    if args.gec_metrics:
        from gec_metrics import GecScorer  # scripts/ is on sys.path when run as a script

        scorer = GecScorer()
    datasets: dict[str, tuple[list[str], list[list[str]]]] = {}
    if args.jfleg:
        datasets["jfleg_test"] = load_jfleg(args.jfleg, args.limit)
    if args.kaggle:
        datasets["kaggle_gc"] = load_kaggle(args.kaggle, args.limit, args.kaggle_split)

    systems: dict[str, Callable[[Sequence[str]], list[str]]] = {}
    for name in args.systems:
        if name == "legacy":
            systems[name] = legacy_system(backend)
        elif name == "speakfix":
            systems[name] = speakfix_system(backend, InputStyle.RAW, args.num_beams)
        elif name == "speakfix-jfleg":
            systems[name] = speakfix_system(backend, InputStyle.JFLEG)

    results: dict[str, dict[str, dict[str, float]]] = {}
    examples: dict[str, list[dict[str, str]]] = {}
    for ds_name, (sources, refs) in datasets.items():
        for sys_name, system in systems.items():
            start = time.perf_counter()
            outputs = system(sources)
            elapsed = time.perf_counter() - start
            metrics = score(sources, outputs, refs, 19 if sys_name == "legacy" else None)
            if scorer is not None:
                metrics.update(scorer.score(sources, outputs, refs))
            metrics["seconds"] = round(elapsed, 1)
            results.setdefault(ds_name, {})[sys_name] = metrics
            examples[f"{ds_name}/{sys_name}"] = [
                {"source": s, "output": o, "reference": r[0]}
                for s, o, r in list(zip(sources, outputs, refs, strict=True))[:25]
            ]
            print(f"{args.label:10} {ds_name:11} {sys_name:15} {metrics}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    existing[args.label] = {
        "model": str(args.model),
        "backend": backend.name,
        "results": results,
        "examples": examples,
    }
    args.out.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
