"""Fine-tune the T5 grammar corrector with leak-free, reproducible data splits.

Fixes the methodological problems of the original experiments:

* duplicates are removed *before* splitting, and splits are made by unique
  source sentence, so no validation/test sentence is ever seen in training
  (the original scenario CSVs leaked 57-65% of validation rows);
* the JFLEG test set and the Kaggle test split are never used for training
  or model selection;
* fixed seeds, saved split manifest -> the run is reproducible;
* identity pairs (correct -> correct) teach the model to leave correct
  sentences alone (less over-correction);
* JFLEG's tokenized style (``do n't`` / ``word .``) is detokenized so training
  text looks like what the service receives (STT output / typed text).

Example (GPU):
    python scripts/train_t5.py --base models/t5-gec --out models/t5-gec-v2 \
        --jfleg-train ../train.csv --jfleg-test ../eval.csv \
        --kaggle "../../FP/Grammar Correction.csv" --epochs 4

Add more data with ``--extra file.csv`` (columns ``input``/``target``), e.g.
the scenario CSVs. Use ``--dry-run`` for a quick smoke test on CPU.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from speech_grammar.text.normalize import detokenize, ensure_terminal_punctuation, normalize_text

PREFIX = "grammar: "


@dataclass(frozen=True)
class Pair:
    source: str
    target: str
    origin: str


def key(text: str) -> str:
    """Comparison key for leakage checks (case/space-insensitive)."""
    return " ".join(text.lower().split())


def clean(text: str) -> str:
    return ensure_terminal_punctuation(detokenize(normalize_text(str(text).removeprefix(PREFIX))))


def load_csv_pairs(path: Path, origin: str) -> list[Pair]:
    df = pd.read_csv(path)
    if "Ungrammatical Statement" in df.columns:
        df = df.rename(columns={"Ungrammatical Statement": "input", "Standard English": "target"})
    df = df.dropna(subset=["input", "target"])
    return [
        Pair(clean(s), clean(t), origin) for s, t in zip(df["input"], df["target"], strict=True)
    ]


def split_by_source(
    pairs: list[Pair], val_ratio: float, test_ratio: float, rng: random.Random
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    sources = sorted({key(p.source) for p in pairs})
    rng.shuffle(sources)
    n_test = int(len(sources) * test_ratio)
    n_val = int(len(sources) * val_ratio)
    test_keys, val_keys = set(sources[:n_test]), set(sources[n_test : n_test + n_val])
    train = [p for p in pairs if key(p.source) not in test_keys | val_keys]
    val = [p for p in pairs if key(p.source) in val_keys]
    test = [p for p in pairs if key(p.source) in test_keys]
    return train, val, test


def dedupe(pairs: list[Pair]) -> list[Pair]:
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        k = (key(p.source), key(p.target))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def build_splits(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    jfleg_test_keys: set[str] = set()
    if args.jfleg_test:
        jfleg_test_keys = {key(p.source) for p in load_csv_pairs(args.jfleg_test, "jfleg_test")}

    jfleg = dedupe(load_csv_pairs(args.jfleg_train, "jfleg")) if args.jfleg_train else []
    jfleg = [p for p in jfleg if key(p.source) not in jfleg_test_keys]
    j_train, j_val, _ = split_by_source(jfleg, args.jfleg_val_ratio, 0.0, rng)

    kaggle = dedupe(load_csv_pairs(args.kaggle, "kaggle")) if args.kaggle else []
    k_train, k_val, k_test = split_by_source(kaggle, args.val_ratio, args.test_ratio, rng)

    protected = jfleg_test_keys | {key(p.source) for p in k_val + k_test + j_val}
    extra: list[Pair] = []
    for path in args.extra or []:
        rows = dedupe(load_csv_pairs(Path(path), f"extra:{Path(path).name}"))
        kept = [p for p in rows if key(p.source) not in protected]
        dropped = len(rows) - len(kept)
        print(
            f"[data] extra {path}: {len(rows)} unique pairs, {dropped} dropped (overlap with eval)"
        )
        extra += kept

    identity: list[Pair] = []
    if args.identity_ratio > 0 and k_train:
        correct = sorted({p.target for p in k_train if key(p.target) not in protected})
        rng.shuffle(correct)
        identity = [
            Pair(t, t, "identity") for t in correct[: int(len(k_train) * args.identity_ratio)]
        ]

    train = dedupe(j_train + k_train + extra + identity)
    val = dedupe(j_val + k_val)
    train_keys = {key(p.source) for p in train}
    leaks = train_keys & (protected | {key(p.source) for p in val})
    assert not leaks, f"{len(leaks)} evaluation sentences leaked into training"
    rng.shuffle(train)
    return {
        "train": train,
        "val": val,
        "kaggle_test": k_test,
        "manifest": {
            "seed": args.seed,
            "counts": {
                "train": len(train),
                "val": len(val),
                "kaggle_test": len(k_test),
                "jfleg_test_sentences": len(jfleg_test_keys),
                "by_origin": pd.Series([p.origin.split(":")[0] for p in train])
                .value_counts()
                .to_dict(),
            },
            "test_sources": [p.source for p in k_test],
            "val_sources": sorted({p.source for p in val}),
        },
    }


def train(args: argparse.Namespace, splits: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.base))

    class PairDataset(Dataset):  # type: ignore[type-arg]
        def __init__(self, pairs: list[Pair]) -> None:
            self.items = [
                {
                    **tokenizer(PREFIX + p.source, max_length=args.max_len, truncation=True),
                    "labels": tokenizer(
                        text_target=p.target, max_length=args.max_len, truncation=True
                    )["input_ids"],
                }
                for p in pairs
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.items[i]

    train_pairs, val_pairs = splits["train"], splits["val"]
    if args.dry_run:
        train_pairs, val_pairs = train_pairs[:64], val_pairs[:32]
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.no_bf16
    print(
        f"[train] device={'cuda' if torch.cuda.is_available() else 'cpu'} bf16={use_bf16} "
        f"train={len(train_pairs)} val={len(val_pairs)}"
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=1 if args.dry_run else args.epochs,
        max_steps=4 if args.dry_run else -1,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        bf16=use_bf16,
        logging_steps=25,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=PairDataset(train_pairs),
        eval_dataset=PairDataset(val_pairs),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    metrics = trainer.evaluate()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    return {
        "train_seconds": round(elapsed, 1),
        "train_loss": round(result.training_loss, 4),
        "best_eval_loss": round(metrics["eval_loss"], 4),
        "epochs_run": trainer.state.epoch,
        "bf16": use_bf16,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", type=Path, default=Path("models/t5-gec"))
    parser.add_argument("--out", type=Path, default=Path("models/t5-gec-v2"))
    parser.add_argument("--jfleg-train", type=Path)
    parser.add_argument(
        "--jfleg-test", type=Path, help="Only used to exclude its sentences from training"
    )
    parser.add_argument("--kaggle", type=Path)
    parser.add_argument("--extra", nargs="*", help="Additional input/target CSV files")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--jfleg-val-ratio", type=float, default=0.05)
    parser.add_argument("--identity-ratio", type=float, default=0.3)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = build_splits(args)
    manifest = splits["manifest"]
    print("[data]", json.dumps(manifest["counts"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = train(args, splits)
    report["data"] = manifest["counts"]
    report["args"] = {k: str(v) for k, v in vars(args).items()}
    (args.out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    seconds, loss = report["train_seconds"], report["best_eval_loss"]
    print(f"[train] done in {seconds}s, best eval_loss={loss} -> {args.out}")


if __name__ == "__main__":
    main()
