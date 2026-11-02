"""Latency benchmark of the T5 runtimes and Whisper.

Examples:
    python scripts/benchmark.py --backends ct2-cpu ct2-cuda torch-cpu torch-cuda --out b.md
    python scripts/benchmark.py --backends --whisper base.en small.en
        --whisper-devices cpu cuda --audio ../recorded.wav    (one line)
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import psutil

from speakfix.config import Device, Settings
from speakfix.domain.ports import Seq2SeqBackend
from speakfix.inference.devices import (
    resolve_device,
    resolve_t5_compute_type,
    resolve_whisper_compute_type,
)
from speakfix.inference.factory import build_edit_classifier
from speakfix.services.corrector import CorrectionConfig, GrammarCorrector
from speakfix.text.normalize import detokenize


def load_backend(spec: str, models_dir: Path, t5_name: str) -> Seq2SeqBackend:
    runtime, _, device = spec.partition("-")
    device = resolve_device(Device(device))  # also exposes pip-installed CUDA libraries
    if runtime == "ct2":
        from speakfix.inference.t5_ct2 import Ct2Seq2SeqBackend

        compute = resolve_t5_compute_type("auto", device)
        return Ct2Seq2SeqBackend(models_dir / f"{t5_name}-ct2", device=device, compute_type=compute)
    from speakfix.inference.t5_torch import TorchSeq2SeqBackend

    return TorchSeq2SeqBackend(models_dir / t5_name, device=device)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def bench_corrector(
    spec: str, sentences: list[str], models_dir: Path, t5_name: str, service: bool
) -> dict[str, str]:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    backend = load_backend(spec, models_dir, t5_name)
    config = CorrectionConfig(max_chars=10_000)
    classifier = None
    if service:  # what the service runs by default: confidence filter + ERRANT error types
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        config = replace(config, min_correction_margin=settings.min_correction_margin)
        classifier = build_edit_classifier(settings)
    load_s = time.perf_counter() - start
    corrector = GrammarCorrector(backend, config=config, classifier=classifier)
    corrector.warmup()
    corrector.correct("Warm up the error typing too.")
    latencies = []
    for sentence in sentences:
        t = time.perf_counter()
        corrector.correct(sentence)
        latencies.append((time.perf_counter() - t) * 1000)
    paragraph = " ".join(sentences[:8])
    t = time.perf_counter()
    corrector.correct(paragraph)
    paragraph_ms = (time.perf_counter() - t) * 1000
    rss_mb = (process.memory_info().rss - rss_before) / 1e6
    return {
        "runtime": spec,
        "backend": backend.name,
        "load_s": f"{load_s:.1f}",
        "p50_ms": f"{statistics.median(latencies):.0f}",
        "p95_ms": f"{percentile(latencies, 0.95):.0f}",
        "paragraph_8_sent_ms": f"{paragraph_ms:.0f}",
        "rss_delta_mb": f"{rss_mb:.0f}",
    }


def bench_whisper(name: str, audio_path: Path, models_dir: Path, device: str) -> dict[str, str]:
    from speakfix.audio.decode import TARGET_SAMPLE_RATE, decode_audio
    from speakfix.audio.preprocess import preprocess
    from speakfix.inference.factory import resolve_whisper_model
    from speakfix.inference.whisper_stt import WhisperSpeechToText

    device = resolve_device(Device(device))
    compute_type = resolve_whisper_compute_type("auto", device)
    model = resolve_whisper_model(name, models_dir / "whisper")
    stt = WhisperSpeechToText(model, device=device, compute_type=compute_type)
    audio, _ = preprocess(decode_audio(audio_path.read_bytes()), TARGET_SAMPLE_RATE)
    stt.transcribe(audio, TARGET_SAMPLE_RATE)  # warm-up
    runs = []
    for _ in range(3):
        t = time.perf_counter()
        stt.transcribe(audio, TARGET_SAMPLE_RATE)
        runs.append(time.perf_counter() - t)
    seconds = audio.size / TARGET_SAMPLE_RATE
    return {
        "model": name,
        "runtime": f"{compute_type}@{device}",
        "audio_s": f"{seconds:.1f}",
        "median_s": f"{statistics.median(runs):.2f}",
        "real_time_factor": f"{statistics.median(runs) / seconds:.3f}",
    }


def to_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r[h] for h in head) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backends", nargs="*", default=["ct2-cpu", "torch-cpu"])
    parser.add_argument("--t5-name", default="t5-gec-v2", help="model folder under --models-dir")
    parser.add_argument(
        "--service-config",
        action="store_true",
        help="measure the service defaults: confidence filter + ERRANT error types",
    )
    parser.add_argument("--whisper", nargs="*", default=[])
    parser.add_argument("--whisper-devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--jfleg", type=Path, default=Path("../eval.csv"))
    parser.add_argument("--sentences", type=int, default=50)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.jfleg)
    sources = df["input"].str.replace(r"^grammar: ", "", regex=True).drop_duplicates().tolist()
    sentences = [detokenize(s) for s in sources[: args.sentences]]

    sections = []
    rows = [
        bench_corrector(spec, sentences, args.models_dir, args.t5_name, args.service_config)
        for spec in args.backends
    ]
    if rows:
        mode = "service defaults" if args.service_config else "generation only"
        title = (
            f"## Grammar correction ({args.t5_name}, {mode}, "
            f"{len(sentences)} JFLEG sentences, beam 5)"
        )
        sections.append(f"{title}\n\n{to_markdown(rows)}")
    if args.whisper and args.audio:
        wrows = [
            bench_whisper(name, args.audio, args.models_dir, device)
            for device in args.whisper_devices
            for name in args.whisper
        ]
        sections.append(f"## Speech-to-text ({args.audio.name})\n\n{to_markdown(wrows)}")
    report = "# Benchmark\n\n" + "\n\n".join(sections) + "\n"
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
