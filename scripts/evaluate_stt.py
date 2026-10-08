"""Evaluate speech-to-text configurations on labelled recordings.

The key question for a *grammar-learning* product is not only "is the
transcript accurate?" (WER) but "does the transcript keep the learner's
mistakes?" (verbatim rate). An STT that silently fixes "didn't passed" into
"didn't pass" hides the error from the grammar checker and from the teacher.

References come from the file name (``She go to school..mp3`` -> ``She go to
school.``) or from a CSV manifest with ``path,reference`` columns.

Example:
    python scripts/evaluate_stt.py --audio-dir ../../FP --models base.en small.en \
        --extra "../recorded.wav=She can sings well." --out reports/stt_eval.md
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

from speakfix.audio.decode import TARGET_SAMPLE_RATE, decode_audio
from speakfix.audio.preprocess import preprocess
from speakfix.config import Device
from speakfix.inference.devices import resolve_device
from speakfix.inference.whisper_stt import WhisperSpeechToText

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"}


def normalize(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def wer(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """Return (edit distance, reference length) at word level."""
    prev = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        cur = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref_word != hyp_word))
        prev = cur
    return prev[-1], len(reference)


@dataclass
class Sample:
    path: Path
    reference: str


def load_samples(audio_dir: Path | None, manifest: Path | None, extra: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    if manifest:
        with manifest.open(encoding="utf-8") as fh:
            samples += [Sample(Path(r["path"]), r["reference"]) for r in csv.DictReader(fh)]
    if audio_dir:
        for path in sorted(audio_dir.iterdir()):
            if path.suffix.lower() in AUDIO_EXT and path.stem.lower() != "recorded":
                samples.append(Sample(path, path.stem))
    for item in extra:
        path, _, ref = item.partition("=")
        samples.append(Sample(Path(path), ref))
    return samples


def run(samples: list[Sample], model: str, prompt: bool, device: str, models_dir: Path) -> dict:
    device = resolve_device(Device(device))  # also exposes pip-installed CUDA libraries
    stt = WhisperSpeechToText(
        model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        download_root=models_dir / "whisper",
        verbatim_prompt=prompt,
    )
    errors = words = exact = 0
    elapsed = audio_seconds = 0.0
    rows = []
    for sample in samples:
        audio = decode_audio(sample.path.read_bytes())
        processed, _ = preprocess(audio, TARGET_SAMPLE_RATE)
        start = time.perf_counter()
        transcript = stt.transcribe(processed, TARGET_SAMPLE_RATE)
        elapsed += time.perf_counter() - start
        audio_seconds += audio.size / TARGET_SAMPLE_RATE
        ref, hyp = normalize(sample.reference), normalize(transcript.text)
        dist, n = wer(ref, hyp)
        errors, words, exact = errors + dist, words + n, exact + (ref == hyp)
        rows.append((sample.reference, transcript.text, dist))
    return {
        "model": model,
        "prompt": prompt,
        "wer": errors / max(words, 1),
        "verbatim": exact / max(len(samples), 1),
        "rtf": elapsed / max(audio_seconds, 1e-9),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extra", nargs="*", default=[], help="path=reference pairs")
    parser.add_argument("--models", nargs="+", default=["base.en", "small.en"])
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    samples = load_samples(args.audio_dir, args.manifest, args.extra)
    if not samples:
        raise SystemExit("No audio samples found.")
    results = [
        run(samples, model, prompt, args.device, args.models_dir)
        for model in args.models
        for prompt in (False, True)
    ]

    lines = [
        f"# STT evaluation ({len(samples)} recordings, device={args.device})",
        "",
        "| Model | Verbatim prompt | WER (lower=better) | Verbatim rate (higher=better) "
        "| Real-time factor |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {'yes' if r['prompt'] else 'no'} | {r['wer']:.3f} | "
            f"{r['verbatim']:.0%} | {r['rtf']:.3f} |"
        )
    for r in results:
        lines += [
            "",
            f"## {r['model']} / prompt={'yes' if r['prompt'] else 'no'}",
            "",
            "| Reference (spoken) | Transcript | Word errors |",
            "|---|---|---|",
        ]
        lines += [f"| {ref} | {hyp} | {d} |" for ref, hyp, d in r["rows"]]
    report = "\n".join(lines) + "\n"
    print("\n".join(lines[: 4 + len(results)]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
