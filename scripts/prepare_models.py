"""Prepare model artifacts under ``models/``.

Examples:
    python scripts/prepare_models.py t5 --source ../saved_model
    python scripts/prepare_models.py whisper --name small.en
    python scripts/prepare_models.py all --source ../saved_model --whisper small.en

Layout produced::

    models/
      t5-gec/          HuggingFace checkpoint (PyTorch backend / fallback / training)
      t5-gec-ct2/      CTranslate2 int8 model + tokenizer.json (production backend)
      whisper/         faster-whisper cache (HuggingFace hub layout)
      MANIFEST.json    checksums and provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link when possible (no extra disk space), else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _update_manifest(models_dir: Path, key: str, entry: dict[str, object]) -> None:
    manifest_path = models_dir / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest[key] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def prepare_t5(source: Path, models_dir: Path, quantization: str, name: str = "t5-gec") -> None:
    """Mirror the HF checkpoint and convert it to CTranslate2."""
    from ctranslate2.converters import TransformersConverter

    if not (source / "config.json").is_file():
        raise SystemExit(f"'{source}' is not a HuggingFace model directory (config.json missing).")
    hf_dir = models_dir / name
    ct2_dir = models_dir / f"{name}-ct2"
    if source.resolve() != hf_dir.resolve():
        for file in source.iterdir():
            if file.is_file():
                _link_or_copy(file, hf_dir / file.name)
    print(f"[t5] HF checkpoint ready in {hf_dir}")

    start = time.perf_counter()
    TransformersConverter(str(hf_dir)).convert(str(ct2_dir), quantization=quantization, force=True)
    for filename in TOKENIZER_FILES:
        if (hf_dir / filename).is_file():
            shutil.copy2(hf_dir / filename, ct2_dir / filename)
    elapsed = time.perf_counter() - start
    print(f"[t5] CTranslate2 ({quantization}) model written to {ct2_dir} in {elapsed:.1f}s")

    _update_manifest(
        models_dir,
        name,
        {
            "source": str(source.resolve()),
            "quantization": quantization,
            "hf_weights_sha256": _sha256(
                next(hf_dir.glob("*.safetensors"), hf_dir / "pytorch_model.bin")
            ),
            "ct2_model_sha256": _sha256(ct2_dir / "model.bin"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def prepare_whisper(name: str, models_dir: Path) -> None:
    """Download a faster-whisper model into the local cache."""
    from faster_whisper import download_model

    cache = models_dir / "whisper"
    cache.mkdir(parents=True, exist_ok=True)
    path = download_model(name, cache_dir=str(cache))
    print(f"[whisper] {name} available at {path}")
    _update_manifest(
        models_dir,
        f"whisper:{name}",
        {"path": str(path), "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", choices=["t5", "whisper", "all"])
    parser.add_argument(
        "--source", type=Path, help="HuggingFace T5 checkpoint (e.g. ../saved_model)"
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--quantization",
        default="int8",
        help="CTranslate2 quantization (int8, int8_float16, float16 ...)",
    )
    parser.add_argument(
        "--whisper", "--name", dest="whisper", default="base.en", help="Whisper model size"
    )
    parser.add_argument(
        "--t5-name", default="t5-gec", help="Output names: models/<name> and models/<name>-ct2"
    )
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.target in ("t5", "all"):
        if args.source is None:
            parser.error("--source is required for the T5 model")
        prepare_t5(args.source, args.models_dir, args.quantization, args.t5_name)
    if args.target in ("whisper", "all"):
        prepare_whisper(args.whisper, args.models_dir)


if __name__ == "__main__":
    main()
