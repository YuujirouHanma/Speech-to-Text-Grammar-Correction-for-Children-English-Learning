"""Pack a trained T5 model into one verified zip that can be stored off the laptop.

The zip holds ``models/<name>/`` (HF checkpoint, without training checkpoints),
``models/<name>-ct2/`` (what the service runs) and ``MANIFEST.json``. After
writing, every member's CRC and the model SHA-256 values in the manifest are
checked against the archive.

Whisper is not included: it is re-downloaded by ``prepare_models.py whisper``.

Examples:
    python scripts/backup_models.py --name t5-gec-v2 --out-dir ../../speakfix-backups
    # restore: unzip into speakfix/ (it recreates models/...), then run the tests
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import zipfile
from pathlib import Path

SKIP_DIRS = {"checkpoints"}  # intermediate training states, several GB


def _sha256_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files_to_pack(models_dir: Path, name: str) -> list[Path]:
    """Model files to back up, relative to ``models_dir``'s parent."""
    files = [models_dir / "MANIFEST.json"]
    for folder in (models_dir / name, models_dir / f"{name}-ct2"):
        if not folder.is_dir():
            raise SystemExit(f"missing {folder}: run prepare_models.py / train_t5.py first")
        files += sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(folder).parts)
        )
    return files


def verify(zip_path: Path, name: str) -> None:
    """Check CRCs and that the archived weights match MANIFEST.json."""
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise SystemExit(f"CRC error in {bad}")
        manifest = json.loads(archive.read("models/MANIFEST.json"))
        entry = manifest.get(name)
        if entry is None:
            raise SystemExit(f"MANIFEST.json has no entry for {name!r}")
        checks = {
            f"models/{name}/model.safetensors": entry["hf_weights_sha256"],
            f"models/{name}-ct2/model.bin": entry["ct2_model_sha256"],
        }
        for member, expected in checks.items():
            if _sha256_member(archive, member) != expected:
                raise SystemExit(f"SHA-256 mismatch for {member}")
    print(f"verified: CRC ok, weights match MANIFEST.json ({len(checks)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="t5-gec-v2", help="models/<name> and models/<name>-ct2")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("backups"))
    args = parser.parse_args()

    files = files_to_pack(args.models_dir, args.name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / f"speakfix-{args.name}-{dt.date.today().isoformat()}.zip"
    root = args.models_dir.parent
    # Weights barely compress, so store them as-is (fast); small text files are deflated.
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in files:
            big = path.stat().st_size > 10_000_000
            archive.write(
                path,
                path.relative_to(root).as_posix(),
                compress_type=zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED,
            )
    size_mb = zip_path.stat().st_size / 1e6
    print(f"{len(files)} files -> {zip_path} ({size_mb:.0f} MB)")
    verify(zip_path, args.name)


if __name__ == "__main__":
    main()
