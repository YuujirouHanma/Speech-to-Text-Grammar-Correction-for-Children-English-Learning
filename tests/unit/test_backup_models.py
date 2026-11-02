"""scripts/backup_models.py: what gets packed and how the archive is verified."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import backup_models


def make_models(root: Path, name: str = "m") -> Path:
    models = root / "models"
    (models / name / "checkpoints" / "step-1").mkdir(parents=True)
    (models / f"{name}-ct2").mkdir()
    (models / name / "model.safetensors").write_bytes(b"hf-weights")
    (models / name / "checkpoints" / "step-1" / "model.safetensors").write_bytes(b"old")
    (models / f"{name}-ct2" / "model.bin").write_bytes(b"ct2-weights")
    manifest = {
        name: {
            "hf_weights_sha256": hashlib.sha256(b"hf-weights").hexdigest(),
            "ct2_model_sha256": hashlib.sha256(b"ct2-weights").hexdigest(),
        }
    }
    (models / "MANIFEST.json").write_text(json.dumps(manifest))
    return models


def test_training_checkpoints_are_not_packed(tmp_path: Path) -> None:
    models = make_models(tmp_path)
    packed = [p.relative_to(tmp_path).as_posix() for p in backup_models.files_to_pack(models, "m")]
    assert packed == [
        "models/MANIFEST.json",
        "models/m/model.safetensors",
        "models/m-ct2/model.bin",
    ]


def test_verify_detects_weights_that_do_not_match_the_manifest(tmp_path: Path) -> None:
    models = make_models(tmp_path)
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        for path in backup_models.files_to_pack(models, "m"):
            archive.write(path, path.relative_to(tmp_path).as_posix())
    backup_models.verify(good, "m")

    (models / "m-ct2" / "model.bin").write_bytes(b"tampered")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        for path in backup_models.files_to_pack(models, "m"):
            archive.write(path, path.relative_to(tmp_path).as_posix())
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        backup_models.verify(bad, "m")
