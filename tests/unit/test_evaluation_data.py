"""scripts/evaluate_gec.py must evaluate every held-out sentence of the training split."""

import json
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_ROOT

pytest.importorskip("sacrebleu")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_gec import load_kaggle
from train_t5 import clean

ROWS = [
    # Non-ASCII: was lost when split.json was read with the Windows default encoding.
    ("Preheat the oven to 350°F then mix it.", "Preheat the oven to 350°F; then mix it."),
    # Cleaning changes the text: the split stores 'great!".' while the CSV has 'great!":'.
    ('She said"the party was great!":', 'She said, "The party was great!"'),
    ("He go home.", "He goes home."),
    ("This one is used for training.", "This one is used for training."),
]


def write_data(tmp_path: Path, test_rows: list[tuple[str, str]]) -> tuple[Path, Path]:
    csv = tmp_path / "data.csv"
    lines = ['"Ungrammatical Statement","Standard English"']
    for source, target in ROWS:
        lines.append('"{}","{}"'.format(source.replace('"', '""'), target.replace('"', '""')))
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    sources = [clean(source) for source, _ in test_rows]
    split.write_text(json.dumps({"test_sources": sources}, ensure_ascii=False), encoding="utf-8")
    return csv, split


def test_every_held_out_sentence_is_loaded(tmp_path: Path) -> None:
    csv, split = write_data(tmp_path, ROWS[:3])
    sources, refs = load_kaggle(csv, None, split)
    assert len(sources) == 3
    assert "Preheat the oven to 350°F then mix it." in sources
    assert 'She said"the party was great!":' in sources
    assert ["He goes home."] in refs
    assert "This one is used for training." not in sources


def test_a_lost_test_sentence_is_an_error_not_a_smaller_test_set(tmp_path: Path) -> None:
    csv, split = write_data(tmp_path, ROWS[:3])
    extra = json.loads(split.read_text(encoding="utf-8"))
    extra["test_sources"].append("A sentence that is not in the file.")
    split.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(SystemExit, match="found 3 of 4"):
        load_kaggle(csv, None, split)
