"""GLEU / ERRANT scoring in scripts/gec_metrics.py.

Expected values were produced by the official tools on the same inputs:
keisks/jfleg ``eval/gleu.py`` and ``errant_parallel`` + ``errant_compare``.
"""

import sys
from importlib.util import find_spec

import pytest

from tests.conftest import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import gec_metrics

SRC = ["He go to school every day .", "She have two cat .", "I am agree with you ."]
HYP = ["He goes to school every day .", "She has two cat .", "I agree with you ."]
REF0 = ["He goes to school every day .", "She has two cats .", "I agree with you ."]
REF1 = ["He went to school every day .", "She has two cats .", "I agree with you ."]


def toks(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines]


def test_corpus_gleu_matches_official_script() -> None:
    refs = [list(pair) for pair in zip(toks(REF0), toks(REF1), strict=True)]
    assert gec_metrics.corpus_gleu(toks(SRC), toks(HYP), refs) == pytest.approx(0.674438, abs=1e-6)
    # Copying the source scores 0: every source-only n-gram is penalised.
    assert gec_metrics.corpus_gleu(toks(SRC), toks(SRC), refs) == 0.0
    one_ref = [[r] for r in toks(REF1)]
    gleu = gec_metrics.corpus_gleu(toks(SRC), toks(HYP), one_ref)
    assert gleu == pytest.approx(0.598539, abs=1e-6)


def test_f_score_follows_errant_compare_conventions() -> None:
    assert gec_metrics.f_score(3, 0, 1) == (1.0, 0.75, 0.9375)
    assert gec_metrics.f_score(0, 0, 0) == (1.0, 1.0, 1.0)  # nothing to correct, nothing proposed
    assert gec_metrics.f_score(0, 2, 3) == (0.0, 0.0, 0.0)


class TableScorer(gec_metrics.GecScorer):
    """Edits come from a lookup table instead of ERRANT."""

    def __init__(self, table: dict[tuple[str, str], dict[gec_metrics.EditKey, list[str]]]) -> None:
        self._table = table

    def edits(self, source: str, corrected: str) -> dict[gec_metrics.EditKey, list[str]]:
        return self._table[(source, corrected)]


def test_errant_picks_the_reference_that_maximises_f05() -> None:
    a, b, c = (1, 2, "goes"), (1, 2, "went"), (5, 5, "every")
    scorer = TableScorer(
        {
            ("src", "hyp"): {a: ["R:VERB:SVA"]},
            ("src", "ref0"): {b: ["R:VERB:TENSE"]},  # tp 0, fp 1, fn 1
            ("src", "ref1"): {a: ["R:VERB:SVA"], c: ["M:ADV"]},  # tp 1, fp 0, fn 1
        }
    )
    scores = scorer.errant(["src"], ["hyp"], [["ref0", "ref1"]])
    assert scores == {"errant_p": 1.0, "errant_r": 0.5, "errant_f05": 0.8333}


@pytest.mark.slow
@pytest.mark.skipif(
    find_spec("errant") is None or find_spec("en_core_web_sm") is None,
    reason="errant / spaCy en_core_web_sm not installed",
)
def test_errant_matches_official_compare() -> None:
    def detok(lines: list[str]) -> list[str]:
        return [line.replace(" .", ".") for line in lines]

    refs = [list(pair) for pair in zip(detok(REF0), detok(REF1), strict=True)]
    scores = gec_metrics.GecScorer().errant(detok(SRC), detok(HYP), refs)
    assert scores == {"errant_p": 1.0, "errant_r": 0.75, "errant_f05": 0.9375}
