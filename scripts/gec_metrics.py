"""Standard GEC metrics: GLEU (the JFLEG metric) and ERRANT span-based F0.5.

Both follow the official scripts, so scores match them on the same tokens:

* GLEU: ``eval/gleu.py`` from github.com/keisks/jfleg (2016 version: 500
  iterations, one randomly chosen reference per sentence, ``seed = 101 * j``,
  mean of the corpus scores).
* ERRANT: ``errant_compare`` defaults (span-based correction, UNK edits
  ignored, per sentence the reference that maximises the running F0.5).

Texts are tokenised with spaCy (through ERRANT), not with JFLEG's original
tokeniser, so absolute GLEU can differ slightly from published numbers.
Requires the ``errant`` extra and ``en_core_web_sm``.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

EditKey = tuple[int, int, str]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) + 1 - n))


def gleu_stats(hyp: Sequence[str], src: Sequence[str], ref: Sequence[str]) -> list[int]:
    """``(len_h, len_r, num_1, den_1, ..., num_4, den_4)`` for one hypothesis/reference pair."""
    stats = [len(hyp), len(ref)]
    for n in range(1, 5):
        h, s, r = _ngrams(hyp, n), _ngrams(src, n), _ngrams(ref, n)
        source_only = Counter({k: v for k, v in s.items() if k not in r})
        stats.append(max(sum((h & r).values()) - sum((h & source_only).values()), 0))
        stats.append(max(len(hyp) + 1 - n, 0))
    return stats


def gleu_from_stats(stats: Sequence[int]) -> float:
    """Corpus GLEU from summed :func:`gleu_stats` (brevity penalty x geometric mean)."""
    if any(x == 0 for x in stats):
        return 0.0
    c, r = stats[0], stats[1]
    log_precision = sum(math.log(x / y) for x, y in zip(stats[2::2], stats[3::2], strict=True)) / 4
    return math.exp(min(0.0, 1 - r / c) + log_precision)


def corpus_gleu(
    sources: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[Sequence[str]]],
    iterations: int = 500,
) -> float:
    """Corpus GLEU of tokenised sentences, averaged over random reference draws."""
    num_refs = max(len(refs) for refs in references)
    per_ref = [
        [gleu_stats(h, s, ref) for ref in refs]
        for h, s, refs in zip(hypotheses, sources, references, strict=True)
    ]
    scores = []
    for j in range(iterations):
        rng = random.Random(j * 101)  # same draws as the official script's random.seed(j * 101)
        totals = [0] * 10
        for stats_by_ref in per_ref:
            stats = stats_by_ref[rng.randint(0, num_refs - 1) % len(stats_by_ref)]
            totals = [a + b for a, b in zip(totals, stats, strict=True)]
        scores.append(gleu_from_stats(totals))
    return sum(scores) / len(scores)


def f_score(tp: int, fp: int, fn: int, beta: float = 0.5) -> tuple[float, float, float]:
    """Precision, recall, F-beta exactly as ``errant_compare`` (rounded to 4 decimals)."""
    p = tp / (tp + fp) if fp else 1.0
    r = tp / (tp + fn) if fn else 1.0
    f = (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def compare_edits(
    hyp: dict[EditKey, list[str]], ref: dict[EditKey, list[str]]
) -> tuple[int, int, int]:
    """True positives, false positives and false negatives of one hypothesis/reference."""
    tp = sum(len(ref[k]) for k in hyp if k in ref)
    fp = sum(len(cats) for k, cats in hyp.items() if k not in ref)
    fn = sum(len(cats) for k, cats in ref.items() if k not in hyp)
    return tp, fp, fn


class GecScorer:
    """Tokenises with spaCy and scores GLEU and ERRANT F0.5 (parses are cached).

    Mirrors the usual official pipeline on pre-tokenised text: the spaCy tokens
    are joined with spaces and parsed like ``errant_parallel`` does (whitespace
    split), which reproduces ``errant_compare`` exactly. Parsing the raw text
    instead changes some POS tags and thus a few alignments (F0.5 +-0.003).
    """

    def __init__(self, annotator: Any = None) -> None:
        if annotator is None:
            import errant

            annotator = errant.load("en")
        self._annotator = annotator
        self._docs: dict[str, Any] = {}

    def tokens(self, text: str) -> list[str]:
        """Tokens of ``text`` from the spaCy tokenizer (no tagging or parsing)."""
        return [t.text for t in self._annotator.nlp.tokenizer(text) if not t.is_space]

    def doc(self, text: str) -> Any:
        """Parsed (POS, lemmas, dependencies) pre-tokenised ``text``, as ERRANT expects."""
        if text not in self._docs:
            self._docs[text] = self._annotator.parse(" ".join(self.tokens(text)))
        return self._docs[text]

    def edits(
        self, source: str, corrected: str, ignore: frozenset[str] = frozenset()
    ) -> dict[EditKey, list[str]]:
        """ERRANT edits keyed by ``(start, end, correction)``, mapped to their types.

        ``ignore`` drops categories such as ``{"PUNCT"}`` (the type without its
        ``M:``/``R:``/``U:`` operation prefix).
        """
        result: dict[EditKey, list[str]] = {}
        for e in self._annotator.annotate(self.doc(source), self.doc(corrected)):
            if e.type != "UNK" and e.type[2:] not in ignore:
                result.setdefault((e.o_start, e.o_end, e.c_str), []).append(e.type)
        return result

    def errant(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str]],
        beta: float = 0.5,
        ignore: frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        """Corpus ERRANT precision, recall and F-beta (best reference per sentence)."""
        total_tp = total_fp = total_fn = 0
        for src, hyp, refs in zip(sources, hypotheses, references, strict=True):
            hyp_edits = self.edits(src, hyp, ignore)
            best_tp, best_fp, best_fn, best_f = 0, 0, 0, -1.0
            for ref in refs:
                tp, fp, fn = compare_edits(hyp_edits, self.edits(src, ref, ignore))
                f = f_score(tp + total_tp, fp + total_fp, fn + total_fn, beta)[2]
                if (f, tp, -fp, -fn) > (best_f, best_tp, -best_fp, -best_fn):
                    best_tp, best_fp, best_fn, best_f = tp, fp, fn, f
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
        p, r, f = f_score(total_tp, total_fp, total_fn, beta)
        return {"errant_p": p, "errant_r": r, "errant_f05": f}

    def score(
        self,
        sources: Sequence[str],
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str]],
    ) -> dict[str, float]:
        """GLEU and ERRANT scores of detokenised hypotheses against references.

        ``errant_*_grammar`` ignores punctuation edits, matching the service's
        default of giving no punctuation feedback on speech transcripts.
        """
        gleu = corpus_gleu(
            [self.tokens(s) for s in sources],
            [self.tokens(h) for h in hypotheses],
            [[self.tokens(r) for r in refs] for refs in references],
        )
        grammar = self.errant(sources, hypotheses, references, ignore=frozenset({"PUNCT"}))
        return {
            "gleu": round(gleu, 4),
            **self.errant(sources, hypotheses, references),
            **{f"{name}_grammar": value for name, value in grammar.items()},
        }
