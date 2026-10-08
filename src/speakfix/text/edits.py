"""Word-level edit extraction and human-readable error descriptions.

Edits are what turns a corrected sentence into *feedback*: they tell the child
(or teacher) exactly which words changed. ``extract_edits`` works everywhere
(pure Python). When ERRANT is installed, :class:`ErrantClassifier` replaces the
alignment with linguistically-typed edits (``R:VERB:SVA`` ...).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from speakfix.domain.models import Edit, EditOp

_TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

_OPS: dict[str, str] = {"M": "Missing", "U": "Unnecessary", "R": "Wrong"}
_CATEGORIES: dict[str, str] = {
    "ADJ": "adjective",
    "ADJ:FORM": "adjective form (comparative/superlative)",
    "ADV": "adverb",
    "CONJ": "conjunction",
    "CONTR": "contraction",
    "DET": "article or determiner",
    "MORPH": "word form",
    "NOUN": "noun",
    "NOUN:INFL": "noun inflection",
    "NOUN:NUM": "singular/plural noun",
    "NOUN:POSS": "possessive noun",
    "ORTH": "capitalization or spacing",
    "OTHER": "word choice",
    "PART": "particle",
    "PREP": "preposition",
    "PRON": "pronoun",
    "PUNCT": "punctuation",
    "SPELL": "spelling",
    "UNK": "unknown",
    "VERB": "verb",
    "VERB:FORM": "verb form",
    "VERB:INFL": "verb inflection",
    "VERB:SVA": "subject-verb agreement",
    "VERB:TENSE": "verb tense",
    "WO": "word order",
}


def describe_error_type(code: str | None) -> str | None:
    """Turn an ERRANT code such as ``R:VERB:SVA`` into plain English.

    Args:
        code: ERRANT error code or ``None``.

    Returns:
        A short description (``"Wrong subject-verb agreement"``) or ``None``.
    """
    if not code:
        return None
    op, _, category = code.partition(":")
    if op not in _OPS:
        return code
    label = _CATEGORIES.get(category, category.lower() or "word")
    if category in {"WO", "ORTH", "SPELL", "VERB:SVA", "VERB:TENSE", "NOUN:NUM"}:
        return label[0].upper() + label[1:]
    return f"{_OPS[op]} {label}"


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def extract_edits(original: str, corrected: str) -> list[Edit]:
    """Align two texts word-by-word and list the differences.

    Args:
        original: Learner text (as transcribed or typed).
        corrected: Model output.

    Returns:
        Edits with character offsets into ``original``.
    """
    a = _tokens(original)
    b = _tokens(corrected)
    matcher = SequenceMatcher(a=[t[0] for t in a], b=[t[0] for t in b], autojunk=False)
    edits: list[Edit] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        op: EditOp = "replace" if tag == "replace" else ("delete" if tag == "delete" else "insert")
        if i1 < i2:
            start, end = a[i1][1], a[i2 - 1][2]
        else:
            start = end = a[i1][1] if i1 < len(a) else len(original)
        new_text = corrected[b[j1][1] : b[j2 - 1][2]] if j1 < j2 else ""
        edits.append(
            Edit(op=op, original=original[start:end], corrected=new_text, start=start, end=end)
        )
    return edits


class ErrantClassifier:
    """Error typing with ERRANT (optional dependency: ``pip install speakfix[errant]``)."""

    def __init__(self, language: str = "en") -> None:
        import errant  # local import keeps the dependency optional

        self._annotator: Any = errant.load(language)

    def classify(self, original: str, corrected: str, edits: Sequence[Edit]) -> list[Edit]:
        """Return ERRANT-aligned edits with error types (falls back to ``edits``)."""
        try:
            # tokenise=True: spaCy splits clitics/punctuation ("don't" -> do n't, "happy." ->
            # happy .); the default whitespace split mislabels such edits (R:OTHER, R:ADJ).
            orig_doc = self._annotator.parse(original, tokenise=True)
            cor_doc = self._annotator.parse(corrected, tokenise=True)
            typed: list[Edit] = []
            for e in self._annotator.annotate(orig_doc, cor_doc):
                if e.o_start < e.o_end:
                    start = orig_doc[e.o_start].idx
                    last = orig_doc[e.o_end - 1]
                    end = last.idx + len(last.text)
                else:
                    start = end = (
                        orig_doc[e.o_start].idx if e.o_start < len(orig_doc) else len(original)
                    )
                op: EditOp = {"M": "insert", "U": "delete"}.get(e.type[:1], "replace")  # type: ignore[assignment]
                typed.append(
                    Edit(
                        op=op,
                        original=original[start:end],
                        corrected=e.c_str,
                        start=start,
                        end=end,
                        error_type=e.type,
                    )
                )
        except Exception:
            return [replace(edit) for edit in edits]
        return typed
