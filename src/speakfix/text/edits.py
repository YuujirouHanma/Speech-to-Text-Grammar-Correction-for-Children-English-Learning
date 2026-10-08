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
from typing import Any, Literal

from speakfix.domain.models import Edit, EditOp

_TOKEN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

FeedbackLanguage = Literal["en", "id"]

# How each ERRANT operation is phrased: M = missing, U = unnecessary, R = replace.
_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {"M": "Missing {label}", "U": "Unnecessary {label}", "R": "Wrong {label}"},
    "id": {"M": "Kurang {label}", "U": "{Label} tidak perlu", "R": "{Label} kurang tepat"},
}
_CATEGORIES: dict[str, dict[str, str]] = {
    "en": {
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
    },
    "id": {
        "ADJ": "kata sifat",
        "ADJ:FORM": "bentuk kata sifat (perbandingan)",
        "ADV": "kata keterangan",
        "CONJ": "kata hubung",
        "CONTR": "bentuk singkat (contraction)",
        "DET": "kata sandang (a/an/the)",
        "MORPH": "bentuk kata",
        "NOUN": "kata benda",
        "NOUN:INFL": "bentuk kata benda",
        "NOUN:NUM": "bentuk tunggal/jamak kata benda",
        "NOUN:POSS": "bentuk kepemilikan ('s)",
        "ORTH": "huruf kapital atau spasi",
        "OTHER": "pilihan kata",
        "PART": "partikel",
        "PREP": "kata depan (in/on/at ...)",
        "PRON": "kata ganti",
        "PUNCT": "tanda baca",
        "SPELL": "ejaan",
        "UNK": "tidak diketahui",
        "VERB": "kata kerja",
        "VERB:FORM": "bentuk kata kerja",
        "VERB:INFL": "bentuk kata kerja",
        "VERB:SVA": "kata kerja harus sesuai dengan subjek",
        "VERB:TENSE": "bentuk waktu (tense) kata kerja",
        "WO": "urutan kata",
    },
}
# Categories whose label already says what is wrong (no "missing/wrong" prefix).
_STANDALONE = {"WO", "ORTH", "SPELL", "VERB:SVA", "VERB:TENSE", "NOUN:NUM"}


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def describe_error_type(code: str | None, language: FeedbackLanguage = "en") -> str | None:
    """Turn an ERRANT code such as ``R:VERB:SVA`` into a short learner-friendly phrase.

    Args:
        code: ERRANT error code or ``None``.
        language: ``en`` (English) or ``id`` (Bahasa Indonesia).

    Returns:
        A short description (``"Subject-verb agreement"``, ``"Kurang kata sandang
        (a/an/the)"``) or ``None``.
    """
    if not code:
        return None
    op, _, category = code.partition(":")
    templates = _TEMPLATES[language]
    if op not in templates:
        return code
    label = _CATEGORIES[language].get(category, category.lower() or "word")
    if category in _STANDALONE:
        return _capitalize(label)
    return templates[op].format(label=label, Label=_capitalize(label))


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
