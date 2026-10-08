"""Text normalization, sentence splitting and model-format conversion.

Two empirical findings about the fine-tuned T5 model drive this module:

* It was fine-tuned on one sentence at a time, so long inputs are split into
  sentences and corrected in a batch (no more truncation at 20 tokens).
* It hallucinates continuations when the input has no terminal punctuation
  (``"I am student"`` -> ``"I am a student and I am a teacher."``), so a period
  is appended before inference when the sentence has none.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
    }
)
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_WHITESPACE = re.compile(r"\s+")
_TERMINAL = re.compile(r"[.!?][\"')\]]*$")
_ABBREVIATIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "a.m.",
    "p.m.",
    "U.S.",
    "U.K.",
)
_ABBR_PLACEHOLDER = "⁣"  # invisible separator, never present in normalized input
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"')\]])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    """Normalize unicode, quotes and whitespace without changing wording.

    Args:
        text: Raw user or transcript text.

    Returns:
        Cleaned single-line text (may be empty).
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _CONTROL.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping common abbreviations intact.

    Args:
        text: Normalized text.

    Returns:
        Non-empty sentences in order.
    """
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _ABBR_PLACEHOLDER))
    parts = _SENTENCE_BOUNDARY.split(protected)
    sentences = [p.replace(_ABBR_PLACEHOLDER, ".").strip() for p in parts]
    return [s for s in sentences if s]


def has_terminal_punctuation(sentence: str) -> bool:
    """Return True when the sentence ends with ``.``, ``!`` or ``?``."""
    return bool(_TERMINAL.search(sentence.strip()))


def ensure_terminal_punctuation(sentence: str) -> str:
    """Append a period when the sentence has no terminal punctuation."""
    sentence = sentence.strip()
    if not sentence or has_terminal_punctuation(sentence):
        return sentence
    return sentence.rstrip(",;:") + "."


def to_jfleg_style(sentence: str) -> str:
    """Tokenize like the JFLEG training data (``do n't``, ``word .``)."""
    s = sentence
    s = re.sub(r"(?i)\b(ca|wo|sha)n't\b", lambda m: m.group(1) + " n't", s)
    s = re.sub(r"(?i)(\w)n't\b", r"\1 n't", s)
    s = re.sub(r"(?i)(\w)'(s|re|ve|ll|d|m)\b", r"\1 '\2", s)
    s = re.sub(r"(?<!\d)([.,!?;:()\"])|([.,!?;:()\"])(?!\d)", lambda m: f" {m.group(0)} ", s)
    return _WHITESPACE.sub(" ", s).strip()


def detokenize(text: str) -> str:
    """Undo JFLEG/PTB-style spacing produced by the model (``test .`` -> ``test.``)."""
    s = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"(?i)\b(ca|wo|sha) n't\b", lambda m: m.group(1) + "n't", s)
    s = re.sub(r"(?i)(\w) n't\b", r"\1n't", s)
    s = re.sub(r"(?i)(\w) '(s|re|ve|ll|d|m)\b", r"\1'\2", s)
    s = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', s)
    return _WHITESPACE.sub(" ", s).strip()


def match_leading_case(source: str, output: str) -> str:
    """Capitalize ``output`` when ``source`` starts with a capital letter."""
    src_first = next((c for c in source if c.isalpha()), "")
    for i, char in enumerate(output):
        if char.isalpha():
            if src_first.isupper() and char.islower():
                return output[:i] + char.upper() + output[i + 1 :]
            return output
    return output


def faithfulness(source: str, output: str) -> float:
    """Word-overlap similarity in [0, 1] between a sentence and its correction.

    A grammar correction keeps most words; a very low score signals a
    hallucination (the model rewrote or invented content).
    """
    a = _WORD.findall(source.lower())
    b = _WORD.findall(output.lower())
    if not a and not b:
        return 1.0
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def sentence_count(text: str) -> int:
    """Number of sentences detected in ``text``."""
    return len(split_sentences(text))
