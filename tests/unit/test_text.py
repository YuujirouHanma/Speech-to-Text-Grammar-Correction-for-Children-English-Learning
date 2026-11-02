import pytest

from speech_grammar.text.edits import describe_error_type, extract_edits
from speech_grammar.text.normalize import (
    detokenize,
    ensure_terminal_punctuation,
    faithfulness,
    has_terminal_punctuation,
    match_leading_case,
    normalize_text,
    same_words,
    split_sentences,
    strip_terminal_punctuation,
    to_jfleg_style,
)


class TestNormalize:
    def test_strip_terminal_punctuation(self) -> None:
        assert strip_terminal_punctuation("They are happy.") == "They are happy"
        assert strip_terminal_punctuation("Really?! ") == "Really"
        assert strip_terminal_punctuation("No period") == "No period"

    def test_same_words_ignores_punctuation_and_case_only(self) -> None:
        assert same_words("if it rains we stay home", "If it rains, we stay home.")
        assert not same_words("She go to school.", "She goes to school.")
        assert not same_words("I can not swim.", "I cannot swim.")
        assert not same_words("I dont know.", "I don't know.")

    def test_quotes_whitespace_and_control_chars(self) -> None:
        assert normalize_text("  She don’t\tlike “apples”\x07  ") == 'She don\'t like "apples"'

    def test_empty(self) -> None:
        assert normalize_text(" \n\t ") == ""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Mr. Smith is here. He is happy!", ["Mr. Smith is here.", "He is happy!"]),
            ("I have 3.5 apples. Yes", ["I have 3.5 apples.", "Yes"]),
            ("first line\nsecond line", ["first line", "second line"]),
            ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
            ("e.g. this works. Fine?", ["e.g. this works.", "Fine?"]),
        ],
    )
    def test_split_sentences(self, text: str, expected: list[str]) -> None:
        assert split_sentences(text) == expected

    @pytest.mark.parametrize(
        ("sentence", "expected"),
        [("I am student", "I am student."), ("Really?", "Really?"), ("Hello,", "Hello."), ("", "")],
    )
    def test_terminal_punctuation(self, sentence: str, expected: str) -> None:
        assert ensure_terminal_punctuation(sentence) == expected

    def test_has_terminal_punctuation_with_quote(self) -> None:
        assert has_terminal_punctuation('He said "stop."')

    @pytest.mark.parametrize(
        "sentence",
        ["She don't like it, right?", "It's 3.5 km (roughly).", "They can't go; we won't."],
    )
    def test_jfleg_round_trip(self, sentence: str) -> None:
        assert detokenize(to_jfleg_style(sentence)) == sentence

    def test_jfleg_style_splits_contractions(self) -> None:
        assert to_jfleg_style("He don't know.") == "He do n't know ."

    def test_detokenize_model_output(self) -> None:
        out = "Many students find it difficult , because of motivations ."
        assert detokenize(out) == "Many students find it difficult, because of motivations."

    def test_match_leading_case(self) -> None:
        assert match_leading_case("She go home.", "she goes home.") == "She goes home."
        assert match_leading_case("she go home.", "She goes home.") == "She goes home."

    def test_faithfulness(self) -> None:
        assert faithfulness("She go home.", "She go home.") == 1.0
        assert faithfulness("She go home.", "She goes home.") > 0.6
        assert faithfulness("She go home.", "Completely unrelated invented text here.") < 0.3


class TestEdits:
    def test_replace(self) -> None:
        original = "She don't like apples."
        edits = extract_edits(original, "She doesn't like apples.")
        assert len(edits) == 1
        edit = edits[0]
        assert (edit.op, edit.original, edit.corrected) == ("replace", "don't", "doesn't")
        assert original[edit.start : edit.end] == "don't"

    def test_insert(self) -> None:
        original = "I am student."
        edits = extract_edits(original, "I am a student.")
        assert [(e.op, e.corrected) for e in edits] == [("insert", "a")]
        assert edits[0].start == original.index("student")

    def test_delete(self) -> None:
        edits = extract_edits("The the cat sleeps.", "The cat sleeps.")
        assert [(e.op, e.original) for e in edits] == [("delete", "the")]

    def test_no_change(self) -> None:
        assert extract_edits("All good.", "All good.") == []

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("R:VERB:SVA", "Subject-verb agreement"),
            ("M:DET", "Missing article or determiner"),
            ("U:PREP", "Unnecessary preposition"),
            ("R:SPELL", "Spelling"),
            (None, None),
            ("X:WEIRD", "X:WEIRD"),
        ],
    )
    def test_describe_error_type(self, code: str | None, expected: str | None) -> None:
        assert describe_error_type(code) == expected

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("R:VERB:SVA", "Kata kerja harus sesuai dengan subjek"),
            ("M:DET", "Kurang kata sandang (a/an/the)"),
            ("U:PREP", "Kata depan (in/on/at ...) tidak perlu"),
            ("R:VERB:FORM", "Bentuk kata kerja kurang tepat"),
            ("R:SPELL", "Ejaan"),
            (None, None),
        ],
    )
    def test_describe_error_type_in_indonesian(
        self, code: str | None, expected: str | None
    ) -> None:
        assert describe_error_type(code, "id") == expected
