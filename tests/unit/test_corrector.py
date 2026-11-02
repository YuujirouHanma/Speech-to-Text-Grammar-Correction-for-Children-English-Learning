from collections.abc import Sequence

import pytest

from speakfix.config import InputStyle
from speakfix.domain.errors import InferenceError, InvalidTextError, ResourceExhaustedError
from speakfix.services.corrector import CorrectionConfig, GrammarCorrector
from tests.conftest import FakeBackend


def test_corrects_each_sentence_and_lists_edits(
    corrector: GrammarCorrector, fake_backend: FakeBackend
) -> None:
    result = corrector.correct("He don't like apples. She go to school every day.")
    assert result.corrected == "He doesn't like apples. She goes to school every day."
    assert result.sentences == 2
    assert result.changed
    assert [(e.original, e.corrected) for e in result.edits] == [
        ("don't", "doesn't"),
        ("go", "goes"),
    ]
    sources, beams, max_new = fake_backend.calls[0]
    assert sources == ["grammar: He don't like apples.", "grammar: She go to school every day."]
    assert beams == 5
    assert max_new >= 16  # dynamic, never the old max_length=20 truncation for long input


def test_adds_terminal_punctuation_before_inference(
    corrector: GrammarCorrector, fake_backend: FakeBackend
) -> None:
    result = corrector.correct("I am student")
    assert fake_backend.calls[0][0] == ["grammar: I am student."]
    assert result.corrected == "I am a student."


def test_max_new_tokens_scales_with_input() -> None:
    backend = FakeBackend()
    corrector = GrammarCorrector(backend, config=CorrectionConfig(max_output_tokens=500))
    corrector.correct(" ".join(["word"] * 60) + ".")
    assert backend.calls[0][2] >= 60


def test_trims_hallucinated_extra_sentence() -> None:
    backend = FakeBackend(transform=lambda s: s + " She is a good student and a teacher.")
    result = GrammarCorrector(backend).correct("She goes to school every day.")
    assert result.corrected == "She goes to school every day."
    assert "trimmed_extra_model_output" in result.warnings


def test_faithfulness_guard_keeps_original() -> None:
    backend = FakeBackend(transform=lambda _s: "Totally unrelated invented content appears now.")
    result = GrammarCorrector(backend).correct("My cat sleep on the sofa.")
    assert result.corrected == "My cat sleep on the sofa."
    assert not result.changed
    assert "low_confidence_correction_skipped" in result.warnings


def test_detokenizes_jfleg_style_output() -> None:
    backend = FakeBackend(transform=lambda _s: "Many students find it difficult , really .")
    result = GrammarCorrector(backend).correct("Many students finds it difficult, really.")
    assert result.corrected == "Many students find it difficult, really."


def test_jfleg_input_style() -> None:
    backend = FakeBackend()
    GrammarCorrector(backend, config=CorrectionConfig(input_style=InputStyle.JFLEG)).correct(
        "He don't know."
    )
    assert backend.calls[0][0] == ["grammar: He do n't know ."]


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_rejected(corrector: GrammarCorrector, text: str) -> None:
    with pytest.raises(InvalidTextError):
        corrector.correct(text)


def test_too_long_text_rejected() -> None:
    corrector = GrammarCorrector(FakeBackend(), config=CorrectionConfig(max_chars=10))
    with pytest.raises(InvalidTextError):
        corrector.correct("This sentence is definitely longer than ten characters.")


def test_long_sentence_is_chunked() -> None:
    backend = FakeBackend()
    corrector = GrammarCorrector(backend, config=CorrectionConfig(max_input_tokens=20))
    corrector.correct(" ".join(["word"] * 40))
    assert len(backend.calls[0][0]) > 1


def test_oom_retries_unbatched_then_succeeds() -> None:
    backend = FakeBackend(fail_times=1)
    result = GrammarCorrector(backend).correct("He don't like apples. I am student.")
    assert result.corrected == "He doesn't like apples. I am a student."
    assert len(backend.calls) == 3  # failed batch + two single retries


def test_persistent_failure_switches_to_fallback() -> None:
    primary = FakeBackend(fail_times=99, name="primary")
    fallback = FakeBackend(name="fallback")
    corrector = GrammarCorrector(primary, fallback_factory=lambda: fallback)
    result = corrector.correct("He don't like apples.")
    assert result.backend == "fallback"
    assert result.corrected == "He doesn't like apples."


def test_oom_then_other_error_still_uses_fallback() -> None:
    class OomThenCudaError(FakeBackend):
        def generate(self, sources, *, num_beams, max_new_tokens):  # type: ignore[no-untyped-def]
            self.calls.append((list(sources), num_beams, max_new_tokens))
            if len(self.calls) == 1:
                raise ResourceExhaustedError("oom")
            raise InferenceError("cuda error after oom")

    fallback = FakeBackend(name="fallback")
    corrector = GrammarCorrector(
        OomThenCudaError(name="primary"), fallback_factory=lambda: fallback
    )
    assert corrector.correct("He don't like apples.").backend == "fallback"


def test_generic_failure_without_fallback_raises() -> None:
    backend = FakeBackend(fail_times=99, error=InferenceError)
    with pytest.raises(InferenceError):
        GrammarCorrector(backend).correct("He don't like apples.")


def test_oom_without_fallback_raises_inference_error() -> None:
    backend = FakeBackend(fail_times=99, error=ResourceExhaustedError)
    with pytest.raises(InferenceError):
        GrammarCorrector(backend).correct("He don't like apples.")


def test_num_beams_override(corrector: GrammarCorrector, fake_backend: FakeBackend) -> None:
    corrector.correct("He don't like apples.", num_beams=2)
    assert fake_backend.calls[0][1] == 2


class _TypingClassifier:
    def classify(self, original, corrected, edits):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        return [replace(e, error_type="R:VERB:SVA") for e in edits]


def test_error_type_classifier_is_applied(fake_backend: FakeBackend) -> None:
    corrector = GrammarCorrector(fake_backend, classifier=_TypingClassifier())
    result = corrector.correct("He don't like apples.")
    assert result.edits[0].error_type == "R:VERB:SVA"


def test_warmup_uses_fallback_when_primary_is_broken() -> None:
    primary = FakeBackend(fail_times=99, error=InferenceError, name="primary")
    corrector = GrammarCorrector(primary, fallback_factory=lambda: FakeBackend(name="fallback"))
    corrector.warmup()  # must not raise


class ScoringBackend(FakeBackend):
    """FakeBackend that can also score outputs (log-probabilities from a table)."""

    def __init__(self, scores: dict[str, float], *, scoring_fails: bool = False) -> None:
        super().__init__()
        self.scores = scores
        self.scoring_fails = scoring_fails
        self.score_calls: list[list[str]] = []

    def score(self, sources: Sequence[str], targets: Sequence[str]) -> list[float]:
        self.score_calls.append(list(targets))
        if self.scoring_fails:
            raise InferenceError("simulated scoring failure")
        return [self.scores[t] for t in targets]


SCORES = {
    "He doesn't like apples.": -0.1,  # margin 4.9 over the input: a confident correction
    "He don't like apples.": -5.0,
    "She goes to school every day.": -1.0,  # margin 0.5: the model barely prefers it
    "She go to school every day.": -1.5,
}
TWO_SENTENCES = "He don't like apples. She go to school every day."


def test_confidence_margin_drops_uncertain_corrections() -> None:
    backend = ScoringBackend(SCORES)
    result = GrammarCorrector(backend, config=CorrectionConfig(min_correction_margin=2.0)).correct(
        TWO_SENTENCES
    )
    assert result.corrected == "He doesn't like apples. She go to school every day."
    assert [(e.original, e.corrected) for e in result.edits] == [("don't", "doesn't")]
    assert "uncertain_correction_skipped" in result.warnings


def test_confidence_margin_scores_only_changed_sentences() -> None:
    backend = ScoringBackend(SCORES)
    config = CorrectionConfig(min_correction_margin=2.0)
    GrammarCorrector(backend, config=config).correct("He don't like apples. All is fine here.")
    assert backend.score_calls == [["He doesn't like apples.", "He don't like apples."]]

    GrammarCorrector(backend).correct(TWO_SENTENCES)  # margin disabled by default
    assert len(backend.score_calls) == 1


def test_a_missing_final_period_is_not_an_uncertain_correction() -> None:
    backend = ScoringBackend(SCORES)
    config = CorrectionConfig(min_correction_margin=2.0)
    result = GrammarCorrector(backend, config=config).correct("All is fine here")
    assert result.corrected == "All is fine here."
    assert result.warnings == ()
    assert backend.score_calls == []


def test_confidence_margin_keeps_corrections_when_scoring_fails() -> None:
    backend = ScoringBackend(SCORES, scoring_fails=True)
    config = CorrectionConfig(min_correction_margin=2.0)
    result = GrammarCorrector(backend, config=config).correct(TWO_SENTENCES)
    assert result.corrected == "He doesn't like apples. She goes to school every day."


def test_confidence_margin_needs_a_scoring_backend() -> None:
    config = CorrectionConfig(min_correction_margin=2.0)
    result = GrammarCorrector(FakeBackend(), config=config).correct(TWO_SENTENCES)
    assert result.corrected == "He doesn't like apples. She goes to school every day."


def test_without_punctuation_feedback_no_period_is_added() -> None:
    backend = FakeBackend()
    result = GrammarCorrector(backend, config=CorrectionConfig(punctuation_feedback=False)).correct(
        "I am student"
    )
    assert backend.calls[0][0] == ["grammar: I am student."]  # the model still sees a period
    assert result.corrected == "I am a student"
    assert [(e.original, e.corrected) for e in result.edits] == [("", "a")]


@pytest.mark.parametrize(
    ("text", "model_output"),
    [
        ("If it rains we stay home.", "If it rains, we stay home."),  # comma only
        ("she is happy.", "She is happy."),  # case only
    ],
)
def test_without_punctuation_feedback_surface_only_changes_are_ignored(
    text: str, model_output: str
) -> None:
    backend = FakeBackend(transform=lambda _s: model_output)
    quiet = GrammarCorrector(backend, config=CorrectionConfig(punctuation_feedback=False))
    assert not quiet.correct(text).changed
    assert GrammarCorrector(backend).correct(text).corrected == model_output


def test_speech_transcripts_never_get_punctuation_feedback() -> None:
    backend = FakeBackend(transform=lambda _s: "If it rains, we stay home.")
    corrector = GrammarCorrector(backend)  # punctuation feedback on (typed text)
    assert corrector.correct("If it rains we stay home.").changed
    assert not corrector.correct("If it rains we stay home.", from_speech=True).changed
