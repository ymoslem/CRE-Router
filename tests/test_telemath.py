"""TeleMath numeric answer parsing and tolerance matching.

TeleMath answers are numerical quantities (floats, scientific notation, and
negatives), scored by a relative tolerance rather than exact match.
"""

import pytest

from cre_router.evaluate import (
    TASKS,
    check_pre_rendered,
    numeric_match,
    parse_telemath_answer,
    score_generations,
)


class TestParseTelemathAnswer:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (r"The result is \boxed{233.333333333333}.", 233.333333333333),
            ("Answer: 6.0", 6.0),
            (r"so \boxed{7.2e-05}", 7.2e-05),
            (r"final \boxed{7.2 \times 10^{-5}}", 7.2e-5),
            (r"\boxed{7.2 \times 10^-5}", 7.2e-5),
            ("Answer: -62.085424660791375", -62.085424660791375),
            ("the current is 0.25 A", 0.25),
            ("Answer: 50000.0", 50000.0),
        ],
    )
    def test_extracts_value(self, text, expected):
        assert parse_telemath_answer(text) == pytest.approx(expected)

    def test_boxed_wins_over_earlier_numbers(self):
        assert parse_telemath_answer(r"tried 12 and 3, so \boxed{204}") == 204.0

    def test_answer_label_wins_over_reasoning(self):
        assert parse_telemath_answer("we get 3.14 then 2.71.\nAnswer: 42.0") == 42.0

    def test_reasoning_is_stripped(self):
        assert parse_telemath_answer(r"<think>maybe 5</think>\boxed{9.0}") == 9.0

    def test_no_number_returns_none(self):
        assert parse_telemath_answer("no numeric answer here") is None

    def test_leading_dot_decimal(self):
        assert parse_telemath_answer(r"\boxed{.5}") == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "text, expected",
        [
            (r"Final answer: $7/6 \approx 1.1667$\n\n$$\frac{7}{6}$$", 7 / 6),
            (r"$$\boxed{\frac{7}{6}}$$", 7 / 6),
            (r"$$\boxed{-\frac{3}{4}}$$", -0.75),
            (r"$$\boxed{-\dfrac{3}{4}}$$", -0.75),
            ("Answer: 7/6", 7 / 6),
        ],
    )
    def test_fraction_answer(self, text, expected):
        # A fraction is one value, not two separate numbers -- the naive
        # "last number in the text" fallback used to read \frac{7}{6} as its
        # denominator alone (6.0) rather than 7/6.
        assert parse_telemath_answer(text) == pytest.approx(expected)

    def test_boxed_with_unit_label(self):
        # A unit after the value inside \boxed{} must not make the whole
        # match fail and fall through to a noisier, unrelated number.
        assert parse_telemath_answer(r"$$\boxed{0.2 \text{ packets/s}}$$") == pytest.approx(0.2)

    def test_boxed_comma_thousands(self):
        assert parse_telemath_answer(r"$$\boxed{1,382,400,000,000}$$") == pytest.approx(1382400000000.0)

    def test_exponent_is_not_mistaken_for_the_answer(self):
        # With no \boxed{} to anchor on, the last-value scan must not end on
        # the "-2" inside e^{-2}; the stated decimal is the answer.
        text = "We get $2e^{-2}$, i.e. 0.2707 in decimal."
        assert parse_telemath_answer(text) == pytest.approx(0.2707)

    def test_later_unparseable_box_falls_back_to_earlier_clean_box(self):
        # A model sometimes boxes the same answer twice: a clean decimal,
        # then a symbolic restatement at the very end. The symbolic box
        # should not shadow the clean one that already answered the question.
        text = (
            r"$$\boxed{1.732} \text{V}$$"
            "\n\nSince this is exact, we can also write it as:\n\n"
            r"$$\boxed{\sqrt{3}}$$"
        )
        assert parse_telemath_answer(text) == pytest.approx(1.732)

    def test_boxed_trailing_math_is_not_truncated(self):
        # \boxed{2e^{-2}} means 2 * e^-2 (approx 0.2707), not the literal
        # digit 2 with "e^{-2}" as an ignorable label -- unlike a unit, this
        # trailing content changes the value, so the leading digit must not
        # be accepted on its own. Matches a real generation: the model
        # restates the decimal after the box, which the parser should prefer
        # over misreading a fragment of the box's own exponent.
        text = (
            r"$$P[X=2] \approx 0.27067056$$"
            "\n\nThe exact numerical answer is $2e^{-2}$.\n\n"
            r"$$\boxed{2e^{-2}}$$ (or approximately 0.2707)"
        )
        assert parse_telemath_answer(text) == pytest.approx(0.2707, rel=1e-3)

    def test_answer_at_end_of_long_reasoning(self):
        # A long completion whose answer sits at the very end still parses; the
        # tail window keeps parsing bounded without clipping a real answer.
        text = "reasoning that goes on and on. " * 4000 + r"\boxed{233.5}"
        assert parse_telemath_answer(text) == pytest.approx(233.5)

    @pytest.mark.parametrize(
        "text",
        [
            "9" * 200_000 + " done",  # long digit run, no closing token
            r"\boxed{" + "1" * 200_000,  # truncated boxed, unclosed brace
            "3." + "3" * 200_000,  # runaway decimal expansion
        ],
    )
    def test_pathological_input_is_fast(self, text):
        # Degenerate/truncated generations used to send the number regex into
        # O(n^2) backtracking and stall for hours. Parsing must stay well under a
        # second regardless of output length.
        import time

        start = time.perf_counter()
        parse_telemath_answer(text)
        assert time.perf_counter() - start < 1.0


class TestNumericMatch:
    def test_exact(self):
        assert numeric_match(233.333333, 233.333333)

    def test_within_one_percent(self):
        # rounding of the same quantity is accepted
        assert numeric_match(233.33, 233.333333)

    def test_outside_tolerance_rejected(self):
        assert not numeric_match(200.0, 233.333)

    def test_scientific_within_tolerance(self):
        assert numeric_match(7.2e-05, 7.21e-05)

    def test_negative(self):
        assert numeric_match(-62.09, -62.085424660791375)

    def test_near_zero_gold(self):
        assert numeric_match(0.0, 0.0)
        assert not numeric_match(1e-10, 0.0)
        assert not numeric_match(0.5, 0.0)

    @pytest.mark.parametrize(
        "pred, gold, expected",
        [
            (5e-10, 5e-10, True),
            (4.99e-10, 5e-10, True),  # within 1%, a genuine answer
            (0.0, 5e-10, False),  # answering zero is not solving for 5e-10
            (1.5e-55, 5e-10, False),
            (0.0, 2e-10, False),
        ],
    )
    def test_tiny_gold_is_not_given_away(self, pred, gold, expected):
        # Several TeleMath golds are smaller than any plausible absolute floor
        # (down to 1e-10). An abs_tol of 1e-9 made every one of them passable by
        # answering 0, inflating accuracy by up to 1pp and most on the strong
        # tier. The comparison is relative only.
        assert numeric_match(pred, gold) is expected

    def test_long_decimal_rounded_by_the_model(self):
        # The common shape: gold stored at full precision, model reports a
        # correctly rounded value. Relative error 4.8e-07, far inside 1%.
        assert numeric_match(8.92339, 8.9233943)

    def test_gold_as_string(self):
        assert numeric_match(6.0, "6.0")

    def test_none_prediction(self):
        assert not numeric_match(None, 6.0)


class TestTelemathTaskWiring:
    def test_task_uses_numeric_match(self):
        assert TASKS["telemath"].match is numeric_match
        assert TASKS["telemath"].parse is parse_telemath_answer

    def test_score_generations_uses_task_matcher(self):
        texts = [r"\boxed{233.33}", r"\boxed{6.0}", r"\boxed{999.0}"]
        gold = [233.333333, 6.0, 6.0]  # third is wrong
        error, correct = score_generations(
            texts, gold, TASKS["telemath"].parse, TASKS["telemath"].match
        )
        assert correct == [True, True, False]
        assert error == pytest.approx(1 / 3)


class TestPreRenderedGuard:
    """A pre_rendered task served raw prompts fails silently at runtime: the
    model never sees its thinking delimiters, so it generates to max_tokens.
    The guard turns that into an immediate error."""

    def _raw(self, n=3):
        return [{"prompt": f"Determine the throughput of a network with {i} nodes.",
                 "answer": 1.0, "cluster": "0", "id": str(i)} for i in range(n)]

    def _templated(self, n=3):
        return [{"prompt": f"<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nQ{i}",
                 "answer": 1.0, "cluster": "0", "id": str(i)} for i in range(n)]

    def test_raw_prompts_rejected_for_pre_rendered_task(self):
        with pytest.raises(ValueError, match="pre_rendered"):
            check_pre_rendered(self._raw(), TASKS["telemath_gemma4"])

    def test_templated_prompts_accepted(self):
        check_pre_rendered(self._templated(), TASKS["telemath_gemma4"])

    @pytest.mark.parametrize("task", ["telemath", "telemath_nothink"])
    def test_non_pre_rendered_tasks_are_unaffected(self, task):
        # These serve raw text by design; the guard must not fire.
        check_pre_rendered(self._raw(), TASKS[task])

    def test_error_names_the_fix(self):
        with pytest.raises(ValueError, match="prep_gemma4_thinking"):
            check_pre_rendered(self._raw(), TASKS["telemath_gemma4"])
