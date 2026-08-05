"""TeleMath numeric answer parsing and tolerance matching.

TeleMath answers are numerical quantities (floats, scientific notation, and
negatives), scored by a relative tolerance rather than exact match.
"""

import pytest

from cre_router.evaluate import (
    TASKS,
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
        assert numeric_match(1e-10, 0.0)
        assert not numeric_match(0.5, 0.0)

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
