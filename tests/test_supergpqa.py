"""SuperGPQA answer parsing, prompt assembly and subset construction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cre_router.evaluate import TASKS, parse_supergpqa_answer, score_generations

_PREP = Path(__file__).resolve().parents[1] / "data" / "prep_supergpqa.py"
_spec = importlib.util.spec_from_file_location("prep_supergpqa", _PREP)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def row(uuid="u1", n_options=4, letter="C", discipline="Science"):
    return {
        "uuid": uuid,
        "question": "What is the capital of France?",
        "options": [f"opt{i}" for i in range(n_options)],
        "answer_letter": letter,
        "discipline": discipline,
        "field": "Physics",
        "difficulty": "middle",
        "is_calculation": False,
    }


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\boxed{C}", 2),
        (r"\boxed{ A }", 0),
        ("Answer: B", 1),
        ("**Answer:** D", 3),
        ("Answer：E", 4),  # full-width colon
        ("The answer is J", 9),
        ("no letter here at all", None),
    ],
)
def test_parses_each_supported_form(text, expected):
    assert parse_supergpqa_answer(text) == expected


def test_boxed_wins_over_a_stray_letter():
    text = "Option A looks plausible, and B too, but \\boxed{D}"
    assert parse_supergpqa_answer(text) == 3


def test_last_match_wins_when_the_model_restates():
    assert parse_supergpqa_answer(r"first \boxed{A} then \boxed{C}") == 2


def test_reasoning_trace_is_stripped_before_parsing():
    """A letter mentioned inside <think> must not be taken as the answer."""
    text = "<think>Maybe A, or possibly B.</think>\n\\boxed{E}"
    assert parse_supergpqa_answer(text) == 4


def test_empty_think_block_does_not_break_parsing():
    """Suppressed replies still carry an empty block; the answer follows it."""
    assert parse_supergpqa_answer("<think>\n\n</think>\n\n\\boxed{B}") == 1


# --- task wiring -----------------------------------------------------------


def test_both_task_entries_exist_with_mode_appropriate_sampling():
    think, nothink = TASKS["supergpqa"], TASKS["supergpqa_nothink"]
    assert (think.temperature, think.top_p) == (0.6, 0.95)
    assert (nothink.temperature, nothink.top_p) == (0.7, 0.8)
    # Same parser and same cap: only the sampling differs between the arms.
    assert think.parse is nothink.parse is parse_supergpqa_answer
    assert think.max_tokens == nothink.max_tokens


def test_cap_leaves_prompt_room_inside_the_native_context():
    """30k output inside a 32,768 context must leave room for the prompt."""
    assert TASKS["supergpqa"].max_tokens < 32768 - 1000


def test_scoring_end_to_end():
    generations = [r"\boxed{C}", "Answer: A", "unparseable"]
    error, correct = score_generations(generations, [2, 0, 1], parse_supergpqa_answer)
    assert correct == [True, True, False]
    assert error == pytest.approx(1 / 3)


# --- subset construction ---------------------------------------------------


def test_gold_index_maps_letters_to_positions():
    assert prep.gold_index(row(letter="A")) == 0
    assert prep.gold_index(row(letter="D")) == 3


def test_gold_index_rejects_a_letter_beyond_the_options():
    with pytest.raises(ValueError, match="outside"):
        prep.gold_index(row(n_options=3, letter="H"))


def test_prompt_letters_every_option():
    text = prep.build_prompt(row(n_options=4))
    for letter in "ABCD":
        assert f"{letter}) opt" in text
    assert "E)" not in text


def test_proportional_sample_preserves_discipline_shares():
    """A 90/10 split in the source should stay roughly 90/10 in the sample."""
    dataset = [row(uuid=f"s{i}", discipline="Science") for i in range(900)]
    dataset += [row(uuid=f"l{i}", discipline="Law") for i in range(100)]

    sample = prep.proportional_sample(dataset, 100, seed=0)

    assert len(sample) == 100
    law = sum(1 for r in sample if r["discipline"] == "Law")
    assert 8 <= law <= 12


def test_proportional_sample_is_deterministic():
    dataset = [row(uuid=f"u{i}") for i in range(200)]
    first = [r["uuid"] for r in prep.proportional_sample(dataset, 50, seed=7)]
    second = [r["uuid"] for r in prep.proportional_sample(dataset, 50, seed=7)]
    assert first == second


def test_proportional_sample_never_exceeds_a_small_discipline():
    """Asking for more than a discipline holds must not oversample it."""
    dataset = [row(uuid=f"s{i}", discipline="Science") for i in range(50)]
    dataset += [row(uuid="only", discipline="Sociology")]
    sample = prep.proportional_sample(dataset, 51, seed=0)
    assert sum(1 for r in sample if r["discipline"] == "Sociology") == 1


def test_nothink_file_prepends_the_switch(tmp_path):
    rows = [row(uuid="u1")]
    plain, switched = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    prep.write_split(rows, plain, no_think=False)
    prep.write_split(rows, switched, no_think=True)

    a = json.loads(plain.read_text())
    b = json.loads(switched.read_text())

    assert not a["prompt"].startswith("/no_think")
    assert b["prompt"].startswith("/no_think ")
    # The switch is the ONLY difference; everything else must match.
    assert b["prompt"].removeprefix("/no_think ") == a["prompt"]
    assert a["answer"] == b["answer"]


def test_written_rows_carry_metadata_for_analysis_not_routing(tmp_path):
    out = tmp_path / "x.jsonl"
    prep.write_split([row()], out, no_think=False)
    record = json.loads(out.read_text())
    assert record["answer"] == 2
    for key in ("uuid", "discipline", "difficulty", "is_calculation", "n_options"):
        assert key in record


def test_gold_survives_the_round_trip(tmp_path):
    """What prep writes as gold must be what the parser recovers."""
    out = tmp_path / "y.jsonl"
    prep.write_split([row(letter="D", n_options=6)], out, no_think=False)
    record = json.loads(out.read_text())
    assert parse_supergpqa_answer(r"\boxed{D}") == record["answer"]
