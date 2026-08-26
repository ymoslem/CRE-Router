"""Selecting one evaluation run from a split that carries all of them.

The eval split drives early stopping and best-checkpoint selection only, so the
convention is one run per question. AIME-router and TeleMath-router bake that
into the data; TeleQnA-router ships all five runs with `run` as a column, and
before it was restructured the default split selection silently picked a single
(cluster, run) slice instead, which is what this filter replaces.
"""

import pytest
from datasets import Dataset

from cre_router.qe.train import select_eval_run


def five_runs(questions=4, runs=5):
    rows = [{"question": f"q{q}", "run": r, "cluster": q % 2}
            for r in range(runs) for q in range(questions)]
    return Dataset.from_list(rows)


class TestWithARunColumn:
    def test_keeps_only_the_named_run(self):
        data, message = select_eval_run(five_runs(), 0, "test_qwen3_4b_2507")
        assert len(data) == 4
        assert set(data["run"]) == {0}
        assert "run 0" in message and "4 of 20" in message

    def test_every_question_survives(self):
        # One run per question, not a slice of the questions.
        data, _ = select_eval_run(five_runs(questions=7), 3, "eval")
        assert sorted(data["question"]) == [f"q{i}" for i in range(7)]

    def test_keeps_every_cluster(self):
        # The old default took cluster 0 run 0; filtering on run must not drop a cluster.
        data, _ = select_eval_run(five_runs(questions=6), 0, "eval")
        assert set(data["cluster"]) == {0, 1}

    def test_a_later_run_is_selectable(self):
        data, _ = select_eval_run(five_runs(), 4, "eval")
        assert set(data["run"]) == {4}

    def test_the_default_keeps_every_run(self):
        # Opt-in: the tool does not filter unless asked.
        data, message = select_eval_run(five_runs(), split_name="eval")
        assert len(data) == 20
        assert "run" not in message.split("(")[0]

    def test_an_absent_run_is_an_error_naming_what_exists(self):
        with pytest.raises(SystemExit, match=r"runs present: \[0, 1, 2, 3, 4\]"):
            select_eval_run(five_runs(), 9, "eval")


class TestWithoutARunColumn:
    def _plain(self):
        return Dataset.from_list([{"question": f"q{i}"} for i in range(3)])

    def test_returned_unchanged(self):
        # AIME-router and TeleMath-router have no `run` column, and neither do the
        # local directories prep_qe.py writes.
        data, message = select_eval_run(self._plain(), 0, "test_e2b")
        assert len(data) == 3
        assert "run 0" not in message

    def test_a_run_request_is_ignored_rather_than_failing(self):
        data, _ = select_eval_run(self._plain(), 3, "test_e2b")
        assert len(data) == 3
