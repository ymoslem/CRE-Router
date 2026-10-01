"""Stage 1+2 cascade evaluation: QE decisions -> per-cluster cascade accuracy
and escalation counts, with no GPU (the classifier is stubbed)."""

from dataclasses import dataclass

import pytest

from cre_router.qe.cascade import (
    compose_cascade,
    run_qe,
    strong_correct_by_qid,
)


def _gen(qid, cluster, run, correct):
    return {"qid": qid, "cluster": cluster, "run": run, "correct": correct,
            "full_output": f"out-{qid}", "num_tokens": 10, "prompt": f"q-{qid}"}


class TestStrongCorrectByQid:
    def test_mean_over_runs(self):
        outcomes = [
            {"qid": "a", "correct": True}, {"qid": "a", "correct": False},
            {"qid": "b", "correct": True}, {"qid": "b", "correct": True},
        ]
        assert strong_correct_by_qid(outcomes) == {"a": 0.5, "b": 1.0}


class TestComposeCascade:
    def test_per_query_composition_and_escalation_count(self):
        # cluster 0, 2 queries x 2 runs. a: weak-correct + accepted; b: weak-wrong + escalated.
        gens = [
            _gen("a", 0, 0, True), _gen("b", 0, 0, False),
            _gen("a", 0, 1, True), _gen("b", 0, 1, False),
        ]
        escalate = [False, True, False, True]
        strong = {"b": 0.5}  # strong right on b half its runs
        report = compose_cascade(gens, escalate, strong)
        # correct: a,a -> 1+1 ; b,b escalated -> 0.5+0.5 ; /4 = 0.75
        assert report["0"]["cascade_accuracy"] == pytest.approx(0.75)
        # 2 escalated rows over 2 runs -> 1 escalation/run
        assert report["0"]["escalations"] == pytest.approx(1.0)
        assert report["0"]["n"] == 4

    def test_multiple_clusters(self):
        gens = [_gen("a", 0, 0, True), _gen("b", 1, 0, False)]
        report = compose_cascade(gens, [False, True], {"b": 1.0})
        assert report["0"]["cascade_accuracy"] == pytest.approx(1.0)
        assert report["1"]["cascade_accuracy"] == pytest.approx(1.0)
        assert report["1"]["escalations"] == pytest.approx(1.0)

    def test_escalation_absent_in_some_runs_divides_by_all_runs(self):
        # AIME C1 shape: one query over 5 runs; escalated in runs 0,1,2 only.
        # escalations must divide by the 5 runs present, not the 3 escalated ones.
        gens = [_gen("x", 1, r, correct=(r >= 3)) for r in range(5)]
        escalate = [True, True, True, False, False]
        report = compose_cascade(gens, escalate, {"x": 1.0})
        assert report["1"]["escalations"] == pytest.approx(0.6)   # 3/5, not 3/3
        assert report["1"]["n"] == 5
        # 3 escalated -> strong 1.0 ; runs 3,4 accepted + weak-correct -> 5/5
        assert report["1"]["cascade_accuracy"] == pytest.approx(1.0)

    def test_misaligned_raises(self):
        with pytest.raises(ValueError, match="align"):
            compose_cascade([_gen("a", 0, 0, True)], [False, True], {})

    def test_missing_strong_outcome_raises(self):
        with pytest.raises(KeyError, match="qid"):
            compose_cascade([_gen("a", 0, 0, False)], [True], strong_correct={})


class TestRunQe:
    def test_stub_classifier_produces_escalate(self):
        @dataclass
        class Decision:
            accept: bool

        class Stub:
            # accept when the output ends in "keep", else route
            def predict_batch(self, items):
                return [Decision(accept=o.endswith("keep")) for _, o, _ in items]

        gens = [
            {"prompt": "q0", "full_output": "... keep", "num_tokens": 5, "cluster": 0, "run": 0, "qid": "0", "correct": True},
            {"prompt": "q1", "full_output": "... drop", "num_tokens": 5, "cluster": 0, "run": 0, "qid": "1", "correct": False},
        ]
        assert run_qe(Stub(), gens, batch_size=1) == [False, True]  # keep->accept->no escalate; drop->route
