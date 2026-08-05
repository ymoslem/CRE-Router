"""Stage 1+2 cascade evaluation: QE decisions -> per-cluster cascade accuracy
and escalation counts, with no GPU (the classifier is stubbed)."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cre_router.qe.cascade import (
    compose_cascade,
    run_qe,
    strong_correct_by_qid,
    write_cascade_stats,
)
from cre_router.routing import cluster_cascade_accuracy

CONFIGS = Path(__file__).parent.parent / "configs"


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


class TestWriteCascadeStats:
    def test_merges_into_routing_json_and_feeds_cre_cascade(self, tmp_path):
        from cre_router.routing import (
            cascade_system_accuracy,
            models_from_stats,
        )

        # a routing-side cascade config, no Stage 2 fields yet
        cfg = tmp_path / "cascade.json"
        cfg.write_text(json.dumps({
            "cluster_sizes": {"0": 2, "1": 2},
            "assignment": {"0": "weak", "1": "strong"},
            "models": {
                "weak": {"errors": {"0": 0.5, "1": 0.5}, "cluster_tpot_ms": {"0": 10.0, "1": 10.0}},
                "strong": {"errors": {"0": 0.1, "1": 0.2}, "cluster_tpot_ms": {"0": 20.0, "1": 20.0}},
            },
        }))
        report = {"0": {"cascade_accuracy": 0.9, "escalations": 1.0, "n": 4}}
        write_cascade_stats(cfg, "strong", report)

        stats = json.loads(cfg.read_text())
        assert stats["escalations"]["0"] == ["strong", 1.0]
        assert stats["cascade_accuracy"]["0"] == 0.9
        # cre cascade composition: C0 cascade 0.9, C1 direct = strong 1-0.2=0.8
        models, sizes = models_from_stats(stats)
        assignment = {str(k): str(v) for k, v in stats["assignment"].items()}
        acc = cascade_system_accuracy(models, assignment, sizes,
                                      {str(k): float(v) for k, v in stats["cascade_accuracy"].items()})
        assert acc == pytest.approx((2 * 0.9 + 2 * 0.8) / 4)


class TestPaperCascadeReconstruction:
    """Locks the session's validation that the composition reproduces the paper's
    per-cluster cascade accuracy from the real per-run data:
    TeleQnA C0 from route.zip + `QE-Route TeleQnA.numbers` + inference_teleqna_log_50;
    AIME C1 from `AIME-Clusters-Test.numbers` + route.zip. See memory
    cre-stage1plus2-metrics-composition."""

    def test_teleqna_c0_reconstructs_0_742(self):
        # per efficient/QE run (route.zip cluster_0_run_0..4): routed count, the
        # accepted-and-efficient-correct count (log_50 "Accept: c/d"), and Gemma-26B
        # accuracy on that routed set (mean of its 5 repeats in QE-Route TeleQnA).
        routed = [199, 205, 206, 200, 202]
        accepted_correct = [305, 295, 301, 310, 304]
        s = [0.671, 0.683, 0.670, 0.654, 0.647]
        N = 590
        per_run = [(a + r * si) / N for r, a, si in zip(routed, accepted_correct, s)]
        mean = sum(per_run) / len(per_run)
        assert mean == pytest.approx(0.742, abs=0.002)   # paper Table teleqna_test 0.743
        cfg = json.loads((CONFIGS / "teleqna_cascade_test.json").read_text())
        assert cfg["cascade_accuracy"]["0"] == pytest.approx(mean, abs=0.005)

    def test_aime_c1_reconstructs_0_96(self):
        # cluster 1: size 10 x 5 runs = 50 instances. VibeThinker acc 0.9 every run
        # = exactly one miss/run (the same hard query); the QE escalates it in 3 of
        # 5 runs (route.zip counts 1,1,1,0,0); Qwen3-30B is correct on those 3.
        weak, strong, escalate = [], [], []
        for run in range(5):
            for q in range(10):
                is_hard = q == 5
                weak.append(not is_hard)              # 9 correct, the hard one wrong
                escalate.append(is_hard and run in {0, 1, 2})
                strong.append(True)                   # strong right on the escalated hard query
        acc = cluster_cascade_accuracy(weak, strong, escalate)
        assert acc == pytest.approx(0.96)             # paper Table aime_test 0.96
        cfg = json.loads((CONFIGS / "aime_cascade_test.json").read_text())
        assert cfg["cascade_accuracy"]["1"] == pytest.approx(acc)
