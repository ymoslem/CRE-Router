"""Serving a model's whole routed traffic as one batch, and composing from it."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cre_router.captures import load_capture
from cre_router.compose import Routing, compose
from cre_router.evaluate import ONE_BATCH, Task, _detail_name, check_one_batch, evaluate_model

RUNS = 2


# ---------------------------------------------------------------- the harness

def _task():
    return Task(name="t", parse=lambda s: s.strip(), temperature=0.6, top_p=0.95,
                top_k=20, min_p=0.0, max_tokens=64)


def test_one_batch_serves_every_row_together_and_keeps_provenance(tmp_path):
    calls = []

    def bench(path, model, task, **kw):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        calls.append(sorted(r["cluster"] for r in rows))
        return {"generated_texts": ["1"] * len(rows), "mean_tpot_ms": 10.0,
                "output_lens": [5] * len(rows)}

    data = [{"id": "a", "prompt": "p", "answer": "1", "cluster": "0", "stage": 1},
            {"id": "b", "prompt": "p", "answer": "1", "cluster": "2", "stage": 1},
            {"id": "c", "prompt": "p", "answer": "1", "cluster": "1", "stage": 2, "eff_run": 3}]
    gens = tmp_path / "g.jsonl"
    m = evaluate_model(data, "org/m", _task(), runs=2, workdir=tmp_path / "s", benchmark=bench,
                       generations_out=gens, one_batch=True)
    assert calls == [["0", "1", "2"], ["0", "1", "2"]]      # one batch per run, all rows
    assert {x.cluster for x in m} == {ONE_BATCH}
    out = {g["qid"]: g for g in map(json.loads, gens.open()) if g["run"] == 0}
    assert out["a"]["cluster"] == "0" and out["a"]["batch"] == ONE_BATCH and out["a"]["stage"] == 1
    assert out["c"]["cluster"] == "1" and out["c"]["stage"] == 2 and out["c"]["eff_run"] == 3
    assert "eff_run" not in out["a"]


@pytest.mark.parametrize("bad, match", [
    ({"id": "a", "prompt": "p", "answer": "1", "cluster": "0"}, "stage"),
    ({"id": "a", "prompt": "p", "answer": "1", "cluster": "0", "stage": 2}, "eff_run"),
    ({"id": "a", "prompt": "p", "answer": "1", "stage": 1}, "true cluster"),
])
def test_one_batch_refuses_rows_that_cannot_be_told_apart(bad, match):
    with pytest.raises(ValueError, match=match):
        check_one_batch([bad])


def test_one_batch_refuses_a_question_twice():
    row = {"id": "a", "prompt": "p", "answer": "1", "cluster": "0", "stage": 1}
    with pytest.raises(ValueError, match="twice"):
        check_one_batch([row, dict(row)])


def test_load_capture_reads_a_one_batch_capture_by_its_batch(tmp_path):
    tag, model, task = "cv_x", "Qwen/Q", "aime"
    d = tmp_path / f"results_{tag}_Qwen_Q"
    d.mkdir()
    stem = f"{task}_Qwen_Q_20260101_000000"
    rows = [{"qid": "1", "cluster": "0", "batch": ONE_BATCH, "stage": 1, "run": 0,
             "num_tokens": 3, "ground_truth_answer": "5", "full_output": r"\boxed{5}"},
            {"qid": "2", "cluster": "1", "batch": ONE_BATCH, "stage": 2, "eff_run": 4, "run": 0,
             "num_tokens": 4, "ground_truth_answer": "6", "full_output": r"\boxed{7}"}]
    (d / f"{stem}_generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / f"{stem}.jsonl").write_text(json.dumps({"cluster": ONE_BATCH, "run": 0, "num_prompts": 2,
                                                 "tpot_ms": 10.0}) + "\n")
    dd = tmp_path / f"detail_{tag}"
    dd.mkdir()
    from cre_router.evaluate import TASKS
    (dd / _detail_name(TASKS[task], model, ONE_BATCH, 0)).write_text(json.dumps(
        {"ttfts": [0.1, 0.2], "itls": [[0.01] * 2, [0.02] * 3], "output_lens": [3, 4]}))
    cap = load_capture(tag, tmp_path)
    assert cap[("1", 0)]["cluster"] == "0" and cap[("1", 0)]["stage"] == 1
    assert cap[("2", 0)]["cluster"] == "1" and cap[("2", 0)]["eff_run"] == 4
    assert cap[("2", 0)]["correct"] is False and cap[("2", 0)]["batch"] == ONE_BATCH
    assert cap[("2", 0)]["e2el_s"] == pytest.approx(0.2 + 0.06)


# ---------------------------------------------------------------- composition
#
# C0 and C2 are routed to A, the strong model; C1 to B, gated. q3 escalates in
# every run, q4 never. A serves C0 + C2 as one Stage 1 batch (a_s1), and in
# Stage 1 + 2 serves them with run r's escalations as one batch (a12_r<r>).

def row(c, n=11, tpot=10.0, e2el=1.0, correct=True, **prov):
    return {"correct": correct, "tokens": n, "tpot_ms": tpot, "e2el_s": e2el, "cluster": c, **prov}


def tier(qs, tpot, e2el, correct):
    """A per-cluster capture: every question, its own batch per cluster."""
    return {(q, r): row(c, tpot=tpot + r, e2el=e2el + r, correct=correct)
            for q, c in qs.items() for r in range(RUNS)}


QS = {"q1": "0", "q2": "2", "q3": "1", "q4": "1"}
A = tier(QS, 20.0, 2.0, True)
B = tier(QS, 5.0, 0.5, False)
A_S1 = {(q, s): row(c, tpot=30.0 + s, e2el=3.0, batch="all", stage=1)
        for q, c in (("q1", "0"), ("q2", "2")) for s in range(RUNS)}


def a12(r, esc=("q3",), eff_run=None):
    cap = {(q, s): row(c, tpot=40.0 + r, e2el=4.0, batch="all", stage=1)
           for q, c in (("q1", "0"), ("q2", "2")) for s in range(RUNS)}
    cap.update({(q, s): row("1", n=21, tpot=50.0, e2el=5.0, batch="all", stage=2,
                            eff_run=r if eff_run is None else eff_run)
                for q in esc for s in range(RUNS)})
    return cap


CAPS = {"a": A, "b": B, "a_s1": A_S1, **{f"a12_r{r}": a12(r) for r in range(RUNS)}}
ROUTING = Routing(tiers={"A": "a", "B": "b"}, assign={"0": "A", "1": "B", "2": "A"},
                  gated={"1": "a12"}, strong="A", tau=0.5, runs=RUNS, basis="per_model",
                  served={"A": "a_s1"}, strong_batch="a12")


def probs(p3=0.2, p4=0.9):
    return {(q, r): {"p_accept": p, "num_tokens": 11} for q, p in (("q3", p3), ("q4", p4))
            for r in range(RUNS)}


def test_each_request_is_priced_on_the_batch_its_model_served():
    res = compose(ROUTING, CAPS.__getitem__, probs())
    i = {q: k for k, q in enumerate(res.qids)}
    t1, t12 = res.grid_stage1[1], res.grid_stage1plus2[1]
    # Stage 1: A's clusters from its one batch a_s1 (strong-run axis s), B from b.
    assert t1[i["q1"], 0, 1] == 31.0
    assert t1[i["q4"], 1, 0] == 5.0
    # Stage 1 + 2: A's own clusters from the batch it shared with run r's escalations.
    assert t12[i["q1"], 1, 0] == 41.0
    # q4 is kept, so it stays B's answer on the efficient run r.
    assert t12[i["q4"], 1, 0] == 6.0
    # q3 pays B's pass on run r and the strong pass, over the strong tokens.
    assert t12[i["q3"], 1, 0] == pytest.approx((6.0 * 10 + 50.0 * 20) / 20)
    assert res.grid_stage1plus2[0][i["q3"]].all() and res.escalated_per_run == {"1": [1, 1]}


def test_kept_answers_of_a_gated_cluster_cancel_in_the_paired_difference():
    res = compose(ROUTING, CAPS.__getitem__, probs())
    i = {q: k for k, q in enumerate(res.qids)}
    for s12, s1p in zip(res.grid_stage1plus2, res.grid_stage1_paired):
        assert np.all((s12 - s1p)[i["q4"]] == 0)


def test_a_model_serving_several_clusters_needs_one_batch():
    r = Routing(**{**ROUTING.__dict__, "served": {}})
    with pytest.raises(ValueError, match="needs one batch"):
        compose(r, CAPS.__getitem__, probs())


def test_a_single_cluster_model_cannot_have_a_second_source():
    r = Routing(**{**ROUTING.__dict__, "served": {"A": "a_s1", "B": "b"}})
    with pytest.raises(ValueError, match="second source"):
        compose(r, CAPS.__getitem__, probs())


def test_a_stage1_batch_missing_a_question_is_refused():
    caps = {**CAPS, "a_s1": {k: v for k, v in A_S1.items() if k[0] != "q2"}}
    with pytest.raises(ValueError, match="Stage 1 rows"):
        compose(ROUTING, caps.__getitem__, probs())


def test_a_capture_not_served_as_one_batch_is_refused():
    caps = {**CAPS, "a_s1": {k: {**v, "batch": None} for k, v in A_S1.items()}}
    with pytest.raises(ValueError, match="one batch"):
        compose(ROUTING, caps.__getitem__, probs())


def test_stage1_rows_taken_from_a_batch_with_escalations_are_refused():
    caps = {**CAPS, "a_s1": a12(0)}
    with pytest.raises(ValueError, match="escalated rows"):
        compose(ROUTING, caps.__getitem__, probs())


def test_escalations_from_another_efficient_run_are_refused():
    caps = {**CAPS, "a12_r1": a12(1, eff_run=0)}
    with pytest.raises(ValueError, match="another efficient run"):
        compose(ROUTING, caps.__getitem__, probs())


def test_a_strong_batch_missing_an_escalation_is_refused():
    caps = {**CAPS, "a12_r0": a12(0, esc=())}
    with pytest.raises(ValueError, match="escalated rows"):
        compose(ROUTING, caps.__getitem__, probs())


def test_the_strong_batch_is_needed_even_when_nothing_escalates():
    # A still serves C0 and C2 in Stage 1 + 2, so its batch must exist in every run.
    caps = {**CAPS, "a12_r0": a12(0, esc=()), "a12_r1": a12(1, esc=())}
    res = compose(ROUTING, caps.__getitem__, probs(p3=0.9))
    assert res.escalated_per_run == {"1": [0, 0]}
    missing = {k: v for k, v in caps.items() if k != "a12_r0"}
    with pytest.raises(KeyError):
        compose(ROUTING, missing.__getitem__, probs(p3=0.9))


def test_gated_clusters_must_map_to_the_strong_batch():
    r = Routing(**{**ROUTING.__dict__, "gated": {"1": "other"}})
    with pytest.raises(ValueError, match="strong_batch"):
        compose(r, CAPS.__getitem__, probs())


def test_scores_of_another_generation_are_refused():
    p = {k: {**v, "num_tokens": 99} for k, v in probs().items()}
    with pytest.raises(ValueError, match="output length"):
        compose(ROUTING, CAPS.__getitem__, p)


def test_per_cluster_routings_refuse_per_model_fields():
    r = Routing(**{**ROUTING.__dict__, "basis": "per_cluster"})
    with pytest.raises(ValueError, match="per_model"):
        compose(r, CAPS.__getitem__, probs())
