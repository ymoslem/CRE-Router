"""Batches that serve a routing as deployed, and the routing file compose reads."""
from __future__ import annotations

import json

import pytest

from cre_router.batches import stage1_batches, strong_batch, write_plan
from cre_router.compose import Routing, compose
from cre_router.evaluate import check_one_batch

RUNS = 2
# C0 and C2 go to S (strong), C1 to E (efficient, gated). Prompts differ per model.
QS = {"a": "0", "b": "1", "c": "1", "d": "2"}
DATA = {m: [{"id": q, "prompt": f"{m}:{q}", "answer": "1", "cluster": c} for q, c in QS.items()]
        for m in ("S", "E")}
ASSIGN = {"0": "S", "1": "E", "2": "S"}


def probs(reject=("b",)):
    return {(q, r): {"p_accept": 0.1 if q in reject else 0.9, "num_tokens": 3}
            for q in ("b", "c") for r in range(RUNS)}


def test_each_model_gets_one_batch_of_exactly_its_clusters():
    b = stage1_batches(ASSIGN, DATA)
    assert sorted(r["id"] for r in b["S"]) == ["a", "d"]
    assert sorted(r["id"] for r in b["E"]) == ["b", "c"]
    assert all(r["stage"] == 1 and r["prompt"].startswith("S:") for r in b["S"])
    check_one_batch(b["S"])


def test_the_strong_batch_adds_run_r_escalations_with_the_strong_prompt():
    rows = strong_batch(ASSIGN, "S", ["1"], DATA, probs(), run=1, tau=0.5)
    by = {r["id"]: r for r in rows}
    assert set(by) == {"a", "b", "d"}
    assert by["b"] == {"id": "b", "prompt": "S:b", "answer": "1", "cluster": "1",
                       "stage": 2, "eff_run": 1}


def test_a_gated_question_without_a_score_is_refused():
    p = probs()
    del p[("c", 0)]
    with pytest.raises(ValueError, match="no accept probability"):
        strong_batch(ASSIGN, "S", ["1"], DATA, p, run=0, tau=0.5)


def test_the_plan_is_written_in_two_steps_and_compose_accepts_it(tmp_path):
    r1 = write_plan(tmp_path, "x", "rt", ASSIGN, DATA, strong="S", gated=["1"])
    assert not (tmp_path / "rt_routing.json").exists()          # Stage 2 not written yet
    assert r1["served"] == {"S": "x_s1_rt_s"} and r1["tiers"]["E"] == "x_s1_rt_e"
    with pytest.raises(FileExistsError):
        write_plan(tmp_path, "x", "rt", ASSIGN, DATA, strong="S", gated=["1"])
    write_plan(tmp_path, "x", "rt", ASSIGN, DATA, strong="S", gated=["1"], probs=probs(), runs=RUNS)
    routing = Routing.from_json(tmp_path / "rt_routing.json")
    assert routing.basis == "per_model" and routing.strong_batch == "x_s12_rt_s"
    assert routing.gated == {"1": "x_s12_rt_s"}

    # Serve every batch (one row per question and run) and compose from the result.
    def served(tag):
        rows = [json.loads(line) for line in (tmp_path / f"{tag}.jsonl").open()]
        return {(r["id"], s): {"correct": r["prompt"].startswith("S:"), "tokens": 3,
                               "tpot_ms": 10.0, "e2el_s": 1.0, "ttft_s": 0.1,
                               "cluster": r["cluster"], "batch": "all", "stage": r["stage"],
                               **({"eff_run": r["eff_run"]} if "eff_run" in r else {})}
                for r in rows for s in range(RUNS)}
    res = compose(routing, served, probs())
    assert res.escalated_per_run == {"1": [1, 1]}
    assert res.stage1[0] == pytest.approx(0.5) and res.stage1plus2[0] == pytest.approx(0.75)


def test_stage1_batches_must_match_the_plan_before_stage2(tmp_path):
    write_plan(tmp_path, "x", "rt", ASSIGN, DATA, strong="S", gated=["1"])
    other = {**ASSIGN, "2": "E"}
    with pytest.raises(ValueError, match="differs from this plan"):
        write_plan(tmp_path, "x", "rt", other, DATA, strong="S", gated=["1"], probs=probs(),
                   runs=RUNS)


def test_an_ungated_routing_is_complete_after_stage1(tmp_path):
    write_plan(tmp_path, "x", "one", {"0": "S", "1": "S", "2": "S"}, DATA)
    routing = Routing.from_json(tmp_path / "one_routing.json")
    assert routing.served == {"S": "x_s1_one_s"} and not routing.gated


def test_cre_batches_reads_the_fit_artifacts(tmp_path):
    from cre_router.artifacts import RouterArtifacts
    from cre_router.cli import main

    RouterArtifacts(routing_table=ASSIGN, escalation_target="S",
                    gated_clusters={"1": "E"}).save(tmp_path / "art")
    args = ["batches", "--router", str(tmp_path / "art"), "--prefix", "x", "--name", "rt",
            "--out", str(tmp_path / "out")]
    for m, rows in DATA.items():
        p = tmp_path / f"{m}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        args += ["--dataset", f"{m}={p}"]
    main(args)
    assert (tmp_path / "out" / "x_s1_rt_e.jsonl").exists()
    pp = tmp_path / "p.jsonl"
    pp.write_text("".join(json.dumps({"qid": q, "run": r, **v}) + "\n" for (q, r), v in probs().items()))
    main(args + ["--probs", str(pp), "--runs", str(RUNS)])
    assert Routing.from_json(tmp_path / "out" / "rt_routing.json").strong_batch == "x_s12_rt_s"


def test_an_exact_tie_escalates_in_the_strong_batch():
    p = probs()
    p[("c", 0)] = {"p_accept": 0.5, "num_tokens": 3}
    assert {r["id"] for r in strong_batch(ASSIGN, "S", ["1"], DATA, p, run=0, tau=0.5)} == {"a", "b", "c", "d"}
