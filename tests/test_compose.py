"""Stage 1 and Stage 1 + 2 composition from captures, and what it refuses."""
from __future__ import annotations

import pytest

from cre_router.compose import Routing, compose

RUNS = 2


def capture(rows: dict[str, tuple[bool, int, float, float, str]]) -> dict:
    """{qid: (correct, tokens, tpot_ms, e2el_s, cluster)}, identical in every run."""
    return {(q, r): {"correct": c, "tokens": n, "tpot_ms": t, "e2el_s": e, "cluster": k}
            for q, (c, n, t, e, k) in rows.items() for r in range(RUNS)}


# Cluster 0 is served by A, always right. Cluster 1 by B, always wrong, gated:
# q3 escalates to C in every run and C gets it right; q4 is accepted.
A = capture({"q1": (True, 11, 10.0, 1.0, "0"), "q2": (True, 11, 10.0, 1.0, "0"),
             "q3": (True, 11, 10.0, 1.0, "1"), "q4": (True, 11, 10.0, 1.0, "1")})
B = capture({"q1": (False, 11, 20.0, 2.0, "0"), "q2": (False, 11, 20.0, 2.0, "0"),
             "q3": (False, 11, 20.0, 2.0, "1"), "q4": (False, 11, 20.0, 2.0, "1")})
ESC = {f"esc_r{r}": {("q3", s): {"correct": True, "tokens": 21, "tpot_ms": 30.0,
                                 "e2el_s": 5.0, "cluster": "1"} for s in range(RUNS)}
       for r in range(RUNS)}
CAPTURES = {"a": A, "b": B, **ESC}
ROUTING = Routing(tiers={"A": "a", "B": "b"}, assign={"0": "A", "1": "B"},
                  gated={"1": "esc"}, strong="C", tau=0.5, runs=RUNS)


def probs(p3=0.2, p4=0.9, tokens=11):
    return {(q, r): {"p_accept": p, "num_tokens": tokens}
            for q, p in (("q3", p3), ("q4", p4)) for r in range(RUNS)}


def load(tag):
    return CAPTURES[tag]


def test_stage1_is_the_assigned_models_alone():
    acc, tpot, e2el = compose(ROUTING, load, probs()).stage1
    assert acc == pytest.approx(0.5)          # A right on 2, B wrong on 2
    assert tpot == pytest.approx(15.0)
    assert e2el == pytest.approx(1.5)


def test_an_escalated_question_takes_the_strong_answer_and_pays_both_passes():
    result = compose(ROUTING, load, probs())
    acc, tpot, e2el = result.stage1plus2
    assert acc == pytest.approx(0.75)         # q3 now right through C
    # q3's E2EL is B's 2 s plus C's 5 s; the others are unchanged.
    assert e2el == pytest.approx((1 + 1 + 7 + 2) / 4)
    # q3's TPOT: B's decode, 20 ms x 10 gaps, plus C's, 30 x 20, over C's 20 gaps.
    assert tpot == pytest.approx((10 + 10 + (200 + 600) / 20 + 20) / 4)
    assert result.escalated_per_run == {"1": [1, 1]}


def test_an_ungated_cluster_is_the_same_in_both_stages():
    routing = Routing(tiers={"A": "a"}, assign={"0": "A", "1": "A"}, runs=RUNS)
    result = compose(routing, load, {})
    assert result.stage1 == pytest.approx(result.stage1plus2)


def test_a_run_that_escalates_nobody_needs_no_capture():
    result = compose(ROUTING, lambda t: CAPTURES[t] if not t.startswith("esc") else
                     pytest.fail("no escalation capture should be read"), probs(p3=0.9))
    assert result.escalated_per_run == {"1": [0, 0]}
    assert result.stage1plus2 == pytest.approx(result.stage1)


def test_a_gated_question_without_a_probability_is_refused():
    p = probs()
    del p[("q4", 1)]
    with pytest.raises(ValueError, match="no accept probability"):
        compose(ROUTING, load, p)


def test_probabilities_from_another_sampling_run_are_refused():
    with pytest.raises(ValueError, match="scored a different sampling run"):
        compose(ROUTING, load, probs(tokens=99))


def test_an_escalation_capture_holding_other_questions_is_refused():
    with pytest.raises(ValueError, match="escalates 2 in cluster 1"):
        compose(ROUTING, load, probs(p4=0.1))
