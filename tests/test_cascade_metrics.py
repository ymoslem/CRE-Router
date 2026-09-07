"""Per-request cascade accounting, including the runs that escalate nothing."""
import numpy as np
import pytest

from cre_router.cascade import per_request_metrics


def cap(correct, e2el, tokens):
    return {"correct": np.array(correct, dtype=float),
            "e2el": np.array(e2el, dtype=float),
            "tokens": np.array(tokens, dtype=float)}


EFF = cap([[1, 0], [0, 0]], [[10.0, 12.0], [20.0, 22.0]], [[100, 120], [200, 220]])
STR = cap([[1, 1], [1, 0]], [[50.0, 55.0], [60.0, 65.0]], [[500, 550], [600, 650]])


def test_no_escalation_is_the_efficient_tier_alone():
    n = np.zeros((2, 2), dtype=bool)
    correct, e2el, tpot = per_request_metrics(~n, n, EFF, STR)
    assert np.array_equal(correct[:, :, 0], EFF["correct"])
    assert np.array_equal(e2el[:, :, 0], EFF["e2el"])
    assert np.allclose(tpot[:, :, 0], EFF["e2el"] / EFF["tokens"] * 1000)


def test_escalated_query_pays_both_tiers_and_delivers_the_strong_answer():
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.array([[True, False], [False, False]])
    correct, e2el, tpot = per_request_metrics(runs_eff, uses_strong, EFF, STR)
    # question 0, efficient run 0, strong run 0: 10 waited, then 50 more
    assert e2el[0, 0, 0] == pytest.approx(60.0)
    # priced over the 500 tokens delivered, not the 100 discarded
    assert tpot[0, 0, 0] == pytest.approx(60.0 / 500 * 1000)
    assert correct[0, 0, 0] == 1.0


def test_a_cluster_routed_straight_to_strong_never_pays_the_efficient_pass():
    runs_eff = np.zeros((2, 2), dtype=bool)
    uses_strong = np.ones((2, 2), dtype=bool)
    _, e2el, _ = per_request_metrics(runs_eff, uses_strong, EFF, STR)
    assert e2el[0, 0, 0] == pytest.approx(50.0)   # strong only
    # charging the unrun efficient pass would give 60.0, the error this guards


def test_a_run_that_escalates_nothing_is_kept_not_dropped():
    """An empty run is an observation, not missing data."""
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.array([[True, False], [True, False]])   # run 1 escalates none
    correct, e2el, _ = per_request_metrics(runs_eff, uses_strong, EFF, [STR, None])
    assert e2el.shape == (2, 2, 2)                           # run 1 still present
    assert np.allclose(e2el[:, 1, :], EFF["e2el"][:, 1][:, None])
    assert np.allclose(correct[:, 1, :], EFF["correct"][:, 1][:, None])
    assert e2el[0, 0, 0] == pytest.approx(10.0 + 50.0)


def test_one_strong_capture_per_efficient_run():
    other = cap([[0, 0], [0, 0]], [[1.0, 1.0], [1.0, 1.0]], [[10, 10], [10, 10]])
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.ones((2, 2), dtype=bool)
    _, e2el, _ = per_request_metrics(runs_eff, uses_strong, EFF, [STR, other])
    assert e2el[0, 0, 0] == pytest.approx(10.0 + 50.0)   # run 0 -> STR
    assert e2el[0, 1, 0] == pytest.approx(12.0 + 1.0)    # run 1 -> other


def test_shape_disagreements_raise_rather_than_broadcast():
    runs_eff = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        per_request_metrics(runs_eff, np.ones((2, 3), dtype=bool), EFF, STR)
    with pytest.raises(ValueError):
        per_request_metrics(runs_eff, runs_eff, EFF, [STR])
