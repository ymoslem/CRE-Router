"""Per-request cascade accounting, including the runs that escalate nothing."""
import numpy as np
import pytest

from cre_router.cascade import per_request_metrics


def cap(correct, e2el, tokens, tpot):
    return {"correct": np.array(correct, dtype=float),
            "e2el": np.array(e2el, dtype=float),
            "tokens": np.array(tokens, dtype=float),
            "tpot": np.array(tpot, dtype=float)}


EFF = cap([[1, 0], [0, 0]], [[10.0, 12.0], [20.0, 22.0]],
          [[101, 121], [201, 221]], [[8.0, 9.0], [10.0, 11.0]])
STR = cap([[1, 1], [1, 0]], [[50.0, 55.0], [60.0, 65.0]],
          [[501, 551], [601, 651]], [[26.0, 27.0], [28.0, 29.0]])


def test_a_single_tier_query_reports_the_measured_tpot_unchanged():
    """The common case does no arithmetic: it hands back what vLLM recorded."""
    n = np.zeros((2, 2), dtype=bool)
    correct, e2el, tpot = per_request_metrics(~n, n, EFF, STR)
    assert np.array_equal(correct[:, :, 0], EFF["correct"])
    assert np.array_equal(e2el[:, :, 0], EFF["e2el"])
    assert np.allclose(tpot[:, :, 0], EFF["tpot"])
    # and it is NOT the whole-wait figure, which folds in time before token one
    assert not np.allclose(tpot[:, :, 0], EFF["e2el"] / EFF["tokens"] * 1000)


def test_escalated_query_pays_both_tiers_and_delivers_the_strong_answer():
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.array([[True, False], [False, False]])
    correct, e2el, tpot = per_request_metrics(runs_eff, uses_strong, EFF, STR)
    # question 0, efficient run 0, strong run 0: 10 waited, then 50 more
    assert e2el[0, 0, 0] == pytest.approx(60.0)
    # decode time of both passes over the 500 gaps in the delivered answer
    assert tpot[0, 0, 0] == pytest.approx((8.0 * 100 + 26.0 * 500) / 500)
    assert correct[0, 0, 0] == 1.0


def test_escalating_costs_more_per_token_than_the_strong_tier_alone():
    """The choice of denominator, pinned.

    The efficient pass's decode time is charged in full, but its tokens are not
    counted, because they were discarded and never reached the user. Counting
    them would put this figure between the two tiers, so a cascade would price
    below the model it escalates to.
    """
    runs_eff = np.ones((2, 2), dtype=bool)
    escalated = per_request_metrics(runs_eff, np.ones((2, 2), dtype=bool), EFF, STR)[2]
    straight = per_request_metrics(~runs_eff, np.ones((2, 2), dtype=bool), EFF, STR)[2]
    assert np.all(escalated > straight)
    assert np.allclose(straight[:, :, 0], STR["tpot"][:, 0][:, None])
    generated = ((8.0 * 100 + 26.0 * 500) / 600)   # the rejected formula
    assert escalated[0, 0, 0] > generated


def test_a_cluster_routed_straight_to_strong_never_pays_the_efficient_pass():
    runs_eff = np.zeros((2, 2), dtype=bool)
    uses_strong = np.ones((2, 2), dtype=bool)
    _, e2el, tpot = per_request_metrics(runs_eff, uses_strong, EFF, STR)
    assert e2el[0, 0, 0] == pytest.approx(50.0)   # strong only
    assert tpot[0, 0, 0] == pytest.approx(26.0)   # the strong tier's own TPOT
    # charging the unrun efficient pass would give 60.0, the error this guards


def test_a_run_that_escalates_nothing_is_kept_not_dropped():
    """An empty run is an observation, not missing data."""
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.array([[True, False], [True, False]])   # run 1 escalates none
    correct, e2el, tpot = per_request_metrics(runs_eff, uses_strong, EFF, [STR, None])
    assert e2el.shape == (2, 2, 2)                           # run 1 still present
    assert np.allclose(e2el[:, 1, :], EFF["e2el"][:, 1][:, None])
    assert np.allclose(correct[:, 1, :], EFF["correct"][:, 1][:, None])
    assert np.allclose(tpot[:, 1, :], EFF["tpot"][:, 1][:, None])
    assert e2el[0, 0, 0] == pytest.approx(10.0 + 50.0)


def test_one_strong_capture_per_efficient_run():
    other = cap([[0, 0], [0, 0]], [[1.0, 1.0], [1.0, 1.0]],
                [[11, 11], [11, 11]], [[2.0, 2.0], [2.0, 2.0]])
    runs_eff = np.ones((2, 2), dtype=bool)
    uses_strong = np.ones((2, 2), dtype=bool)
    _, e2el, _ = per_request_metrics(runs_eff, uses_strong, EFF, [STR, other])
    assert e2el[0, 0, 0] == pytest.approx(10.0 + 50.0)   # run 0 -> STR
    assert e2el[0, 1, 0] == pytest.approx(12.0 + 1.0)    # run 1 -> other


def test_a_one_token_answer_has_no_tpot_and_is_refused():
    """Zero would be a plausible-looking lie, so it raises instead."""
    tiny = cap([[1, 1], [1, 1]], [[5.0, 5.0], [5.0, 5.0]],
               [[1, 1], [1, 1]], [[0.0, 0.0], [0.0, 0.0]])
    n = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="no TPOT"):
        per_request_metrics(~n, n, tiny, STR)


def test_shape_disagreements_raise_rather_than_broadcast():
    runs_eff = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError):
        per_request_metrics(runs_eff, np.ones((2, 3), dtype=bool), EFF, STR)
    with pytest.raises(ValueError):
        per_request_metrics(runs_eff, runs_eff, EFF, [STR])


def test_a_capture_without_tpot_is_refused_not_reconstructed():
    """0.3.0 captures lack `tpot`. Falling back would reinstate the bug."""
    old = {k: v for k, v in EFF.items() if k != "tpot"}
    n = np.zeros((2, 2), dtype=bool)
    with pytest.raises(KeyError):
        per_request_metrics(~n, n, old, STR)
