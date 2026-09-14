"""FrugalGPT cascade baseline: the cascade semantics and the budget constraint."""
import numpy as np
import pytest

from cre_router.baselines.frugalgpt import (
    CascadeFit,
    _first_acceptance,
    _thresholds_from_quantiles,
    apply_cascade,
    fit_cascade,
    optimise_thresholds,
)


def test_only_the_first_clearing_stage_answers():
    """A cascade stops at the first acceptance; later stages never run."""
    d = np.array([[0.1, 0.1, 0.1],      # stage 0 accepts, 1 and 2 must not
                  [0.9, 0.2, 0.1],      # stage 0 escalates, stage 1 accepts
                  [0.9, 0.9, 0.5]])     # only the last stage accepts
    answered = _first_acceptance(d, np.array([0.5, 0.5, 1.0]))
    assert answered.sum(axis=1).tolist() == [1, 1, 1]
    assert answered.argmax(axis=1).tolist() == [0, 1, 2]


def test_the_last_stage_always_answers():
    """Its threshold is pinned to 1.0, so no query can fall off the end."""
    d = np.array([[0.99, 0.99, 0.99]])
    thresholds = _thresholds_from_quantiles(d, np.array([0.5, 0.5]))
    assert thresholds[-1] == 1.0
    assert _first_acceptance(d, thresholds).any()


def test_thresholds_are_quantiles_of_the_queries_that_reach_a_stage():
    """Not of every query. A later stage only sees what earlier stages rejected."""
    d = np.column_stack([np.linspace(0, 1, 11), np.full(11, 0.4)])
    thresholds = _thresholds_from_quantiles(d, np.array([0.5]))
    # stage 0 keeps the more confident half
    assert thresholds[0] == pytest.approx(np.quantile(d[:, 0], 0.5))


def test_a_budget_below_the_cheapest_model_is_refused_not_silently_met():
    L = np.ones((4, 2))
    C = np.column_stack([np.full(4, 10.0), np.full(4, 20.0)])
    d = np.zeros((4, 2))
    acc, _, _ = optimise_thresholds(L, C, d, budget=1.0)
    assert acc == -np.inf


def test_the_fit_respects_the_budget_it_was_given():
    rng = np.random.default_rng(0)
    n = 60
    correct = (rng.random((n, 3)) < np.array([0.4, 0.6, 0.9])).astype(float)
    cost = np.tile(np.array([1.0, 5.0, 20.0]), (n, 1))
    # a scorer that knows something: confident exactly when the answer is right
    score = correct * 0.9 + (1 - correct) * 0.1
    fit = fit_cascade(correct, cost, score, budget=8.0, depth=2)
    assert fit.cost <= 8.0 + 1e-9
    assert len(fit.models) == 2


def test_a_useful_scorer_beats_the_cheap_model_alone():
    """The point of a cascade: escalate the failures, keep the successes."""
    rng = np.random.default_rng(1)
    n = 200
    weak = (rng.random(n) < 0.5).astype(float)
    strong = np.ones(n)
    correct = np.column_stack([weak, strong])
    cost = np.tile(np.array([1.0, 10.0]), (n, 1))
    score = np.column_stack([weak * 0.95 + (1 - weak) * 0.05, np.full(n, 0.99)])
    fit = fit_cascade(correct, cost, score, budget=8.0, depth=2)
    assert fit.accuracy > weak.mean()


def test_apply_uses_the_fitted_thresholds_rather_than_refitting():
    rng = np.random.default_rng(2)
    n = 40
    correct = (rng.random((n, 2)) < 0.5).astype(float)
    cost = np.tile(np.array([1.0, 10.0]), (n, 1))
    score = np.column_stack([np.full(n, 0.8), np.full(n, 0.9)])
    fit = CascadeFit((0, 1), np.array([0.0, 1.0]), np.array([0.0]), 0.0, 0.0)
    # threshold 0.0 accepts nobody at stage 0, so everything escalates
    acc, spend, stage = apply_cascade(fit, correct, cost, score)
    assert (stage == 1).all()
    assert spend == pytest.approx(11.0)          # paid for both passes
    assert acc.tolist() == correct[:, 1].tolist()


def test_shape_disagreements_raise():
    correct = np.ones((5, 3))
    with pytest.raises(ValueError):
        fit_cascade(correct, np.ones((5, 2)), np.ones((5, 3)), budget=1.0, depth=2)
    with pytest.raises(ValueError):
        fit_cascade(correct, np.ones((5, 3)), np.ones((5, 3)), budget=1.0, depth=4)


def test_every_ordered_list_is_considered_not_just_cheapest_first():
    """The released code enumerates permutations; order is searched, not assumed.

    Here the dearer model is the better first stage, because its scorer is
    informative and the cheap model's is not. Cheapest-first would lose.
    """
    n = 120
    rng = np.random.default_rng(3)
    good = (rng.random(n) < 0.8).astype(float)
    poor = (rng.random(n) < 0.3).astype(float)
    correct = np.column_stack([poor, good])
    cost = np.tile(np.array([1.0, 2.0]), (n, 1))
    score = np.column_stack([np.full(n, 0.5),                       # uninformative
                             good * 0.95 + (1 - good) * 0.05])      # informative
    fit = fit_cascade(correct, cost, score, budget=3.0, depth=2)
    assert fit.models[0] == 1
