"""The Stage 2 plan `cre fit` derives reproduces every shipped routing spec.

Each spec records the escalation target and the gated clusters of a reported
system. `stage2_plan` must derive both from the fitted stats alone: the target
is the most accurate model the routing uses, and every cluster served by a
cheaper model is gated. The k = 1 ablations route to one model and borrow the
full system's target, which `escalate_to` supplies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cre_router.compose import Routing
from cre_router.routing import ModelStats, models_from_stats, stage2_plan

CONFIGS = Path(__file__).parents[1] / "configs"
SPECS = sorted((CONFIGS / "routings").glob("*.json"))
#: Stats-file names for the short names the TeleMath and TeleQnA specs use.
ALIAS = {"Q30B": "Qwen3-30B-A3B-Thinking-2507", "E2B": "Gemma4-E2B", "26B": "Gemma4-26B-A4B",
         "26B-think": "Gemma4-26B-A4B-think", "E4B-think": "Gemma4-E4B-think",
         "Q4-Instruct": "Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct": "Qwen3-4B"}


def _fit_inputs(path: Path):
    bench, *_rest = path.stem.split("_")
    term = "e2el" if "_e2el_" in path.stem else "tpot"
    cards = next(p for p in path.stem.split("_") if p.endswith("A100"))
    stats = json.loads((CONFIGS / f"{bench}_stats_{cards}_Sep2026.json").read_text())
    return models_from_stats(stats, term)


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_plan_reproduces_the_spec(path):
    spec = Routing.from_json(path)
    models, sizes = _fit_inputs(path)
    name = lambda n: ALIAS.get(n, n)  # noqa: E731
    assign = {c: name(m) for c, m in spec.assign.items()}
    # A one-model routing has nothing of its own to escalate to; the ablation
    # names the full system's target, as `cre fit --escalate-to` would.
    override = name(spec.strong) if len(set(assign.values())) == 1 and spec.strong else None
    plan = stage2_plan(models, assign, sizes, escalate_to=override)
    assert plan.target == (name(spec.strong) if spec.gated else None)
    assert set(plan.gated) == set(spec.gated)


def _m(name, cost, err):
    return ModelStats(name=name, tpot_ms=cost, errors={"0": err, "1": err})


SIZES = {"0": 10, "1": 10}


def test_target_is_the_most_accurate_model_used():
    models = [_m("small", 5, 0.5), _m("mid", 10, 0.3), _m("big", 30, 0.1)]
    plan = stage2_plan(models, {"0": "small", "1": "mid"}, SIZES)
    # "big" is more accurate but unused, so the target is "mid".
    assert plan.target == "mid" and plan.gated == {"0": "small"}


def test_a_costlier_model_is_never_gated():
    # "slow" is less accurate than "best" but costs more, so it is not gated.
    models = [_m("best", 10, 0.1), _m("slow", 20, 0.2)]
    plan = stage2_plan(models, {"0": "best", "1": "slow"}, SIZES)
    assert plan.target is None and plan.gated == {}


def test_one_model_routing_has_no_stage2_without_an_override():
    models = [_m("small", 5, 0.5), _m("big", 30, 0.1)]
    assert stage2_plan(models, {"0": "small", "1": "small"}, SIZES).target is None
    plan = stage2_plan(models, {"0": "small", "1": "small"}, SIZES, escalate_to="big")
    assert plan.target == "big" and plan.estimators == {"small": ["0", "1"]}


def test_accuracy_ties_go_to_the_cheaper_model():
    three = {"0": 10, "1": 10, "2": 10}
    models = [ModelStats(name=n, tpot_ms=c, errors=dict.fromkeys(three, e))
              for n, c, e in (("a", 5, 0.2), ("b", 9, 0.2), ("c", 3, 0.6))]
    plan = stage2_plan(models, {"0": "a", "1": "b", "2": "c"}, three)
    assert plan.target == "a" and plan.gated == {"2": "c"}


def test_unknown_override_is_refused():
    with pytest.raises(ValueError, match="not in the stats"):
        stage2_plan([_m("a", 5, 0.2)], {"0": "a", "1": "a"}, SIZES, escalate_to="zzz")
