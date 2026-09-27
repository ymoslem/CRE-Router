"""Every shipped routing spec is well formed, so `cre compose` can read it."""
from __future__ import annotations

from pathlib import Path

import pytest

from cre_router.compose import Routing

SPECS = sorted((Path(__file__).parents[1] / "configs" / "routings").glob("*.json"))


def test_there_are_routing_specs():
    assert SPECS


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_a_spec_is_consistent(path):
    r = Routing.from_json(path)
    assert r.name, "a spec names the system it describes"
    assert set(r.assign.values()) <= set(r.tiers), "every assigned model has a capture"
    assert set(r.gated) <= set(r.assign), "only a served cluster can be gated"
    assert 0 < r.tau < 1 and r.runs >= 1
    if r.gated:
        assert r.strong, "a gated system names its strong model"
