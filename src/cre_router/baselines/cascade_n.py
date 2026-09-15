"""Cost accounting for a cascade of more than two stages.

`cre_router.cascade.per_request_metrics` prices *our* cascade, which is two
tiers by construction: one efficient model per cluster and one escalation
target. FrugalGPT learns an ordered list of up to `depth` models, so a query can
run three passes. This module generalises the same arithmetic to any depth and
lives under `baselines/` because that is whose requirement it is; the router's
own accounting is not changed to accommodate a baseline.

**The rule is the one `cre_router.cascade` documents, unchanged.** vLLM defines
TPOT as ``(e2el - ttft) / (tokens - 1)``, so a pass spends ``tpot * (tokens - 1)``
decoding. A cascade's TPOT is therefore total decode time over the tokens the
user actually received, which are the answering stage's::

    sum_i tpot_i * (n_i - 1)  /  (n_answer - 1)

Every pass is charged in full, exactly as E2EL adds every pass's wall time, but
a discarded pass contributes no tokens to spread its time over. Summing each
pass's TPOT is a different and wrong quantity; that error is what cre-router
v0.3.1 was released to fix, and reproducing it here would put it straight back.

At depth 2 this must agree with `per_request_metrics` exactly, and
`test_baseline_cascade_n.py` asserts that on random inputs. If the two ever
disagree, the router's version is right and this one is wrong.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = ["FIELDS", "n_stage_metrics"]

#: Every stage grid must carry these, per (question, run).
FIELDS = ("correct", "e2el", "tokens", "tpot")


def n_stage_metrics(
    answered_at: np.ndarray,
    first: Mapping[str, np.ndarray],
    later: Sequence[Sequence[Mapping[str, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correctness, E2EL and TPOT on a (question, first run, later run) grid.

    ``answered_at`` is (question, first run) and holds the zero-based index of
    the stage that answered, so 0 means stage 1 kept its own answer.

    ``first`` is the stage 1 grids, (question, first run): that stage serves
    every query, and its run index is the one the scorer's decision was made
    from. ``later[i]`` holds one grid per first-run ``r`` for stage ``i + 2``,
    each (question, later run), because each escalated subset was served as its
    own batch once per ``r``.

    The two run axes are deliberately not symmetric. A query that never
    escalates is priced on ``r`` and broadcast, matching how the router's own
    accounting treats a query the gate accepts.
    """
    answered_at = np.asarray(answered_at, dtype=int)
    n_q, n_first = answered_at.shape
    depth = len(later) + 1
    if answered_at.max(initial=0) > depth - 1:
        raise ValueError(f"a query is answered at stage {answered_at.max() + 1}, "
                         f"but only {depth} stages were given")
    for f in FIELDS:
        if f not in first:
            raise ValueError(f"stage 1 grid is missing {f!r}")
        if np.shape(first[f]) != (n_q, n_first):
            raise ValueError(f"stage 1 field {f!r} has shape {np.shape(first[f])}, "
                             f"expected {(n_q, n_first)}")
    for i, per_run in enumerate(later):
        if len(per_run) != n_first:
            raise ValueError(f"stage {i + 2} has {len(per_run)} run grids, "
                             f"expected {n_first}")

    # A stage the fit never reaches has no grid for that run, and at a loose
    # enough budget it is never reached for ANY run: the cascade degenerates to
    # its earlier stages, which price it exactly. So the later-run axis is sized
    # from the first grid that exists rather than from `later[0][0]`, which is
    # None in exactly that case.
    grids = [g for per_run in later for g in per_run if g is not None]
    n_later = np.shape(grids[0]["correct"])[1] if grids else n_first
    for g in grids:
        for f in FIELDS:
            if f not in g:
                raise ValueError(f"a later-stage grid is missing {f!r}")
            if np.shape(g[f]) != (n_q, n_later):
                raise ValueError(f"a later-stage field {f!r} has shape "
                                 f"{np.shape(g[f])}, expected {(n_q, n_later)}")

    def bcast(a):
        return np.repeat(np.asarray(a, dtype=float)[:, :, None], n_later, axis=2)

    # Stage 1 runs for every query, so it seeds both the decode time and the
    # delivered-token denominator.
    n_first_tok = np.maximum(bcast(first["tokens"]) - 1, 1)
    decode = bcast(first["tpot"]) * n_first_tok
    delivered = n_first_tok.copy()
    e2el = bcast(first["e2el"])
    correct = bcast(first["correct"])

    for i, per_run in enumerate(later):
        stage = i + 1
        for r, g in enumerate(per_run):
            if g is None:
                continue
            ran = (answered_at[:, r] >= stage)[:, None]
            n_tok = np.maximum(np.asarray(g["tokens"], dtype=float) - 1, 1)
            decode[:, r, :] += np.where(ran, np.asarray(g["tpot"], dtype=float) * n_tok, 0.0)
            e2el[:, r, :] += np.where(ran, np.asarray(g["e2el"], dtype=float), 0.0)
            # The answering stage's tokens are the ones delivered, so a later
            # stage REPLACES the denominator rather than adding to it.
            delivered[:, r, :] = np.where(ran, n_tok, delivered[:, r, :])
            correct[:, r, :] = np.where(ran, np.asarray(g["correct"], dtype=float),
                                        correct[:, r, :])

    return correct, e2el, decode / delivered
