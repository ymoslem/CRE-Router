"""Cluster, Route, Escalate: cascaded framework for cost-aware LLM serving."""

__version__ = "0.2.0"

from cre_router.routing import (  # noqa: F401
    ModelStats,
    Region,
    assign,
    crossover_candidates,
    eta,
    normalized_costs,
    pareto_prune,
    routing_regions,
    select_lambda,
    system_metrics,
)
