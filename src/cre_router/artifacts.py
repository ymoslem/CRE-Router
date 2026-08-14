"""Persistence for router artifacts produced offline and consumed at serving time.

An artifacts directory contains:
  centroids.npy  -- k-means centroids in embedding space (``cre cluster``)
  router.json    -- embedding model, routing table, lambda*, budget, cost metric,
                    and pool stats (``cre fit``)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cre_router.clustering import DEFAULT_EMBEDDING_MODEL

CENTROIDS_FILE = "centroids.npy"
ROUTER_FILE = "router.json"


@dataclass
class RouterArtifacts:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    centroids: np.ndarray | None = None
    routing_table: dict[str, str] = field(default_factory=dict)
    lambda_star: float | None = None
    budget_ms: float | None = None
    # Which measurement ``cre fit`` treated as Cost. Serving reads it back so the
    # escalation ladder is ordered by the same metric that produced the routing
    # table. Absent from artifacts written before this was recorded, hence the
    # default, which is also the fit default.
    cost_metric: str = "tpot"
    stats: dict = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.centroids is not None:
            np.save(directory / CENTROIDS_FILE, np.asarray(self.centroids))
        payload = asdict(self)
        payload.pop("centroids")
        (directory / ROUTER_FILE).write_text(json.dumps(payload, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "RouterArtifacts":
        directory = Path(directory)
        payload = json.loads((directory / ROUTER_FILE).read_text())
        centroids_path = directory / CENTROIDS_FILE
        centroids = np.load(centroids_path) if centroids_path.exists() else None
        return cls(centroids=centroids, **payload)
