"""Stage 1: semantic clustering of queries (paper Sec. 4.2).

Queries are embedded with a sentence-transformer (all-MiniLM-L6-v2 in the
paper) and partitioned with k-means; the cluster count is selected by
maximising the mean Silhouette score. Centroids are fixed on training data;
at inference each query is assigned to its nearest centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)
    )


@dataclass
class ClusteringResult:
    k: int
    centroids: np.ndarray
    labels: np.ndarray
    silhouette_by_k: dict[int, float]
    inertia_by_k: dict[int, float]


def fit_clusters(
    embeddings: np.ndarray,
    k_range: range = range(2, 10),
    k: int | None = None,
    seed: int = 0,
    n_init: int = 20,
) -> ClusteringResult:
    """Fit k-means, selecting k by max Silhouette score unless ``k`` is forced.

    The default range covers k in [2, 9] inclusive, matching the clustering
    notebook used for the paper.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    silhouette_by_k: dict[int, float] = {}
    inertia_by_k: dict[int, float] = {}
    if k is None:
        for candidate in k_range:
            km = KMeans(n_clusters=candidate, random_state=seed, n_init=n_init)
            labels = km.fit_predict(embeddings)
            silhouette_by_k[candidate] = float(silhouette_score(embeddings, labels))
            inertia_by_k[candidate] = float(km.inertia_)
        k = max(silhouette_by_k, key=silhouette_by_k.__getitem__)

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(embeddings)
    return ClusteringResult(
        k=k,
        centroids=km.cluster_centers_,
        labels=km.labels_,
        silhouette_by_k=silhouette_by_k,
        inertia_by_k=inertia_by_k,
    )


def assign_clusters(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid assignment (Euclidean, identical to KMeans.predict)."""
    distances = np.linalg.norm(
        np.asarray(embeddings)[:, None, :] - np.asarray(centroids)[None, :, :], axis=2
    )
    return distances.argmin(axis=1)
