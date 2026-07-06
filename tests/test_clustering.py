"""Clustering and artifact round-trip tests on synthetic data."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from cre_router.artifacts import RouterArtifacts
from cre_router.clustering import assign_clusters, fit_clusters


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=0)
    return X, y


class TestFitClusters:
    def test_silhouette_selects_true_k(self, blobs):
        X, _ = blobs
        result = fit_clusters(X, k_range=range(2, 6))
        assert result.k == 3
        assert set(result.silhouette_by_k) == {2, 3, 4, 5}

    def test_forced_k_skips_selection(self, blobs):
        X, _ = blobs
        result = fit_clusters(X, k=4)
        assert result.k == 4
        assert result.silhouette_by_k == {}

    def test_assign_matches_training_labels(self, blobs):
        X, _ = blobs
        result = fit_clusters(X, k=3)
        assert (assign_clusters(X, result.centroids) == result.labels).all()


class TestArtifacts:
    def test_round_trip(self, tmp_path, blobs):
        X, _ = blobs
        result = fit_clusters(X, k=3)
        artifacts = RouterArtifacts(
            embedding_model="test-model",
            centroids=result.centroids,
            routing_table={"0": "a", "1": "b", "2": "a"},
            lambda_star=0.06,
            budget_ms=20.0,
            stats={"cluster_sizes": {"0": 100, "1": 100, "2": 100}},
        )
        artifacts.save(tmp_path)
        loaded = RouterArtifacts.load(tmp_path)
        assert loaded.embedding_model == "test-model"
        assert loaded.routing_table == artifacts.routing_table
        assert loaded.lambda_star == 0.06
        np.testing.assert_allclose(loaded.centroids, result.centroids)
