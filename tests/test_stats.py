"""Stats files rebuilt from captures: one measurement per batch, the same path as cre evaluate."""
from __future__ import annotations

import pytest

from cre_router.stats import build_stats, load_clusters, measurements, relabel


def capture(errors_by_cluster, runs=2, per_cluster=4, tpot=10.0):
    """Questions q<c><i>; in each cluster the first `wrong` answers are wrong."""
    cap = {}
    for c, wrong in errors_by_cluster.items():
        for r in range(runs):
            for i in range(per_cluster):
                cap[(f"q{c}{i}", r)] = {
                    "cluster": c, "correct": i >= wrong,
                    "batch_tpot_ms": tpot + r, "batch_ttft_ms": 100.0, "batch_e2el_ms": 900.0 + r}
    return cap


def test_one_measurement_per_cluster_and_run_with_error_regraded():
    ms = measurements(capture({"0": 1, "1": 2}))
    assert [(m.cluster, m.run) for m in ms] == [("0", 0), ("0", 1), ("1", 0), ("1", 1)]
    assert [m.error for m in ms] == [0.25, 0.25, 0.5, 0.5]
    assert ms[1].tpot_ms == 11.0 and ms[1].e2el_ms == 901.0


def test_rows_of_one_batch_must_share_its_means():
    cap = capture({"0": 0})
    cap[("q00", 0)] = {**cap[("q00", 0)], "batch_tpot_ms": 99.0}
    with pytest.raises(ValueError, match="not served as one batch"):
        measurements(cap)


def test_build_stats_averages_runs_like_cre_evaluate():
    pool = {"name": "p", "_note": "n", "models": {"A": "a", "B": "b"}}
    caps = {"a": capture({"0": 1, "1": 2}), "b": capture({"0": 0, "1": 4}, tpot=20.0)}
    stats = build_stats(pool, caps.__getitem__)
    assert stats["cluster_sizes"] == {"0": 4, "1": 4}
    assert stats["models"]["A"]["errors"] == {"0": 0.25, "1": 0.5}
    assert stats["models"]["A"]["cluster_tpot_ms"] == {"0": 10.5, "1": 10.5}
    assert stats["models"]["B"]["cluster_e2el_ms"] == {"0": 900.5, "1": 900.5}
    assert stats["_note"] == "n" and stats["built_from"] == {"A": "a", "B": "b"}


def test_models_over_different_questions_are_refused():
    pool = {"models": {"A": "a", "B": "b"}}
    caps = {"a": capture({"0": 0}), "b": capture({"0": 0}, per_cluster=5)}
    with pytest.raises(ValueError, match="cluster sizes"):
        build_stats(pool, caps.__getitem__)


def test_a_run_missing_questions_is_refused():
    cap = capture({"0": 0})
    del cap[("q01", 1)]
    with pytest.raises(ValueError, match="not every run answered every question"):
        build_stats({"models": {"A": "a"}}, {"a": cap}.__getitem__)


# ---------------------------------------------------------------- one batch

def one_batch(runs=2):
    """Clusters 0 and 1 served together as one batch: per-request costs differ by cluster."""
    cap = {}
    for r in range(runs):
        for c, tpot, wrong in (("0", 10.0, 1), ("1", 30.0, 2)):
            for i in range(4):
                cap[(f"q{c}{i}", r)] = {
                    "cluster": c, "batch": "all", "stage": 1, "correct": i >= wrong,
                    "tpot_ms": tpot + i + r, "ttft_s": 0.1, "e2el_s": 1.0 + i,
                    "batch_tpot_ms": 99.0, "batch_ttft_ms": 99.0, "batch_e2el_ms": 99.0}
    return cap


def test_a_one_batch_capture_prices_each_cluster_by_its_own_requests():
    ms = measurements(one_batch())
    assert [(m.cluster, m.run) for m in ms] == [("0", 0), ("0", 1), ("1", 0), ("1", 1)]
    assert [m.error for m in ms] == [0.25, 0.25, 0.5, 0.5]
    assert ms[0].tpot_ms == 11.5 and ms[3].tpot_ms == 32.5       # not the batch's 99
    assert ms[0].e2el_ms == pytest.approx(2500.0) and ms[0].ttft_ms == pytest.approx(100.0)


def test_per_cluster_means_weight_back_to_the_batch_mean():
    cap = one_batch()
    for (q, r), row in cap.items():
        row["batch_tpot_ms"] = sum(v["tpot_ms"] for (_, s), v in cap.items() if s == r) / 8
    ms = [m for m in measurements(cap) if m.run == 0]
    assert sum(m.tpot_ms * m.num_prompts for m in ms) / 8 == pytest.approx(cap[("q00", 0)]["batch_tpot_ms"])


def test_a_one_batch_capture_without_request_detail_is_refused():
    cap = one_batch()
    cap[("q00", 0)] = {**cap[("q00", 0)], "tpot_ms": None}
    with pytest.raises(ValueError, match="per-request detail"):
        measurements(cap)


def test_escalated_rows_are_refused_in_training_statistics():
    cap = one_batch()
    cap[("q00", 0)] = {**cap[("q00", 0)], "stage": 2, "eff_run": 0}
    with pytest.raises(ValueError, match="escalated rows"):
        measurements(cap)


def test_one_batch_and_per_cluster_rows_are_not_mixed_in_a_cluster():
    cap = one_batch()
    cap[("q00", 0)] = {**cap[("q00", 0)], "batch": None}
    with pytest.raises(ValueError, match="some rows come from one batch"):
        measurements(cap)


def test_a_one_batch_capture_is_priced_under_another_clustering():
    k1 = {q: "0" for q, _ in one_batch()}
    ms = measurements(relabel(one_batch(), k1))
    assert [(m.cluster, m.run, m.num_prompts) for m in ms] == [("0", 0, 8), ("0", 1, 8)]
    assert ms[0].error == pytest.approx(3 / 8)
    assert ms[0].tpot_ms == pytest.approx((11.5 * 4 + 31.5 * 4) / 8)


def test_build_stats_with_a_clustering_counts_every_question_once():
    k1 = {q: "0" for q, _ in one_batch()}
    stats = build_stats({"models": {"A": "a"}}, {"a": one_batch()}.__getitem__, k1)
    assert stats["cluster_sizes"] == {"0": 8}
    assert stats["models"]["A"]["errors"] == {"0": pytest.approx(3 / 8)}


def test_per_cluster_captures_cannot_be_relabelled():
    with pytest.raises(ValueError, match="only a capture served as one batch"):
        relabel(capture({"0": 0}), {f"q0{i}": "1" for i in range(4)})


@pytest.mark.parametrize("drop, extra, match", [(True, False, "has no cluster"),
                                                (False, True, "does not hold")])
def test_a_clustering_must_cover_exactly_the_captured_questions(drop, extra, match):
    k = {q: "0" for q, _ in one_batch()}
    if drop:
        k.pop("q00")
    if extra:
        k["zz"] = "0"
    with pytest.raises(ValueError, match=match):
        relabel(one_batch(), k)


def test_load_clusters_reads_ids_and_refuses_duplicates(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "a", "cluster": 1}\n{"id": "b", "cluster": 0}\n')
    assert load_clusters(p) == {"a": "1", "b": "0"}
    p.write_text('{"id": "a", "cluster": 1}\n{"id": "a", "cluster": 0}\n')
    with pytest.raises(ValueError, match="twice"):
        load_clusters(p)
