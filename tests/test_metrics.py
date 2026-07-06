"""Serving metrics: in-memory tallies and the decision-log record (no fastapi)."""

from dataclasses import dataclass

from cre_router.server.app import RouterMetrics, decision_record


@dataclass
class FakeMeta:
    cluster: int
    path: list
    p_accept: float | None = None

    @property
    def final_model(self):
        return self.path[-1]

    @property
    def escalated(self):
        return len(self.path) > 1


class TestRouterMetrics:
    def test_counts_and_escalation_rate(self):
        m = RouterMetrics()
        m.record(FakeMeta(0, ["weak"]))
        m.record(FakeMeta(0, ["weak", "strong"]))  # escalated
        m.record(FakeMeta(1, ["strong"]))
        snap = m.snapshot()
        assert snap["total_requests"] == 3
        assert snap["escalations"] == 1
        assert snap["escalation_rate"] == 1 / 3
        assert snap["by_cluster"] == {"0": 2, "1": 1}
        assert snap["by_final_model"] == {"weak": 1, "strong": 2}
        assert snap["by_path"] == {"weak": 1, "weak -> strong": 1, "strong": 1}

    def test_empty_snapshot_has_zero_rate(self):
        assert RouterMetrics().snapshot()["escalation_rate"] == 0.0

    def test_dropped_log_records_reported(self):
        m = RouterMetrics()
        m.log_dropped = 3
        assert m.snapshot()["decision_log_dropped"] == 3


class TestDecisionRecord:
    def test_shape(self):
        rec = decision_record(FakeMeta(2, ["weak", "mid"], p_accept=0.3))
        assert rec["cluster"] == 2
        assert rec["path"] == ["weak", "mid"]
        assert rec["final_model"] == "mid"
        assert rec["escalated"] is True
        assert rec["p_accept"] == 0.3
        assert isinstance(rec["ts"], float)
