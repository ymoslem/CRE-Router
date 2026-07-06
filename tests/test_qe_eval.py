"""QE classifier evaluation metrics (paper Appendix D), pure and GPU-free."""

from cre_router.qe.evaluate import qe_metrics, to_binary_label


class TestToBinaryLabel:
    def test_continue_folds_to_route(self):
        assert to_binary_label(2) == 0

    def test_route_and_accept_unchanged(self):
        assert to_binary_label(0) == 0
        assert to_binary_label(1) == 1

    def test_accepts_strings(self):
        assert to_binary_label("2") == 0

    def test_accepts_float_formatted_labels(self):
        # released datasets store the label as "0.0" / "1.0" / "2.0"
        assert to_binary_label("0.0") == 0
        assert to_binary_label("1.0") == 1
        assert to_binary_label("2.0") == 0
        assert to_binary_label(1.0) == 1


class TestQEMetrics:
    def test_perfect_predictions(self):
        y = [1, 1, 0, 0]
        m = qe_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["false_escalations"] == 0
        assert m["missed_escalations"] == 0

    def test_escalation_confusion_counts(self):
        # true: accept, accept, route, route
        # pred: accept, route,  route, accept
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        m = qe_metrics(y_true, y_pred)
        assert m["n"] == 4
        assert m["n_accept_true"] == 2
        # one correct answer escalated unnecessarily (accept -> route)
        assert m["false_escalations"] == 1
        # one wrong answer wrongly accepted (route -> accept)
        assert m["missed_escalations"] == 1
        # one true escalation (route -> route)
        assert m["true_escalations"] == 1
        assert m["escalations"] == 2  # true + false
        assert m["accuracy"] == 0.5
