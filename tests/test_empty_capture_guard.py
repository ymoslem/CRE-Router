"""A capture where every request failed must not be returned as a result."""
import pytest

from cre_router.evaluate import (
    EmptyCaptureError,
    RunMeasurement,
    assert_capture_generated_something,
)


def meas(tokens, **kw):
    d = dict(cluster="0", run=0, error=0.1, tpot_ms=10.0, num_prompts=5,
             mean_output_tokens=tokens)
    d.update(kw)
    return RunMeasurement(**d)


def test_a_capture_with_no_generated_tokens_is_refused():
    """The failure this guards: every request rejected on length.

    vLLM reports each rejection, the harness averages them, and the stats file
    reads error 1.0 at 0.0 ms. Nothing downstream can tell it apart from a real
    result, so it has to be caught here.
    """
    caps = [meas(0.0, error=1.0, tpot_ms=0.0, run=r) for r in range(5)]
    with pytest.raises(EmptyCaptureError, match="0 output tokens"):
        assert_capture_generated_something(caps)


def test_a_normal_capture_passes():
    assert_capture_generated_something([meas(1200.0), meas(950.0)]) is None


def test_one_empty_cluster_among_good_ones_is_allowed():
    """Only a wholly empty capture is refused.

    A single (cluster, run) can legitimately generate nothing at a tight cap;
    refusing on that would block real runs.
    """
    assert_capture_generated_something([meas(0.0), meas(1200.0)]) is None


def test_a_capture_that_reports_no_token_counts_is_not_refused():
    """Older captures predate mean_output_tokens; absence is not emptiness."""
    assert_capture_generated_something([meas(None), meas(None)]) is None


def test_no_measurements_at_all_is_not_this_error():
    assert_capture_generated_something([]) is None
