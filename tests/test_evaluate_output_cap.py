"""The task's output cap, and the one case that may lower it."""
import pytest

from cre_router.evaluate import TASKS, model_entry, RunMeasurement


def meas(**kw):
    d = dict(cluster="0", run=0, error=0.1, tpot_ms=10.0, num_prompts=5)
    d.update(kw)
    return RunMeasurement(**d)


def test_the_cap_is_recorded_on_the_stats_entry():
    """Two captures at different caps are not comparable, so the file says which."""
    entry = model_entry([meas()], max_output_tokens=38912)
    assert entry["max_output_tokens"] == 38912


def test_an_entry_without_a_cap_omits_the_key_rather_than_guessing():
    assert "max_output_tokens" not in model_entry([meas()])


def test_the_aime_cap_is_the_one_the_pool_was_measured_at():
    assert TASKS["aime"].max_tokens == 40960


@pytest.mark.parametrize("task", sorted(TASKS))
def test_every_task_caps_output(task):
    assert TASKS[task].max_tokens > 0
