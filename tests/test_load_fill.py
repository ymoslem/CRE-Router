"""Tests for the load filler that holds a server at a fixed in-flight count."""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from cre_router.evaluate import Task, evaluate_model
from cre_router.load_fill import (RUNNING, SUCCESS, WAITING, LoadFiller,
                                  fillers_to_launch, parse_metrics)


def test_parse_metrics_sums_over_labels_and_skips_comments():
    text = (
        "# HELP vllm:num_requests_running Number of requests\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:num_requests_running{engine="1",model_name="m"} 2.0\n'
        'vllm:request_success_total{finished_reason="stop",model_name="m"} 10.0\n'
        'vllm:request_success_total{finished_reason="length",model_name="m"} 4.0\n'
        "vllm:num_requests_waiting 1\n"
    )
    m = parse_metrics(text)
    assert m[RUNNING] == 5.0
    assert m[SUCCESS] == 14.0
    assert m[WAITING] == 1.0


def test_no_fillers_while_the_benchmark_may_still_have_requests_queued():
    # 100 measured requests at a limit of 32: its queue is empty only once 68 finished.
    assert fillers_to_launch(running=31, waiting=0, measured_done=67, n_measured=100, target=32) == 0
    assert fillers_to_launch(running=31, waiting=0, measured_done=68, n_measured=100, target=32) == 1


def test_fillers_top_up_to_target_and_never_go_negative():
    assert fillers_to_launch(3, 0, 0, n_measured=3, target=32) == 29
    assert fillers_to_launch(30, 1, 0, n_measured=3, target=32) == 1
    assert fillers_to_launch(33, 2, 0, n_measured=3, target=32) == 0


def test_request_body_chat_versus_pre_rendered():
    chat = LoadFiller("http://h:1/", "m", ["q"], n_measured=1, chat=True,
                      sampling={"max_tokens": 5, "top_k": 20, "min_p": None})
    url, body = chat.request("q")
    assert url == "http://h:1/v1/chat/completions"
    assert body == {"model": "m", "messages": [{"role": "user", "content": "q"}],
                    "stream": False, "max_tokens": 5, "top_k": 20}
    raw = LoadFiller("http://h:1", "m", ["<bos>q"], n_measured=1, chat=False,
                     sampling={"skip_special_tokens": False})
    url, body = raw.request("<bos>q")
    assert url == "http://h:1/v1/completions"
    assert body == {"model": "m", "prompt": "<bos>q", "stream": False, "skip_special_tokens": False}


class FakeServer:
    """Running count = measured requests in flight + fillers in flight."""

    def __init__(self, measured_running: int, success_base: float = 100.0):
        self.measured_running = measured_running
        self.success = success_base
        self.fillers = 0
        self.sent: list[str] = []

    async def fetch(self):
        return {RUNNING: float(self.measured_running + self.fillers), WAITING: 0.0,
                SUCCESS: self.success}

    async def send(self, prompt):
        self.sent.append(prompt)
        self.fillers += 1
        try:
            await asyncio.sleep(3600)
        finally:
            self.fillers -= 1


def test_filler_holds_target_replays_prompts_and_cancels_on_exit(tmp_path):
    srv = FakeServer(measured_running=3)
    trace = tmp_path / "fill" / "detail_x_c1_r0.json"
    with LoadFiller("http://h", "m", ["a", "b"], n_measured=3, target=8, poll_s=0.01,
                    fetch_metrics=srv.fetch, send=srv.send, trace_path=trace) as f:
        time.sleep(0.2)
        assert srv.fillers == 5          # 3 measured + 5 fillers = target
    assert srv.fillers == 0              # every filler cancelled on exit
    assert srv.sent[:4] == ["a", "b", "a", "b"]
    assert f.stats.launched == 5 and f.stats.cancelled == 5 and f.stats.finished == 0
    rec = json.loads(trace.read_text())
    assert rec["target"] == 8 and rec["n_measured"] == 3 and rec["launched"] == 5
    assert rec["trace"] and all(r["measured_done"] == 0 for r in rec["trace"])


def test_filler_error_is_raised_not_swallowed():
    async def broken():
        raise ConnectionError("no server")
    with pytest.raises(RuntimeError, match="failed to start"):
        with LoadFiller("http://h", "m", ["a"], n_measured=1, fetch_metrics=broken,
                        send=lambda p: None):
            pass


def _task(pre_rendered=False):
    return Task(name="t", parse=lambda s: s.strip(), temperature=0.6, top_p=0.95,
                top_k=20, min_p=0.0, max_tokens=64, pre_rendered=pre_rendered)


def _bench(path, model, task, **kw):
    n = sum(1 for line in open(path) if line.strip())
    return {"generated_texts": ["1"] * n, "mean_tpot_ms": 10.0, "output_lens": [1] * n}


def test_evaluate_model_wraps_each_run_in_a_filler(tmp_path):
    calls = []

    class Spy:
        def __init__(self, base_url, model, prompts, **kw):
            calls.append((base_url, model, list(prompts), kw))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    data = [{"prompt": "p1", "answer": "1", "cluster": "0"},
            {"prompt": "p2", "answer": "1", "cluster": "0"}]
    evaluate_model(data, "org/m", _task(), runs=2, port=9, workdir=tmp_path / "s",
                   benchmark=_bench, detail_dir=tmp_path / "d", fill_load=True, filler=Spy)
    assert len(calls) == 2
    base, model, prompts, kw = calls[0]
    assert base == "http://127.0.0.1:9" and model == "org/m" and prompts == ["p1", "p2"]
    assert kw["n_measured"] == 2 and kw["target"] == 32 and kw["chat"] is True
    assert kw["sampling"]["max_tokens"] == 64 and "skip_special_tokens" not in kw["sampling"]
    assert kw["trace_path"].parent == tmp_path / "d" / "fill"
    assert kw["trace_path"].name.endswith("_c0_r0.json")


def test_fill_load_requires_detail_dir(tmp_path):
    with pytest.raises(ValueError, match="detail_dir"):
        evaluate_model([{"prompt": "p", "answer": "1", "cluster": "0"}], "m", _task(),
                       runs=1, workdir=tmp_path, benchmark=_bench, fill_load=True)
