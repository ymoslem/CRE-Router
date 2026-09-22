"""The capture reader: grading, the request-to-question join, and refusals."""
from __future__ import annotations

import json

import pytest

from cre_router.captures import (
    CaptureError,
    load_capture,
    load_generations,
    task_of,
)
from cre_router.evaluate import _detail_name, TASKS


def write_capture(root, tag, task, model, rows, detail=None, cluster="0", run=0,
                  batch=True):
    """Write one capture the way `cre evaluate` leaves it on disk."""
    d = root / f"results_{tag}_{model.replace('/', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{task}_{model.replace('/', '_')}_20260101_000000"
    with (d / f"{stem}_generations.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if batch:
        with (d / f"{stem}.jsonl").open("w") as f:
            f.write(json.dumps({"cluster": cluster, "run": run, "error": 0.0,
                                "num_prompts": len(rows), "ttft_ms": 100.0,
                                "tpot_ms": 10.0, "e2el_ms": 300.0}) + "\n")
    if detail is None:
        detail = {
            "ttfts": [0.1] * len(rows),
            "itls": [[0.01] * (r["num_tokens"] - 1) for r in rows],
            "output_lens": [r["num_tokens"] for r in rows],
        }
    dd = root / f"detail_{tag}"
    dd.mkdir(parents=True, exist_ok=True)
    name = _detail_name(TASKS[task], model, cluster, run)
    (dd / name).write_text(json.dumps(detail))
    return d


def row(qid, gold, output, tokens=3, cluster="0", run=0, correct=True):
    return {"qid": qid, "cluster": cluster, "run": run, "num_tokens": tokens,
            "ground_truth_answer": gold, "full_output": output, "correct": correct}


class TestGrading:
    def test_task_comes_from_the_filename(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", [row("1", "385", r"\boxed{385}")])
        rows, report = load_generations("t", tmp_path)
        assert report["task"] == "aime"

    def test_aime_is_not_graded_by_telemath_tolerance(self, tmp_path):
        """384 is within 1% of 385: TeleMath's matcher accepts it, AIME's must not."""
        write_capture(tmp_path, "t", "aime", "Qwen/Q", [row("1", "385", r"\boxed{384}")])
        rows, _ = load_generations("t", tmp_path)
        assert rows[0]["correct"] is False

    def test_telemath_keeps_its_tolerance(self, tmp_path):
        write_capture(tmp_path, "t", "telemath", "Qwen/Q",
                      [row("1", "1000", r"\boxed{1005}")])
        rows, _ = load_generations("t", tmp_path)
        assert rows[0]["correct"] is True

    def test_stored_verdict_is_kept_but_not_trusted(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q",
                      [row("1", "385", r"\boxed{384}", correct=True)])
        rows, report = load_generations("t", tmp_path)
        assert (rows[0]["correct"], rows[0]["correct_as_run"]) == (False, True)
        assert report["regraded_flips"] == 1

    def test_unknown_task_is_refused(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", [row("1", "1", "x")])
        bad = next((tmp_path / "results_t_Qwen_Q").glob("*_generations.jsonl"))
        bad.rename(bad.with_name("mystery_Qwen_Q_1_generations.jsonl"))
        with pytest.raises(CaptureError, match="no task in the filename"):
            load_generations("t", tmp_path)


class TestJoin:
    def test_cost_comes_from_the_matching_request(self, tmp_path):
        rows = [row("1", "385", r"\boxed{385}", tokens=3),
                row("2", "12", r"\boxed{12}", tokens=5)]
        detail = {"ttfts": [0.2, 0.4],
                  "itls": [[0.01, 0.01], [0.02, 0.02, 0.02, 0.02]],
                  "output_lens": [3, 5]}
        write_capture(tmp_path, "t", "aime", "Qwen/Q", rows, detail)
        cap = load_capture("t", tmp_path)
        assert cap[("1", 0)]["tpot_ms"] == pytest.approx(10.0)
        assert cap[("2", 0)]["e2el_s"] == pytest.approx(0.48)
        assert cap[("2", 0)]["tokens"] == 5

    def test_a_mismatched_detail_file_is_refused(self, tmp_path):
        """The detail file has no qid, so a wrong one must be caught on length."""
        rows = [row("1", "385", r"\boxed{385}", tokens=3)]
        detail = {"ttfts": [0.2], "itls": [[0.01] * 8], "output_lens": [9]}
        write_capture(tmp_path, "t", "aime", "Qwen/Q", rows, detail)
        with pytest.raises(CaptureError, match="positional join does not hold"):
            load_capture("t", tmp_path)

    def test_a_missing_detail_file_is_refused(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", [row("1", "1", "x")])
        for f in (tmp_path / "detail_t").glob("*.json"):
            f.unlink()
        with pytest.raises(CaptureError, match="cannot carry a cost"):
            load_capture("t", tmp_path)

    def test_single_token_answers_report_no_tpot(self, tmp_path):
        rows = [row("1", "385", r"\boxed{385}", tokens=1)]
        detail = {"ttfts": [0.2], "itls": [[]], "output_lens": [1]}
        write_capture(tmp_path, "t", "aime", "Qwen/Q", rows, detail)
        assert load_capture("t", tmp_path)[("1", 0)]["tpot_ms"] == 0.0


class TestBatchOnly:
    """Runs that kept only batch means: refused by default, readable on request."""

    def rows(self):
        return [row("1", "385", r"\boxed{385}"), row("2", "12", r"\boxed{12}")]

    def capture(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", self.rows())
        for f in (tmp_path / "detail_t").glob("*.json"):
            f.unlink()
        return tmp_path

    def test_refused_by_default(self, tmp_path):
        with pytest.raises(CaptureError, match="cannot carry a cost"):
            load_capture("t", self.capture(tmp_path))

    def test_read_on_request_with_batch_means(self, tmp_path):
        cap = load_capture("t", self.capture(tmp_path), require_cost=False)
        assert cap[("1", 0)]["batch_tpot_ms"] == 10.0
        assert cap[("1", 0)]["batch_e2el_ms"] == 300.0
        assert cap[("1", 0)]["tpot_ms"] is None
        assert cap[("1", 0)]["correct"] is True

    def test_batch_means_ride_along_with_per_request_cost(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", self.rows())
        cap = load_capture("t", tmp_path)
        assert cap[("1", 0)]["batch_tpot_ms"] == 10.0
        assert cap[("1", 0)]["tpot_ms"] == pytest.approx(10.0)

    def test_a_batch_of_the_wrong_size_is_refused(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", self.rows())
        d = next(tmp_path.glob("results_t_*"))
        f = d / "aime_Qwen_Q_20260101_000000.jsonl"
        f.write_text(json.dumps({"cluster": "0", "run": 0, "num_prompts": 7,
                                 "ttft_ms": 1.0, "tpot_ms": 1.0, "e2el_ms": 1.0}) + "\n")
        with pytest.raises(CaptureError, match="served 7 prompts"):
            load_capture("t", tmp_path)

    def test_a_missing_batch_file_is_refused(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", self.rows(), batch=False)
        with pytest.raises(CaptureError, match="expected one batch-means file"):
            load_capture("t", tmp_path)


class TestShape:
    def test_expectations_refuse_a_partial_capture(self, tmp_path):
        write_capture(tmp_path, "t", "aime", "Qwen/Q", [row("1", "1", "x")])
        with pytest.raises(CaptureError, match="1 questions, expected 2"):
            load_capture("t", tmp_path, expect_questions=2)
        with pytest.raises(CaptureError, match=r"runs \[0\], expected \[0, 1\]"):
            load_capture("t", tmp_path, expect_runs=(0, 1))

    def test_a_longer_tag_is_not_matched(self, tmp_path):
        write_capture(tmp_path, "ours_eff_memm", "aime", "Qwen/Q", [row("1", "1", "x")])
        with pytest.raises(CaptureError, match="found 0"):
            load_capture("ours_eff", tmp_path)
