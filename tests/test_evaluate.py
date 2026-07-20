"""Stage A measurement: pure functions plus an end-to-end per-cluster run with
an injected fake benchmark (no GPU, no vLLM, no server)."""

import json

import pytest

from cre_router.evaluate import (
    TASKS,
    RunMeasurement,
    aggregate_runs,
    answers_match,
    cluster_sizes,
    evaluate_model,
    merge_model_into_stats,
    model_entry,
    parse_aime_answer,
    parse_telelogs_answer,
    parse_teleqna_answer,
    score_generations,
    split_by_cluster,
    split_thinking,
)


class TestAnswerParsing:
    def test_boxed_wins(self):
        assert parse_aime_answer("blah \\boxed{204} end") == 204

    def test_answer_label(self):
        assert parse_aime_answer("Answer: 42") == 42

    def test_last_short_integer_fallback(self):
        assert parse_aime_answer("the result is 17.") == 17

    def test_ignores_long_numbers(self):
        assert parse_aime_answer("id 123456 answer 7") == 7

    def test_none_when_no_number(self):
        assert parse_aime_answer("no digits here") is None

    def test_strips_thinking_block(self):
        assert split_thinking("<think>99 is wrong</think> Answer: 3") == "Answer: 3"

    def test_aime_ignores_reasoning_numbers(self):
        assert parse_aime_answer("<think>try 500 then 600</think>\\boxed{7}") == 7

    def test_teleqna_answer_label(self):
        assert parse_teleqna_answer("Explanation: foo\nAnswer: 2") == 2


class TestTeleLogsParsing:
    def test_boxed_number(self):
        assert parse_telelogs_answer("The root cause is \\boxed{3}.") == 3

    def test_boxed_tolerates_leading_c(self):
        assert parse_telelogs_answer("Final answer: \\boxed{C7}") == 7

    def test_answer_label_fallback(self):
        assert parse_telelogs_answer("Reasoning...\nAnswer: C2") == 2

    def test_cause_reference_last_resort(self):
        assert parse_telelogs_answer("Ruling out C1, the cause is C5.") == 5

    def test_ignores_out_of_range_and_data_numbers(self):
        # PCI 737, throughput 600 Mbps and a timestamp must not be read as the
        # answer; only the boxed 4 counts.
        text = "PCI 737 drops to 600 at 10:25:34; cause \\boxed{4}"
        assert parse_telelogs_answer(text) == 4

    def test_none_when_no_valid_label(self):
        assert parse_telelogs_answer("throughput 737 at PCI 919") is None

    def test_strips_thinking_block(self):
        assert parse_telelogs_answer("<think>maybe C1 or C2</think>\\boxed{6}") == 6


class TestAnswersMatch:
    def test_int_string_equivalence(self):
        assert answers_match(42, "42")

    def test_mismatch(self):
        assert not answers_match(42, 43)

    def test_none_predicted(self):
        assert not answers_match(None, 1)


class TestScoreGenerations:
    def test_error_rate_and_flags(self):
        texts = ["Answer: 1", "Answer: 9", "Answer: 3"]
        gold = [1, 2, 3]
        error, correct = score_generations(texts, gold, parse_teleqna_answer)
        assert correct == [True, False, True]
        assert error == pytest.approx(1 / 3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="generations"):
            score_generations(["Answer: 1"], [1, 2], parse_teleqna_answer)


class TestAggregation:
    def _measurements(self):
        return [
            RunMeasurement("0", 0, error=0.2, tpot_ms=10.0, num_prompts=5),
            RunMeasurement("0", 1, error=0.4, tpot_ms=12.0, num_prompts=5),
            RunMeasurement("1", 0, error=0.1, tpot_ms=30.0, num_prompts=3),
        ]

    def test_runs_averaged_per_cluster(self):
        agg = aggregate_runs(self._measurements())
        assert agg["0"]["error"] == pytest.approx(0.3)
        assert agg["0"]["tpot_ms"] == pytest.approx(11.0)
        assert agg["1"]["tpot_ms"] == pytest.approx(30.0)

    def test_model_entry_shape(self):
        entry = model_entry(self._measurements())
        assert set(entry) == {"errors", "cluster_tpot_ms"}
        assert entry["errors"]["0"] == pytest.approx(0.3)

    def test_split_and_sizes(self):
        dataset = [
            {"cluster": 0, "prompt": "a"},
            {"cluster": 1, "prompt": "b"},
            {"cluster": 0, "prompt": "c"},
        ]
        assert set(split_by_cluster(dataset)) == {"0", "1"}
        assert cluster_sizes(dataset) == {"0": 2, "1": 1}


class TestMerge:
    def test_merges_models_and_rewrites_sizes(self, tmp_path):
        path = tmp_path / "stats.json"
        merge_model_into_stats(path, "modelA", {"errors": {"0": 0.1}}, {"0": 5})
        merge_model_into_stats(path, "modelB", {"errors": {"0": 0.2}}, {"0": 5})
        stats = json.loads(path.read_text())
        assert set(stats["models"]) == {"modelA", "modelB"}
        assert stats["cluster_sizes"] == {"0": 5}


class TestEvaluateModelWithFakeBenchmark:
    def test_per_cluster_runs_and_stats(self, tmp_path):
        dataset = [
            {"prompt": "q0", "answer": 1, "cluster": 0},
            {"prompt": "q1", "answer": 2, "cluster": 0},
            {"prompt": "q2", "answer": 9, "cluster": 1},
        ]
        calls = []

        def fake_benchmark(dataset_path, model, task, *, host, port, max_concurrency, seed, download_dir):
            # Read the split vLLM would have been given and answer each prompt.
            rows = [json.loads(line) for line in open(dataset_path)]
            calls.append((str(dataset_path), seed, [r["prompt"] for r in rows]))
            replies = {"q0": "Answer: 1", "q1": "Answer: 3", "q2": "Answer: 9"}
            return {
                "generated_texts": [replies[r["prompt"]] for r in rows],
                "mean_tpot_ms": 10.0 + seed,  # varies per run so averaging is visible
            }

        measurements = evaluate_model(
            dataset,
            model="test-model",
            task=TASKS["teleqna"],
            runs=2,
            workdir=tmp_path / "splits",
            benchmark=fake_benchmark,
        )

        # 2 clusters x 2 runs = 4 measurements
        assert len(measurements) == 4
        entry = model_entry(measurements)
        # cluster 0: q0 correct, q1 wrong -> error 0.5 both runs
        assert entry["errors"]["0"] == pytest.approx(0.5)
        assert entry["errors"]["1"] == pytest.approx(0.0)
        # tpot averaged over seeds 0 and 1 -> 10.5
        assert entry["cluster_tpot_ms"]["0"] == pytest.approx(10.5)
        # seed advances per run
        assert sorted({seed for _, seed, _ in calls}) == [0, 1]

    def test_telelogs_task_end_to_end(self, tmp_path):
        """The telelogs Task wired through the per-cluster machinery: boxed and C<n>
        answer forms, integer gold, per-cluster error aggregation."""
        dataset = [
            {"prompt": "q0", "answer": 1, "cluster": 0},
            {"prompt": "q1", "answer": 2, "cluster": 0},
            {"prompt": "q2", "answer": 8, "cluster": 1},
        ]

        def fake_benchmark(dataset_path, model, task, *, host, port, max_concurrency, seed, download_dir):
            rows = [json.loads(line) for line in open(dataset_path)]
            replies = {
                "q0": "<think>weighing causes</think> the cause is \\boxed{1}",
                "q1": "Answer: C5",            # parses to 5, gold is 2 -> wrong
                "q2": "over-shooting, so \\boxed{8}",
            }
            return {"generated_texts": [replies[r["prompt"]] for r in rows], "mean_tpot_ms": 12.0}

        measurements = evaluate_model(
            dataset, model="m", task=TASKS["telelogs"], runs=1,
            workdir=tmp_path / "splits", benchmark=fake_benchmark,
        )
        entry = model_entry(measurements)
        assert entry["errors"]["0"] == pytest.approx(0.5)  # q0 right, q1 wrong
        assert entry["errors"]["1"] == pytest.approx(0.0)  # q2 right

    def test_telelogs_task_uses_reasoning_preset(self):
        task = TASKS["telelogs"]
        assert (task.temperature, task.top_p, task.top_k, task.min_p) == (0.6, 0.95, 20, 0.0)
        assert task.max_tokens == 16384

    def test_missing_tpot_raises(self, tmp_path):
        def bad_benchmark(dataset_path, model, task, **kwargs):
            rows = [json.loads(line) for line in open(dataset_path)]
            return {"generated_texts": ["Answer: 1"] * len(rows)}  # no mean_tpot_ms

        with pytest.raises(ValueError, match="mean_tpot_ms"):
            evaluate_model(
                [{"prompt": "q", "answer": 1, "cluster": 0}],
                model="m",
                task=TASKS["teleqna"],
                runs=1,
                workdir=tmp_path / "splits",
                benchmark=bad_benchmark,
            )
