"""Stage A measurement: pure functions plus an end-to-end per-cluster run with
an injected fake benchmark (no GPU, no vLLM, no server)."""

import json

import pytest

from cre_router.evaluate import (
    SCORER_VERSION,
    TASKS,
    RunMeasurement,
    aggregate_runs,
    answers_match,
    cluster_sizes,
    evaluate_model,
    merge_model_into_stats,
    model_entry,
    parse_aime_answer,
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

    def test_strips_gemma4_channel_block(self):
        # Gemma 4's marker pair, distinct from Qwen's <think>...</think>.
        text = "<|channel>thought\nmaybe 99\n<channel|> Answer: 3"
        assert split_thinking(text) == "Answer: 3"

    def test_no_marker_passes_through(self):
        # Gemma 4's E2B/E4B variants emit no channel markers at all when
        # thinking is disabled, unlike Qwen (always <think></think>) and
        # unlike Gemma 4's own larger siblings (empty channel block).
        assert split_thinking("Answer: 3") == "Answer: 3"

    def test_aime_ignores_reasoning_numbers(self):
        assert parse_aime_answer("<think>try 500 then 600</think>\\boxed{7}") == 7

    def test_teleqna_answer_label(self):
        assert parse_teleqna_answer("Explanation: foo\nAnswer: 2") == 2


class TestGemma4TaskWiring:
    def test_gemma4_tasks_are_pre_rendered(self):
        # Only the Gemma 4 arms bake enable_thinking into the prompt text;
        # every other task lets the benchmark apply the chat template.
        assert TASKS["telemath_gemma4"].pre_rendered
        pre_rendered = {"telemath_gemma4"}
        for name, task in TASKS.items():
            if name not in pre_rendered:
                assert not task.pre_rendered, name


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

    def test_per_question_outcomes_saved(self, tmp_path):
        dataset = [
            {"id": "a", "prompt": "q0", "answer": 1, "cluster": 0},
            {"id": "b", "prompt": "q1", "answer": 2, "cluster": 0},
            {"id": "c", "prompt": "q2", "answer": 9, "cluster": 1},
        ]

        def fake_benchmark(dataset_path, model, task, *, host, port, max_concurrency, seed, download_dir):
            rows = [json.loads(line) for line in open(dataset_path)]
            replies = {"q0": "Answer: 1", "q1": "Answer: 3", "q2": "Answer: 9"}
            return {
                "generated_texts": [replies[r["prompt"]] for r in rows],
                "mean_tpot_ms": 10.0,
                "output_lens": [5] * len(rows),
            }

        out = tmp_path / "outcomes.jsonl"
        evaluate_model(
            dataset, model="m", task=TASKS["teleqna"], runs=2,
            workdir=tmp_path / "splits", benchmark=fake_benchmark, outcomes_out=out,
        )
        recs = [json.loads(line) for line in open(out)]
        assert len(recs) == 3 * 2  # questions x runs
        for r in recs:
            assert set(r) == {
                "qid", "cluster", "run", "correct", "output_len", "scorer_version",
            }
            assert r["output_len"] == 5
            # An outcomes row keeps no text, so the stamp is the only record of
            # which grading rules produced its verdict.
            assert r["scorer_version"] == SCORER_VERSION
        by_qid: dict = {}
        for r in recs:
            by_qid.setdefault(r["qid"], []).append(r["correct"])
        assert by_qid["a"] == [True, True]   # q0 -> 1 correct
        assert by_qid["b"] == [False, False]  # q1 -> 3 wrong
        assert by_qid["c"] == [True, True]   # q2 -> 9 correct
        assert sorted({r["run"] for r in recs}) == [0, 1]

    def test_generations_saved(self, tmp_path):
        dataset = [
            {"id": "a", "prompt": "q0", "answer": 1, "cluster": 0},
            {"id": "b", "prompt": "q1", "answer": 2, "cluster": 0},
            {"id": "c", "prompt": "q2", "answer": 9, "cluster": 1},
        ]

        def fake_benchmark(dataset_path, model, task, *, host, port, max_concurrency, seed, download_dir):
            rows = [json.loads(line) for line in open(dataset_path)]
            replies = {"q0": "Answer: 1", "q1": "Answer: 3", "q2": "Answer: 9"}
            return {
                "generated_texts": [replies[r["prompt"]] for r in rows],
                "mean_tpot_ms": 10.0,
                "output_lens": [7] * len(rows),
            }

        gens = tmp_path / "generations.jsonl"
        evaluate_model(
            dataset, model="m", task=TASKS["teleqna"], runs=1,
            workdir=tmp_path / "splits", benchmark=fake_benchmark, generations_out=gens,
        )
        recs = [json.loads(line) for line in open(gens)]
        assert len(recs) == 3  # questions x 1 run
        for r in recs:
            assert set(r) == {
                "qid", "cluster", "run", "question", "prompt", "ground_truth_answer",
                "answer", "full_output", "num_tokens", "correct", "scorer_version",
            }
            assert r["num_tokens"] == 7
            assert r["scorer_version"] == SCORER_VERSION
        by_qid = {r["qid"]: r for r in recs}
        # no explicit question field -> falls back to the (raw) prompt
        assert by_qid["a"]["question"] == "q0" and by_qid["a"]["prompt"] == "q0"
        # a: gold 1, model says "Answer: 1" -> parsed 1, correct
        assert by_qid["a"]["full_output"] == "Answer: 1" and by_qid["a"]["correct"] is True
        assert by_qid["a"]["ground_truth_answer"] == 1 and by_qid["a"]["answer"] == 1
        # b: gold 2, model says "Answer: 3" -> parsed 3, wrong
        assert by_qid["b"]["correct"] is False
        assert by_qid["b"]["ground_truth_answer"] == 2 and by_qid["b"]["answer"] == 3

    def test_generations_question_is_raw_for_prerendered(self, tmp_path):
        # a pre_rendered item carries the templated text in `prompt` and the raw
        # query in `question`; generations must store the raw question for the QE.
        dataset = [
            {"id": "a", "question": "what is 2+2?",
             "prompt": "<start>what is 2+2?<gen>", "answer": 4, "cluster": 0},
        ]

        def fake_benchmark(dataset_path, model, task, **kw):
            rows = [json.loads(line) for line in open(dataset_path)]
            return {"generated_texts": ["Answer: 4"] * len(rows),
                    "mean_tpot_ms": 10.0, "output_lens": [5] * len(rows)}

        gens = tmp_path / "g.jsonl"
        evaluate_model(
            dataset, model="m", task=TASKS["teleqna"], runs=1,
            workdir=tmp_path / "s", benchmark=fake_benchmark, generations_out=gens,
        )
        r = json.loads(open(gens).readline())
        assert r["question"] == "what is 2+2?"          # raw, QE-facing
        assert r["prompt"] == "<start>what is 2+2?<gen>"  # exact served input

    def test_generations_off_by_default(self, tmp_path):
        def fake_benchmark(dataset_path, model, task, **kw):
            rows = [json.loads(line) for line in open(dataset_path)]
            return {"generated_texts": ["Answer: 1"] * len(rows), "mean_tpot_ms": 10.0}

        # default generations_out=None: no file, no error
        evaluate_model(
            [{"prompt": "q", "answer": 1, "cluster": 0}], model="m",
            task=TASKS["teleqna"], runs=1, workdir=tmp_path / "splits",
            benchmark=fake_benchmark,
        )

    def test_question_id_uses_id_then_hash(self):
        from cre_router.evaluate import question_id
        assert question_id({"id": 7, "prompt": "x"}) == "7"
        h = question_id({"prompt": "x"})
        assert len(h) == 12 and question_id({"prompt": "x"}) == h  # stable
        assert question_id({"prompt": "y"}) != h  # different prompt -> different id

    def test_outcomes_off_by_default(self, tmp_path):
        def fake_benchmark(dataset_path, model, task, **kw):
            rows = [json.loads(line) for line in open(dataset_path)]
            return {"generated_texts": ["Answer: 1"] * len(rows), "mean_tpot_ms": 10.0}

        # default outcomes_out=None: no file, no error
        evaluate_model(
            [{"prompt": "q", "answer": 1, "cluster": 0}], model="m",
            task=TASKS["teleqna"], runs=1, workdir=tmp_path / "splits",
            benchmark=fake_benchmark,
        )

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
