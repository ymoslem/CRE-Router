"""QE-dataset builder: generations JSONL -> qe-train {train,test}.jsonl."""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "prep_qe", Path(__file__).parent.parent / "data" / "prep_qe.py"
)
prep_qe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_qe)


def _gen(qid, cluster, correct, out="the answer is 4", ntok=12):
    return {
        "qid": qid, "cluster": cluster, "run": 0, "prompt": f"question {qid}?",
        "ground_truth_answer": 4, "answer": 4 if correct else 3,
        "full_output": out, "num_tokens": ntok, "correct": correct,
    }


class TestQeRow:
    def test_correct_maps_to_accept(self):
        # No task, so the stored verdict is copied and the caller is warned.
        with pytest.warns(RuntimeWarning, match="copying the stored"):
            row = prep_qe.qe_row(_gen("a", 0, True))
        assert row["decision_label"] == 1 and row["decision_str"] == "accept"
        assert row["score"] == 1.0 and row["accuracy"] == 1.0

    def test_stale_scorer_version_is_refused(self):
        """A log naming an older scorer must not become a training label."""
        gen = _gen("a", 0, True) | {"scorer_version": 1}
        with pytest.raises(ValueError, match="scorer_version"):
            prep_qe.qe_row(gen)

    def test_wrong_maps_to_route(self):
        row = prep_qe.qe_row(_gen("b", 1, False))
        assert row["decision_label"] == 0 and row["decision_str"] == "route"
        assert row["score"] == 0.0

    def test_schema_and_derived_fields(self):
        row = prep_qe.qe_row(_gen("a", 0, True, out="one two three", ntok=7))
        # columns cre qe-train needs plus the router-parity extras
        assert {"question", "full_output", "num_tokens", "decision_label"} <= set(row)
        assert row["question"] == "question a?"
        assert row["num_words"] == 3 and row["num_tokens"] == 7
        assert row["cluster"] == 0 and row["qid"] == "a"

    def test_question_prefers_explicit_field(self):
        row = prep_qe.qe_row({**_gen("a", 0, True), "question": "raw?"})
        assert row["question"] == "raw?"


class TestBuild:
    def test_writes_pooled_train_test_splits(self, tmp_path):
        trainf = tmp_path / "train_gen.jsonl"
        testf = tmp_path / "test_gen.jsonl"
        trainf.write_text("\n".join(json.dumps(_gen(f"t{i}", i % 2, i % 2 == 0)) for i in range(4)))
        testf.write_text("\n".join(json.dumps(_gen(f"e{i}", 0, True)) for i in range(2)))

        out = tmp_path / "router"
        sizes = prep_qe.build([str(trainf)], [str(testf)], str(out))
        assert sizes == {"train": 4, "test": 2}

        train_rows = [json.loads(l) for l in open(out / "train.jsonl")]
        test_rows = [json.loads(l) for l in open(out / "test.jsonl")]
        assert len(train_rows) == 4 and len(test_rows) == 2
        # labels present and binary
        assert all(r["decision_label"] in (0, 1) for r in train_rows)
        # i even -> correct -> accept(1); i odd -> route(0)
        assert [r["decision_label"] for r in train_rows] == [1, 0, 1, 0]

    def test_pools_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.jsonl"; f1.write_text(json.dumps(_gen("a", 0, True)))
        f2 = tmp_path / "b.jsonl"; f2.write_text(json.dumps(_gen("b", 0, False)))
        out = tmp_path / "router"
        sizes = prep_qe.build([str(f1), str(f2)], [str(f1)], str(out))
        assert sizes["train"] == 2

    def test_drops_rows_with_null_num_tokens(self, tmp_path):
        good = _gen("a", 0, True)
        bad = {**_gen("b", 0, False), "num_tokens": None}
        f = tmp_path / "g.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in [good, bad]))
        out = tmp_path / "router"
        sizes = prep_qe.build([str(f)], [str(f)], str(out))
        assert sizes["train"] == 1  # the null-num_tokens row is dropped
        rows = [json.loads(l) for l in open(out / "train.jsonl")]
        assert [r["qid"] for r in rows] == ["a"]
