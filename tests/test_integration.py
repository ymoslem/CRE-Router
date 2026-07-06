"""GPU / running-server smoke tests, skipped by default.

These exercise the two surfaces that cannot run without real resources: loading
a trained QE classifier and running vLLM's benchmark. Enable them by setting
the environment variables below and running ``pytest -m integration``.

  CRE_TEST_QE_CHECKPOINT   a QE classifier checkpoint (HF id or local path)
  CRE_TEST_VLLM_MODEL      a model id served by a running vLLM server
  CRE_TEST_VLLM_HOST       vLLM host (default 127.0.0.1)
  CRE_TEST_VLLM_PORT       vLLM port (default 8000)
"""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.environ.get("CRE_TEST_QE_CHECKPOINT"),
    reason="set CRE_TEST_QE_CHECKPOINT to a QE classifier to run",
)
def test_qe_classifier_roundtrip():
    from cre_router.qe import QEClassifier

    classifier = QEClassifier(
        model_name=os.environ["CRE_TEST_QE_CHECKPOINT"],
        base_tokenizer=os.environ.get("CRE_TEST_QE_TOKENIZER", "answerdotai/ModernBERT-base"),
        max_length=512,
    )
    decision = classifier.predict("What is 2 + 2?", "Answer: 4", 4)
    assert isinstance(decision.accept, bool)
    assert 0.0 <= decision.p_accept <= 1.0
    assert decision.p_accept == pytest.approx(1.0 - decision.p_route, abs=1e-4)


@pytest.mark.skipif(
    not os.environ.get("CRE_TEST_VLLM_MODEL"),
    reason="set CRE_TEST_VLLM_MODEL and run a vLLM server to run",
)
def test_vllm_evaluate_smoke(tmp_path):
    from cre_router.evaluate import TASKS, evaluate_model, model_entry

    # Two trivial AIME-style prompts, one per cluster.
    dataset = [
        {"prompt": "What is 2 + 2? Answer as a number.", "answer": 4, "cluster": 0},
        {"prompt": "What is 3 + 5? Answer as a number.", "answer": 8, "cluster": 1},
    ]
    measurements = evaluate_model(
        dataset,
        model=os.environ["CRE_TEST_VLLM_MODEL"],
        task=TASKS["aime"],
        host=os.environ.get("CRE_TEST_VLLM_HOST", "127.0.0.1"),
        port=int(os.environ.get("CRE_TEST_VLLM_PORT", "8000")),
        runs=1,
        workdir=tmp_path,
    )
    entry = model_entry(measurements)
    assert set(entry["errors"]) == {"0", "1"}
    assert all(t > 0 for t in entry["cluster_tpot_ms"].values())
