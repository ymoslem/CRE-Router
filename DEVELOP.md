# Development

Local setup for working on the package and running its tests.

```bash
git clone https://github.com/ymoslem/CRE-Router
cd CRE-Router
pip install -e ".[qe,serve,dev]"
pytest
```

The `dev` extra adds the test dependencies (`pytest`, plus `fastapi` and
`httpx` for the serve smoke test); `qe` and `serve` bring the runtime the
tests exercise.

The default `pytest` run is CPU-only and needs no model weights. Two smoke
tests that require real resources are marked `integration` and skipped unless
you point them at a GPU/server via environment variables, e.g.:

```bash
# QE classifier load + predict
CRE_TEST_QE_CHECKPOINT=ymoslem/ModernBERT-base-AIME-1983-2023-instruct-qe-classifier-binary-10ep-lr5e-05 \
  pytest -m integration
# vLLM measurement (needs a running `vllm serve <model>`)
CRE_TEST_VLLM_MODEL=WeiboAI/VibeThinker-1.5B pytest -m integration
```
