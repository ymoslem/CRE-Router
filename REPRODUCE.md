# Reproducing the paper

This document covers the paper-specific material, regenerating the reported
numbers, the released Hugging Face artifacts, the pinned environment, and how
to cite the work. For the tool itself (install, CLI, serving) see the
[README](README.md).

## System design

The framework is a two-stage cascade. **Stage 1 (clustering-based routing)**
embeds each query, assigns it to a semantic cluster, and routes the cluster to
the model that minimizes a cost-adjusted score `Error + lambda * Cost` under a
latency (TPOT) budget; this produces the routing table and the budgeted
$\lambda^*$. **Stage 2 (quality-estimation cascade)** inspects an efficient model's
output with a lightweight ModernBERT classifier and escalates low-quality
answers to a stronger model.

<p align="center"><img src="img/system.svg" alt="Two-stage cascaded routing system" width="640"></p>

The stages map onto the `cre` commands:

- **Stage 1** — cluster the training queries (`cre cluster`), measure each
  model per cluster (`cre evaluate`), then fit the routing table and $\lambda^*$
  (`cre fit`). The measured stats are checked into [`configs/`](configs), so
  `cre fit` reproduces the routing tables with no GPU (below).
- **Stage 2** — train and evaluate the accept/escalate classifier
  (`cre qe-train`, `cre qe-eval`).

Deploying both stages together is `cre serve`.

## Installation (from source)

Everything below runs from a checkout, not from a wheel. `pip install
cre-router` ships the library alone: the configs, datasets and this document
live in the repository and in the source distribution, so the commands here
resolve their `configs/...` paths only if you have one of those.

```bash
git clone https://github.com/ymoslem/CRE-Router.git
cd CRE-Router
pip install -e ".[qe,serve,eval]"
```

The `[eval]` extra pulls vLLM, which the GPU sections need; the no-GPU sections
below need only the base install. To pin the exact environment behind the
reported numbers rather than current releases, see
[requirements-paper.txt](requirements-paper.txt) — read the header first, it is
provenance rather than something to install over your working environment.

## Routing tables and $\lambda^*$ (no GPU)

The per-cluster error rates and TPOT measured on the training corpora are
checked in under [`configs/`](configs), so the Stage 1 routing tables and
budgeted $\lambda^*$ selection reproduce directly:

```bash
cre fit --stats configs/aime_stats_1xA100_Sep2026.json --budget 30
cre fit --stats configs/teleqna_stats_1xA100_Sep2026.json --budget 20
```

This prints the Pareto analysis, the full $\lambda$ sweep (routing regions), and
the budget-feasible $\lambda^*$ selection. Expected values are pinned as tests in
[`tests/test_routing.py`](tests/test_routing.py): AIME crossovers
$\lambda$ = 0.061 / 0.074 / 0.103 with $\lambda^*$ = 0.061 at B = 30 ms, and a
TeleQnA pool where **nothing is Pareto-dominated** and the sweep has six regions.

**Every config names the configuration that produced it, and there is no
basis-less default.** Cost is a property of the serving setup, so a config
without its basis in the name is a number waiting to be misread:

| config | basis |
| --- | --- |
| `*_1xA100_Sep2026.json` | 1 x A100 at concurrency 32, vLLM 0.19.0 -- what the paper reports |
| `*_2xA100_Jun2026.json` | 2 x A100, vLLM 0.17.0, the June 2026 preprint, kept reproducible |

Every name is basis plus measurement month, so two files never differ only in
hardware when they also differ in date and serving stack. The suffixes go away
once a single basis remains.

The two bases are genuinely different pools rather than the same pool priced
twice. On two cards Gemma4-E2B and Gemma4-E4B are dominated and the TeleQnA
sweep has three regions; on one card nothing is dominated and it has six. The
archived configs record the vLLM version **per model**, because the submitted
TeleQnA pool served Qwen3-4B-Instruct under 0.17.0 and the three Gemmas under
0.19.0.

The training-set cluster sizes used for the system-level accuracy and TPOT
come from the paper's clustering (AIME train 194 / 405 / 322; TeleQnA train
5,211 / 3,789).

### Stage 1+2 cascade latency (no GPU)

The combined Stage 1+2 system TPOT and E2EL are composed from the test-split
per-cluster measurements plus the measured QE escalation counts, checked in
under [`configs/`](configs):

```bash
cre cascade --stats configs/aime_cascade_test_2xA100_Jun2026.json
cre cascade --stats configs/teleqna_cascade_test_2xA100_Jun2026.json
```

Each escalated query is charged both passes: for TPOT, per delivered token
(`TPOT_strong + TPOT_eff * L_eff / L_strong`, following vLLM's per-request
Mean TPOT convention); for E2EL, as the sum `E2EL_eff + E2EL_strong`, since
Stage 2 inspects the complete efficient-model output before escalating. This
gives 9.75 ms / 156,303 ms (AIME) and 23.65 ms / 1,127 ms (TeleQnA), matching
the paper's Tables `aime_test` and `teleqna_test` Stage 1+2 latency (9.7 and
23.8 ms) to within rounding. Expected values are pinned in
[`tests/test_routing.py`](tests/test_routing.py). The escalated queries'
accuracy recovery is measured separately (`cre qe-eval`, Appendix D).

### Reproducing the clustering

The first step in the paper's Stage 1 is to cluster the training queries. The released datasets already include the paper's clustering in the `cluster` column, so you can skip this step and use the released datasets directly. If you want to reproduce the clustering, you can run the following command:

```bash
# Download the training split of the AIME dataset
python data/download.py --dataset ymoslem/AIME-clustered --split train --output data/aime_train.jsonl

# Compute embeddings and cluster the training queries
cre cluster --input data/aime_train.jsonl --embeddings-field embeddings --output artifacts/aime
```

Note: The embeddings were computed with the all-MiniLM-L6-v2 model on Apple Silicon (MPS), which might result in slightly different cluster sizes on other hardware. For a guaranteed exact cluster split on any hardware, skip re-embedding and cluster the frozen embeddings the released datasets ship in their `embeddings` column, using the command above.



## Regenerating the stats from scratch (GPU)

To rebuild the stats files, fit centroids and then measure each model per
cluster against a running vLLM server:

```bash
# 0. fetch the training split (skip if already downloaded above)
python data/download.py --dataset ymoslem/AIME-clustered --split train --output data/aime_train.jsonl

# 1. fit clustering centroids on the training queries (frozen embeddings for
#    the exact paper split; drop --embeddings-field to re-embed from scratch)
cre cluster --input data/aime_train.jsonl --embeddings-field embeddings --output artifacts/aime

# 2. serve each model, then measure it per cluster (run once per model).
#    --max-model-len must cover input + the 40,960-token AIME generation.
vllm serve WeiboAI/VibeThinker-1.5B --port 8000 --max-model-len 42000
cre evaluate --task aime --model WeiboAI/VibeThinker-1.5B \
    --dataset data/aime_train.jsonl --artifacts artifacts/aime \
    --stats-out configs/aime_stats_1xA100_Sep2026.json --runs 5

vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 --port 8000 --max-model-len 42000
cre evaluate --task aime --model Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
    --dataset data/aime_train.jsonl --artifacts artifacts/aime \
    --stats-out configs/aime_stats_1xA100_Sep2026.json --runs 5

# 3. compute the routing table and lambda*
cre fit --stats configs/aime_stats_1xA100_Sep2026.json --budget 30 --output artifacts/aime
```

`cre evaluate` runs vLLM's benchmark per cluster, averages over `--runs`, and
saves the raw per-(cluster, run) measurements under `results/` so every number
in the stats traces back to a benchmark run. Sampling follows the paper's
Appendix A (AIME thinking-mode 0.6 / 0.95 / 20; TeleQnA 0.7 / 0.8 / 20).

## QE classifier (Stage 2)

Train and evaluate the accept/escalate classifier on the released datasets:

```bash
cre qe-train --dataset ymoslem/AIME-router --output-dir qe-aime \
    --max-length 4096 --learning-rate 5e-5
cre qe-eval --classifier <checkpoint> --dataset ymoslem/AIME-clustered-output \
    --split test_vibethinker_cluster_0_run_0 --max-length 4096
```

`cre qe-eval` reports the accuracy, macro-F1, and escalation confusion counts
(true / unnecessary / missed escalations) used in the QE appendices. For
TeleQnA use `--max-length 512` and learning rate 2e-5.

### Building QE data for a new pool

For a pool other than the released ones, the QE data comes from the efficient
model's own generations:

```bash
# capture generations alongside the per-question outcomes
cre evaluate ... --save-generations

# convert them to the schema cre qe-train reads
python data/prep_qe.py --train <train_generations.jsonl> \
    --test <test_generations.jsonl> --out qe-data/<name>

# replay a trained classifier over the gated clusters
cre qe-cascade --classifier <checkpoint> --generations <test_generations.jsonl> \
    --clusters 1,3 --strong-outcomes <strong_outcomes.jsonl> \
    --strong-model <name> --out configs/<pool>_cascade_test.json
```

`--save-generations` adds the full outputs the classifier judges; per-question
outcomes are always written. `cre qe-cascade` writes the per-cluster cascade
accuracy and escalation counts into the cascade config that `cre cascade` reads.

## Serving the paper's pools

Two ready-made serving configs are provided:
[`example_config_aime24.yaml`](src/cre_router/server/example_config_aime24.yaml) for the
two-model AIME pool (VibeThinker-1.5B escalating to Qwen3-30B-A3B) and
[`example_config_teleqna.yaml`](src/cre_router/server/example_config_teleqna.yaml)
for the TeleQnA pool (Qwen3-4B escalating to Gemma4-26B). Point each at your
running vLLM servers and the released QE classifiers, then `cre serve
--config <file>`.

## Released artifacts

Datasets and QE classifier checkpoints are public on the Hugging Face Hub:
<https://huggingface.co/collections/ymoslem/routing>.

| Purpose | HF Hub ID |
|---|---|
| AIME QE training data | `ymoslem/AIME-router` |
| TeleQnA QE training data | `ymoslem/TeleQnA-router` |
| AIME clustered test outputs | `ymoslem/AIME-clustered-output` |
| TeleQnA clustered test outputs | `ymoslem/TeleQnA-clustered-output` |
| AIME QE classifier | `ymoslem/ModernBERT-base-AIME-1983-2023-instruct-qe-classifier-binary-10ep-lr5e-05` |
| TeleQnA QE classifier | `ymoslem/ModernBERT-base-TeleQnA-router-qe-classifier-binary-10ep-lr2e-05-qwen4b-5runs_1eval` |

Fetch any split as JSONL with `python data/download.py --dataset <id>`.

## Pinned environment

Package versions are pinned in
[`requirements-paper.txt`](requirements-paper.txt). The reported numbers were
measured on 2x A100 SXM 80 GB under Python 3.11 with 32 concurrent requests,
averaged over 5 runs. TPOT is hardware- and version-specific and will shift on
newer vLLM releases or different hardware (e.g. H100 with full W8A8 FP8
support), which can also change the selected $\lambda^*$. Efficient ModernBERT
training additionally used `flash-attn==2.8.3`.

Install order matters for the Gemma models: installing the pinned vLLM pulls
transformers 4.57.6, which does **not** recognize the `gemma4` architecture.
Upgrade with `pip install transformers==5.5.3` afterwards (it serves both the
Qwen and Gemma pools; vLLM's `transformers<5` pin is conservative).

## Citation

```bibtex
@article{moslem2026clusterrouteescalate,
      title={Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving}, 
      author={Yasmin Moslem and Magdalena Kacmajor and Vasudevan Nedumpozhimana and Ammar Abbas and Solmaz Panahi and David Lynch and Zhuangzhuang Nie and Alexandros Agapitos and Aleksandar Milenovic and Hongmeng Song and Yucheng Shi and Yue Pan and Patricia Buffini and John D. Kelleher},
      year={2026},
      eprint={2606.27457},
      archivePrefix={arXiv},
      primaryClass={cs.PF},
      url={https://arxiv.org/abs/2606.27457}, 
}
```

## Acknowledgements

This work is funded by ADAPT Centre, Trinity College Dublin, and Huawei Ireland.
