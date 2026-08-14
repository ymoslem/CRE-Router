# CRE-Router

Implementation of the paper [Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving](https://arxiv.org/abs/2606.27457).

Production LLM serving trades accuracy against cost. CRE-Router routes each
query to the most cost-effective model in a pool, then escalates low-quality
outputs to a stronger model:

- **Stage 1 (clustering-based routing).** Queries are embedded and clustered
  offline; each cluster is routed to the model that minimizes a cost-adjusted
  error score (Error + lambda * Cost), with lambda tuned once to satisfy a
  latency (TPOT or E2EL) budget.
- **Stage 2 (quality-estimation cascade).** A lightweight ModernBERT
  classifier inspects each efficient-model output and escalates low-quality
  answers up an ordered ladder of stronger models.

Both stages train only on task-correctness labels obtainable from standard
benchmark evaluation; no extra annotation is required.

## Install

```bash
pip install "cre-router[full]"
```

`full` installs the whole pipeline on one machine. Narrower installs are
available through the `serve`, `qe`, and `eval` extras; see the
[installation guide](https://github.com/ymoslem/CRE-Router#installation).

## Usage

The workflow is driven by the `cre` CLI, one stage per step:

`cre cluster` → `cre evaluate` → `cre fit` → `cre qe-train` → `cre serve`

Run `cre <stage> --help` for options. Runnable end-to-end quickstarts (a
no-GPU routing-table demo and a full serving walkthrough) are in the
[README](https://github.com/ymoslem/CRE-Router#readme).

## Links

- **Source and documentation:** https://github.com/ymoslem/CRE-Router
- **Paper:** https://arxiv.org/abs/2606.27457
- **Reproducing the paper:** https://github.com/ymoslem/CRE-Router/blob/main/REPRODUCE.md

Apache-2.0 licensed.
