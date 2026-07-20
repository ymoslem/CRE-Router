"""``cre`` command-line interface.

  cre cluster   embed training queries, select k, fit centroids
  cre evaluate  measure per-cluster accuracy and TPOT for a model (requires [eval])
  cre fit       Pareto-prune the pool, sweep lambda, select lambda* for a budget
  cre qe-train  fine-tune the accept/escalate QE classifier (requires [qe])
  cre qe-eval   evaluate a trained QE classifier on a split (requires [qe])
  cre serve     run the cascade router (requires [serve])

End to end: cluster -> evaluate (per model) -> fit -> qe-train -> serve.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from cre_router.artifacts import RouterArtifacts
from cre_router.clustering import DEFAULT_EMBEDDING_MODEL
from cre_router.routing import (
    eta,
    models_from_stats,
    pareto_prune,
    routing_regions,
    select_lambda,
    system_metrics,
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_cluster(args: argparse.Namespace) -> None:
    import numpy as np

    from cre_router.clustering import embed, fit_clusters

    rows = _read_jsonl(Path(args.input))
    if args.embeddings_field:
        # Reproduction aid: cluster precomputed embeddings (e.g. the frozen
        # `embeddings` column of the released clustered datasets) instead of
        # re-embedding, which reproduces the paper's exact split even for
        # boundary-sensitive datasets like AIME.
        embeddings = np.asarray([row[args.embeddings_field] for row in rows], dtype=np.float32)
        print(f"Using {len(embeddings)} precomputed embeddings from '{args.embeddings_field}'")
    else:
        texts = [row[args.text_field] for row in rows]
        print(f"Embedding {len(texts)} queries with {args.embedding_model}...")
        embeddings = embed(texts, model_name=args.embedding_model, show_progress=True)

    result = fit_clusters(
        embeddings, k_range=range(args.k_min, args.k_max + 1), k=args.k, seed=args.seed
    )
    if result.silhouette_by_k:
        print("\nSilhouette scores by k:")
        for k, score in sorted(result.silhouette_by_k.items()):
            marker = "  <-- selected" if k == result.k else ""
            print(f"  k={k}: {score:.4f}{marker}")
    counts = {int(c): int((result.labels == c).sum()) for c in range(result.k)}
    print(f"\nk={result.k}, cluster sizes: {counts}")

    out_dir = Path(args.output)
    artifacts = (
        RouterArtifacts.load(out_dir)
        if (out_dir / "router.json").exists()
        else RouterArtifacts()
    )
    artifacts.embedding_model = args.embedding_model
    artifacts.centroids = result.centroids
    artifacts.save(out_dir)

    assignments_path = out_dir / "train_assignments.jsonl"
    with assignments_path.open("w") as f:
        for i, label in enumerate(result.labels):
            f.write(json.dumps({"index": i, "cluster": int(label)}) + "\n")
    print(f"Saved centroids and assignments to {out_dir}/")


def cmd_fit(args: argparse.Namespace) -> None:
    stats = json.loads(Path(args.stats).read_text())
    models, cluster_sizes = models_from_stats(stats)

    efficient, dominated = pareto_prune(models)
    print("Pareto analysis:")
    for m in sorted(models, key=lambda m: m.tpot_ms):
        status = "dominated" if m in dominated else "efficient"
        print(f"  {m.name:<24} TPOT {m.tpot_ms:7.3f} ms  {status}")
    if dominated:
        print(f"Pruned {len(dominated)} dominated model(s); "
              f"routing over: {[m.name for m in efficient]}")

    clusters = sorted(cluster_sizes)
    print(f"\nRouting regions (lambda sweep) over clusters {clusters}:")
    header = f"  {'lambda range':>16}  " + "  ".join(f"{('C' + c):>24}" for c in clusters) \
        + f"  {'Acc':>7}  {'TPOT':>8}  {'eta':>6}"
    print(header)
    for region in routing_regions(efficient):
        acc, tpot = system_metrics(efficient, region.assignment, cluster_sizes)
        e = eta(efficient, region.assignment, cluster_sizes)
        row = f"  {region.interval_str:>16}  " + "  ".join(
            f"{region.assignment[c]:>24}" for c in clusters
        ) + f"  {acc:>6.1%}  {tpot:>6.1f}ms  " + (f"{e:>6.2f}" if e is not None else "   ---")
        print(row)

    selection = select_lambda(efficient, cluster_sizes, args.budget)
    print(f"\nBudget B = {args.budget} ms  ->  lambda* = {selection.lambda_star}")
    print(f"  assignment: {selection.region.assignment}")
    print(f"  training accuracy {selection.accuracy:.1%} at {selection.tpot_ms:.1f} ms TPOT"
          + (f", eta {selection.eta:.2f} pp/ms" if selection.eta is not None else ""))

    if args.output:
        out_dir = Path(args.output)
        artifacts = (
            RouterArtifacts.load(out_dir)
            if (out_dir / "router.json").exists()
            else RouterArtifacts()
        )
        artifacts.routing_table = selection.region.assignment
        artifacts.lambda_star = selection.lambda_star
        artifacts.budget_ms = args.budget
        artifacts.stats = stats
        artifacts.save(out_dir)
        print(f"Saved routing table to {out_dir}/router.json")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from cre_router.evaluate import (
        TASKS,
        cluster_sizes,
        evaluate_model,
        merge_model_into_stats,
        model_entry,
        save_raw_measurements,
    )

    if args.task not in TASKS:
        raise SystemExit(f"unknown task {args.task!r}; choose from {sorted(TASKS)}")
    task = TASKS[args.task]

    dataset = _read_jsonl(Path(args.dataset))
    if args.limit is not None:
        dataset = dataset[: args.limit]
    if any("cluster" not in row for row in dataset):
        if not args.artifacts:
            raise SystemExit(
                "dataset rows lack a 'cluster' field; pass --artifacts to assign "
                "them via the fitted centroids"
            )
        _assign_clusters_into(dataset, args.artifacts, args.text_field)

    print(
        f"Evaluating {args.model} on {args.task}: {len(dataset)} queries across "
        f"{len(cluster_sizes(dataset))} clusters x {args.runs} run(s) via vLLM "
        f"benchmark at {args.host}:{args.port}"
    )

    measurements = evaluate_model(
        dataset,
        model=args.model,
        task=task,
        host=args.host,
        port=args.port,
        runs=args.runs,
        max_concurrency=args.concurrency,
        base_seed=args.seed,
        workdir=Path(args.results_dir) / "splits",
        download_dir=args.download_dir,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_")
    raw_path = Path(args.results_dir) / f"{args.task}_{safe_model}_{timestamp}.jsonl"
    save_raw_measurements(measurements, raw_path)

    entry = model_entry(measurements)
    sizes = cluster_sizes(dataset)
    merge_model_into_stats(args.stats_out, args.model, entry, sizes)

    print(f"\nPer-cluster results for {args.model}:")
    for c in sorted(entry["errors"]):
        print(
            f"  C{c}: error {entry['errors'][c]:.3f}  "
            f"TPOT {entry['cluster_tpot_ms'][c]:.3f} ms"
        )
    print(f"\nRaw measurements: {raw_path}")
    print(f"Updated stats:    {args.stats_out}")


def _assign_clusters_into(dataset: list[dict], artifacts_dir: str, text_field: str) -> None:
    from cre_router.artifacts import RouterArtifacts
    from cre_router.clustering import assign_clusters, embed

    artifacts = RouterArtifacts.load(artifacts_dir)
    if artifacts.centroids is None:
        raise SystemExit(f"{artifacts_dir} has no centroids.npy; run `cre cluster` first")
    embeddings = embed(
        [row[text_field] for row in dataset], model_name=artifacts.embedding_model
    )
    labels = assign_clusters(embeddings, artifacts.centroids)
    for row, label in zip(dataset, labels):
        row["cluster"] = int(label)


def cmd_qe_train(args: argparse.Namespace, extra: list[str]) -> None:
    from cre_router.qe.train import main as qe_train_main

    qe_train_main(extra)


def cmd_serve(args: argparse.Namespace) -> None:
    from cre_router.server.app import serve

    serve(args.config, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cre", description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cluster", help="fit k-means centroids over training queries")
    p.add_argument("--input", required=True, help="JSONL file of training queries")
    p.add_argument("--text-field", default="prompt", help="JSONL field to embed")
    p.add_argument("--embeddings-field", default=None,
                   help="cluster precomputed embeddings from this field instead of re-embedding "
                        "(reproduction aid; e.g. the released datasets' 'embeddings' column)")
    p.add_argument("--output", required=True, help="artifacts directory")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--k", type=int, default=None, help="force k instead of Silhouette selection")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=9, help="inclusive; paper uses k in [2, 9]")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser(
        "evaluate",
        help="measure per-cluster accuracy and TPOT for one model (requires [eval] + a running vLLM server)",
    )
    p.add_argument("--task", required=True, help="dataset task: aime, teleqna, or telelogs")
    p.add_argument("--model", required=True, help="served model name (stats key and vLLM model id)")
    p.add_argument("--dataset", required=True, help="JSONL with prompt, answer[, cluster]")
    p.add_argument("--stats-out", required=True, help="stats JSON to create/update")
    p.add_argument("--host", default="127.0.0.1", help="running vLLM server host")
    p.add_argument("--port", type=int, default=8000, help="running vLLM server port")
    p.add_argument("--artifacts", default=None, help="artifacts dir for cluster assignment if dataset lacks 'cluster'")
    p.add_argument("--text-field", default="prompt", help="field to embed for cluster assignment")
    p.add_argument("--runs", type=int, default=5, help="inference repetitions to average")
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N queries (for quick smoke tests)")
    p.add_argument("--concurrency", type=int, default=32, help="vLLM benchmark max concurrency")
    p.add_argument("--seed", type=int, default=0, help="base seed; run r uses seed+r")
    p.add_argument("--download-dir", default=None, help="vLLM model download/cache directory")
    p.add_argument("--results-dir", default="results", help="where to write raw measurements and cluster splits")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("fit", help="compute the routing table from model stats")
    p.add_argument("--stats", required=True, help="JSON stats file (see configs/)")
    p.add_argument("--budget", type=float, required=True, help="TPOT budget B in ms")
    p.add_argument("--output", default=None, help="artifacts directory to update")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser(
        "qe-train",
        help="fine-tune the QE classifier (all flags forwarded, see --help)",
        add_help=False,
    )

    p = sub.add_parser(
        "qe-eval",
        help="evaluate a trained QE classifier on a split (all flags forwarded, see --help)",
        add_help=False,
    )

    p = sub.add_parser("serve", help="run the cascade router")
    p.add_argument("--config", required=True, help="YAML serving config")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=4000)
    p.set_defaults(func=cmd_serve)

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "qe-train":
        from cre_router.qe.train import main as qe_train_main

        qe_train_main(argv[1:])
        return
    if argv and argv[0] == "qe-eval":
        from cre_router.qe.evaluate import main as qe_eval_main

        qe_eval_main(argv[1:])
        return

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
