"""
sweep.py - Parameter sweep for BERTopic with MLflow logging

Requires saved embeddings (run rq2_bertopic.py --embed-only first).
Iterates over UMAP + HDBSCAN parameter combinations, fitting BERTopic
each time and logging coherence, diversity, and outlier metrics to MLflow.

Usage:
    # Run sweep with default grid:
    python sweep.py

    # Custom sweep config:
    python sweep.py --sweep-config configs/sweep_configs.yaml

    # Sweep on specific era only:
    python sweep.py --era pre_gpt

    # Sweep with custom sample size:
    python sweep.py --sample 100000
"""

import argparse, os, time, itertools
import pandas as pd
import numpy as np
import mlflow

from mlflow_config import (
    load_config, merge_configs, apply_sweep_params, flatten_config,
    init_mlflow, get_git_hash,
    compute_all_metrics, log_metrics_to_mlflow,
)

from rq2_bertopic import (
    build_bertopic, load_embeddings, generate_embeddings,
    print_topic_summary, timer,
)


def build_sweep_grid(sweep_config: dict) -> list[dict]:
    """
    Build a list of flat param dicts from the sweep config grid.
    Example output: [{"hdbscan.min_cluster_size": 30, "umap.n_neighbors": 10, ...}, ...]
    """
    
    sweep_params = sweep_config.get("sweep", {})

    # Flatten the nested sweep params: {"hdbscan": {"min_cluster_size": [30,50]}}
    # -> [("hdbscan.min_cluster_size", [30, 50]), ...]
    param_lists = []
    for section, params in sweep_params.items():
        for param_name, values in params.items():
            param_lists.append((f"{section}.{param_name}", values))

    if not param_lists:
        print("  WARNING: No sweep parameters found in config.")
        return []

    # Cartesian product of all parameter values
    keys = [k for k, _ in param_lists]
    value_lists = [v for _, v in param_lists]

    grid = []
    for combo in itertools.product(*value_lists):
        grid.append(dict(zip(keys, combo)))

    return grid


def run_single_config(docs: list[str], embeddings: np.ndarray, era: str,
                      config: dict, combo_idx: int, total: int) -> dict:
    """Run BERTopic with a single parameter configuration and return metrics."""
    
    mcs = config["hdbscan"]["min_cluster_size"]
    nn = config["umap"]["n_neighbors"]
    nc = config["umap"]["n_components"]
    run_name = f"sweep_{era}_mcs{mcs}_nn{nn}_nc{nc}"

    print(f"\n  --- Config {combo_idx+1}/{total}: mcs={mcs}, nn={nn}, nc={nc} ---")
    t0 = time.time()

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("era", era)
        mlflow.set_tag("run_type", "sweep")
        mlflow.set_tag("git_commit", get_git_hash())

        # Log params
        model_params = {
            "hdbscan.min_cluster_size": mcs,
            "umap.n_neighbors": nn,
            "umap.n_components": nc,
            "umap.min_dist": config["umap"]["min_dist"],
            "umap.metric": config["umap"]["metric"],
            "hdbscan.metric": config["hdbscan"]["metric"],
            "hdbscan.cluster_selection_method": config["hdbscan"]["cluster_selection_method"],
            "embedding.model": config["embedding"]["model"],
            "data.sample": config["data"].get("sample", "full"),
            "data.num_docs": len(docs),
        }
        mlflow.log_params(model_params)

        # Fit
        _, _, topic_model = build_bertopic(config)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        topics_list = topics if isinstance(topics, list) else topics.tolist()

        elapsed = time.time() - t0

        # Evaluate
        metrics = compute_all_metrics(topic_model, docs, topics_list)
        metrics["fit_time_sec"] = round(elapsed, 1)
        log_metrics_to_mlflow(metrics)

        # Quick summary
        print(f"  Result: {metrics['num_topics']} topics, "
              f"{metrics['outlier_pct']:.1%} outliers, "
              f"coherence={metrics['coherence_cv']:.4f}, "
              f"diversity={metrics['topic_diversity']:.4f} "
              f"({timer(t0)})")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="BERTopic parameter sweep with MLflow")
    parser.add_argument("--base-config", default="configs/default.yaml",
                        help="Base config to override with sweep params")
    parser.add_argument("--sweep-config", default="configs/sweep_configs.yaml",
                        help="Sweep grid definition")
    parser.add_argument("--era", default=None, choices=["pre_gpt", "post_gpt"],
                        help="Run sweep for one era only (default: both)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Override sample size for sweep")
    args = parser.parse_args()

    # Load configs
    base_config = load_config(args.base_config)
    sweep_config = load_config(args.sweep_config)

    # Apply fixed overrides from sweep config
    fixed = sweep_config.get("fixed_overrides", {})
    if fixed:
        base_config = merge_configs(base_config, fixed)

    if args.sample is not None:
        base_config["data"]["sample"] = args.sample

    out_dir = base_config["data"]["output_dir"]
    init_mlflow(base_config)

    # Build grid
    grid = build_sweep_grid(sweep_config)
    print(f"\nSweep grid: {len(grid)} configurations")
    for i, combo in enumerate(grid):
        print(f"  [{i+1}] {combo}")

    # Load data
    print(f"\nLoading {base_config['data']['input']} ...")
    df = pd.read_parquet(base_config["data"]["input"])

    eras = [args.era] if args.era else ["pre_gpt", "post_gpt"]

    for era in eras:
        era_col = "pre-GPT" if era == "pre_gpt" else "post-GPT"
        era_df = df[df["era"] == era_col].copy()

        sample_n = base_config["data"].get("sample")
        if sample_n:
            era_df = era_df.sample(n=min(sample_n, len(era_df)), random_state=42)

        docs = era_df["text_clean"].tolist()

        # Load embeddings (must exist; run rq2_bertopic.py --embed-only first)
        embeddings = load_embeddings(era, out_dir, len(docs))
        if embeddings is None:
            print(f"\n  ERROR: No embeddings found for {era} with {len(docs)} docs.")
            print(f"  Run first: python rq2_bertopic.py --embed-only --sample {sample_n or ''}")
            continue

        print(f"\n{'='*60}")
        print(f"SWEEP: {era.upper()} ({len(docs):,} docs, {len(grid)} configs)")
        print(f"{'='*60}")

        # Parent run for this era's sweep
        with mlflow.start_run(run_name=f"sweep_{era}"):
            mlflow.set_tag("run_type", "sweep_parent")
            mlflow.set_tag("era", era)
            mlflow.set_tag("num_configs", len(grid))

            all_results = []
            for i, combo in enumerate(grid):
                config = apply_sweep_params(base_config, combo)
                metrics = run_single_config(docs, embeddings, era, config, i, len(grid))
                all_results.append({**combo, **metrics})

            # Summary table
            results_df = pd.DataFrame(all_results)
            print(f"\n{'='*60}")
            print(f"SWEEP RESULTS: {era}")
            print(f"{'='*60}")

            # Sort by coherence (primary) then by outlier_pct (lower is better)
            results_df = results_df.sort_values(
                ["coherence_cv", "outlier_pct"],
                ascending=[False, True]
            )

            display_cols = [c for c in results_df.columns
                           if c in ["hdbscan.min_cluster_size", "umap.n_neighbors",
                                    "umap.n_components", "num_topics", "outlier_pct",
                                    "coherence_cv", "topic_diversity", "fit_time_sec"]]
            print(results_df[display_cols].to_string(index=False))

            # Save results CSV
            csv_path = os.path.join(out_dir, f"sweep_results_{era}.csv")
            results_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            print(f"\n  Saved sweep results to {csv_path}")

            # Highlight best config
            best = results_df.iloc[0]
            print(f"\n  BEST CONFIG (by coherence):")
            for col in display_cols:
                print(f"    {col}: {best[col]}")


if __name__ == "__main__":
    main()