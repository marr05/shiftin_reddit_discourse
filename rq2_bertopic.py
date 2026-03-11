"""
rq2_bertopic.py - BERTopic pipeline with MLflow experiment tracking

   1: Generate embeddings (or load saved)
   2: BERTopic clustering (UMAP + HDBSCAN)
   3: Evaluate (coherence, diversity, outlier metrics)
   4: Log details to MLflow
   5: Output docs for human labeling

Usage:
    # Quick sample run (logs to MLflow):
    python rq2_bertopic.py --sample 50000

    # Full dataset run:
    python rq2_bertopic.py --config configs/full_run.yaml

    # Reuse saved embeddings (for parameter tuning):
    python rq2_bertopic.py --reuse-embeddings --sample 100000

    # Generate embeddings only (for example, on GPU, then transfer):
    python rq2_bertopic.py --embed-only

    # Custom config:
    python rq2_bertopic.py --config configs/default.yaml --sample 50000

Dependencies:
    pip install bertopic sentence-transformers umap-learn hdbscan mlflow gensim pyyaml
"""

import argparse, os, json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from mlflow_config import (
    load_config, merge_configs, flatten_config,
    init_mlflow, log_run_context, get_git_hash,
    compute_all_metrics, log_metrics_to_mlflow, log_artifact_safe,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# Helpers 

def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def timer(start):
    elapsed = time.time() - start
    return f"{elapsed/60:.1f} min" if elapsed > 60 else f"{elapsed:.0f} sec"


# Embedding 

def generate_embeddings(docs: list[str], era: str, config: dict, out_dir: str) -> np.ndarray:
    """Generate or load sentence embeddings."""
    
    emb_path = os.path.join(out_dir, f"embeddings_{era}.npy")
    emb_cfg = config["embedding"]

    print(f"  [{era}] Loading embedding model: {emb_cfg['model']} ")
    model = SentenceTransformer(emb_cfg["model"])

    print(f"  [{era}] Encoding {len(docs):,} documents ")
    t0 = time.time()
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        batch_size=emb_cfg["batch_size"],
    )
    print(f"  [{era}] Embeddings done in {timer(t0)}")

    np.save(emb_path, embeddings)
    print(f"  [{era}] Saved to {emb_path}")
    return embeddings


def load_embeddings(era: str, out_dir: str, expected_len: int) -> np.ndarray | None:
    """Load saved embeddings if they exist and match expected doc count."""
    
    emb_path = os.path.join(out_dir, f"embeddings_{era}.npy")
    if not os.path.exists(emb_path):
        print(f"  [{era}] No saved embeddings at {emb_path}")
        return None

    embeddings = np.load(emb_path)
    if len(embeddings) != expected_len:
        print(f"  [{era}] Embedding count ({len(embeddings)}) != doc count ({expected_len}). re-embedding")
        return None

    print(f"  [{era}] Loaded embeddings from {emb_path} ({len(embeddings):,} vectors)")
    return embeddings


# BERTopic Fitting 

def build_bertopic(config: dict) -> tuple[UMAP, HDBSCAN, BERTopic]:
    """Construct UMAP, HDBSCAN, and BERTopic from config."""
    
    umap_cfg = config["umap"]
    hdb_cfg = config["hdbscan"]
    bt_cfg = config["bertopic"]

    umap_model = UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        metric=hdb_cfg["metric"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=True,
    )

    embedding_model = SentenceTransformer(config["embedding"]["model"])

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=bt_cfg["top_n_words"],
        verbose=True,
    )

    return umap_model, hdbscan_model, topic_model


def fit_and_evaluate(docs: list[str], embeddings: np.ndarray, era: str,
                     config: dict, out_dir: str, fig_dir: str,
                     run_type: str = "experiment"):
    """
    Fit BERTopic, compute metrics, log to MLflow, export artifacts. Entry point for a single era's analysis.
    """
    
    run_name = f"{era}_mcs{config['hdbscan']['min_cluster_size']}_nn{config['umap']['n_neighbors']}_nc{config['umap']['n_components']}"

    with mlflow.start_run(run_name=run_name, nested=True):
        log_run_context(era, run_type, config)
        t0 = time.time()

        # Build and fit
        print(f"\n  [{era}] Fitting BERTopic ")
        _, _, topic_model = build_bertopic(config)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
        elapsed = time.time() - t0
        print(f"  [{era}] Fit complete in {timer(t0)}")

        # Convert topics to list for .count() etc.
        topics_list = topics if isinstance(topics, list) else topics.tolist()

        # Evaluate
        metrics = compute_all_metrics(topic_model, docs, topics_list)
        metrics["fit_time_sec"] = round(elapsed, 1)
        log_metrics_to_mlflow(metrics)

        # Print topic summary
        print_topic_summary(topic_model, era)

        # Export labeling JSON
        labeling_path = export_representative_docs(
            topic_model, docs, topics_list, era, out_dir, config
        )
        log_artifact_safe(labeling_path)

        # Generate and log figures
        fig_path = plot_topic_overview(topic_model, era, fig_dir)
        log_artifact_safe(fig_path)

        # Save model (only for final runs, not parametersweeps)
        if run_type == "production":
            model_path = os.path.join(out_dir, f"bertopic_{era}")
            topic_model.save(
                model_path, serialization="safetensors",
                save_ctfidf=True, save_embedding_model=config["embedding"]["model"]
            )
            print(f"  [{era}] Model saved to {model_path}")

        return topic_model, topics_list, probs, metrics


# Export for Manual Labeling

def export_representative_docs(topic_model, docs, topics, era, out_dir, config):
    """Export representative documents per topic for human labeling."""
    
    n_rep = config["bertopic"]["n_representative_docs"]
    topic_info = topic_model.get_topic_info()
    output = []

    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue

        rep_docs = topic_model.get_representative_docs(tid) or []

        # Additional random samples
        topic_indices = [i for i, t in enumerate(topics) if t == tid]
        n_extra = max(0, n_rep - len(rep_docs))
        extra_docs = []
        if n_extra > 0 and len(topic_indices) > len(rep_docs):
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(topic_indices, size=min(n_extra, len(topic_indices)), replace=False)
            extra_docs = [docs[i][:1000] for i in sample_idx]

        topic_words = topic_model.get_topic(tid)
        keywords = [w for w, _ in topic_words[:10]] if topic_words else []

        output.append({
            "topic_id": tid,
            "topic_keywords": keywords,
            "count": int(row["Count"]),
            "representative_docs": [d[:1000] for d in rep_docs],
            "additional_samples": extra_docs,
            "manual_label": "",
            "manual_theme": "",
            "labeler_notes": "",
        })

    output.sort(key=lambda x: x["count"], reverse=True)

    path = os.path.join(out_dir, f"topics_for_labeling_{era}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  [{era}] Exported {len(output)} topics for labeling -> {path}")
    return path


# Visualization 

def plot_topic_overview(topic_model, era, fig_dir):
    """Top 20 topics bar chart."""
    
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].head(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [f"Topic {row['Topic']}: {row['Name'][:40]}" for _, row in topic_info.iterrows()]
    ax.barh(labels[::-1], topic_info["Count"].values[::-1], color="#2c7bb6")
    ax.set_xlabel("Number of Documents")
    ax.set_title(f"Top 20 Topics ({era})")
    path = os.path.join(fig_dir, f"rq2_top_topics_{era}.png")
    savefig(fig, path)

    # HTML visualizations (non-critical)
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=15)
        fig_bar.write_html(os.path.join(fig_dir, f"rq2_barchart_{era}.html"))
    except Exception:
        pass

    try:
        fig_map = topic_model.visualize_topics()
        fig_map.write_html(os.path.join(fig_dir, f"rq2_intertopic_{era}.html"))
    except Exception:
        pass

    return path


def print_topic_summary(topic_model, era):
    """Print top 25 topics to console."""
    
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1]

    print(f" ")
    print(f"  TOPIC SUMMARY: {era} ({len(topic_info)} topics)")
    print(f"--------------------------------")
    for _, row in topic_info.head(25).iterrows():
        tid = row["Topic"]
        words = topic_model.get_topic(tid)
        kw = ", ".join([w for w, _ in words[:8]]) if words else "N/A"
        print(f"  Topic {tid:3d} ({row['Count']:>5,} docs): {kw}")


# Main 

def main():
    parser = argparse.ArgumentParser(description="BERTopic with MLflow tracking")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N docs per era (overrides config)")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Load saved embeddings instead of recomputing")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only generate and save embeddings, then exit")
    parser.add_argument("--run-type", default="experiment",
                        choices=["experiment", "production", "debug"],
                        help="Run type tag for MLflow")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.sample is not None:
        config["data"]["sample"] = args.sample

    out_dir = config["data"]["output_dir"]
    fig_dir = config["data"]["figures_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Init MLflow
    init_mlflow(config)

    # Load data
    input_path = config["data"]["input"]
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)

    pre_df = df[df["era"] == "pre-GPT"].copy()
    post_df = df[df["era"] == "post-GPT"].copy()

    sample_n = config["data"].get("sample")
    if sample_n:
        pre_df = pre_df.sample(n=min(sample_n, len(pre_df)), random_state=42)
        post_df = post_df.sample(n=min(sample_n, len(post_df)), random_state=42)
        print(f"  Sampled to {len(pre_df):,} pre-GPT, {len(post_df):,} post-GPT")
    else:
        print(f"  Full dataset: {len(pre_df):,} pre-GPT, {len(post_df):,} post-GPT")

    pre_docs = pre_df["text_clean"].tolist()
    post_docs = post_df["text_clean"].tolist()

    # Embeddings 
    for era, docs in [("pre_gpt", pre_docs), ("post_gpt", post_docs)]:
        if args.reuse_embeddings:
            emb = load_embeddings(era, out_dir, len(docs))
            if emb is None:
                generate_embeddings(docs, era, config, out_dir)
        else:
            generate_embeddings(docs, era, config, out_dir)

    if args.embed_only:
        print("\n  --embed-only flag set. Embeddings saved. Exiting.")
        return

    # Fit both eras inside a parent MLflow run
    with mlflow.start_run(run_name=f"bertopic_{args.run_type}_{config['hdbscan']['min_cluster_size']}"):
        mlflow.set_tag("run_type", args.run_type)
        mlflow.set_tag("git_commit", get_git_hash())

        for era, docs in [("pre_gpt", pre_docs), ("post_gpt", post_docs)]:
            print(f"\n{'='*60}")
            print(f"{era.upper().replace('_', '-')} TOPIC MODEL")
            print(f"{'='*60}")

            embeddings = load_embeddings(era, out_dir, len(docs))
            if embeddings is None:
                embeddings = generate_embeddings(docs, era, config, out_dir)

            model, topics, probs, metrics = fit_and_evaluate(
                docs, embeddings, era, config, out_dir, fig_dir,
                run_type=args.run_type,
            )

        # Comparison summary 
        print(f" ")
        print("COMPARISON SUMMARY")
        print(f"--------------------------------")
        print(f"  Config: mcs={config['hdbscan']['min_cluster_size']}, "
              f"nn={config['umap']['n_neighbors']}, "
              f"nc={config['umap']['n_components']}")
        print(f"  View results: mlflow ui --port 5000")
        print(f"  Then open: http://localhost:5000")


if __name__ == "__main__":
    main()