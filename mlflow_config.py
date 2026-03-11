"""
mlflow_config.py - MLflow setup, config loading, and evaluation helpers

Provides:
  - YAML config loading with CLI overrides
  - MLflow experiment initialization
  - Topic model evaluation metrics (coherence, diversity, outlier stats)
  - Param flattening for MLflow logging
"""

import os
import yaml
import subprocess
import mlflow
import numpy as np
from copy import deepcopy
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


# Config Loading 

def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config file."""
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, overrides: dict) -> dict:
    """Deep merge overrides into base config."""
    
    result = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def apply_sweep_params(base_config: dict, sweep_params: dict) -> dict:
    """
    Apply a single sweep parameter combination to base config.
    sweep_params is a flat dict like {"hdbscan.min_cluster_size": 75, "umap.n_neighbors": 10}
    """
    
    config = deepcopy(base_config)
    for dotted_key, value in sweep_params.items():
        keys = dotted_key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config


def flatten_config(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for MLflow log_params. {'umap': {'n_neighbors': 15}} -> {'umap.n_neighbors': 15}"""
    
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_config(v, key))
        else:
            items[key] = v
    return items


# MLflow Setup

def init_mlflow(config: dict) -> str:
    """Initialize MLflow tracking. Returns experiment ID."""
    
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "./mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "career-anxiety-bertopic")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    print(f"  MLflow experiment: {experiment_name} (tracking: {tracking_uri})")
    return experiment.experiment_id


def get_git_hash() -> str:
    """Get current git commit hash for tagging runs."""
    
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def log_run_context(era: str, run_type: str, config: dict):
    """Log standard tags and params at the start of a run."""
    
    mlflow.set_tag("era", era)
    mlflow.set_tag("run_type", run_type)
    mlflow.set_tag("git_commit", get_git_hash())

    # Log all config params (flattened)
    params = flatten_config(config)
    # MLflow has a 500-param limit; filter out non-model params
    model_keys = {"embedding", "umap", "hdbscan", "bertopic", "data.sample"}
    filtered = {k: v for k, v in params.items()
                if any(k.startswith(mk) for mk in model_keys)}
    mlflow.log_params(filtered)


# Evaluation Metrics

def compute_coherence_cv(topic_model, docs: list[str], top_n: int = 10,
                         max_docs: int = 20000) -> float:
    """
    Compute c_v coherence score using Gensim.
    Standard metric for topic model evaluation in NLP papers.

    Uses a random subsample of docs for the co-occurrence matrix when
    len(docs) > max_docs, since coherence scores stabilize well before
    using the full corpus. We can use this for shorter runs.

    Returns the mean c_v coherence across all non-outlier topics.
    """
    
    topics = topic_model.get_topics()
    if not topics:
        return 0.0

    # Build topic word lists (excluding outlier topic -1)
    topic_words = []
    for tid, words in topics.items():
        if tid == -1:
            continue
        topic_words.append([w for w, _ in words[:top_n]])

    if not topic_words:
        return 0.0

    # Subsample docs for coherence 
    if len(docs) > max_docs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(docs), size=max_docs, replace=False)
        docs_sample = [docs[i] for i in indices]
    else:
        docs_sample = docs

    # Tokenize docs for Gensim
    tokenized = [doc.split() for doc in docs_sample]

    # Build dictionary and compute coherence
    dictionary = Dictionary(tokenized)
    try:
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
    except Exception as e:
        print(f"  WARNING: Coherence computation failed: {e}")
        score = 0.0

    return score


def compute_topic_diversity(topic_model, top_n: int = 10) -> float:
    """
    Topic diversity: fraction of unique words across all topic representations.
    Range [0, 1]. Higher = more diverse topics (less redundancy).
    """
    
    topics = topic_model.get_topics()
    all_words = []
    for tid, words in topics.items():
        if tid == -1:
            continue
        all_words.extend([w for w, _ in words[:top_n]])

    if not all_words:
        return 0.0

    return len(set(all_words)) / len(all_words)


def compute_all_metrics(topic_model, docs: list[str], topics: list[int]) -> dict:
    """Compute all evaluation metrics for a single BERTopic run."""
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1)
    outlier_pct = n_outliers / len(topics) if topics else 0

    # Topic sizes (excluding outliers)
    from collections import Counter
    counts = Counter(t for t in topics if t != -1)
    sizes = list(counts.values())
    avg_topic_size = np.mean(sizes) if sizes else 0
    median_topic_size = np.median(sizes) if sizes else 0

    print(f"  Computing c_v coherence (this may take a minute) ..")
    coherence = compute_coherence_cv(topic_model, docs)
    diversity = compute_topic_diversity(topic_model)

    metrics = {
        "num_topics": n_topics,
        "num_outliers": n_outliers,
        "outlier_pct": round(outlier_pct, 4),
        "avg_topic_size": round(avg_topic_size, 1),
        "median_topic_size": round(median_topic_size, 1),
        "coherence_cv": round(coherence, 4),
        "topic_diversity": round(diversity, 4),
        "num_docs": len(docs),
    }

    print(f"  Metrics: {n_topics} topics | {outlier_pct:.1%} outliers | "
          f"coherence={coherence:.4f} | diversity={diversity:.4f}")

    return metrics


def log_metrics_to_mlflow(metrics: dict):
    """Log a metrics dict to the active MLflow run."""
    
    mlflow.log_metrics(metrics)


def log_artifact_safe(path: str):
    """Log an artifact if it exists."""
    
    if os.path.exists(path):
        mlflow.log_artifact(path)