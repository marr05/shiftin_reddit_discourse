# Shifting Reddit Discourse

Analyzing shifts in career-anxiety discussions on Reddit before and after the release of ChatGPT (November 30, 2022). The pipeline filters Reddit data by relevant keywords, preprocesses and labels it, then answers two research questions through statistical analysis and topic modeling.

## Project Structure

### Scripts

| File | Description |
|---|---|
| `reddit_data.py` | Filters raw Reddit data dumps (ZST-compressed) by career-anxiety keywords and date range (2020–2024). Outputs matched posts/comments as JSONL. Adapted from an external data-source repository. |
| `preprocess.py` | Loads the filtered JSONL files, cleans text, assigns era labels (pre-GPT vs post-GPT), runs VADER sentiment analysis, applies rule-based theme tags, and exports a single analysis-ready Parquet file. |
| `rq1_analysis.py` | **RQ1 — Volume & Sentiment.** Produces monthly volume time series, sentiment comparisons (Mann-Whitney U), TF-IDF distinguishing terms per era, and keyword surge tracking around the ChatGPT release. |
| `rq2_bertopic.py` | **RQ2 — Topic Modeling.** Runs a BERTopic pipeline (sentence embeddings → UMAP → HDBSCAN) on each era, evaluates topics via coherence and diversity metrics, and exports representative docs for manual labeling. All runs are logged to MLflow. |
| `sweep.py` | Performs a parameter sweep over UMAP and HDBSCAN hyperparameters, logging all results to MLflow. Requires pre-computed embeddings (`rq2_bertopic.py --embed-only`). |
| `mlflow_config.py` | Shared utilities — YAML config loading, MLflow experiment initialization, topic-model evaluation metrics (c_v coherence, diversity, outlier stats), and parameter flattening for logging. |

### Configs

YAML configuration files live in `configs/`:

- `default.yaml` — Default hyperparameters for embedding, UMAP, HDBSCAN, and BERTopic.
- `full_run.yaml` — Final production config with parameters selected from the sweep.
- `sweep_configs.yaml` — Defines the parameter grid used by `sweep.py`.

## Data

The `data/` folder is **not** checked into the repository — it is stored on **OneDrive** due to file size. Sync or download the shared OneDrive folder and place (or symlink) it as `./data/` in the project root before running the pipeline.

Key data artifacts:

- **Raw Reddit dumps** (`.zst` files) — input to `reddit_data.py`
- **Filtered JSONL** — output of `reddit_data.py`, input to `preprocess.py`
- **`processed.parquet`** — output of `preprocess.py`, used by all analysis scripts

## Typical Workflow

```

1. Filter raw Reddit ZST dumps with `reddit_data.py`.
2. Preprocess and label with `preprocess.py`.
3. Run RQ1 statistical analyses with `rq1_analysis.py`.
4. Generate embeddings and fit topic models with `rq2_bertopic.py`.
5. (Optional) Tune hyperparameters with `sweep.py`, then re-run with the best config.
