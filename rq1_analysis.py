"""
rq1_analysis.py - Volume trends, sentiment comparison, keyword shifts (Pre vs Post GPT)

Reads the parquet file from preprocess.py and produces:
  - Monthly post volume time series (overall + by theme)
  - Sentiment distributions per era with Mann-Whitney U tests
  - TF-IDF distinguishing terms per era
  - Keyword surge analysis around the GPT release date
  - Publication-ready figures

Usage:
    python rq1_analysis.py --input ./data/processed.parquet --figures ./figures/
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

sns.set_theme(style="whitegrid", font_scale=1.1)
GPT_RELEASE = "2022-11-30"


# Helpers
def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# 1. Volume over time 
def plot_volume_timeseries(df, fig_dir):
    """Monthly post counts with vertical line at GPT release."""
    
    monthly = df.groupby("year_month").size().reset_index(name="count")
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly["year_month"], monthly["count"], linewidth=1.5, color="#2c7bb6")
    ax.axvline(pd.Timestamp(GPT_RELEASE), color="red", linestyle="--", alpha=0.8, label="ChatGPT Release")
    ax.fill_between(monthly["year_month"], monthly["count"], alpha=0.15, color="#2c7bb6")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Posts/Comments")
    ax.set_title("Career Anxiety Discussion Volume Over Time")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    savefig(fig, os.path.join(fig_dir, "rq1_volume_timeseries.png"))


def plot_volume_by_theme(df, fig_dir):
    """Monthly volume broken out by rule-based theme."""
    
    df_ex = df.explode("themes")
    df_ex = df_ex[df_ex["themes"] != "Other"]
    monthly_theme = (
        df_ex.groupby(["year_month", "themes"])
        .size()
        .reset_index(name="count")
    )
    monthly_theme["year_month"] = pd.to_datetime(monthly_theme["year_month"])

    fig, ax = plt.subplots(figsize=(14, 6))
    for theme, grp in monthly_theme.groupby("themes"):
        ax.plot(grp["year_month"], grp["count"], label=theme, linewidth=1.2)
    ax.axvline(pd.Timestamp(GPT_RELEASE), color="red", linestyle="--", alpha=0.7, label="ChatGPT Release")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts/Comments")
    ax.set_title("Discussion Volume by Theme Over Time")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    savefig(fig, os.path.join(fig_dir, "rq1_volume_by_theme.png"))


# 2. Sentiment comparison 
def sentiment_comparison(df, fig_dir):
    """Boxplots + Mann-Whitney U for overall and per-theme sentiment."""
    
    pre = df[df["era"] == "pre-GPT"]["sentiment"]
    post = df[df["era"] == "post-GPT"]["sentiment"]

    # Overall test
    u_stat, p_val = stats.mannwhitneyu(pre, post, alternative="two-sided")
    print(f"\n  Overall Sentiment Mann-Whitney U:")
    print(f"    pre-GPT  mean={pre.mean():.4f}  median={pre.median():.4f}  n={len(pre):,}")
    print(f"    post-GPT mean={post.mean():.4f}  median={post.median():.4f}  n={len(post):,}")
    print(f"    U={u_stat:,.0f}  p={p_val:.2e}")
    # Effect size (rank-biserial correlation)
    r_rb = 1 - (2 * u_stat) / (len(pre) * len(post))
    print(f"    Rank-biserial r={r_rb:.4f}")

    # Boxplot overall
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="era", y="sentiment", order=["pre-GPT", "post-GPT"],
                palette=["#abd9e9", "#fdae61"], ax=ax)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Sentiment by Era  (U={u_stat:,.0f}, p={p_val:.2e})")
    ax.set_ylabel("VADER Compound Score")
    ax.set_xlabel("")
    savefig(fig, os.path.join(fig_dir, "rq1_sentiment_boxplot.png"))

    # Per-theme tests
    df_ex = df.explode("themes")
    themes = [t for t in df_ex["themes"].unique() if t != "Other"]
    print("\n  Per-Theme Sentiment Tests:")
    results = []
    for t in sorted(themes):
        s_pre = df_ex[(df_ex["themes"] == t) & (df_ex["era"] == "pre-GPT")]["sentiment"]
        s_post = df_ex[(df_ex["themes"] == t) & (df_ex["era"] == "post-GPT")]["sentiment"]
        if len(s_pre) < 10 or len(s_post) < 10:
            continue
        u, p = stats.mannwhitneyu(s_pre, s_post, alternative="two-sided")
        r = 1 - (2 * u) / (len(s_pre) * len(s_post))
        results.append({"theme": t, "pre_mean": s_pre.mean(), "post_mean": s_post.mean(),
                         "U": u, "p": p, "r_rb": r, "n_pre": len(s_pre), "n_post": len(s_post)})
        print(f"    {t:25s}  pre={s_pre.mean():.4f}  post={s_post.mean():.4f}  U={u:,.0f}  p={p:.2e}  r={r:.4f}")

    # Per-theme boxplot
    df_plot = df_ex[df_ex["themes"].isin(themes)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_plot, x="themes", y="sentiment", hue="era",
                hue_order=["pre-GPT", "post-GPT"], palette=["#abd9e9", "#fdae61"], ax=ax)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Sentiment by Theme and Era")
    ax.set_ylabel("VADER Compound Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, os.path.join(fig_dir, "rq1_sentiment_by_theme.png"))

    return pd.DataFrame(results)


# 3. TF-IDF distinguishing terms 
def tfidf_distinguishing_terms(df, fig_dir, top_n=25):
    """Find terms that most distinguish pre-GPT from post-GPT using TF-IDF."""
    
    # Treat each era as a single document
    pre_text = " ".join(df[df["era"] == "pre-GPT"]["text_clean"].values)
    post_text = " ".join(df[df["era"] == "post-GPT"]["text_clean"].values)

    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          ngram_range=(1, 2))
    tfidf_matrix = vec.fit_transform([pre_text, post_text])
    features = vec.get_feature_names_out()

    pre_scores = tfidf_matrix[0].toarray().flatten()
    post_scores = tfidf_matrix[1].toarray().flatten()

    # diff:  positive = more distinctive of post-GPT
    diff = post_scores - pre_scores
    idx_sorted = np.argsort(diff)

    pre_top = [(features[i], pre_scores[i], diff[i]) for i in idx_sorted[:top_n]]
    post_top = [(features[i], post_scores[i], diff[i]) for i in idx_sorted[-top_n:][::-1]]

    print(f"\n  Top {top_n} terms distinguishing PRE-GPT:")
    for term, score, d in pre_top:
        print(f"    {term:30s}  tfidf={score:.4f}  diff={d:.4f}")

    print(f"\n  Top {top_n} terms distinguishing POST-GPT:")
    for term, score, d in post_top:
        print(f"    {term:30s}  tfidf={score:.4f}  diff={d:+.4f}")

    # viz: horizontal diverging bar chart
    terms = [t for t, _, _ in pre_top[:15]] + [t for t, _, _ in post_top[:15]]
    diffs = [d for _, _, d in pre_top[:15]] + [d for _, _, d in post_top[:15]]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2c7bb6" if d < 0 else "#d7191c" for d in diffs]
    ax.barh(terms[::-1], [d for d in diffs[::-1]], color=colors[::-1])
    ax.set_xlabel("TF-IDF Difference (post-GPT minus pre-GPT)")
    ax.set_title("Most Distinguishing Terms by Era")
    ax.axvline(0, color="black", linewidth=0.5)
    
    # Add legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2c7bb6", label="Pre-GPT"),
                        Patch(color="#d7191c", label="Post-GPT")])
    savefig(fig, os.path.join(fig_dir, "rq1_tfidf_distinguishing.png"))


# 4. Keyword surge analysis
def keyword_surge(df, fig_dir):
    """Track specific anxiety-related keywords monthly."""
    
    track_terms = ["chatgpt", "gpt", "llm", "replaced", "automation",
                   "job security", "layoff", "laid off", "obsolete", "upskill"]

    monthly = df.groupby("year_month")["text_clean"].apply(list).reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    results = {t: [] for t in track_terms}
    for _, row in monthly.iterrows():
        combined = " ".join(row["text_clean"])
        for term in track_terms:
            # Count posts containing the term 
            count = sum(1 for txt in row["text_clean"] if term in txt)
            results[term].append(count)

    fig, ax = plt.subplots(figsize=(14, 7))
    for term in track_terms:
        ax.plot(monthly["year_month"], results[term], label=term, linewidth=1.0, alpha=0.85)
    ax.axvline(pd.Timestamp(GPT_RELEASE), color="red", linestyle="--", alpha=0.7, label="ChatGPT Release")
    ax.set_xlabel("Month")
    ax.set_ylabel("Posts Mentioning Term")
    ax.set_title("Keyword Frequency Over Time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    savefig(fig, os.path.join(fig_dir, "rq1_keyword_surge.png"))


# Main 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/processed.parquet")
    parser.add_argument("--figures", default="./figures/")
    args = parser.parse_args()

    os.makedirs(args.figures, exist_ok=True)
    print(f"Loading {args.input} ")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} records  |  pre-GPT: {(df['era']=='pre-GPT').sum():,}  post-GPT: {(df['era']=='post-GPT').sum():,}")

    print(" ")
    print("1. VOLUME TIME SERIES")
    print("--------------------------------")
    plot_volume_timeseries(df, args.figures)
    plot_volume_by_theme(df, args.figures)

    print(" ")
    print("2. SENTIMENT COMPARISON")
    print("--------------------------------")
    results_df = sentiment_comparison(df, args.figures)

    print(" ")
    print("3. TF-IDF DISTINGUISHING TERMS")
    print("--------------------------------")
    tfidf_distinguishing_terms(df, args.figures)

    print(" ")
    print("4. KEYWORD SURGE ANALYSIS")
    print("--------------------------------")
    keyword_surge(df, args.figures)

    print("\nRQ1 analysis complete.")


if __name__ == "__main__":
    main()