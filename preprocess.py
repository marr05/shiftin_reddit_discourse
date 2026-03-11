"""
preprocess.py - Data loading and preprocessing 

This script:
1. Loads filtered JSONL files (output of reddit_data.py)
2. Preserves metadata (timestamp, subreddit, score, etc.)
3. Cleans text
4. Assigns era labels (pre-GPT vs post-GPT)
5. Runs VADER sentiment
6. Applies rule-based theme tags
7. Exports a single analysis-ready parquet file

Usage:
    python preprocess.py --input /path/to/jsonl/folder --output ./data/processed.parquet
"""

import json
import re
import os
import argparse
from datetime import datetime, timezone

import pandas as pd
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup 
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
ANALYZER = SentimentIntensityAnalyzer()

# ChatGPT public release date
GPT_RELEASE = datetime(2022, 11, 30, tzinfo=timezone.utc)

# Theme keywords 
THEMES = {
    "AI_Anxiety": [
        "ai", "automation", "replaced", "displaced", "obsolete",
        "future-proof", "job security", "ai taking my job",
        "worried about ai", "outsmart", "eradicated",
    ],
    "Learning_Barriers": [
        "bootcamp", "coursera", "udemy", "certificate", "expensive",
        "cost", "unstructured", "outdated", "low quality",
        "not worth it", "free resources", "unwilling to pay",
    ],
    "Learning_Styles": [
        "project-based", "hands-on", "portfolio", "projects",
        "video", "tutorial", "visual learner", "learn by doing",
        "step-by-step",
    ],
    "Upskilling_Motivation": [
        "upskill", "reskill", "career change", "career switch",
        "pivot", "new skills", "mid-career", "learn to code",
        "necessity",
    ],
}

THEME_KW_FLAT = set(kw for kws in THEMES.values() for kw in kws)


# Loading data
def load_jsonl_folder(folder: str) -> list[dict]:
    """Load all .txt / .jsonl files produced by reddit_data.py"""
    
    records = []
    files = [f for f in os.listdir(folder) if f.endswith((".txt", ".jsonl"))]
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl files in {folder}")

    print(f"Found {len(files)} file(s) in {folder}")
    for fname in sorted(files):
        path = os.path.join(folder, fname)
        print(f"  Loading {fname} ")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Distinguish submissions vs comments
                is_submission = "title" in obj and "selftext" in obj

                title = obj.get("title", "")
                body = obj.get("body", obj.get("selftext", ""))
                text = f"{title} {body}".strip()
                if not text:
                    continue

                # Parse timestamp
                ts_raw = obj.get("created_utc")
                if ts_raw is None:
                    continue
                try:
                    ts = datetime.fromtimestamp(int(float(ts_raw)), tz=timezone.utc)
                except (ValueError, OSError):
                    continue

                records.append({
                    "id": obj.get("id", ""),
                    "subreddit": obj.get("subreddit", ""),
                    "author": obj.get("author", ""),
                    "created_utc": ts,
                    "score": obj.get("score", 0),
                    "num_comments": obj.get("num_comments", 0) if is_submission else None,
                    "is_submission": is_submission,
                    "title": title,
                    "body": body,
                    "text_raw": text,
                })

    print(f" Total records loaded: {len(records):,}")
    return records


# Cleaning
def clean_text(text: str) -> str:
    """Lowercase, strip URLs, collapse whitespace. Keeps punctuation for VADER."""
    
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)       # remove URLs
    text = re.sub(r"/r/\w+|/u/\w+", "", text)      # remove reddit links
    text = re.sub(r"&amp;?#?x200b;?", "", text)    # zero-width spaces
    text = re.sub(r"&amp;", "and", text)            # HTML ampersand
    text = re.sub(r"&gt;?", "", text)               # HTML greater-than (quote markers)
    text = re.sub(r"&lt;?", "", text)               # HTML less-than
    text = re.sub(r"&nbsp;?", " ", text)            # non-breaking spaces
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def clean_for_ngrams(text: str) -> list[str]:
    """Cleans for n-gram / embedding input: no punctuation, no stopwords."""
    
    text = re.sub(r"[^\w\s]", "", text)
    return [w for w in text.split() if w not in STOP_WORDS or w in THEME_KW_FLAT]


# Tagging
def assign_era(ts: datetime) -> str:
    return "post-GPT" if ts >= GPT_RELEASE else "pre-GPT"


def assign_themes(text: str) -> list[str]:
    """Rule-based multi-label theme tagging."""
    
    found = []
    for theme, keywords in THEMES.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                found.append(theme)
                break
    return found if found else ["Other"]


def vader_compound(text: str) -> float:
    return ANALYZER.polarity_scores(text)["compound"]


# Main pipeline
def build_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    print("Cleaning text")
    df["text_clean"] = df["text_raw"].apply(clean_text)

    print("Assigning eras")
    df["era"] = df["created_utc"].apply(assign_era)

    print("Running VADER sentiment")
    df["sentiment"] = df["text_clean"].apply(vader_compound)

    print("Tagging themes (rule-based)")
    df["themes"] = df["text_clean"].apply(assign_themes)

    # Convenience columns
    df["year_month"] = df["created_utc"].dt.to_period("M")
    df["year_quarter"] = df["created_utc"].dt.to_period("Q")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total records:   {len(df):>10,}")
    print(f"  pre-GPT:       {(df['era']=='pre-GPT').sum():>10,}")
    print(f"  post-GPT:      {(df['era']=='post-GPT').sum():>10,}")
    print(f"Date range:      {df['created_utc'].min().date()} to {df['created_utc'].max().date()}")
    print(f"Subreddits:      {df['subreddit'].nunique()}")
    print(f"{'='*50}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Reddit JSONL for research paper")
    parser.add_argument("--input", required=True, help="Folder with JSONL files from reddit_data.py")
    parser.add_argument("--output", default="./data/processed.parquet", help="Output parquet path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records = load_jsonl_folder(args.input)
    df = build_dataframe(records)

    # Save as parquet - preserves dtypes, much faster to reload than CSV
    # Convert period columns to strings for parquet compatibility
    
    df["year_month"] = df["year_month"].astype(str)
    df["year_quarter"] = df["year_quarter"].astype(str)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}  ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()