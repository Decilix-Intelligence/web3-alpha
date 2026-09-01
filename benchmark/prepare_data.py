#!/usr/bin/env python3
"""
Prepare the Web-Rec benchmark dataset from raw CSV snapshots.

Outputs:
- data/interactions.csv
- data/user_mapping.csv
- data/item_mapping.csv
- data/item_features.csv
- data/train.csv
- data/valid.csv
- data/test.csv
- data/stats.csv
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DATA_DIR = (BASE_DIR.parent / "data" / "raw" / "chain_cms_backup").resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "data").resolve()
DEFAULT_CUTOFF_DATE = "2026-01-15"
DEFAULT_MIN_USER_INTERACTIONS = 5
DEFAULT_MIN_ITEM_INTERACTIONS = 5

ARTICLE_COLUMNS = [
    "id",
    "title",
    "ai_synopsis",
    "synopsis",
    "keywords",
    "language",
    "create_time",
    "is_show",
    "data_status",
]


@dataclass
class BenchmarkStats:
    cutoff_date: str
    min_user_interactions: int
    min_item_interactions: int
    n_interactions: int
    n_users: int
    n_items: int
    density: float
    dropped_eval_users_valid: int
    dropped_eval_users_test: int
    n_train: int
    n_valid: int
    n_test: int
    date_range: str


def _parse_timestamp(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    millis_mask = numeric.notna() & (numeric.abs() >= 10**11)
    seconds_mask = numeric.notna() & ~millis_mask

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if millis_mask.any():
        parsed.loc[millis_mask] = pd.to_datetime(
            numeric.loc[millis_mask], unit="ms", errors="coerce"
        )
    if seconds_mask.any():
        parsed.loc[seconds_mask] = pd.to_datetime(
            numeric.loc[seconds_mask], unit="s", errors="coerce"
        )

    string_mask = parsed.isna()
    if string_mask.any():
        parsed.loc[string_mask] = pd.to_datetime(series.loc[string_mask], errors="coerce")
    return parsed


def _load_articles(raw_dir: Path, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    article_path = raw_dir / "cms_article.csv"
    articles = pd.read_csv(article_path, usecols=lambda c: c in ARTICLE_COLUMNS)
    articles["create_time"] = _parse_timestamp(articles["create_time"])
    articles["title"] = articles["title"].fillna("").astype(str)
    for col in ["ai_synopsis", "synopsis", "keywords", "language"]:
        if col in articles:
            articles[col] = articles[col].fillna("").astype(str)

    if "is_show" in articles:
        articles["is_show"] = (
            pd.to_numeric(articles["is_show"], errors="coerce").fillna(0).astype(int)
        )
    if "data_status" in articles:
        articles["data_status"] = (
            pd.to_numeric(articles["data_status"], errors="coerce").fillna(0).astype(int)
        )

    articles = articles[
        articles["id"].notna()
        & (articles["create_time"] <= cutoff_ts)
        & (articles["is_show"] == 1)
        & (articles["data_status"] == 1)
    ].copy()
    articles["id"] = articles["id"].astype(int)
    articles = articles.drop_duplicates(subset=["id"], keep="last")
    return articles


def _load_single_interaction(path: Path, interaction_type: str, weight: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = _parse_timestamp(df["create_time"])
    df = df.dropna(subset=["create_by", "article_id", "timestamp"]).copy()
    df["user_id"] = pd.to_numeric(df["create_by"], errors="coerce")
    df["item_id"] = pd.to_numeric(df["article_id"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["interaction_type"] = interaction_type
    df["weight"] = weight
    return df[["user_id", "item_id", "timestamp", "interaction_type", "weight"]]


def load_raw_interactions(
    raw_dir: Path, cutoff_ts: pd.Timestamp, articles: pd.DataFrame
) -> pd.DataFrame:
    article_ids = set(articles["id"].tolist())
    parts = [
        _load_single_interaction(raw_dir / "cms_article_like.csv", "like", 1.0),
        _load_single_interaction(raw_dir / "cms_article_collect.csv", "collect", 2.0),
        _load_single_interaction(raw_dir / "cms_article_comment.csv", "comment", 1.5),
    ]
    interactions = pd.concat(parts, ignore_index=True)
    interactions = interactions[interactions["timestamp"] <= cutoff_ts].copy()
    interactions = interactions[interactions["item_id"].isin(article_ids)].copy()
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(
        drop=True
    )
    return interactions


def iterative_k_core(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    filtered = interactions.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_id").size()
        filtered = filtered[
            filtered["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)
        ]
        item_counts = filtered.groupby("item_id").size()
        filtered = filtered[
            filtered["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)
        ]
        if len(filtered) == before:
            break
    return filtered.reset_index(drop=True)


def remap_ids(
    interactions: pd.DataFrame, articles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Raw identifiers are used only in memory. Release mapping files deliberately
    # contain self-maps of the anonymous contiguous indices, never source IDs.
    raw_users = sorted(interactions["user_id"].unique())
    raw_items = sorted(interactions["item_id"].unique())
    user_id_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_users)}
    item_id_to_idx = {raw_id: idx for idx, raw_id in enumerate(raw_items)}
    user_mapping = pd.DataFrame(
        {"user_idx": range(len(raw_users)), "user_id": range(len(raw_users))}
    )
    item_mapping = pd.DataFrame(
        {"item_idx": range(len(raw_items)), "item_id": range(len(raw_items))}
    )

    remapped = interactions.copy()
    remapped["user_idx"] = remapped["user_id"].map(user_id_to_idx)
    remapped["item_idx"] = remapped["item_id"].map(item_id_to_idx)
    remapped = remapped.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)

    item_features = articles[articles["id"].isin(raw_items)].copy()
    item_features["item_idx"] = item_features["id"].map(item_id_to_idx)
    item_features["item_id"] = item_features["item_idx"]
    item_features = item_features[
        [
            "item_idx",
            "item_id",
            "title",
            "ai_synopsis",
            "synopsis",
            "keywords",
            "language",
            "create_time",
        ]
    ].sort_values("item_idx")

    return remapped, user_mapping, item_mapping, item_features


def leave_one_out_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    ranked = interactions.sort_values(["user_idx", "timestamp", "item_idx"]).copy()
    ranked["rev_rank"] = ranked.groupby("user_idx").cumcount(ascending=False)

    train = ranked[ranked["rev_rank"] >= 2].copy()
    valid = ranked[ranked["rev_rank"] == 1].copy()
    test = ranked[ranked["rev_rank"] == 0].copy()

    train_users = set(train["user_idx"].unique())
    train_items = set(train["item_idx"].unique())
    valid_filtered = valid[
        valid["user_idx"].isin(train_users) & valid["item_idx"].isin(train_items)
    ].copy()
    test_filtered = test[
        test["user_idx"].isin(train_users) & test["item_idx"].isin(train_items)
    ].copy()

    dropped_valid = len(valid) - len(valid_filtered)
    dropped_test = len(test) - len(test_filtered)
    cols = ["user_idx", "item_idx", "timestamp", "interaction_type", "weight"]
    return train[cols], valid_filtered[cols], test_filtered[cols], dropped_valid, dropped_test


def build_stats(
    interactions: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    cutoff_date: str,
    min_user_interactions: int,
    min_item_interactions: int,
    dropped_valid: int,
    dropped_test: int,
) -> BenchmarkStats:
    date_min = interactions["timestamp"].min().strftime("%Y-%m-%d")
    date_max = interactions["timestamp"].max().strftime("%Y-%m-%d")
    return BenchmarkStats(
        cutoff_date=cutoff_date,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        n_interactions=len(interactions),
        n_users=interactions["user_idx"].nunique(),
        n_items=interactions["item_idx"].nunique(),
        density=len(interactions)
        / (interactions["user_idx"].nunique() * interactions["item_idx"].nunique()),
        dropped_eval_users_valid=dropped_valid,
        dropped_eval_users_test=dropped_test,
        n_train=len(train),
        n_valid=len(valid),
        n_test=len(test),
        date_range=f"{date_min} ~ {date_max}",
    )


def prepare_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    cutoff_date: str,
    min_user_interactions: int,
    min_item_interactions: int,
) -> BenchmarkStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff_ts = pd.Timestamp(cutoff_date)

    articles = _load_articles(raw_data_dir, cutoff_ts)
    interactions = load_raw_interactions(raw_data_dir, cutoff_ts, articles)
    interactions = iterative_k_core(interactions, min_user_interactions, min_item_interactions)
    interactions, user_mapping, item_mapping, item_features = remap_ids(interactions, articles)
    train, valid, test, dropped_valid, dropped_test = leave_one_out_split(interactions)

    interactions_to_save = interactions[
        ["user_idx", "item_idx", "timestamp", "interaction_type", "weight"]
    ]
    interactions_to_save.to_csv(output_dir / "interactions.csv", index=False)
    user_mapping.to_csv(output_dir / "user_mapping.csv", index=False)
    item_mapping.to_csv(output_dir / "item_mapping.csv", index=False)
    item_features.to_csv(output_dir / "item_features.csv", index=False)
    train.to_csv(output_dir / "train.csv", index=False)
    valid.to_csv(output_dir / "valid.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    stats = build_stats(
        interactions_to_save,
        train,
        valid,
        test,
        cutoff_date,
        min_user_interactions,
        min_item_interactions,
        dropped_valid,
        dropped_test,
    )
    pd.DataFrame([asdict(stats)]).to_csv(output_dir / "stats.csv", index=False)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Web-Rec benchmark dataset.")
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--min-user-interactions", type=int, default=DEFAULT_MIN_USER_INTERACTIONS)
    parser.add_argument("--min-item-interactions", type=int, default=DEFAULT_MIN_ITEM_INTERACTIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        cutoff_date=args.cutoff_date,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    print("Prepared benchmark dataset:")
    for key, value in asdict(stats).items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
