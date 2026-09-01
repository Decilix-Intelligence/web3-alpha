"""Shared split and evaluation helpers for the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TOP_KS = (10, 20, 50)


@dataclass
class EvaluationSummary:
    model: str
    auc: float
    mrr: float
    evaluated_users: int
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]


def iterative_k_core(
    interactions: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 5
) -> pd.DataFrame:
    filtered = interactions.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("user_idx").size()
        filtered = filtered[
            filtered["user_idx"].isin(user_counts[user_counts >= min_user_interactions].index)
        ]
        item_counts = filtered.groupby("item_idx").size()
        filtered = filtered[
            filtered["item_idx"].isin(item_counts[item_counts >= min_item_interactions].index)
        ]
        if len(filtered) == before:
            break
    return filtered.reset_index(drop=True)


def load_benchmark_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    interactions = pd.read_csv(data_dir / "interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    loaded = {"interactions": interactions}
    for split in ("train", "valid", "test"):
        split_path = data_dir / f"{split}.csv"
        if split_path.exists():
            df = pd.read_csv(split_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            loaded[split] = df
    return loaded


def leave_one_out_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = interactions.sort_values(["user_idx", "timestamp", "item_idx"]).copy()
    ranked["rev_rank"] = ranked.groupby("user_idx").cumcount(ascending=False)

    train = ranked[ranked["rev_rank"] >= 2].copy()
    valid = ranked[ranked["rev_rank"] == 1].copy()
    test = ranked[ranked["rev_rank"] == 0].copy()

    train_users = set(train["user_idx"].unique())
    train_items = set(train["item_idx"].unique())
    valid = valid[valid["user_idx"].isin(train_users) & valid["item_idx"].isin(train_items)].copy()
    test = test[test["user_idx"].isin(train_users) & test["item_idx"].isin(train_items)].copy()

    cols = ["user_idx", "item_idx", "timestamp", "interaction_type", "weight"]
    return train[cols], valid[cols], test[cols]


def _rank_metrics(
    rank: int, num_candidates: int, top_ks: Iterable[int]
) -> tuple[float, float, dict[int, float], dict[int, float], dict[int, float]]:
    auc = 1.0 if num_candidates <= 1 else (num_candidates - rank) / (num_candidates - 1)
    mrr = 1.0 / rank
    precision = {}
    recall = {}
    ndcg = {}
    for k in top_ks:
        hit = 1.0 if rank <= k else 0.0
        precision[k] = hit / k
        recall[k] = hit
        ndcg[k] = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
    return auc, mrr, precision, recall, ndcg


def evaluate_model(
    model,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    top_ks: Iterable[int] = TOP_KS,
    max_eval_users: int | None = None,
) -> EvaluationSummary:
    user_history = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    item_pool = sorted(train_df["item_idx"].unique().tolist())

    auc_values: list[float] = []
    mrr_values: list[float] = []
    precision_values = {k: [] for k in top_ks}
    recall_values = {k: [] for k in top_ks}
    ndcg_values = {k: [] for k in top_ks}

    for idx, row in enumerate(eval_df.itertuples(index=False)):
        if max_eval_users is not None and idx >= max_eval_users:
            break
        user_id = int(row.user_idx)
        ground_truth = int(row.item_idx)
        history = user_history.get(user_id)
        if not history:
            continue

        candidate_items = [item for item in item_pool if item not in history]
        if ground_truth not in candidate_items:
            continue

        ranked_items = model.recommend(
            user_id=user_id,
            user_history=history,
            candidate_items=candidate_items,
            n=len(candidate_items),
        )
        if not ranked_items:
            continue

        try:
            rank = ranked_items.index(ground_truth) + 1
        except ValueError:
            continue

        auc, mrr, precision, recall, ndcg = _rank_metrics(rank, len(candidate_items), top_ks)
        auc_values.append(auc)
        mrr_values.append(mrr)
        for k in top_ks:
            precision_values[k].append(precision[k])
            recall_values[k].append(recall[k])
            ndcg_values[k].append(ndcg[k])

    return EvaluationSummary(
        model=model.name,
        auc=float(np.mean(auc_values)) if auc_values else 0.0,
        mrr=float(np.mean(mrr_values)) if mrr_values else 0.0,
        evaluated_users=len(auc_values),
        precision={
            k: float(np.mean(precision_values[k])) if precision_values[k] else 0.0 for k in top_ks
        },
        recall={k: float(np.mean(recall_values[k])) if recall_values[k] else 0.0 for k in top_ks},
        ndcg={k: float(np.mean(ndcg_values[k])) if ndcg_values[k] else 0.0 for k in top_ks},
    )
