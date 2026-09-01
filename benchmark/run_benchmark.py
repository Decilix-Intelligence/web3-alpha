#!/usr/bin/env python3
"""Run the lightweight local benchmark baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines import ItemCFRecommender, PopularityRecommender, UserCFRecommender
from evaluator import (
    TOP_KS,
    evaluate_model,
    iterative_k_core,
    leave_one_out_split,
    load_benchmark_data,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (BASE_DIR / "data").resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "results").resolve()


def get_models() -> list:
    return [
        PopularityRecommender(),
        UserCFRecommender(n_neighbors=50),
        ItemCFRecommender(n_neighbors=50),
    ]


def build_result_rows(summary) -> list[dict]:
    rows = []
    for k in TOP_KS:
        rows.append(
            {
                "model": summary.model,
                "metric": f"Precision@{k}",
                "value": summary.precision[k],
                "evaluated_users": summary.evaluated_users,
            }
        )
        rows.append(
            {
                "model": summary.model,
                "metric": f"Recall@{k}",
                "value": summary.recall[k],
                "evaluated_users": summary.evaluated_users,
            }
        )
        rows.append(
            {
                "model": summary.model,
                "metric": f"NDCG@{k}",
                "value": summary.ndcg[k],
                "evaluated_users": summary.evaluated_users,
            }
        )
    rows.append(
        {
            "model": summary.model,
            "metric": "AUC",
            "value": summary.auc,
            "evaluated_users": summary.evaluated_users,
        }
    )
    rows.append(
        {
            "model": summary.model,
            "metric": "MRR",
            "value": summary.mrr,
            "evaluated_users": summary.evaluated_users,
        }
    )
    return rows


def print_summary(summary) -> None:
    print(f"\n[{summary.model}] users={summary.evaluated_users}")
    print(f"  AUC={summary.auc:.6f}  MRR={summary.mrr:.6f}")
    for k in TOP_KS:
        print(
            f"  @{k:<2} Precision={summary.precision[k]:.6f} "
            f"Recall={summary.recall[k]:.6f} NDCG={summary.ndcg[k]:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local benchmark baselines.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--max-eval-users", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_benchmark_data(args.data_dir)
    train_df = loaded.get("train")
    test_df = loaded.get("test")
    if train_df is None or test_df is None:
        filtered = iterative_k_core(
            loaded["interactions"],
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
        )
        train_df, _, test_df = leave_one_out_split(filtered)

    results = []
    for model in get_models():
        print(f"Training {model.name} ...")
        model.fit(train_df)
        summary = evaluate_model(
            model, train_df, test_df, top_ks=TOP_KS, max_eval_users=args.max_eval_users
        )
        print_summary(summary)
        results.extend(build_result_rows(summary))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output_dir / "benchmark_results.csv", index=False)
    print(f"\nSaved results to {args.output_dir / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
