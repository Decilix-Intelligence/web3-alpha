#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Web-Rec data to Prompt4NR pickle format.")
    parser.add_argument(
        "--newsreclib-data-dir",
        type=Path,
        required=True,
        help="Path to newsreclib_data root, e.g. .../webrec_mind_v2/newsreclib_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Prompt4NR pickle files",
    )
    return parser.parse_args()


def _load_behaviors(path: Path) -> tuple[list[int], list[str], list[str], list[list[list[str]]]]:
    df = pd.read_table(path)
    imp_ids: list[int] = []
    users: list[str] = []
    times: list[str] = []
    behaviors: list[list[list[str]]] = []

    for imp_id, row in enumerate(df.itertuples(index=False), start=1):
        history = ast.literal_eval(row.history)
        candidates = ast.literal_eval(row.candidates)
        labels = ast.literal_eval(row.labels)
        positives = [nid for nid, label in zip(candidates, labels) if int(label) == 1]
        negatives = [nid for nid, label in zip(candidates, labels) if int(label) == 0]

        if not positives or not negatives:
            continue

        imp_ids.append(imp_id)
        users.append(str(row.uid))
        times.append("")
        behaviors.append([history, positives, negatives])

    return imp_ids, users, times, behaviors


def _load_news(path: Path) -> dict[str, dict[str, str]]:
    df = pd.read_table(path)
    return {
        str(row.nid): {
            "title": "" if pd.isna(row.title) else str(row.title),
            "abstract": "" if pd.isna(row.abstract) else str(row.abstract),
            "category": "" if pd.isna(row.category) else str(row.category),
            "subcategory": "" if pd.isna(row.subcategory) else str(row.subcategory),
        }
        for row in df.itertuples(index=False)
    }


def _dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    train_dir = args.newsreclib_data_dir / "MINDsmall_train"
    dev_dir = args.newsreclib_data_dir / "MINDsmall_dev"

    news = _load_news(train_dir / "parsed_news.tsv")
    train = _load_behaviors(train_dir / "train_parsed_behaviors.tsv")
    val = _load_behaviors(train_dir / "val_parsed_behaviors.tsv")
    test = _load_behaviors(dev_dir / "parsed_behaviors.tsv")

    _dump_pickle(args.output_dir / "news.txt", news)
    _dump_pickle(args.output_dir / "train.txt", train)
    _dump_pickle(args.output_dir / "val.txt", val)
    _dump_pickle(args.output_dir / "test.txt", test)

    print(f"Exported Prompt4NR data to {args.output_dir}")
    print(
        f"news={len(news)} train_impressions={len(train[0])} val_impressions={len(val[0])} test_impressions={len(test[0])}"
    )


if __name__ == "__main__":
    main()
