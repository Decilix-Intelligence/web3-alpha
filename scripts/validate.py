#!/usr/bin/env python3
"""Validate the code-only repository and an optional local dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_REQUIRED = (
    "README.md",
    "pyproject.toml",
    "w3alpha/bridge.py",
    "sentinels/backtest/runner.py",
    "scripts/run_integrated_pipeline.py",
    "scripts/check_anonymity.py",
)
DATA_REQUIRED = (
    "data/articles/cms_article.parquet",
    "data/behaviors/click.parquet",
    "data/behaviors/like.parquet",
    "data/behaviors/collect.parquet",
    "data/behaviors/comment.parquet",
    "data/behaviors/dislike.parquet",
    "splits/impressions_train.parquet",
    "splits/impressions_val.parquet",
    "splits/impressions_test.parquet",
    "data/webrec_v1/interactions.csv",
    "data/webrec_v1/item_features.csv",
    "data/webrec_v1/train.csv",
    "data/webrec_v1/valid.csv",
    "data/webrec_v1/test.csv",
)
ARTICLE_FIELDS = {"id", "title", "create_time", "language"}
BEHAVIOR_FIELDS = {"create_by", "article_id", "create_time"}
IMPRESSION_FIELDS = {"user_id", "article_id", "timestamp", "behavior", "label"}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"(?:/Users/|/home/)[^\s]+")


def pass_(message: str) -> None:
    print(f"  [PASS] {message}")


def fail(message: str, errors: list[str]) -> None:
    print(f"  [FAIL] {message}")
    errors.append(message)


def check_repository(root: Path, errors: list[str]) -> None:
    print("\n=== Code-only repository ===")
    for relative in REPOSITORY_REQUIRED:
        path = root / relative
        if path.is_file():
            pass_(relative)
        else:
            fail(f"missing repository file: {relative}", errors)

    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if "docs/DATA.md" in text:
            pass_("README links to the dataset file contract")
        else:
            fail("README is missing the dataset file contract link", errors)


def check_required_data(root: Path, errors: list[str]) -> bool:
    print("\n=== Dataset files ===")
    all_present = True
    for relative in DATA_REQUIRED:
        path = root / relative
        if path.is_file():
            pass_(relative)
        else:
            all_present = False
            fail(f"missing dataset file: {relative}", errors)
    return all_present


def check_articles(root: Path, errors: list[str]) -> None:
    print("\n=== Article schema and PII sample ===")
    path = root / "data/articles/cms_article.parquet"
    if not path.is_file():
        return
    table = pq.read_table(path)
    missing = ARTICLE_FIELDS.difference(table.column_names)
    if missing:
        fail(f"article table missing fields: {sorted(missing)}", errors)
        return
    pass_("article schema contains required fields")

    text_columns = [
        name
        for name in ("title", "ai_synopsis", "synopsis", "content")
        if name in table.column_names
    ]
    sample = table.select(text_columns).slice(0, min(5000, len(table))).to_pydict()
    suspect = []
    for column, values in sample.items():
        for row, value in enumerate(values):
            text = str(value or "")
            if EMAIL_RE.search(text) or LOCAL_PATH_RE.search(text):
                suspect.append(f"{column}[{row}]")
                if len(suspect) >= 5:
                    break
        if len(suspect) >= 5:
            break
    if suspect:
        fail(f"article sample contains PII-like text at {', '.join(suspect)}", errors)
    else:
        pass_("no email/local-path patterns in first 5,000 article rows")


def check_behavior_tables(root: Path, errors: list[str]) -> None:
    print("\n=== Behavior schemas ===")
    for name in ("click", "like", "collect", "comment", "dislike"):
        path = root / f"data/behaviors/{name}.parquet"
        if not path.is_file():
            continue
        table = pq.read_table(path)
        if set(table.column_names) != BEHAVIOR_FIELDS:
            fail(f"{name} schema mismatch: {table.column_names}", errors)
            continue
        user_field = table.schema.field("create_by")
        if not str(user_field.type).startswith(("int", "uint")):
            fail(f"{name}.create_by must be an anonymous integer index", errors)
        else:
            pass_(f"{name} schema uses numeric anonymous identifiers")


def check_impressions(root: Path, errors: list[str]) -> None:
    print("\n=== Impression splits ===")
    sizes: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = root / f"splits/impressions_{split}.parquet"
        if not path.is_file():
            continue
        table = pq.read_table(path)
        if set(table.column_names) != IMPRESSION_FIELDS:
            fail(f"impressions_{split} schema mismatch: {table.column_names}", errors)
            continue
        sizes[split] = len(table)
        pass_(f"impressions_{split}: {len(table):,} rows")

    if sizes and sum(sizes.values()) > 0:
        total = sum(sizes.values())
        expected = {"train": 0.8, "val": 0.1, "test": 0.1}
        for split, target in expected.items():
            actual = sizes.get(split, 0) / total
            if abs(actual - target) > 0.02:
                fail(f"{split} ratio {actual:.3f} differs from {target:.2f}", errors)
            else:
                pass_(f"{split} ratio {actual:.3f}")


def check_release_mappings(root: Path, errors: list[str]) -> None:
    print("\n=== Anonymous release mappings ===")
    mapping_specs = (
        ("user_mapping.csv", "user_idx", "user_id"),
        ("item_mapping.csv", "item_idx", "item_id"),
    )
    for filename, index_column, legacy_column in mapping_specs:
        path = root / "data/webrec_v1" / filename
        if not path.is_file():
            print(f"  [SKIP] {filename} is not part of this release")
            continue
        frame = pd.read_csv(path)
        if not {index_column, legacy_column}.issubset(frame.columns):
            fail(f"{filename} missing mapping columns", errors)
            continue
        left = pd.to_numeric(frame[index_column], errors="coerce")
        right = pd.to_numeric(frame[legacy_column], errors="coerce")
        expected = pd.Series(range(len(frame)), dtype="int64")
        if (
            left.isna().any()
            or right.isna().any()
            or not (
                left.astype("int64").reset_index(drop=True).equals(expected)
                and right.astype("int64").reset_index(drop=True).equals(expected)
            )
        ):
            fail(f"{filename} contains a non-anonymous or non-contiguous mapping", errors)
        else:
            pass_(f"{filename} contains anonymous self-maps only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="optional root containing local data/ and splits/ files for validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    root = args.root.resolve()
    check_repository(root, errors)

    if args.data_root is None:
        print("\n[SKIP] No --data-root supplied; dataset validation was not requested.")
    else:
        data_root = args.data_root.resolve()
        check_required_data(data_root, errors)
        check_articles(data_root, errors)
        check_behavior_tables(data_root, errors)
        check_impressions(data_root, errors)
        check_release_mappings(data_root, errors)

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        return 1
    print("ALL REQUESTED CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
