"""Load a local article Parquet with Hugging Face Datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--articles",
        type=Path,
        default=Path("data/articles/cms_article.parquet"),
        help="local article Parquet from the released dataset",
    )
    args = parser.parse_args()
    if not args.articles.is_file():
        raise SystemExit(
            f"Dataset file not found: {args.articles}. "
            "Provide a local article Parquet with --articles; see docs/DATA.md."
        )

    articles = load_dataset("parquet", data_files={"full": str(args.articles)}, split="full")
    print(f"articles: {len(articles):,}")
    if articles:
        available = set(articles.column_names)
        fields = [name for name in ("id", "title", "language", "create_time") if name in available]
        print({name: articles[0][name] for name in fields})


if __name__ == "__main__":
    main()
