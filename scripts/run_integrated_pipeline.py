#!/usr/bin/env python3
"""Run W3 popularity recommendation and optionally Sentinels sentiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from w3alpha.bridge import (  # noqa: E402
    BridgeError,
    SentimentDependencyError,
    run_integrated_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge anonymous W3 Web-Rec data to News2Alpha inputs using the "
            "pandas-only popularity recommender, then optionally run sentiment."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "webrec_v1",
        help="Directory containing train.csv and item_features.csv",
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        help="Recommendation date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--user-id",
        "--user-idx",
        dest="user_idx",
        type=int,
        required=True,
        help="Anonymous zero-based user_idx (the legacy --user-id spelling is retained)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Maximum unseen items to recommend")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "integrated",
        help="Destination for articles.csv and daily_recommendations.csv",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Stop after the offline popularity/export stage (no torch, models, or network)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_integrated_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            user_idx=args.user_idx,
            top_k=args.top_k,
            target_date=args.target_date,
            skip_sentiment=args.skip_sentiment,
        )
    except (BridgeError, SentimentDependencyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: integrated pipeline failed: {exc}", file=sys.stderr)
        return 1

    artifacts = result.artifacts
    print(f"Interaction source: {artifacts.source_interactions_path}")
    print(f"Anonymous user_idx: {artifacts.user_idx}")
    print(f"Recommended item_idx: {','.join(map(str, artifacts.item_indices))}")
    print(f"Articles: {artifacts.articles_path}")
    print(f"Daily recommendations: {artifacts.recommendations_path}")

    if args.skip_sentiment:
        print("Sentiment: skipped (offline lightweight run complete)")
    else:
        context = result.sentiment_context
        metadata = context.get_metadata()
        if metadata["errors"]:
            print(
                f"error: sentiment pipeline reported {len(metadata['errors'])} error(s)",
                file=sys.stderr,
            )
            return 1
        sentiment = context.get("sentiment_result")
        print(
            "Sentiment: "
            f"mean={sentiment.sentiment_mean:+.4f}, articles={sentiment.articles_count}"
        )
        if context.has("prompt_text"):
            print("\nStrategy memo:\n")
            print(context.get("prompt_text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
