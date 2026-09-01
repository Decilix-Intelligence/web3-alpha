#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse
import logging

# Add the project path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="News2Alpha daily sentiment pipeline: news -> factors -> strategy memo"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD). If not specified, uses latest available date.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        help="User ID for news recommendation. If not specified, uses config default.",
    )
    parser.add_argument("--articles", help="Article CSV/Parquet produced by the bridge")
    parser.add_argument(
        "--recommendations", help="Daily recommendations CSV produced by the bridge"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    os.environ.setdefault("FINBERT_TRANSLATE", "0")
    if args.articles:
        os.environ["NEWS_ARTICLES_PATH"] = str(Path(args.articles).resolve())
    if args.recommendations:
        os.environ["NEWS_RECOMMENDATIONS_PATH"] = str(Path(args.recommendations).resolve())

    # Configure logging
    setup_logging(args.log_level)

    from sentinels.pipelines.daily import DailySentimentPipeline
    from sentinels.config import get_config

    # Load config
    config = get_config()

    # Create and run the pipeline
    pipeline = DailySentimentPipeline(config=config)

    try:
        context = pipeline.run(target_date=args.date, user_id=args.user_id)

        # Print a results summary
        print("\n" + "=" * 70)
        print("Pipeline Execution Summary")
        print("=" * 70)

        if context.has("target_date"):
            print(f"  Target Date: {context.get('target_date')}")

        if context.has("sentiment_result"):
            sentiment = context.get("sentiment_result")
            print(f"  Sentiment Mean: {sentiment.sentiment_mean:+.4f}")
            print(f"  Articles Analyzed: {sentiment.articles_count}")

        if context.has("prompt_text"):
            print(f"  Memo Length: {len(context.get('prompt_text'))} chars")

        metadata = context.get_metadata()
        if metadata["errors"]:
            print(f"  [!] Errors: {len(metadata['errors'])}")
            for error in metadata["errors"]:
                print(f"    - {error['message']}")
        else:
            print("  [OK] All steps completed successfully")

        print("=" * 70)

        # Return the status code
        return 0 if not metadata["errors"] else 1

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
