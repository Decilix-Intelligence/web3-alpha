"""Offline tests for the W3-to-Sentinels bridge."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sentinels.data.loader import NewsDataLoader
from w3alpha.bridge import (
    BridgeError,
    build_sentiment_config,
    export_news2alpha_inputs,
)


def _write_snapshot(data_dir: Path, interaction_name: str = "train.csv") -> None:
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"user_idx": 0, "item_idx": 0, "weight": 1.0},
            {"user_idx": 0, "item_idx": 2, "weight": 1.0},
            {"user_idx": 1, "item_idx": 1, "weight": 5.0},
            {"user_idx": 1, "item_idx": 3, "weight": 2.0},
            {"user_idx": 2, "item_idx": 3, "weight": 4.0},
        ]
    ).to_csv(data_dir / interaction_name, index=False)
    pd.DataFrame(
        [
            {
                "item_idx": idx,
                "item_id": 900_000 + idx,  # Must never reach generated files.
                "title": f"Anonymous article {idx}",
                "ai_synopsis": f"AI synopsis {idx}",
                "synopsis": f"Synopsis {idx}",
                "language": "en",
            }
            for idx in range(5)
        ]
    ).to_csv(data_dir / "item_features.csv", index=False)


def test_exports_weighted_popularity_without_raw_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "webrec_v1"
    _write_snapshot(data_dir)

    artifacts = export_news2alpha_inputs(
        data_dir=data_dir,
        output_dir=tmp_path / "generated",
        user_idx=0,
        top_k=3,
        target_date="2026-01-15",
    )

    # User 0 has seen 0 and 2.  Global weighted popularity ranks 3, then 1;
    # never-interacted item 4 remains a valid cold candidate.
    assert artifacts.item_indices == (3, 1, 4)
    assert artifacts.source_interactions_path.name == "train.csv"

    articles = pd.read_csv(artifacts.articles_path)
    assert articles.columns.tolist() == [
        "id",
        "title",
        "ai_synopsis",
        "synopsis",
        "url",
        "language",
    ]
    assert articles["id"].tolist() == [3, 1, 4]
    assert "item_id" not in articles.columns
    assert (
        not articles.astype(str)
        .apply(lambda column: column.str.contains("90000", regex=False).any())
        .any()
    )
    assert articles["url"].fillna("").tolist() == ["", "", ""]

    recommendations = pd.read_csv(artifacts.recommendations_path, dtype=str)
    assert recommendations.to_dict("records") == [
        {"updated_at": "2026-01-15", "user_id": "0", "article_ids": "3,1,4"}
    ]

    loader = NewsDataLoader(
        str(artifacts.articles_path),
        str(artifacts.recommendations_path),
        max_news=3,
    )
    loaded = loader.get_news_for_date("2026-01-15", user_id=0)
    assert [article.id for article in loaded] == [3, 1, 4]


def test_in_memory_config_uses_generated_paths(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "webrec_v1"
    _write_snapshot(data_dir)
    artifacts = export_news2alpha_inputs(data_dir, tmp_path / "out", 1, top_k=2)

    assert artifacts.source_interactions_path.name == "train.csv"
    config = build_sentiment_config(artifacts)
    assert config.get_articles_path() == str(artifacts.articles_path)
    assert config.get_recommendations_path() == str(artifacts.recommendations_path)
    assert config.get_target_user_id() == 1
    assert config.get_max_news() == 2
    assert config.get_llm_api_key() == ""


def test_rejects_full_interactions_without_a_training_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "webrec_v1"
    _write_snapshot(data_dir, interaction_name="interactions.csv")

    with pytest.raises(BridgeError, match="validation/test events"):
        export_news2alpha_inputs(data_dir, tmp_path / "out", 0)


def test_rejects_non_anonymous_item_indices(tmp_path: Path) -> None:
    data_dir = tmp_path / "webrec_v1"
    _write_snapshot(data_dir)
    features = pd.read_csv(data_dir / "item_features.csv")
    features["item_idx"] = features["item_id"]
    features.to_csv(data_dir / "item_features.csv", index=False)

    with pytest.raises(BridgeError, match="zero-based and contiguous"):
        export_news2alpha_inputs(data_dir, tmp_path / "out", 0, target_date="2026-01-15")


def test_lightweight_cli_is_offline_and_reports_bad_inputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "webrec_v1"
    _write_snapshot(data_dir)
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_integrated_pipeline.py"
    output_dir = tmp_path / "cli-output"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--date",
            "2026-01-15",
            "--user-id",
            "0",
            "--top-k",
            "2",
            "--output-dir",
            str(output_dir),
            "--skip-sentiment",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Recommended item_idx: 3,1" in completed.stdout
    assert "Sentiment: skipped" in completed.stdout

    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--user-id",
            "99",
            "--skip-sentiment",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 2
    assert "does not occur" in failed.stderr
