"""Bridge the W3 recommendation snapshot to the Sentinels news pipeline.

The W3 snapshot uses anonymous, zero-based ``user_idx`` and ``item_idx``
values.  Sentinels predates that convention and expects CSV columns named
``user_id`` and ``id``.  This module writes the legacy column names for file
compatibility, but their values always come from the anonymous indices.  Raw
``user_id``/``item_id`` mapping values are never copied to bridge artifacts.

The lightweight path needs only pandas.  Sentiment dependencies are imported
only when :func:`run_sentiment_pipeline` is explicitly called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


class BridgeError(ValueError):
    """Raised when the W3 snapshot cannot be bridged safely."""


class SentimentDependencyError(RuntimeError):
    """Raised when the optional Sentinels model dependencies are unavailable."""


@dataclass(frozen=True)
class BridgeArtifacts:
    """Paths and anonymous indices emitted by the recommendation bridge."""

    articles_path: Path
    recommendations_path: Path
    source_interactions_path: Path
    user_idx: int
    item_indices: tuple[int, ...]
    target_date: str


@dataclass(frozen=True)
class IntegratedRun:
    """Result of the recommendation bridge and optional sentiment stage."""

    artifacts: BridgeArtifacts
    sentiment_context: Optional[Any] = None


class _GeneratedInputConfig:
    """Small in-memory config adapter for ``DailySentimentPipeline``.

    Keeping the generated file paths here avoids editing ``config/config.yaml``
    and prevents unrelated local configuration values from entering a run.
    """

    def __init__(self, artifacts: BridgeArtifacts, max_news: int):
        self._artifacts = artifacts
        self._max_news = max_news

    def get_articles_path(self) -> str:
        return str(self._artifacts.articles_path)

    def get_recommendations_path(self) -> str:
        return str(self._artifacts.recommendations_path)

    def get_target_user_id(self) -> int:
        # Sentinels' historical name; the value is the anonymous user_idx.
        return self._artifacts.user_idx

    def get_max_news(self) -> int:
        return self._max_news

    def get_finbert_model(self) -> str:
        return "ProsusAI/finbert"

    def get_finbert_batch_size(self) -> int:
        return 32

    def get_finbert_device(self) -> str:
        return "auto"

    # FinBERT's optional translator checks these methods.  Deliberately return
    # no credentials: a bridge run must never pull secrets from a local YAML.
    def get_llm_provider(self) -> str:
        return "openai"

    def get_llm_base_url(self) -> str:
        return ""

    def get_llm_model(self) -> str:
        return ""

    def get_llm_api_key(self) -> str:
        return ""


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise BridgeError(f"Missing {label}: {path}") from exc
    except Exception as exc:
        raise BridgeError(f"Could not read {label} at {path}: {exc}") from exc


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise BridgeError(f"{label} is missing required column(s): {', '.join(missing)}")


def _integer_indices(series: pd.Series, column: str, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() | (numeric % 1 != 0) | (numeric < 0)
    if invalid.any():
        bad_rows = ", ".join(str(value) for value in series[invalid].head(3).tolist())
        raise BridgeError(
            f"{label}.{column} must contain non-negative integer anonymous indices"
            f" (examples of invalid values: {bad_rows})"
        )
    return numeric.astype("int64")


def _validate_target_date(target_date: Optional[str]) -> str:
    if target_date is None:
        return calendar_date.today().isoformat()
    try:
        parsed = calendar_date.fromisoformat(target_date)
    except (TypeError, ValueError) as exc:
        raise BridgeError("--date must use ISO format YYYY-MM-DD") from exc
    if parsed.isoformat() != target_date:
        raise BridgeError("--date must use ISO format YYYY-MM-DD")
    return target_date


def _resolve_snapshot_files(data_dir: Path) -> tuple[Path, Path]:
    train_path = data_dir / "train.csv"
    if not train_path.is_file():
        raise BridgeError(
            f"Missing training split: {train_path}. The bridge will not fall back "
            "to interactions.csv because it may contain validation/test events."
        )

    features_path = data_dir / "item_features.csv"
    if not features_path.is_file():
        raise BridgeError(f"Missing item features: {features_path}")
    return train_path, features_path


def _load_snapshot(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    source_path, features_path = _resolve_snapshot_files(data_dir)
    interactions = _read_csv(source_path, "interaction source")
    features = _read_csv(features_path, "item features")

    _require_columns(interactions, ["user_idx", "item_idx"], source_path.name)
    _require_columns(features, ["item_idx", "title"], features_path.name)

    interactions = interactions.copy()
    features = features.copy()
    interactions["user_idx"] = _integer_indices(
        interactions["user_idx"], "user_idx", source_path.name
    )
    interactions["item_idx"] = _integer_indices(
        interactions["item_idx"], "item_idx", source_path.name
    )
    features["item_idx"] = _integer_indices(features["item_idx"], "item_idx", features_path.name)

    if interactions.empty:
        raise BridgeError(f"{source_path.name} contains no interactions")
    if features.empty:
        raise BridgeError("item_features.csv contains no items")
    if features["item_idx"].duplicated().any():
        duplicate = int(features.loc[features["item_idx"].duplicated(), "item_idx"].iloc[0])
        raise BridgeError(f"item_features.csv has duplicate item_idx {duplicate}")

    feature_indices = set(int(value) for value in features["item_idx"])
    expected_indices = set(range(len(feature_indices)))
    if feature_indices != expected_indices:
        raise BridgeError(
            "item_features.csv item_idx values must be zero-based and contiguous; "
            "raw item_id values are not accepted as anonymous indices"
        )

    missing_features = sorted(set(interactions["item_idx"]).difference(feature_indices))
    if missing_features:
        sample = ", ".join(str(value) for value in missing_features[:5])
        raise BridgeError(
            "Interactions reference item_idx values absent from item_features.csv: " f"{sample}"
        )

    if "weight" not in interactions:
        interactions["weight"] = 1.0
    else:
        weights = pd.to_numeric(interactions["weight"], errors="coerce")
        if weights.isna().any():
            raise BridgeError(f"{source_path.name}.weight must contain numeric values")
        interactions["weight"] = weights.astype(float)

    return interactions, features, source_path


def recommend_popular_items(
    interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_idx: int,
    top_k: int,
) -> list[int]:
    """Recommend unseen items by weighted global popularity.

    Ties are resolved by ascending ``item_idx`` so identical inputs always
    produce identical artifacts.  This implementation intentionally uses no
    scikit-learn components.
    """

    if isinstance(user_idx, bool) or not isinstance(user_idx, int) or user_idx < 0:
        raise BridgeError("--user-id must be a non-negative anonymous user_idx")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise BridgeError("--top-k must be a positive integer")

    _require_columns(interactions, ["user_idx", "item_idx", "weight"], "interactions")
    _require_columns(item_features, ["item_idx"], "item_features")

    user_rows = interactions[interactions["user_idx"] == user_idx]
    if user_rows.empty:
        available = sorted(int(value) for value in interactions["user_idx"].unique())
        preview = ", ".join(str(value) for value in available[:10])
        raise BridgeError(
            f"Anonymous user_idx {user_idx} does not occur in the interaction source"
            + (f"; available values begin with: {preview}" if preview else "")
        )

    seen = set(int(value) for value in user_rows["item_idx"])
    popularity = interactions.groupby("item_idx", sort=False)["weight"].sum()
    candidates = item_features[["item_idx"]].copy()
    candidates = candidates[~candidates["item_idx"].isin(seen)]
    candidates["popularity"] = candidates["item_idx"].map(popularity).fillna(0.0)
    candidates = candidates.sort_values(
        ["popularity", "item_idx"], ascending=[False, True], kind="mergesort"
    )

    ranked = [int(value) for value in candidates["item_idx"].head(top_k).tolist()]
    if not ranked:
        raise BridgeError(f"Anonymous user_idx {user_idx} has no unseen candidate items")
    return ranked


def _text_column(features: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in features:
        return pd.Series([default] * len(features), index=features.index, dtype="object")
    return features[column].fillna(default).astype(str)


def export_news2alpha_inputs(
    data_dir: str | Path,
    output_dir: str | Path,
    user_idx: int,
    top_k: int = 10,
    target_date: Optional[str] = None,
) -> BridgeArtifacts:
    """Create Sentinels-compatible CSVs from an anonymized W3 snapshot."""

    data_path = Path(data_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    normalized_date = _validate_target_date(target_date)
    interactions, features, source_path = _load_snapshot(data_path)
    ranked = recommend_popular_items(interactions, features, user_idx=user_idx, top_k=top_k)

    selected = features.set_index("item_idx", drop=False).loc[ranked].copy()
    # These are the exact columns NewsDataLoader consumes.  ``id`` is populated
    # only from item_idx; a source ``item_id`` column is intentionally ignored.
    articles = pd.DataFrame(
        {
            "id": selected["item_idx"].astype("int64"),
            "title": _text_column(selected, "title"),
            "ai_synopsis": _text_column(selected, "ai_synopsis"),
            "synopsis": _text_column(selected, "synopsis"),
            "url": _text_column(selected, "url"),
            "language": _text_column(selected, "language"),
        }
    )
    recommendations = pd.DataFrame(
        [
            {
                "updated_at": normalized_date,
                # Legacy compatibility name; value is the anonymous user_idx.
                "user_id": user_idx,
                "article_ids": ",".join(str(item_idx) for item_idx in ranked),
            }
        ],
        columns=["updated_at", "user_id", "article_ids"],
    )

    output_path.mkdir(parents=True, exist_ok=True)
    articles_path = output_path / "articles.csv"
    recommendations_path = output_path / "daily_recommendations.csv"
    articles.to_csv(articles_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    return BridgeArtifacts(
        articles_path=articles_path,
        recommendations_path=recommendations_path,
        source_interactions_path=source_path,
        user_idx=user_idx,
        item_indices=tuple(ranked),
        target_date=normalized_date,
    )


def build_sentiment_config(
    artifacts: BridgeArtifacts,
    max_news: Optional[int] = None,
) -> _GeneratedInputConfig:
    """Build an in-memory Sentinels config bound to generated bridge files."""

    effective_max = max_news if max_news is not None else len(artifacts.item_indices)
    if effective_max <= 0:
        raise BridgeError("Sentiment max_news must be positive")
    return _GeneratedInputConfig(artifacts, effective_max)


def run_sentiment_pipeline(artifacts: BridgeArtifacts) -> Any:
    """Run Sentinels against the generated files without changing YAML config."""

    try:
        # Lazy import preserves the pandas-only recommendation path.
        from sentinels.pipelines.daily import DailySentimentPipeline
    except (ImportError, ModuleNotFoundError) as exc:
        raise SentimentDependencyError(
            "The sentiment stage requires optional model packages. "
            "Install them with: pip install -e '.[sentiment]'"
        ) from exc

    config = build_sentiment_config(artifacts)
    pipeline = DailySentimentPipeline(config=config)
    return pipeline.run(target_date=artifacts.target_date, user_id=artifacts.user_idx)


def run_integrated_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    user_idx: int,
    top_k: int = 10,
    target_date: Optional[str] = None,
    skip_sentiment: bool = False,
) -> IntegratedRun:
    """Run W3 popularity recommendation, export, and optional sentiment."""

    artifacts = export_news2alpha_inputs(
        data_dir=data_dir,
        output_dir=output_dir,
        user_idx=user_idx,
        top_k=top_k,
        target_date=target_date,
    )
    context = None if skip_sentiment else run_sentiment_pipeline(artifacts)
    return IntegratedRun(artifacts=artifacts, sentiment_context=context)
