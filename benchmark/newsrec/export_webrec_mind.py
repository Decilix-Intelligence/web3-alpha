#!/usr/bin/env python3
"""Export Web-Rec interactions to a MIND-style dataset for news models."""

from __future__ import annotations

import argparse
import json
import re
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.prepare_data import _load_articles, load_raw_interactions

WORD_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


@dataclass(frozen=True)
class Impression:
    impid: int
    user_id: int
    time: pd.Timestamp
    history: list[str]
    positive: str
    negatives: list[str]


def _format_nid(value: int | str) -> str:
    token = str(int(value))
    return token if token.startswith("N") else f"N{token}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Web-Rec as a MIND-style benchmark dataset."
    )
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", default="2026-01-15")
    parser.add_argument("--language", default="cn")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)
    parser.add_argument("--train-neg-ratio", type=int, default=4)
    parser.add_argument("--eval-neg-ratio", type=int, default=19)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _slugify_keyword(value: str) -> str:
    token = re.sub(r"\W+", "_", value.strip().lower())
    return token.strip("_") or "kw"


def _split_keywords(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in re.split(r"[;,锛屻€亅/]\s*", value) if part.strip()]


def _hash_normal_vector(key: str, dim: int) -> np.ndarray:
    seed = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=dim).astype(np.float32)


def _entity_json_from_keywords(keywords: list[str]) -> str:
    entities = []
    for idx, keyword in enumerate(keywords):
        entities.append(
            {
                "WikidataId": f"KW_{_slugify_keyword(keyword)}",
                "OccurrenceOffsets": [idx],
                "Confidence": 1.0,
                "SurfaceForms": [keyword],
            }
        )
    return json.dumps(entities, ensure_ascii=False)


def _tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return WORD_PATTERN.findall(text.lower())


def _build_news_rows(articles: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    vocab: set[str] = set()
    entities: set[str] = set()
    rows = []

    for row in articles.itertuples(index=False):
        title = str(getattr(row, "title", "") or "")
        abstract = str(getattr(row, "synopsis", "") or "")
        keywords = _split_keywords(str(getattr(row, "keywords", "") or ""))
        category = keywords[0] if keywords else "unknown"
        subcategory = keywords[1] if len(keywords) > 1 else "unknown"
        entity_json = _entity_json_from_keywords(keywords)

        vocab.update(_tokenize_text(title))
        vocab.update(_tokenize_text(abstract))
        vocab.update(_tokenize_text(category))
        vocab.update(_tokenize_text(subcategory))
        for keyword in keywords:
            entities.add(f"KW_{_slugify_keyword(keyword)}")

        rows.append(
            {
                "nid": _format_nid(getattr(row, "id")),
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
                "url": "",
                "title_entities": entity_json,
                "abstract_entities": entity_json,
            }
        )

    return pd.DataFrame(rows), sorted(vocab), sorted(entities)


def _sample_negatives(
    rng: np.random.Generator,
    catalog: np.ndarray,
    blocked: set[str],
    sample_size: int,
) -> list[str]:
    if sample_size <= 0:
        return []
    candidates = [item for item in catalog.tolist() if item not in blocked]
    if not candidates:
        return []
    if len(candidates) <= sample_size:
        return candidates
    indices = rng.choice(len(candidates), size=sample_size, replace=False)
    return [candidates[int(idx)] for idx in indices]


def _build_impressions(
    interactions: pd.DataFrame,
    train_neg_ratio: int,
    eval_neg_ratio: int,
    seed: int,
) -> tuple[list[Impression], list[Impression], str]:
    catalog = np.array(
        sorted(interactions["item_id"].astype(int).map(_format_nid).unique().tolist())
    )
    grouped = interactions.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id")
    train_rows: list[Impression] = []
    dev_rows: list[Impression] = []
    train_times: list[pd.Timestamp] = []
    impid = 1

    for user_id, user_df in grouped:
        items = user_df["item_id"].astype(int).map(_format_nid).tolist()
        times = user_df["timestamp"].tolist()
        if len(items) < 3:
            continue

        rng = np.random.default_rng(seed + int(user_id))
        for idx in range(1, len(items) - 1):
            history = items[:idx]
            positive = items[idx]
            blocked = set(history)
            blocked.add(positive)
            negatives = _sample_negatives(rng, catalog, blocked, train_neg_ratio)
            if not negatives:
                continue
            impression = Impression(
                impid=impid,
                user_id=int(user_id),
                time=pd.Timestamp(times[idx]),
                history=history,
                positive=positive,
                negatives=negatives,
            )
            train_rows.append(impression)
            train_times.append(impression.time)
            impid += 1

        history = items[:-1]
        positive = items[-1]
        blocked = set(history)
        blocked.add(positive)
        negatives = _sample_negatives(rng, catalog, blocked, eval_neg_ratio)
        if negatives:
            dev_rows.append(
                Impression(
                    impid=impid,
                    user_id=int(user_id),
                    time=pd.Timestamp(times[-1]),
                    history=history,
                    positive=positive,
                    negatives=negatives,
                )
            )
            impid += 1

    if not train_times:
        raise ValueError("No train impressions generated.")

    valid_time_split = pd.Series(train_times).quantile(0.9)
    return train_rows, dev_rows, pd.Timestamp(valid_time_split).strftime("%Y-%m-%d %H:%M:%S")


def _serialize_behaviors(rows: list[Impression], output_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = []
    for row in rows:
        candidates = [(candidate, 0) for candidate in row.negatives] + [(row.positive, 1)]
        order = rng.permutation(len(candidates))
        shuffled = [candidates[int(idx)] for idx in order]
        impressions = " ".join(f"{nid}-{label}" for nid, label in shuffled)
        history = " ".join(row.history)
        time_str = pd.Timestamp(row.time).strftime("%m/%d/%Y %I:%M:%S %p")
        lines.append(f"{row.impid}\tU{row.user_id}\t{time_str}\t{history}\t{impressions}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_news_tsv(news_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    news_df.to_csv(output_path, sep="\t", index=False, header=False)


def _write_entity_embeddings(entity_ids: list[str], output_path: Path) -> None:
    lines = []
    for entity_id in entity_ids:
        vector = _hash_normal_vector(entity_id, 100)
        lines.append(entity_id + "\t" + "\t".join(f"{value:.6f}" for value in vector))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_pseudo_glove(vocab: list[str], glove_txt_path: Path) -> None:
    glove_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with glove_txt_path.open("w", encoding="utf-8") as handle:
        for token in vocab:
            vector = _hash_normal_vector(token, 300)
            handle.write(token + " " + " ".join(f"{value:.6f}" for value in vector) + "\n")


def _iterative_k_core_by_id(
    interactions: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    filtered = interactions.copy()
    while True:
        start_size = len(filtered)
        user_counts = filtered.groupby("user_id").size()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered["user_id"].isin(keep_users)]

        item_counts = filtered.groupby("item_id").size()
        keep_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered["item_id"].isin(keep_items)]

        if len(filtered) == start_size:
            return filtered.reset_index(drop=True)


def _zip_newsreclib_split(split_dir: Path, split_name: str, zip_path: Path) -> None:
    nested_news = split_dir / split_name / "news.tsv"
    behaviors = split_dir / "behaviors.tsv"
    entity_embedding = split_dir / "entity_embedding.vec"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(nested_news, arcname=f"{split_name}/news.tsv")
        archive.write(behaviors, arcname="behaviors.tsv")
        archive.write(entity_embedding, arcname="entity_embedding.vec")


def main() -> None:
    args = parse_args()
    cutoff_ts = pd.Timestamp(args.cutoff_date) + pd.Timedelta(days=1)

    articles = _load_articles(args.raw_data_dir, cutoff_ts)
    articles = articles[
        articles["language"].fillna("").astype(str).str.lower() == args.language.lower()
    ].copy()
    interactions = load_raw_interactions(args.raw_data_dir, cutoff_ts, articles)
    interactions = _iterative_k_core_by_id(
        interactions,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    interactions = interactions[
        interactions["item_id"].isin(set(articles["id"].astype(int).tolist()))
    ].copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    kept_items = set(interactions["item_id"].astype(int).tolist())
    articles = articles[articles["id"].isin(kept_items)].copy()

    news_df, vocab, entity_ids = _build_news_rows(articles)
    train_rows, dev_rows, valid_time_split = _build_impressions(
        interactions=interactions,
        train_neg_ratio=args.train_neg_ratio,
        eval_neg_ratio=args.eval_neg_ratio,
        seed=args.seed,
    )

    output_dir = args.output_dir.resolve()
    newsrec_root = output_dir / "newsreclib_data"
    train_root = newsrec_root / "MINDsmall_train"
    dev_root = newsrec_root / "MINDsmall_dev"
    legommenders_root = output_dir / "legommenders_mind"

    for root, split_name in ((train_root, "MINDsmall_train"), (dev_root, "MINDsmall_dev")):
        (root / split_name).mkdir(parents=True, exist_ok=True)
        _write_news_tsv(news_df, root / split_name / "news.tsv")
        _write_entity_embeddings(entity_ids, root / "entity_embedding.vec")

    _serialize_behaviors(train_rows, train_root / "behaviors.tsv", seed=args.seed)
    _serialize_behaviors(dev_rows, dev_root / "behaviors.tsv", seed=args.seed + 1)

    (legommenders_root / "train").mkdir(parents=True, exist_ok=True)
    (legommenders_root / "dev").mkdir(parents=True, exist_ok=True)
    _write_news_tsv(news_df, legommenders_root / "train" / "news.tsv")
    _write_news_tsv(news_df, legommenders_root / "dev" / "news.tsv")
    _serialize_behaviors(train_rows, legommenders_root / "train" / "behaviors.tsv", seed=args.seed)
    _serialize_behaviors(dev_rows, legommenders_root / "dev" / "behaviors.tsv", seed=args.seed + 1)

    glove_txt = newsrec_root / "glove" / "glove.840B.300d.txt"
    _write_pseudo_glove(vocab, glove_txt)
    with zipfile.ZipFile(
        newsrec_root / "glove.840B.300d.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(glove_txt, arcname="glove.840B.300d.txt")

    _zip_newsreclib_split(train_root, "MINDsmall_train", newsrec_root / "MINDsmall_train.zip")
    _zip_newsreclib_split(dev_root, "MINDsmall_dev", newsrec_root / "MINDsmall_dev.zip")

    metadata = {
        "language": args.language,
        "cutoff_date": args.cutoff_date,
        "valid_time_split": valid_time_split,
        "n_articles": int(len(news_df)),
        "n_interactions": int(len(interactions)),
        "n_train_behaviors": int(len(train_rows)),
        "n_dev_behaviors": int(len(dev_rows)),
        "train_neg_ratio": args.train_neg_ratio,
        "eval_neg_ratio": args.eval_neg_ratio,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
