"""把 5 个 behavior 表合并为 impression logs，按时序 80/10/10 切分。

输出：splits/impressions_{train,val,test}.parquet
列：user_id, article_id, timestamp, label, behavior

论文 §D.2.4：
  - 训练 80% / 验证 10% / 测试 10%
  - 按 timestamp 升序后切片
  - 原 label 字段 0=train, 1=val, 2=test —— 这里我们直接切 3 个文件，对 HF 更友好
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BEHAVIOR_FILES = {
    "click": "click.parquet",
    "like": "like.parquet",
    "collect": "collect.parquet",
    "comment": "comment.parquet",
    "dislike": "dislike.parquet",
}

IMPRESSION_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("article_id", pa.int64()),
        ("timestamp", pa.timestamp("ms", tz="UTC")),
        ("behavior", pa.string()),
        ("label", pa.int8()),  # 1 for any positive interaction; dislike → 0
    ]
)


def _load_behaviors(behaviors_dir: Path) -> pd.DataFrame:
    frames = []
    for name, fname in BEHAVIOR_FILES.items():
        p = behaviors_dir / fname
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        df = pd.read_parquet(p)
        df = df.rename(
            columns={
                "create_by": "user_id",
                "create_time": "timestamp",
            }
        )
        df["behavior"] = name
        df["label"] = 0 if name == "dislike" else 1
        frames.append(df[["user_id", "article_id", "timestamp", "behavior", "label"]])
        print(f"[load] {name}: {len(df):,} rows")
    if not frames:
        raise SystemExit("No behavior parquet found in " + str(behaviors_dir))
    return pd.concat(frames, ignore_index=True)


def _filter_known_articles(df: pd.DataFrame, articles_path: Path) -> pd.DataFrame:
    article_ids = set(pd.read_parquet(articles_path, columns=["id"])["id"].tolist())
    before = len(df)
    df = df[df["article_id"].isin(article_ids)].copy()
    print(f"[filter] kept {len(df):,}/{before:,} rows referencing known articles")
    return df


def _split_chronological(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    return {
        "train": df.iloc[:train_end],
        "validation": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True, help="path to cms_article.parquet")
    p.add_argument("--behaviors-dir", required=True, help="dir containing behavior parquets")
    p.add_argument("--output-dir", required=True, help="where to write impressions_*.parquet")
    args = p.parse_args()

    behaviors_dir = Path(args.behaviors_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_behaviors(behaviors_dir)
    df = _filter_known_articles(df, Path(args.articles))
    df = df.dropna(subset=["timestamp"])

    splits = _split_chronological(df)
    name_map = {"train": "train", "validation": "val", "test": "test"}
    for split_name, frame in splits.items():
        out = out_dir / f"impressions_{name_map[split_name]}.parquet"
        arrow = pa.Table.from_pandas(frame, schema=IMPRESSION_SCHEMA, preserve_index=False)
        pq.write_table(arrow, out, compression="zstd", compression_level=9)
        print(f"[write] {split_name}: {len(frame):,} rows → {out}")


if __name__ == "__main__":
    main()
