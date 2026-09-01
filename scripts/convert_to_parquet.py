"""W3 dataset → Parquet 转换脚本。

支持的输入格式：
  - SQL dump (.sql)          → 解析 INSERT INTO 语句
  - CSV / TSV (.csv, .tsv)
  - JSON / JSONL (.json, .jsonl, .ndjson)
  - Excel (.xlsx, .xls)
  - Parquet (.parquet)        → 仅做 schema 归一化

用法示例：
  python3 scripts/convert_to_parquet.py \
    --input raw/articles.csv --output data/articles.parquet --table articles
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# 目标 schema（论文 Table 12 / 13）
# --------------------------------------------------------------------------- #

ARTICLES_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("category_id", pa.int64()),
        ("title", pa.string()),
        ("ai_synopsis", pa.string()),
        ("synopsis", pa.string()),
        ("currency_id", pa.int64()),
        ("keywords", pa.string()),
        ("is_show", pa.bool_()),
        ("content", pa.string()),
        ("create_time", pa.timestamp("ms", tz="UTC")),
        ("update_time", pa.timestamp("ms", tz="UTC")),
        ("language", pa.string()),
        ("data_status", pa.int32()),
        ("visit_rank", pa.float32()),
    ]
)

BEHAVIOR_SCHEMA = pa.schema(
    [
        ("create_by", pa.int64()),
        ("article_id", pa.int64()),
        ("create_time", pa.timestamp("ms", tz="UTC")),
    ]
)

SCHEMAS = {"articles": ARTICLES_SCHEMA, "behavior": BEHAVIOR_SCHEMA}


# --------------------------------------------------------------------------- #
# 输入解析
# --------------------------------------------------------------------------- #


def _read_csv_like(path: Path, sep: str | None = None) -> pd.DataFrame:
    """容错读 CSV / TSV，支持多编码。"""
    if sep is None:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path} with any common encoding")


def _read_json_like(path: Path) -> pd.DataFrame:
    """JSON / JSONL 自适应读取。"""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return pd.DataFrame(json.loads(text))
    # JSONL / NDJSON
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows)


_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+`?(?P<table>\w+)`?\s*\((?P<cols>[^)]+)\)\s*VALUES\s*(?P<values>.+?);",
    re.IGNORECASE | re.DOTALL,
)
_VALUE_TUPLE_RE = re.compile(r"\((?:[^()']|'(?:\\.|[^'\\])*')*\)")


def _read_sql_dump(path: Path) -> pd.DataFrame:
    """从 SQL dump 输出里抽取 INSERT INTO 行。

    粗糙但够用：假定单文件对应单表；多表会拼到一起，可能不是用户想要的。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(_INSERT_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"No INSERT INTO statements found in {path}")

    all_rows: list[list[object]] = []
    columns: list[str] = []
    for m in matches:
        cols = [c.strip().strip("`") for c in m.group("cols").split(",")]
        if not columns:
            columns = cols
        for tup in _VALUE_TUPLE_RE.findall(m.group("values")):
            row = _parse_sql_tuple(tup)
            if len(row) == len(columns):
                all_rows.append(row)
    return pd.DataFrame(all_rows, columns=columns)


def _parse_sql_tuple(tup: str) -> list[object]:
    """把 `(1, 'a', NULL, '2024-01-01 00:00:00')` 拆成 Python 值。"""
    body = tup[1:-1]
    out: list[object] = []
    buf = []
    in_str = False
    i = 0
    while i < len(body):
        c = body[i]
        if in_str:
            if c == "\\" and i + 1 < len(body):
                buf.append(body[i + 1])
                i += 2
                continue
            if c == "'":
                in_str = False
                out.append("".join(buf))
                buf = []
            else:
                buf.append(c)
            i += 1
            continue
        if c == "'":
            in_str = True
            i += 1
            continue
        if c == ",":
            token = "".join(buf).strip()
            if buf:
                out.append(_coerce(token))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    token = "".join(buf).strip()
    if token:
        out.append(_coerce(token))
    return out


def _coerce(token: str) -> object:
    if token.upper() == "NULL":
        return None
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


# --------------------------------------------------------------------------- #
# 字段归一化
# --------------------------------------------------------------------------- #


def _strip_html(s: object) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    text = str(s)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    text = re.sub(r"https?://\S+", "", text)  # 去 URL
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _to_utc_ms(s: object) -> pd.Timestamp | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, (int, float)):
        unit = "ms" if s > 1e11 else "s"
        return pd.Timestamp(s, unit=unit, tz="UTC")
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize(df: pd.DataFrame, table: str, column_map: dict[str, str] | None) -> pd.DataFrame:
    if column_map:
        df = df.rename(columns=column_map)

    schema = SCHEMAS[table]
    target_cols = [f.name for f in schema]

    # 补齐缺列
    for col in target_cols:
        if col not in df.columns:
            df[col] = None

    df = df[target_cols].copy()

    # 类型转换
    if table == "articles":
        df["content"] = df["content"].map(_strip_html)
        df["ai_synopsis"] = df["ai_synopsis"].map(_strip_html)
        df["synopsis"] = df["synopsis"].map(_strip_html)
        df["title"] = df["title"].map(lambda s: None if s is None else str(s).strip())
        df["keywords"] = df["keywords"].map(lambda s: None if s is None else str(s).strip())
        df["language"] = df["language"].fillna("zh").map(str)
        df["is_show"] = df["is_show"].map(lambda v: bool(int(v)) if v not in (None, "") else None)
        for c in ("id", "category_id", "currency_id", "data_status"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        df["visit_rank"] = pd.to_numeric(df["visit_rank"], errors="coerce").astype("Float32")
        df["create_time"] = df["create_time"].map(_to_utc_ms)
        df["update_time"] = df["update_time"].map(_to_utc_ms)

        before = len(df)
        df = df[df["content"].str.len().fillna(0) >= 10]
        print(f"[clean] dropped {before - len(df)} articles with body < 10 chars")
    else:
        for c in ("create_by", "article_id"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        df["create_time"] = df["create_time"].map(_to_utc_ms)

    df = df.dropna(subset=["id"] if table == "articles" else ["create_by", "article_id"])
    return df


def _write_parquet(df: pd.DataFrame, table: str, output: Path) -> None:
    schema = SCHEMAS[table]
    output.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(arrow_table, output, compression="zstd", compression_level=9)
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"[done] wrote {len(df):,} rows → {output}  ({size_mb:.1f} MB, zstd-9)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _read_csv_like(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _read_json_like(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".sql":
        return _read_sql_dump(path)
    sys.exit(f"Unsupported file extension: {suffix}")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Convert raw W3 data to normalized Parquet.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to input file (.sql/.csv/.tsv/.json/.jsonl/.xlsx/.parquet)",
    )
    p.add_argument("--output", required=True, help="Output Parquet path")
    p.add_argument(
        "--table",
        required=True,
        choices=["articles", "behavior"],
        help="Which target schema to apply",
    )
    p.add_argument("--column-map", help='JSON dict to rename columns, e.g. \'{"news_id":"id"}\'')
    args = p.parse_args(list(argv) if argv is not None else None)

    column_map = json.loads(args.column_map) if args.column_map else None

    raw = _load(args)
    print(f"[read] {len(raw):,} rows, columns: {list(raw.columns)}")
    cleaned = _normalize(raw, args.table, column_map)
    _write_parquet(cleaned, args.table, Path(args.output))


if __name__ == "__main__":
    main()
