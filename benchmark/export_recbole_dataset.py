#!/usr/bin/env python3
"""Export the benchmark data into RecBole atomic files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluator import iterative_k_core, leave_one_out_split

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (BASE_DIR / "data").resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "recbole_data" / "webrec").resolve()
DATASET_NAME = "webrec"


def _format_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["user_idx", "item_idx", "timestamp", "weight"]].copy()
    out["user_id:token"] = out["user_idx"].astype(str)
    out["item_id:token"] = out["item_idx"].astype(str)
    out["timestamp:float"] = pd.to_datetime(out["timestamp"]).astype("int64") / 1e9
    out["rating:float"] = out["weight"].astype(float)
    return out[["user_id:token", "item_id:token", "rating:float", "timestamp:float"]]


def export_recbole_dataset(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {split: data_dir / f"{split}.csv" for split in ("train", "valid", "test")}
    if all(path.exists() for path in split_paths.values()):
        split_frames = {split: pd.read_csv(path) for split, path in split_paths.items()}
        full_df = pd.concat(
            [split_frames["train"], split_frames["valid"], split_frames["test"]], ignore_index=True
        )
    else:
        interactions = pd.read_csv(data_dir / "interactions.csv")
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
        filtered = iterative_k_core(interactions, min_user_interactions=5, min_item_interactions=5)
        train_df, valid_df, test_df = leave_one_out_split(filtered)
        split_frames = {"train": train_df, "valid": valid_df, "test": test_df}
        full_df = filtered

    full_df = full_df.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)
    full_formatted = _format_split(full_df)
    full_formatted.to_csv(output_dir / f"{DATASET_NAME}.inter", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark splits to RecBole format.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_recbole_dataset(args.data_dir, args.output_dir)
    print(f"Exported RecBole dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
