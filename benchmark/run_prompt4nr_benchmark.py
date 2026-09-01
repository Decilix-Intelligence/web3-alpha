#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import re
import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NEWSREC_DATA_DIR = (
    BASE_DIR / "newsrec" / "artifacts" / "webrec_mind_v2" / "newsreclib_data"
).resolve()
DEFAULT_PROMPT4NR_DATA_DIR = (
    BASE_DIR / "newsrec" / "artifacts" / "webrec_mind_v2" / "prompt4nr_data"
).resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "newsrec" / "runs" / "prompt4nr_discrete_relevance").resolve()
DEFAULT_RESULTS_PATH = (BASE_DIR / "results" / "llm_results.csv").resolve()
METRIC_PATTERN = re.compile(
    r"Test: AUC: (?P<AUC>[0-9.]+)\tMRR: (?P<MRR>[0-9.]+)\tnDCG@5: (?P<NDCG5>[0-9.]+)\tnDCG@10: (?P<NDCG10>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prompt4NR benchmark on Web-Rec export.")
    parser.add_argument("--prompt4nr-root", type=Path, required=True)
    parser.add_argument("--newsreclib-data-dir", type=Path, default=DEFAULT_NEWSREC_DATA_DIR)
    parser.add_argument("--prompt4nr-data-dir", type=Path, default=DEFAULT_PROMPT4NR_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_prompt4nr_data(args: argparse.Namespace) -> None:
    if args.skip_export and (args.prompt4nr_data_dir / "test.txt").exists():
        return
    run(
        [
            "python",
            str(BASE_DIR / "newsrec" / "export_prompt4nr_data.py"),
            "--newsreclib-data-dir",
            str(args.newsreclib_data_dir),
            "--output-dir",
            str(args.prompt4nr_data_dir),
        ]
    )


def build_run_script(args: argparse.Namespace) -> str:
    return f"""set -euo pipefail
cd "{args.prompt4nr_root / 'Discrete-Relevance'}"
export CUDA_VISIBLE_DEVICES="{args.gpu_id}"
python main-multigpu.py \
  --data_path "{args.prompt4nr_data_dir}" \
  --model_name "{args.model_name}" \
  --epochs {args.epochs} \
  --batch_size {args.batch_size} \
  --test_batch_size {args.test_batch_size} \
  --wd 1e-3 \
  --max_tokens {args.max_tokens} \
  --log True \
  --model_save True \
  > "{args.output_dir / 'train.log'}" 2>&1
python predict.py \
  --data_path "{args.prompt4nr_data_dir}" \
  --model_name "{args.model_name}" \
  --test_batch_size {args.test_batch_size} \
  --max_tokens {args.max_tokens} \
  --model_file ./temp/BestModel.pt \
  --log True \
  > "{args.output_dir / 'test.log'}" 2>&1
"""


def parse_test_metrics(test_log: Path) -> dict[str, float]:
    text = test_log.read_text(encoding="utf-8", errors="ignore")
    match = None
    for match in METRIC_PATTERN.finditer(text):
        pass
    if match is None:
        raise RuntimeError(f"Could not parse Prompt4NR metrics from {test_log}")
    groups = match.groupdict()
    return {
        "AUC": float(groups["AUC"]),
        "MRR": float(groups["MRR"]),
        "NDCG@5": float(groups["NDCG5"]),
        "NDCG@10": float(groups["NDCG10"]),
    }


def find_prompt4nr_metric_log(args: argparse.Namespace) -> Path:
    """Find the log file containing Prompt4NR's final test metric line."""
    candidates = [args.output_dir / "test.log"]
    discrete_root = args.prompt4nr_root / "Discrete-Relevance"
    for log_dir_name in ("log-Test-Small", "log-Test"):
        log_dir = discrete_root / log_dir_name
        if log_dir.exists():
            candidates.extend(
                sorted(log_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            )

    for path in candidates:
        if not path.exists():
            continue
        try:
            parse_test_metrics(path)
            return path
        except RuntimeError:
            continue
    raise RuntimeError(
        "Could not parse Prompt4NR metrics from runner test.log or Prompt4NR log-Test directories"
    )


def infer_evaluated_users(prompt4nr_data_dir: Path) -> int:
    with (prompt4nr_data_dir / "test.txt").open("rb") as f:
        test = pickle.load(f)
    return len(test[0])


def upsert_results(results_path: Path, rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if results_path.exists():
        old_df = pd.read_csv(results_path)
        new_keys = set(zip(new_df["model"], new_df["metric"]))
        keep = [(m, metric) not in new_keys for m, metric in zip(old_df["model"], old_df["metric"])]
        old_df = old_df.loc[keep]
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(["model", "metric"]).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(results_path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_prompt4nr_data(args)
    if args.export_only:
        print(f"Exported Prompt4NR data to {args.prompt4nr_data_dir}")
        return

    run(["bash", "-lc", build_run_script(args)])
    metric_log = find_prompt4nr_metric_log(args)
    metrics = parse_test_metrics(metric_log)
    evaluated_users = infer_evaluated_users(args.prompt4nr_data_dir)
    rows = [
        {
            "model": "Prompt4NR",
            "family": "llm-track",
            "protocol": "impression-ranking",
            "metric": metric,
            "value": value,
            "evaluated_users": evaluated_users,
        }
        for metric, value in metrics.items()
    ]
    upsert_results(args.results_path, rows)
    print(f"Saved Prompt4NR metrics to {args.results_path}")
    for metric, value in metrics.items():
        print(f"{metric}={value:.6f}")
    print(f"evaluated_users={evaluated_users}")


if __name__ == "__main__":
    main()
