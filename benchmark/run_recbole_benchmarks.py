#!/usr/bin/env python3
"""Train RecBole models and evaluate them with batched full-sort scoring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

# RecBole 1.2.1 still expects NumPy scalar aliases removed in NumPy 2.x.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.abstract_recommender import AbstractRecommender, SequentialRecommender
from recbole.trainer import Trainer
from recbole.utils import get_model, init_logger, init_seed

from evaluator import TOP_KS, iterative_k_core, leave_one_out_split
from export_recbole_dataset import DATASET_NAME, export_recbole_dataset

if hasattr(sp, "dok_matrix"):

    def _dok_update_compat(self, data_dict):
        for key, value in data_dict.items():
            self[key] = value

    sp.dok_matrix._update = _dok_update_compat


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (BASE_DIR / "data").resolve()
DEFAULT_RECB0LE_DATA_DIR = (BASE_DIR / "recbole_data" / DATASET_NAME).resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "results").resolve()

MODEL_NAME_MAP = {
    "BPR": "BPR",
    "NCF": "NeuMF",
    "GRU4Rec": "GRU4Rec",
    "LightGCN": "LightGCN",
    "SASRec": "SASRec",
}
MODEL_LIST = list(MODEL_NAME_MAP.keys())


@dataclass
class Summary:
    model: str
    auc: float
    mrr: float
    evaluated_users: int
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]


def _ordered_history(train_df: pd.DataFrame) -> tuple[dict[int, set[int]], dict[int, list[int]]]:
    ordered = (
        train_df.sort_values(["user_idx", "timestamp", "item_idx"])
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
    )
    sets = {user: set(items) for user, items in ordered.items()}
    return sets, ordered


def build_config_dict(display_name: str, data_root: Path) -> dict:
    config = {
        "data_path": str(data_root.parent),
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "field_separator": "\t",
        "metrics": ["Recall", "MRR", "NDCG", "Precision"],
        "topk": list(TOP_KS),
        "valid_metric": "MRR@20",
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        },
        "epochs": 30,
        "stopping_step": 10,
        "eval_step": 1,
        "train_batch_size": 1024,
        "eval_batch_size": 2048,
        "learning_rate": 0.001,
        "reproducibility": True,
        "seed": 2026,
        "checkpoint_dir": str((BASE_DIR / "saved").resolve()),
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "state": "ERROR",
        "use_gpu": False,
    }
    if display_name in {"GRU4Rec", "SASRec"}:
        config["MAX_ITEM_LIST_LENGTH"] = 50
        config["loss_type"] = "CE"
        config["train_neg_sample_args"] = None
    if display_name == "NCF":
        config["mf_embedding_size"] = 64
        config["mlp_embedding_size"] = 64
    return config


def load_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    interactions = pd.read_csv(data_dir / "interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    filtered = iterative_k_core(interactions, min_user_interactions=5, min_item_interactions=5)
    train_df, _, test_df = leave_one_out_split(filtered)
    return train_df, test_df


def train_single_model(display_name: str, recbole_data_dir: Path, overrides: dict | None = None):
    recbole_model_name = MODEL_NAME_MAP[display_name]
    config_dict = build_config_dict(display_name, recbole_data_dir)
    if overrides:
        config_dict.update(overrides)
    config = Config(model=recbole_model_name, dataset=DATASET_NAME, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    model_class = get_model(recbole_model_name)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, verbose=False, saved=False, show_progress=False)
    return config, train_data.dataset, model


def _build_general_interaction(dataset, model, user_tokens: list[str]) -> Interaction:
    uid_field = dataset.uid_field
    user_internal = [dataset.token2id(uid_field, token) for token in user_tokens]
    return Interaction({uid_field: torch.tensor(user_internal, dtype=torch.long)}).to(model.device)


def _build_sequential_interaction(
    dataset, model, user_tokens: list[str], train_sequences: dict[int, list[int]]
) -> Interaction:
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    seq_field = model.ITEM_SEQ
    seq_len_field = model.ITEM_SEQ_LEN
    user_internal = []
    seqs = []
    seq_lens = []
    max_len = model.max_seq_length

    for token in user_tokens:
        user_id = int(token)
        user_internal.append(dataset.token2id(uid_field, token))
        history_items = train_sequences.get(user_id, [])
        history_internal = [
            dataset.token2id(iid_field, str(item))
            for item in history_items
            if str(item) in dataset.field2token_id[iid_field]
        ][-max_len:]
        seq_len = len(history_internal)
        padded = [0] * (max_len - seq_len) + history_internal
        seqs.append(padded)
        seq_lens.append(seq_len)

    return Interaction(
        {
            uid_field: torch.tensor(user_internal, dtype=torch.long),
            seq_field: torch.tensor(seqs, dtype=torch.long),
            seq_len_field: torch.tensor(seq_lens, dtype=torch.long),
        }
    ).to(model.device)


def _supports_full_sort_predict(model) -> bool:
    return type(model).full_sort_predict is not AbstractRecommender.full_sort_predict


def _predict_scores_via_predict(
    model,
    interaction: Interaction,
    iid_field: str,
    item_count: int,
    max_pairs: int = 262144,
) -> np.ndarray:
    batch_size = len(interaction)
    scores = np.full((batch_size, item_count), -np.inf, dtype=np.float32)
    candidate_items = torch.arange(1, item_count, dtype=torch.long, device=model.device)
    item_chunk_size = max(1, min(len(candidate_items), max_pairs // max(batch_size, 1)))

    for start in range(0, len(candidate_items), item_chunk_size):
        item_chunk = candidate_items[start : start + item_chunk_size]
        chunk_interaction = interaction.repeat_interleave(len(item_chunk))
        chunk_interaction.update(Interaction({iid_field: item_chunk.repeat(batch_size)}))
        chunk_scores = model.predict(chunk_interaction)
        chunk_scores = chunk_scores.detach().view(batch_size, -1).cpu().numpy()
        scores[:, item_chunk.detach().cpu().numpy()] = chunk_scores

    return scores


def evaluate_recbole_model_batch(
    display_name: str,
    dataset,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 64,
) -> Summary:
    train_history, train_sequences = _ordered_history(train_df)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    topk_precision = {k: [] for k in TOP_KS}
    topk_recall = {k: [] for k in TOP_KS}
    topk_ndcg = {k: [] for k in TOP_KS}
    auc_values = []
    mrr_values = []

    eval_rows = []
    for row in test_df.itertuples(index=False):
        user_id = int(row.user_idx)
        item_id = int(row.item_idx)
        if user_id not in train_history:
            continue
        user_token = str(user_id)
        item_token = str(item_id)
        if user_token not in dataset.field2token_id[uid_field]:
            continue
        if item_token not in dataset.field2token_id[iid_field]:
            continue
        eval_rows.append((user_id, item_id, user_token))

    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_rows), batch_size):
            batch = eval_rows[start : start + batch_size]
            user_tokens = [token for _, _, token in batch]
            if isinstance(model, SequentialRecommender):
                interaction = _build_sequential_interaction(
                    dataset, model, user_tokens, train_sequences
                )
            else:
                interaction = _build_general_interaction(dataset, model, user_tokens)

            if _supports_full_sort_predict(model):
                scores = model.full_sort_predict(interaction)
                if scores.dim() == 1:
                    scores = scores.view(len(batch), -1)
                scores = scores.detach().cpu().numpy()
                scores[:, 0] = -np.inf
            else:
                scores = _predict_scores_via_predict(
                    model,
                    interaction,
                    iid_field,
                    dataset.item_num,
                )

            for row_idx, (user_id, item_id, _) in enumerate(batch):
                history = train_history[user_id]
                history_internal = [
                    dataset.token2id(iid_field, str(hist_item))
                    for hist_item in history
                    if str(hist_item) in dataset.field2token_id[iid_field]
                ]
                if history_internal:
                    scores[row_idx, history_internal] = -np.inf

                gt_internal = dataset.token2id(iid_field, str(item_id))
                gt_score = scores[row_idx, gt_internal]
                if not np.isfinite(gt_score):
                    continue

                row_scores = scores[row_idx]
                better = int(np.sum(row_scores > gt_score))
                rank = better + 1
                candidate_count = int(np.sum(np.isfinite(row_scores)))

                auc = (
                    1.0
                    if candidate_count <= 1
                    else (candidate_count - rank) / (candidate_count - 1)
                )
                mrr = 1.0 / rank
                auc_values.append(auc)
                mrr_values.append(mrr)

                for k in TOP_KS:
                    hit = 1.0 if rank <= k else 0.0
                    topk_precision[k].append(hit / k)
                    topk_recall[k].append(hit)
                    topk_ndcg[k].append((1.0 / np.log2(rank + 1)) if rank <= k else 0.0)

    return Summary(
        model=display_name,
        auc=float(np.mean(auc_values)) if auc_values else 0.0,
        mrr=float(np.mean(mrr_values)) if mrr_values else 0.0,
        evaluated_users=len(auc_values),
        precision={
            k: float(np.mean(topk_precision[k])) if topk_precision[k] else 0.0 for k in TOP_KS
        },
        recall={k: float(np.mean(topk_recall[k])) if topk_recall[k] else 0.0 for k in TOP_KS},
        ndcg={k: float(np.mean(topk_ndcg[k])) if topk_ndcg[k] else 0.0 for k in TOP_KS},
    )


def summary_to_rows(summary: Summary) -> list[dict]:
    rows = [
        {
            "model": summary.model,
            "metric": "AUC",
            "value": summary.auc,
            "evaluated_users": summary.evaluated_users,
        }
    ]
    rows.append(
        {
            "model": summary.model,
            "metric": "MRR",
            "value": summary.mrr,
            "evaluated_users": summary.evaluated_users,
        }
    )
    for k in TOP_KS:
        rows.extend(
            [
                {
                    "model": summary.model,
                    "metric": f"Precision@{k}",
                    "value": summary.precision[k],
                    "evaluated_users": summary.evaluated_users,
                },
                {
                    "model": summary.model,
                    "metric": f"Recall@{k}",
                    "value": summary.recall[k],
                    "evaluated_users": summary.evaluated_users,
                },
                {
                    "model": summary.model,
                    "metric": f"NDCG@{k}",
                    "value": summary.ndcg[k],
                    "evaluated_users": summary.evaluated_users,
                },
            ]
        )
    return rows


def upsert_results(output_path: Path, rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        replacement_keys = set(zip(new_df["model"], new_df["metric"]))
        keep_mask = [
            (model, metric) not in replacement_keys
            for model, metric in zip(existing_df["model"], existing_df["metric"])
        ]
        existing_df = existing_df.loc[keep_mask]
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df

    merged_df = merged_df.sort_values(["model", "metric"]).reset_index(drop=True)
    merged_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RecBole benchmark models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--recbole-data-dir", type=Path, default=DEFAULT_RECB0LE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", default=MODEL_LIST)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_recbole_dataset(args.data_dir, args.recbole_data_dir)
    train_df, test_df = load_train_test(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "recbole_results.csv"
    for model_name in args.models:
        print(f"Training {model_name} in project web-rec environment ...")
        _, dataset, model = train_single_model(
            model_name,
            args.recbole_data_dir,
            {
                "use_gpu": args.use_gpu,
                "gpu_id": args.gpu_id,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
            },
        )
        summary = evaluate_recbole_model_batch(
            model_name,
            dataset,
            model,
            train_df,
            test_df,
            batch_size=args.eval_batch_size,
        )
        print(
            f"[{model_name}] AUC={summary.auc:.6f} MRR={summary.mrr:.6f} users={summary.evaluated_users}"
        )
        upsert_results(output_path, summary_to_rows(summary))
        print(f"Updated results at {output_path}")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
