#!/usr/bin/env python3
"""Text-only modality sequential recommendation for canonical Web-Rec data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, Dataset

TOP_KS = (10, 20, 50)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (BASE_DIR / "datasets" / "webrec_v1").resolve()
DEFAULT_OUTPUT_DIR = (BASE_DIR / "results").resolve()


@dataclass
class Summary:
    model: str
    auc: float
    mrr: float
    evaluated_users: int
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_optional_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["user_idx", "item_idx", "timestamp", "interaction_type", "weight"]
        )
    return load_split(path)


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_optional_split(data_dir / "train.csv")
    valid_df = load_optional_split(data_dir / "valid.csv")
    test_df = load_optional_split(data_dir / "test.csv")
    interaction_df = load_optional_split(data_dir / "interactions.csv")
    item_df = pd.read_csv(data_dir / "item_features.csv")
    return train_df, valid_df, test_df, interaction_df, item_df


def select_training_frame(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> pd.DataFrame:
    if args.train_source == "split":
        return train_df
    if args.train_source == "interactions":
        if interaction_df.empty:
            raise ValueError("--train-source interactions requires interactions.csv in --data-dir")
        return interaction_df
    frames = [df for df in (train_df, valid_df, test_df) if not df.empty]
    if not frames:
        raise ValueError("--train-source all_splits requires at least one split CSV")
    return pd.concat(frames, ignore_index=True)


def build_tfidf_text_embeddings(
    item_df: pd.DataFrame,
    max_item_idx: int,
    text_dim: int,
    max_features: int,
    min_df: int,
) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    item_df = item_df.copy()
    item_df["title"] = item_df["title"].fillna("").astype(str)
    item_df["synopsis"] = item_df["synopsis"].fillna("").astype(str)
    item_df["text"] = (item_df["title"] + " " + item_df["synopsis"]).str.strip()

    ordered_text = [""] * (max_item_idx + 1)
    for row in item_df.itertuples(index=False):
        ordered_text[int(row.item_idx)] = row.text

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(ordered_text)
    svd_dim = min(text_dim, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=2026)
    dense = svd.fit_transform(tfidf).astype(np.float32)
    dense = normalize(dense, norm="l2", copy=False).astype(np.float32)
    if svd_dim < text_dim:
        padded = np.zeros((dense.shape[0], text_dim), dtype=np.float32)
        padded[:, :svd_dim] = dense
        dense = padded

    with_pad = np.zeros((max_item_idx + 2, text_dim), dtype=np.float32)
    with_pad[1:] = dense
    return with_pad, vectorizer, svd


def ordered_item_text(item_df: pd.DataFrame, max_item_idx: int) -> tuple[list[str], pd.DataFrame]:
    item_df = item_df.copy()
    item_df["title"] = item_df["title"].fillna("").astype(str)
    item_df["synopsis"] = item_df["synopsis"].fillna("").astype(str)
    item_df["text"] = (item_df["title"] + " " + item_df["synopsis"]).str.strip()
    ordered_text = [""] * (max_item_idx + 1)
    for row in item_df.itertuples(index=False):
        ordered_text[int(row.item_idx)] = row.text
    return ordered_text, item_df


def build_bert_text_embeddings(
    item_df: pd.DataFrame,
    max_item_idx: int,
    bert_model: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    cache_path: Path | None,
    refresh_cache: bool,
) -> tuple[np.ndarray, dict]:
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        dense = np.load(cache_path).astype(np.float32)
        return dense, {"type": "bert", "model": bert_model, "cache_path": str(cache_path)}

    from transformers import AutoModel, AutoTokenizer

    ordered_text, _ = ordered_item_text(item_df, max_item_idx)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    encoder = AutoModel.from_pretrained(bert_model).to(device)
    encoder.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(ordered_text), batch_size):
            batch_text = ordered_text[start : start + batch_size]
            encoded = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model_out = encoder(**encoded)
            token_embeddings = model_out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))

    dense = np.concatenate(outputs, axis=0).astype(np.float32)
    dense = normalize(dense, norm="l2", copy=False).astype(np.float32)
    with_pad = np.zeros((max_item_idx + 2, dense.shape[1]), dtype=np.float32)
    with_pad[1:] = dense
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, with_pad)
    return with_pad, {"type": "bert", "model": bert_model, "max_length": max_length}


def ordered_sequences(df: pd.DataFrame) -> dict[int, list[int]]:
    return (
        df.sort_values(["user_idx", "timestamp", "item_idx"])
        .groupby("user_idx")["item_idx"]
        .apply(lambda values: [int(v) for v in values])
        .to_dict()
    )


class PrefixDataset(Dataset):
    def __init__(
        self,
        train_df: pd.DataFrame,
        max_seq_len: int,
        max_examples: int | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.examples: list[tuple[list[int], int]] = []
        for seq in ordered_sequences(train_df).values():
            if len(seq) < 2:
                continue
            for pos in range(1, len(seq)):
                prefix = seq[max(0, pos - max_seq_len) : pos]
                target = seq[pos]
                self.examples.append(([item + 1 for item in prefix], target + 1))
        if max_examples is not None and len(self.examples) > max_examples:
            rng = random.Random(2026)
            self.examples = rng.sample(self.examples, max_examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_prefixes(
    batch: list[tuple[list[int], int]], max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = torch.zeros((len(batch), max_seq_len), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    for row, (seq, target) in enumerate(batch):
        seq = seq[-max_seq_len:]
        length = len(seq)
        if length:
            seqs[row, -length:] = torch.tensor(seq, dtype=torch.long)
        lengths[row] = length
        targets[row] = target
    return seqs, lengths, targets


class TextMoRecSASRec(nn.Module):
    def __init__(
        self,
        text_embeddings: np.ndarray,
        hidden_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
        self.register_buffer("text_embeddings", text_tensor)
        self.text_proj = nn.Linear(text_tensor.shape[1], hidden_size)
        self.output_proj = nn.Linear(text_tensor.shape[1], hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def encode_sequence(self, seqs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        text = self.text_embeddings[seqs]
        hidden = self.text_proj(text)
        positions = torch.arange(self.max_seq_len, device=seqs.device).unsqueeze(0)
        hidden = self.layer_norm(hidden + self.pos_embedding(positions))
        hidden = self.dropout(hidden)
        padding_mask = seqs.eq(0)
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        last_index = (lengths.clamp(min=1) - 1) + (self.max_seq_len - lengths.clamp(min=1))
        batch_index = torch.arange(seqs.size(0), device=seqs.device)
        return encoded[batch_index, last_index]

    def candidate_representations(self) -> torch.Tensor:
        return self.output_proj(self.text_embeddings)

    def forward(self, seqs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        user_repr = self.encode_sequence(seqs, lengths)
        item_repr = self.candidate_representations()
        logits = user_repr @ item_repr.t()
        logits[:, 0] = -1e9
        return logits


def train_one_epoch(
    model: TextMoRecSASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    progress_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    loss_fn = nn.CrossEntropyLoss()
    for step, (seqs, lengths, targets) in enumerate(loader, start=1):
        seqs = seqs.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(seqs, lengths)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * seqs.size(0)
        total_examples += seqs.size(0)
        if progress_interval > 0 and step % progress_interval == 0:
            running_loss = total_loss / max(total_examples, 1)
            print(
                f"epoch={epoch} step={step}/{len(loader)} " f"running_loss={running_loss:.6f}",
                flush=True,
            )
    return total_loss / max(total_examples, 1)


def make_eval_batch(
    histories: list[list[int]], max_seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = torch.zeros((len(histories), max_seq_len), dtype=torch.long, device=device)
    lengths = torch.zeros(len(histories), dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        seq = [item + 1 for item in history[-max_seq_len:]]
        length = len(seq)
        if length:
            seqs[row, -length:] = torch.tensor(seq, dtype=torch.long, device=device)
        lengths[row] = length
    return seqs, lengths


def rank_to_metrics(
    rank: int, num_candidates: int
) -> tuple[float, float, dict[int, float], dict[int, float], dict[int, float]]:
    auc = 1.0 if num_candidates <= 1 else (num_candidates - rank) / (num_candidates - 1)
    mrr = 1.0 / rank
    precision = {}
    recall = {}
    ndcg = {}
    for k in TOP_KS:
        hit = 1.0 if rank <= k else 0.0
        precision[k] = hit / k
        recall[k] = hit
        ndcg[k] = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    return auc, mrr, precision, recall, ndcg


def evaluate(
    model: TextMoRecSASRec,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    model_name: str,
    max_eval_users: int | None = None,
) -> Summary:
    train_sequences = ordered_sequences(train_df)
    train_history = {user: set(items) for user, items in train_sequences.items()}
    item_pool = sorted(int(item) for item in train_df["item_idx"].unique())
    item_to_col = {item: col for col, item in enumerate(item_pool)}
    item_pool_tensor = torch.tensor(
        [item + 1 for item in item_pool], dtype=torch.long, device=device
    )

    eval_rows: list[tuple[int, int, list[int]]] = []
    for row in eval_df.itertuples(index=False):
        user_id = int(row.user_idx)
        target = int(row.item_idx)
        history = train_sequences.get(user_id)
        if not history or target not in item_to_col:
            continue
        eval_rows.append((user_id, target, history))
        if max_eval_users is not None and len(eval_rows) >= max_eval_users:
            break

    auc_values: list[float] = []
    mrr_values: list[float] = []
    precision_values = {k: [] for k in TOP_KS}
    recall_values = {k: [] for k in TOP_KS}
    ndcg_values = {k: [] for k in TOP_KS}

    model.eval()
    with torch.no_grad():
        item_repr = model.candidate_representations()[item_pool_tensor]
        for start in range(0, len(eval_rows), batch_size):
            batch = eval_rows[start : start + batch_size]
            histories = [history for _, _, history in batch]
            seqs, lengths = make_eval_batch(histories, max_seq_len, device)
            user_repr = model.encode_sequence(seqs, lengths)
            scores = user_repr @ item_repr.t()
            scores_np = scores.detach().cpu().numpy()

            for row_idx, (user_id, target, _) in enumerate(batch):
                row_scores = scores_np[row_idx].copy()
                history = train_history[user_id]
                for hist_item in history:
                    col = item_to_col.get(hist_item)
                    if col is not None:
                        row_scores[col] = -np.inf
                target_col = item_to_col[target]
                gt_score = row_scores[target_col]
                if not np.isfinite(gt_score):
                    continue
                better = int(np.sum(row_scores > gt_score))
                rank = better + 1
                candidate_count = int(np.sum(np.isfinite(row_scores)))
                auc, mrr, precision, recall, ndcg = rank_to_metrics(rank, candidate_count)
                auc_values.append(auc)
                mrr_values.append(mrr)
                for k in TOP_KS:
                    precision_values[k].append(precision[k])
                    recall_values[k].append(recall[k])
                    ndcg_values[k].append(ndcg[k])

    return Summary(
        model=model_name,
        auc=float(np.mean(auc_values)) if auc_values else 0.0,
        mrr=float(np.mean(mrr_values)) if mrr_values else 0.0,
        evaluated_users=len(auc_values),
        precision={
            k: float(np.mean(precision_values[k])) if precision_values[k] else 0.0 for k in TOP_KS
        },
        recall={k: float(np.mean(recall_values[k])) if recall_values[k] else 0.0 for k in TOP_KS},
        ndcg={k: float(np.mean(ndcg_values[k])) if ndcg_values[k] else 0.0 for k in TOP_KS},
    )


def summary_to_rows(summary: Summary) -> list[dict]:
    rows = [
        {
            "model": summary.model,
            "metric": "AUC",
            "value": summary.auc,
            "evaluated_users": summary.evaluated_users,
        },
        {
            "model": summary.model,
            "metric": "MRR",
            "value": summary.mrr,
            "evaluated_users": summary.evaluated_users,
        },
    ]
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


def write_results(output_dir: Path, summary: Summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_to_rows(summary)).to_csv(
        output_dir / "text_moRec_results.csv", index=False
    )


def count_parameter_groups(model: nn.Module) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {
        "trainable": trainable,
        "frozen": frozen,
        "buffers": buffers,
        "total_tensors": trainable + frozen + buffers,
    }


def save_artifacts(
    output_dir: Path,
    model: TextMoRecSASRec,
    encoder_artifact: dict,
    item_df: pd.DataFrame,
    text_embeddings: np.ndarray,
    args: argparse.Namespace,
) -> None:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "state_dict": model.state_dict(),
            "config": {
                "max_seq_len": args.max_seq_len,
                "text_dim": args.text_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "max_features": args.max_features,
                "min_df": args.min_df,
                "seed": args.seed,
                "train_source": args.train_source,
                "text_encoder": args.text_encoder,
                "bert_model": args.bert_model if args.text_encoder == "bert" else None,
                "bert_max_length": args.bert_max_length if args.text_encoder == "bert" else None,
            },
        },
        artifact_dir / "text_moRec_checkpoint.pt",
    )
    if args.text_encoder == "tfidf":
        dump(encoder_artifact["vectorizer"], artifact_dir / "tfidf_vectorizer.joblib")
        dump(encoder_artifact["svd"], artifact_dir / "text_svd.joblib")
        encoder_config = {"type": "tfidf", "max_features": args.max_features, "min_df": args.min_df}
    else:
        encoder_config = dict(encoder_artifact)
    np.save(artifact_dir / "item_text_embeddings.npy", text_embeddings)
    (artifact_dir / "text_encoder_config.json").write_text(
        json.dumps(encoder_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    item_df[["item_idx", "item_id", "title", "synopsis"]].to_csv(
        artifact_dir / "item_text_index.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-only MoRec sequential benchmark.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="TextMoRec-SASRec")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--text-encoder", choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--bert-model", default="bert-base-chinese")
    parser.add_argument("--bert-batch-size", type=int, default=128)
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--cache-text-embeddings", type=Path, default=None)
    parser.add_argument("--refresh-text-embeddings", action="store_true")
    parser.add_argument(
        "--train-source", choices=["split", "all_splits", "interactions"], default="split"
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    train_df, valid_df, test_df, interaction_df, item_df = load_data(args.data_dir)
    train_source_df = select_training_frame(args, train_df, valid_df, test_df, interaction_df)
    max_item_idx = int(max(item_df["item_idx"].max(), train_source_df["item_idx"].max()))
    for frame in (train_df, valid_df, test_df, interaction_df):
        if not frame.empty:
            max_item_idx = int(max(max_item_idx, frame["item_idx"].max()))

    print(
        f"Building title+synopsis {args.text_encoder} embeddings for {max_item_idx + 1} items ...",
        flush=True,
    )
    if args.text_encoder == "bert":
        text_embeddings, encoder_artifact = build_bert_text_embeddings(
            item_df=item_df,
            max_item_idx=max_item_idx,
            bert_model=args.bert_model,
            batch_size=args.bert_batch_size,
            max_length=args.bert_max_length,
            device=device,
            cache_path=args.cache_text_embeddings,
            refresh_cache=args.refresh_text_embeddings,
        )
        args.text_dim = int(text_embeddings.shape[1])
    else:
        text_embeddings, vectorizer, svd = build_tfidf_text_embeddings(
            item_df=item_df,
            max_item_idx=max_item_idx,
            text_dim=args.text_dim,
            max_features=args.max_features,
            min_df=args.min_df,
        )
        encoder_artifact = {"type": "tfidf", "vectorizer": vectorizer, "svd": svd}
    dataset = PrefixDataset(
        train_source_df, max_seq_len=args.max_seq_len, max_examples=args.max_train_examples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: collate_prefixes(batch, args.max_seq_len),
        pin_memory=device.type == "cuda",
    )
    print(f"Training examples: {len(dataset)} device={device}", flush=True)
    print(f"Steps per epoch: {len(loader)}", flush=True)

    model = TextMoRecSASRec(
        text_embeddings=text_embeddings,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    param_counts = count_parameter_groups(model)
    print(
        "Parameter counts: "
        f"trainable={param_counts['trainable']} "
        f"buffers={param_counts['buffers']} "
        f"total_tensors={param_counts['total_tensors']}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_auc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch, args.progress_interval)
        if args.skip_eval:
            print(f"epoch={epoch} loss={loss:.6f}", flush=True)
            continue
        valid_summary = evaluate(
            model,
            train_df,
            valid_df,
            args.max_seq_len,
            args.eval_batch_size,
            device,
            args.model_name,
            max_eval_users=args.max_eval_users,
        )
        print(
            f"epoch={epoch} loss={loss:.6f} valid_auc={valid_summary.auc:.6f} "
            f"valid_mrr={valid_summary.mrr:.6f} users={valid_summary.evaluated_users}",
            flush=True,
        )
        if valid_summary.auc > best_valid_auc:
            best_valid_auc = valid_summary.auc
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.skip_eval:
        if args.save_artifacts:
            save_artifacts(args.output_dir, model, encoder_artifact, item_df, text_embeddings, args)
            print(f"Saved artifacts to {args.output_dir / 'artifacts'}", flush=True)
        return
    test_summary = evaluate(
        model,
        train_df,
        test_df,
        args.max_seq_len,
        args.eval_batch_size,
        device,
        args.model_name,
        max_eval_users=args.max_eval_users,
    )
    print(
        f"[{args.model_name}] AUC={test_summary.auc:.6f} MRR={test_summary.mrr:.6f} "
        f"NDCG@10={test_summary.ndcg[10]:.6f} Recall@10={test_summary.recall[10]:.6f} "
        f"users={test_summary.evaluated_users}",
        flush=True,
    )
    write_results(args.output_dir, test_summary)
    if args.save_artifacts:
        save_artifacts(args.output_dir, model, encoder_artifact, item_df, text_embeddings, args)
    print(f"Saved results to {args.output_dir / 'text_moRec_results.csv'}", flush=True)
    if args.save_artifacts:
        print(f"Saved artifacts to {args.output_dir / 'artifacts'}", flush=True)


if __name__ == "__main__":
    main()
