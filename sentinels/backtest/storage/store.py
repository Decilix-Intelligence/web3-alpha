"""On-disk layout of a run.

    output/mini_backtest/<run_id>/
        run.json            config, state and summary for the whole run
        checkpoint.json     last saved bar index + account state (for --resume)
        equity.jsonl        one line per decision bar
        trades.jsonl        one line per fill, including stops and liquidations
        metrics.json        final performance summary
        ai_cache.json       decisions keyed by context hash
        decisions/NNNN.json full record of one decision cycle

Streams are JSONL and appended as the run progresses, so a run that is killed
part way through still leaves a readable, truncation-safe record.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sentinels.backtest.core.types import Checkpoint, EquityPoint, RunMetadata

_DECISION_ARTIFACT = re.compile(r"^\d{4,}\.json(?:\.tmp)?$")


class RunStore:
    def __init__(self, base_dir: str | Path, run_id: str):
        self.run_id = run_id
        self.root = Path(base_dir) / run_id
        self.decisions_dir = self.root / "decisions"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------

    @property
    def run_path(self) -> Path:
        return self.root / "run.json"

    @property
    def equity_path(self) -> Path:
        return self.root / "equity.jsonl"

    @property
    def trades_path(self) -> Path:
        return self.root / "trades.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def ai_cache_path(self) -> Path:
        return self.root / "ai_cache.json"

    # ---------- streams ----------

    def append_equity(self, point: EquityPoint) -> None:
        _append_jsonl(self.equity_path, point.to_dict())

    def append_trade(self, fill: Dict[str, Any]) -> None:
        _append_jsonl(self.trades_path, fill)

    def read_equity(self) -> List[EquityPoint]:
        return [EquityPoint.from_dict(row) for row in _read_jsonl(self.equity_path)]

    def read_trades(self) -> List[Dict[str, Any]]:
        return list(_read_jsonl(self.trades_path))

    def truncate_streams(self) -> None:
        """Drop equity/trade history — used when a resume rewinds to a checkpoint."""
        for path in (self.equity_path, self.trades_path):
            if path.exists():
                path.unlink()

    def reset_run(self) -> None:
        """Remove this run's generated artifacts before a fresh invocation.

        The run directory may contain operator notes or other unrelated files,
        so this deliberately names every root artifact and only recognises the
        numeric filenames produced by :meth:`write_decision` below.  The
        directory itself is retained.
        """
        documents = (
            self.run_path,
            self.equity_path,
            self.trades_path,
            self.metrics_path,
            self.checkpoint_path,
            self.ai_cache_path,
        )
        atomic_documents = (
            self.run_path,
            self.metrics_path,
            self.checkpoint_path,
            self.ai_cache_path,
        )
        for path in documents:
            _unlink_file(path)
        for path in atomic_documents:
            _unlink_file(path.with_suffix(path.suffix + ".tmp"))

        if self.decisions_dir.is_dir():
            for path in self.decisions_dir.iterdir():
                if _DECISION_ARTIFACT.fullmatch(path.name):
                    _unlink_file(path)

    # ---------- documents ----------

    def write_decision(self, cycle: int, record: Dict[str, Any]) -> Path:
        path = self.decisions_dir / f"{cycle:04d}.json"
        _write_json(path, record)
        return path

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        _write_json(self.metrics_path, metrics)

    def write_run(self, meta: RunMetadata) -> None:
        payload = meta.to_dict()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.run_path, payload)

    def read_run(self) -> Optional[Dict[str, Any]]:
        return _read_json(self.run_path)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        _write_json(self.checkpoint_path, checkpoint.to_dict())

    def load_checkpoint(self) -> Optional[Checkpoint]:
        raw = _read_json(self.checkpoint_path)
        return Checkpoint.from_dict(raw) if raw else None


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from an interrupted run


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class InMemoryStore:
    """A Storage that keeps everything in RAM.

    For tests and for sweeps where only the metrics matter and writing a
    directory per configuration would be noise.
    """

    def __init__(self, run_id: str = "memory"):
        self.run_id = run_id
        self.equity: List[EquityPoint] = []
        self.trades: List[Dict[str, Any]] = []
        self.decisions: Dict[int, Dict[str, Any]] = {}
        self.metrics: Optional[Dict[str, Any]] = None
        self.run: Optional[Dict[str, Any]] = None
        self.checkpoint: Optional[Checkpoint] = None
        self.ai_cache_path = None

    def append_equity(self, point: EquityPoint) -> None:
        self.equity.append(point)

    def append_trade(self, fill: Dict[str, Any]) -> None:
        self.trades.append(fill)

    def read_equity(self) -> List[EquityPoint]:
        return list(self.equity)

    def read_trades(self) -> List[Dict[str, Any]]:
        return list(self.trades)

    def truncate_streams(self) -> None:
        self.equity.clear()
        self.trades.clear()

    def reset_run(self) -> None:
        self.equity.clear()
        self.trades.clear()
        self.decisions.clear()
        self.metrics = None
        self.run = None
        self.checkpoint = None

    def write_decision(self, cycle: int, record: Dict[str, Any]) -> int:
        self.decisions[cycle] = record
        return cycle

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        self.metrics = metrics

    def write_run(self, meta: RunMetadata) -> None:
        self.run = meta.to_dict()

    def read_run(self) -> Optional[Dict[str, Any]]:
        return self.run

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint

    def load_checkpoint(self) -> Optional[Checkpoint]:
        return self.checkpoint


def _unlink_file(path: Path) -> None:
    """Unlink a regular file or symlink without touching a same-named directory."""
    if path.is_file() or path.is_symlink():
        path.unlink()
