"""Discovering past runs on disk.

There is no database and no server here: a run is a directory, and the registry
is whatever `run.json` files are sitting under the output root. That is enough
to list what has been run, compare two variants, or find the run whose cache a
`--replay-only` invocation should reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentinels.backtest.storage.store import RunStore
from sentinels.backtest.core.types import HEADLINE_METRICS, RunState


@dataclass
class RunRecord:
    run_id: str
    state: RunState
    path: Path
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    summary: Dict[str, Any] = None
    metrics: Dict[str, Any] = None

    @property
    def store(self) -> RunStore:
        return RunStore(self.path.parent, self.run_id)

    def headline(self) -> Dict[str, Any]:
        return {k: (self.metrics or {}).get(k) for k in HEADLINE_METRICS}

    def __str__(self) -> str:
        m = self.metrics or {}
        ret = m.get("TotalReturnPct")
        trades = m.get("TotalTrades")
        tail = f"  {ret:+.2f}%  {trades} trades" if ret is not None else ""
        return f"{self.run_id:<28} {str(self.state):<11}{tail}"


def list_runs(base_dir: str | Path) -> List[RunRecord]:
    """Every run under `base_dir`, newest first. Unreadable directories are skipped."""
    root = Path(base_dir)
    if not root.is_dir():
        return []

    records: List[RunRecord] = []
    for run_path in sorted(root.iterdir()):
        record = load_run(root, run_path.name)
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r.updated_at or r.created_at or datetime.min, reverse=True)
    return records


def load_run(base_dir: str | Path, run_id: str) -> Optional[RunRecord]:
    store = RunStore(base_dir, run_id)
    meta = store.read_run()
    if not meta:
        return None
    try:
        state = RunState(meta.get("state", "created"))
    except ValueError:
        state = RunState.CREATED
    return RunRecord(
        run_id=meta.get("run_id", run_id),
        state=state,
        path=store.root,
        created_at=_dt(meta.get("created_at")),
        updated_at=_dt(meta.get("updated_at")),
        summary=meta.get("summary") or {},
        metrics=meta.get("metrics") or {},
    )


def _dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
