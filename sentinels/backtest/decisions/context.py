"""Assembles everything the model is allowed to see at one decision point.

The system prompt carries the strategy and the risk rules; this module builds
the user turn, which is pure state: the account, the open positions, and the
market as of `ts`. The news memo is *not* repeated here — it is injected once
into the system prompt so its guidance frames the whole session rather than
competing with the price data for the model's attention.

Every number here comes from a DataFeed snapshot taken at `ts`, so the prompt
inherits the feed's no-look-ahead guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sentinels.backtest.execution.account import Account
from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.data.feed import SymbolSnapshot

PRIMARY_KLINES = 24
LONGER_KLINES = 12


@dataclass
class DecisionContext:
    timestamp: datetime
    cycle: int
    runtime_minutes: int
    account: Dict[str, Any]
    positions: List[Dict[str, Any]]
    candidates: List[str]
    market: Dict[str, SymbolSnapshot]
    timeframes: List[str] = field(default_factory=list)
    primary_tf: str = "15m"

    # ---------- prompt ----------

    def to_user_prompt(self) -> str:
        out: List[str] = []
        out.append("# Trading Decision Request")
        out.append("")
        out.append(
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
            f"Cycle: #{self.cycle} | Runtime: {self.runtime_minutes} minutes"
        )
        out.append("")
        out.extend(self._account_section())
        out.extend(self._positions_section())
        out.extend(self._market_section())
        out.append("Reply with <reasoning> then <decision> containing the JSON array.")
        return "\n".join(out)

    def _account_section(self) -> List[str]:
        a = self.account
        lines = ["## Account", ""]
        lines.append(
            f"Equity: {a['equity']:.2f} USDT | Available: {a['cash']:.2f} USDT | "
            f"Margin used: {a['margin_used']:.2f} USDT ({a['margin_used_pct']:.1f}%)"
        )
        lines.append(
            f"Realized PnL: {a['realized_pnl']:+.2f} USDT | "
            f"Unrealized PnL: {a['unrealized_pnl']:+.2f} USDT | "
            f"Total: {a['total_pnl']:+.2f} USDT ({a['total_pnl_pct']:+.2f}%)"
        )
        lines.append(f"Open positions: {a['position_count']}")
        if a["margin_used_pct"] > 70:
            lines.append("")
            lines.append("**Risk alert**: margin usage above 70%.")
        elif a["margin_used_pct"] > 50:
            lines.append("")
            lines.append("**Note**: margin usage above 50%, be selective about new entries.")
        lines.append("")
        return lines

    def _positions_section(self) -> List[str]:
        lines = ["## Open Positions", ""]
        if not self.positions:
            lines.append("None — the account is flat.")
            lines.append("")
            return lines
        for i, p in enumerate(self.positions, start=1):
            lines.append(
                f"{i}. {p['symbol']} {p['side'].upper()} | qty {p['quantity']:.6f} | "
                f"entry {p['entry_price']:.4f} | mark {p['mark_price']:.4f} | "
                f"{p['leverage']}x | margin {p['margin']:.2f} USDT"
            )
            lines.append(
                f"   uPnL {p['unrealized_pnl']:+.2f} USDT ({p['unrealized_pnl_pct']:+.2f}% on margin) | "
                f"peak {p['peak_pnl_pct']:+.2f}% | liquidation {p['liquidation_price']:.4f}"
            )
            protection = []
            if p.get("stop_loss"):
                protection.append(f"stop {p['stop_loss']:.4f}")
            if p.get("take_profit"):
                protection.append(f"target {p['take_profit']:.4f}")
            lines.append(
                f"   {' | '.join(protection)}"
                if protection
                else "   no stop or target attached — the position is unprotected"
            )
            drawdown = p["peak_pnl_pct"] - p["unrealized_pnl_pct"]
            if p["peak_pnl_pct"] > 2.0 and drawdown > 0.3 * p["peak_pnl_pct"]:
                lines.append(
                    f"   Gave back {drawdown:.2f}% of a {p['peak_pnl_pct']:.2f}% peak — "
                    "consider taking profit."
                )
            lines.append("")
        return lines

    def _market_section(self) -> List[str]:
        lines = ["## Market", ""]
        lines.append(f"Candidates: {', '.join(self.candidates)}")
        lines.append("")
        for symbol in self.candidates:
            snap = self.market.get(symbol)
            if snap is None:
                continue
            lines.append(f"### {symbol} — {snap.current_price:.4f}")
            lines.append("")
            for tf in sorted(
                snap.by_tf, key=lambda t: self.timeframes.index(t) if t in self.timeframes else 99
            ):
                tfs = snap.by_tf[tf]
                lines.append(f"**{tf} indicators**: {_indicator_line(tfs)}")
            lines.append("")
            for tf in self.timeframes:
                tfs = snap.by_tf.get(tf)
                if tfs is None:
                    continue
                count = PRIMARY_KLINES if tf == self.primary_tf else LONGER_KLINES
                lines.append(f"**{tf} klines** (oldest to newest, last {count}):")
                lines.append("```")
                lines.append("time              open      high      low       close     volume")
                for bar in tfs.bars[-count:]:
                    lines.append(
                        f"{bar.open_ts.strftime('%m-%d %H:%M')}    "
                        f"{bar.open:<9.2f} {bar.high:<9.2f} {bar.low:<9.2f} "
                        f"{bar.close:<9.2f} {bar.volume:.0f}"
                    )
                lines.append("```")
                lines.append("")
        return lines

    # ---------- caching ----------

    def cache_payload(self) -> Dict[str, Any]:
        """Canonical view of this context, used to key the AI cache.

        Only decision-relevant state goes in: the same account facing the same
        market must produce the same key, and any difference that could change
        the answer must change it.
        """
        return {
            "ts": self.timestamp.isoformat(),
            "cycle": self.cycle,
            "account": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in sorted(self.account.items())
            },
            "positions": [
                {k: (round(v, 8) if isinstance(v, float) else v) for k, v in sorted(p.items())}
                for p in self.positions
            ],
            "candidates": list(self.candidates),
            "market": {symbol: self.market[symbol].to_dict() for symbol in sorted(self.market)},
        }


def build_context(
    cfg: BacktestConfig,
    account: Account,
    market: Dict[str, SymbolSnapshot],
    ts: datetime,
    cycle: int,
) -> DecisionContext:
    price_map = {s: snap.current_price for s, snap in market.items()}
    equity, unrealized, margin_used = account.mark_to_market(price_map)

    account_view = {
        "equity": equity,
        "cash": account.cash,
        "margin_used": margin_used,
        "margin_used_pct": (margin_used / equity * 100.0) if equity > 0 else 0.0,
        "realized_pnl": account.realized_pnl,
        "unrealized_pnl": unrealized,
        "total_pnl": equity - account.initial_balance,
        "total_pnl_pct": (equity - account.initial_balance) / account.initial_balance * 100.0,
        "position_count": account.position_count(),
    }

    positions = []
    for pos in sorted(account.positions(), key=lambda p: p.key):
        mark = price_map.get(pos.symbol, pos.entry_price)
        view = pos.to_dict(mark)
        view.pop("opened_at", None)
        view.pop("entry_fees", None)
        view.pop("notional", None)
        positions.append(view)

    runtime = int((ts - cfg.start_ts).total_seconds() // 60)

    return DecisionContext(
        timestamp=ts,
        cycle=cycle,
        runtime_minutes=max(runtime, 0),
        account=account_view,
        positions=positions,
        candidates=list(cfg.symbols),
        market=market,
        timeframes=list(cfg.timeframes),
        primary_tf=cfg.decision_timeframe,
    )


def _indicator_line(tfs) -> str:
    parts = [
        f"EMA20 {_fmt(tfs.ema20)}",
        f"EMA50 {_fmt(tfs.ema50)}",
        f"MACD {_fmt(tfs.macd, 4)}/{_fmt(tfs.macd_signal, 4)} hist {_fmt(tfs.macd_hist, 4)}",
        f"RSI7 {_fmt(tfs.rsi7, 1)}",
        f"RSI14 {_fmt(tfs.rsi14, 1)}",
        f"ATR14 {_fmt(tfs.atr14, 4)}",
    ]
    return " | ".join(parts)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"
