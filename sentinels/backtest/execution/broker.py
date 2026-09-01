"""The only component that places orders against the account.

It owns three things the loop should not have to: what price an order fills at,
how large it is once free cash is taken into account, and whether a bar's range
took out a position before the model got another turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sentinels.backtest.execution.account import Account, AccountError, Fill, Position
from sentinels.backtest.core.config import BacktestConfig
from sentinels.backtest.data.feed import Bar, DataFeed
from sentinels.backtest.core.schema import Decision
from sentinels.backtest.execution.fills import execution_price

logger = logging.getLogger(__name__)


@dataclass
class ProtectiveExit:
    position: Position
    price: float
    reason: str
    liquidation: bool


def protective_exit_for(position: Position, bar: Bar) -> Optional[ProtectiveExit]:
    """Which level, if any, this bar's range took out.

    Checked most-adverse first: a bar whose path crosses several levels is
    assumed to have crossed the worst one, since the bar alone cannot say in
    what order the prices occurred.
    """
    if position.side == "long":
        if position.liquidation_price > 0 and bar.low <= position.liquidation_price:
            return ProtectiveExit(position, position.liquidation_price, "liquidated", True)
        if position.stop_loss and bar.low <= position.stop_loss:
            return ProtectiveExit(position, position.stop_loss, "stop_loss", False)
        if position.take_profit and bar.high >= position.take_profit:
            return ProtectiveExit(position, position.take_profit, "take_profit", False)
        return None

    if position.liquidation_price > 0 and bar.high >= position.liquidation_price:
        return ProtectiveExit(position, position.liquidation_price, "liquidated", True)
    if position.stop_loss and bar.high >= position.stop_loss:
        return ProtectiveExit(position, position.stop_loss, "stop_loss", False)
    if position.take_profit and bar.low <= position.take_profit:
        return ProtectiveExit(position, position.take_profit, "take_profit", False)
    return None


class Broker:
    def __init__(self, cfg: BacktestConfig, account: Account, feed: DataFeed):
        self.cfg = cfg
        self.account = account
        self.feed = feed

    # ---------- involuntary exits ----------

    def protective_exits(self, ts: datetime) -> Tuple[List[Fill], str]:
        """Resolve stops, targets and liquidations against each symbol's own bar."""
        fills: List[Fill] = []
        notes: List[str] = []

        for position in list(self.account.positions()):
            bar, _ = self.feed.decision_bar(position.symbol, ts)
            if bar is None:
                continue
            exit_ = protective_exit_for(position, bar)
            if exit_ is None:
                continue
            try:
                fill = self.account.close(
                    position.symbol,
                    position.side,
                    None,
                    exit_.price,
                    timestamp=ts,
                    action="liquidated" if exit_.liquidation else exit_.reason,
                    exact_price=True,
                    liquidation=exit_.liquidation,
                    note=f"{exit_.reason} at {exit_.price:.4f}",
                )
            except AccountError as exc:
                logger.warning("could not close %s on %s: %s", position.key, exit_.reason, exc)
                continue

            fills.append(fill)
            logger.info(
                "%s %s %s @ %.4f, realized %+.2f",
                ts.isoformat(),
                position.key,
                exit_.reason,
                exit_.price,
                fill.realized_pnl,
            )
            if exit_.liquidation:
                notes.append(f"{position.key} liquidated at {exit_.price:.4f}")

        return fills, "; ".join(notes)

    # ---------- voluntary orders ----------

    def submit(
        self, decision: Decision, ts: datetime, price_map: Dict[str, float]
    ) -> Tuple[Optional[Fill], str]:
        """Place one validated decision. Returns (fill, human-readable note)."""
        if decision.action in ("hold", "wait"):
            return None, decision.action

        mark = price_map.get(decision.symbol)
        if not mark:
            return None, f"no price for {decision.symbol}"

        signal_bar, next_bar = self.feed.decision_bar(decision.symbol, ts)
        resolved = execution_price(self.cfg.fill_policy, signal_bar, next_bar, mark)
        note = f"filled at {resolved.source}"
        if resolved.fell_back:
            note += " (no next bar; fell back to the signal bar's close)"

        try:
            if decision.is_opening:
                quantity, sizing_note = self.size_order(decision, resolved.price)
                if quantity <= 0:
                    return None, sizing_note
                fill = self.account.open(
                    decision.symbol,
                    decision.side,
                    quantity,
                    decision.leverage,
                    resolved.price,
                    timestamp=ts,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                )
                if sizing_note:
                    note += f"; {sizing_note}"
            else:
                fill = self.account.close(
                    decision.symbol, decision.side, None, resolved.price, timestamp=ts
                )
        except AccountError as exc:
            return None, str(exc)

        logger.info(
            "%s %s %s qty=%.6f @ %.4f fee=%.4f realized=%+.4f",
            ts.isoformat(),
            fill.action,
            fill.symbol,
            fill.quantity,
            fill.price,
            fill.fee,
            fill.realized_pnl,
        )
        return fill, note

    def size_order(self, decision: Decision, price: float) -> Tuple[float, str]:
        """Convert a USD notional into a quantity the free cash can actually margin.

        The validator has already checked this size against the equity-based caps;
        this is the separate question of whether the cash on hand can post the
        margin *and* the opening fee right now.

        Opening notional N at leverage L costs N/L in margin plus N*fee_rate in
        fees, so the largest affordable N solves N/L + N*fee = cash exactly.
        Deriving it beats reserving an arbitrary fraction of cash as a buffer:
        the limit is then the real constraint rather than a guess at it.
        """
        size_usd = decision.position_size_usd
        note = ""

        cost_per_unit_notional = 1.0 / decision.leverage + self.account.fee_rate
        affordable = (
            self.account.cash / cost_per_unit_notional if cost_per_unit_notional > 0 else 0.0
        )
        if size_usd > affordable:
            note = (
                f"size trimmed from {size_usd:.2f} to {affordable:.2f} USDT "
                f"by available cash {self.account.cash:.2f} at {decision.leverage}x"
            )
            size_usd = affordable

        minimum = self.cfg.risk.min_position_size_for(decision.symbol)
        if size_usd < minimum:
            return (
                0.0,
                f"affordable size {size_usd:.2f} USDT is below the {minimum:.0f} USDT minimum",
            )

        return size_usd / price, note

    def flatten(self, ts: datetime, note: str = "end of run") -> List[Fill]:
        """Close whatever is still open, so the reported metrics are realised."""
        prices = self.feed.price_map(ts)
        fills: List[Fill] = []
        for position in list(self.account.positions()):
            price = prices.get(position.symbol)
            if not price:
                continue
            try:
                fill = self.account.close(
                    position.symbol, position.side, None, price, timestamp=ts, note=note
                )
            except AccountError as exc:
                logger.warning("could not flatten %s: %s", position.key, exc)
                continue
            fills.append(fill)
            logger.info(
                "flattened %s at %.4f, realized %+.2f", position.key, price, fill.realized_pnl
            )
        return fills
