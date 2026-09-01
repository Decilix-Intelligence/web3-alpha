"""Cash, margin and position accounting for a simulated cross-margin futures account.

Conventions used throughout:
  * `side` is "long" or "short"; a symbol may hold one position per side at once.
  * Slippage always moves against the holder: it worsens the entry when opening
    and worsens the exit when closing.
  * `entry_fees` carries the opening fees still attributable to the open size, so
    a partial close can charge its share of them instead of double counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

EPS = 1e-9


class AccountError(Exception):
    """Base class for order rejections raised by the account."""


class InvalidOrder(AccountError):
    """The order is malformed (non-positive size, bad leverage, unknown side)."""


class InsufficientMargin(AccountError):
    """Not enough free cash to post margin plus fees for this order."""


class NoSuchPosition(AccountError):
    """Asked to reduce a position that is not open."""


def liquidation_price(entry: float, leverage: int, side: str) -> float:
    """Price at which the initial margin of an isolated position is exhausted.

    A long is wiped out after an adverse move of 1/leverage, a short after the
    same move in the other direction. Maintenance margin is not modelled, which
    makes this the optimistic end of the range: a real venue liquidates slightly
    earlier than this.
    """
    if leverage <= 0:
        return 0.0
    if side == "long":
        return entry * (1.0 - 1.0 / leverage)
    return entry * (1.0 + 1.0 / leverage)


def apply_slippage(price: float, rate: float, side: str, opening: bool) -> float:
    """Move `price` against the holder of `side` by `rate`."""
    if rate <= 0:
        return price
    worse_up = (side == "long") == opening  # long opens higher, long closes lower
    return price * (1.0 + rate) if worse_up else price * (1.0 - rate)


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    leverage: int
    margin: float
    notional: float
    liquidation_price: float
    opened_at: Optional[datetime] = None
    entry_fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    peak_pnl_pct: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.side}"

    def unrealized(self, mark: float) -> float:
        if self.side == "long":
            return (mark - self.entry_price) * self.quantity
        return (self.entry_price - mark) * self.quantity

    def pnl_pct(self, mark: float) -> float:
        """Return on posted margin, i.e. leverage-amplified."""
        if self.margin <= EPS:
            return 0.0
        return self.unrealized(mark) / self.margin * 100.0

    def observe(self, mark: float) -> None:
        """Track the best unrealized return this position has seen."""
        self.peak_pnl_pct = max(self.peak_pnl_pct, self.pnl_pct(mark))

    def to_dict(self, mark: Optional[float] = None) -> Dict[str, object]:
        data = {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "leverage": self.leverage,
            "margin": self.margin,
            "notional": self.notional,
            "liquidation_price": self.liquidation_price,
            "entry_fees": self.entry_fees,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "peak_pnl_pct": self.peak_pnl_pct,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }
        if mark is not None:
            data["mark_price"] = mark
            data["unrealized_pnl"] = self.unrealized(mark)
            data["unrealized_pnl_pct"] = self.pnl_pct(mark)
        return data


@dataclass
class Fill:
    """One executed order. `fee` on a close includes the closed share of entry fees."""

    timestamp: Optional[datetime]
    symbol: str
    side: str
    action: str
    quantity: float
    price: float
    fee: float
    realized_pnl: float
    slippage: float
    order_value: float
    leverage: int
    position_after: float
    liquidation: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "ts": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "qty": self.quantity,
            "price": self.price,
            "fee": self.fee,
            "realized_pnl": self.realized_pnl,
            "slippage": self.slippage,
            "order_value": self.order_value,
            "leverage": self.leverage,
            "position_after": self.position_after,
            "liquidation": self.liquidation,
            "note": self.note,
        }


class Account:
    def __init__(self, initial_balance: float, fee_bps: float, slippage_bps: float):
        if initial_balance <= 0:
            raise InvalidOrder("initial_balance must be positive")
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.fee_rate = fee_bps / 10_000.0
        self.slippage_rate = slippage_bps / 10_000.0
        self.realized_pnl = 0.0
        self._positions: Dict[str, Position] = {}

    # ---------- inspection ----------

    def positions(self) -> List[Position]:
        return list(self._positions.values())

    def get(self, symbol: str, side: str) -> Optional[Position]:
        return self._positions.get(f"{symbol.upper()}:{side}")

    def position_count(self) -> int:
        return len(self._positions)

    def margin_used(self) -> float:
        return sum(p.margin for p in self._positions.values())

    def unrealized(self, price_map: Dict[str, float]) -> float:
        return sum(
            p.unrealized(price_map.get(p.symbol, p.entry_price)) for p in self._positions.values()
        )

    def equity(self, price_map: Dict[str, float]) -> float:
        """Cash + posted margin + mark-to-market on open positions."""
        return self.cash + self.margin_used() + self.unrealized(price_map)

    def mark_to_market(self, price_map: Dict[str, float]) -> Tuple[float, float, float]:
        """(equity, unrealized, margin_used), also refreshing each position's peak."""
        for pos in self._positions.values():
            mark = price_map.get(pos.symbol)
            if mark is not None:
                pos.observe(mark)
        unrealized = self.unrealized(price_map)
        margin = self.margin_used()
        return self.cash + margin + unrealized, unrealized, margin

    # ---------- orders ----------

    def open(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int,
        price: float,
        timestamp: Optional[datetime] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Fill:
        """Open or add to a position. Adding blends the entry at weighted average."""
        symbol = symbol.upper()
        if side not in ("long", "short"):
            raise InvalidOrder(f"unknown side '{side}'")
        if quantity <= 0:
            raise InvalidOrder("quantity must be positive")
        if leverage <= 0:
            raise InvalidOrder("leverage must be positive")
        if price <= 0:
            raise InvalidOrder("price must be positive")

        exec_price = apply_slippage(price, self.slippage_rate, side, opening=True)
        notional = exec_price * quantity
        margin = notional / leverage
        fee = notional * self.fee_rate

        if margin + fee > self.cash + EPS:
            raise InsufficientMargin(
                f"need {margin + fee:.2f} USDT (margin {margin:.2f} + fee {fee:.2f}), "
                f"have {self.cash:.2f}"
            )

        self.cash -= margin + fee

        key = f"{symbol}:{side}"
        pos = self._positions.get(key)
        if pos is None:
            pos = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=exec_price,
                leverage=leverage,
                margin=margin,
                notional=notional,
                liquidation_price=liquidation_price(exec_price, leverage, side),
                opened_at=timestamp,
                entry_fees=fee,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self._positions[key] = pos
        else:
            blended = (pos.entry_price * pos.quantity + exec_price * quantity) / (
                pos.quantity + quantity
            )
            if leverage != pos.leverage:
                # notional-weighted effective leverage across the merged position
                pos.leverage = max(1, round((pos.notional + notional) / (pos.margin + margin)))
            pos.entry_price = blended
            pos.quantity += quantity
            pos.notional += notional
            pos.margin += margin
            pos.entry_fees += fee
            pos.liquidation_price = liquidation_price(pos.entry_price, pos.leverage, pos.side)
            if stop_loss is not None:
                pos.stop_loss = stop_loss
            if take_profit is not None:
                pos.take_profit = take_profit

        return Fill(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            action=f"open_{side}",
            quantity=quantity,
            price=exec_price,
            fee=fee,
            realized_pnl=0.0,
            slippage=exec_price - price,
            order_value=notional,
            leverage=pos.leverage,
            position_after=pos.quantity,
        )

    def close(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float],
        price: float,
        timestamp: Optional[datetime] = None,
        action: Optional[str] = None,
        exact_price: bool = False,
        liquidation: bool = False,
        note: str = "",
    ) -> Fill:
        """Reduce or flatten a position.

        `quantity=None` closes the whole position. A partial close releases margin,
        notional and entry fees in proportion to the closed share, so the remaining
        position still carries exactly the costs it is responsible for.

        `exact_price=True` fills at `price` untouched — used for stop-loss,
        take-profit and liquidation, where the trigger level *is* the fill.
        """
        symbol = symbol.upper()
        key = f"{symbol}:{side}"
        pos = self._positions.get(key)
        if pos is None or pos.quantity <= EPS:
            raise NoSuchPosition(f"no open {side} position on {symbol}")
        if price <= 0:
            raise InvalidOrder("price must be positive")

        if quantity is None or quantity <= 0 or quantity > pos.quantity - EPS:
            quantity = pos.quantity
        share = quantity / pos.quantity

        exec_price = (
            price if exact_price else apply_slippage(price, self.slippage_rate, side, opening=False)
        )
        exit_fee = exec_price * quantity * self.fee_rate
        entry_fee_share = pos.entry_fees * share
        total_fee = exit_fee + entry_fee_share

        gross = (
            (exec_price - pos.entry_price) * quantity
            if side == "long"
            else (pos.entry_price - exec_price) * quantity
        )

        margin_share = pos.margin * share
        notional_share = pos.notional * share

        # The entry fee already left cash when the position was opened, so only the
        # exit fee is deducted here; realized_pnl reports the round-trip cost.
        self.cash += margin_share + gross - exit_fee
        self.realized_pnl += gross - total_fee

        pos.quantity -= quantity
        pos.margin -= margin_share
        pos.notional -= notional_share
        pos.entry_fees -= entry_fee_share
        if pos.quantity <= EPS:
            del self._positions[key]

        return Fill(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            action=action or f"close_{side}",
            quantity=quantity,
            price=exec_price,
            fee=total_fee,
            realized_pnl=gross - total_fee,
            slippage=0.0 if exact_price else exec_price - price,
            order_value=exec_price * quantity,
            leverage=pos.leverage,
            position_after=max(pos.quantity, 0.0),
            liquidation=liquidation,
            note=note,
        )

    # ---------- persistence ----------

    def snapshot(self, price_map: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
        price_map = price_map or {}
        return [p.to_dict(price_map.get(p.symbol)) for p in self._positions.values()]

    def restore(self, cash: float, realized_pnl: float, positions: List[Dict[str, object]]) -> None:
        self.cash = float(cash)
        self.realized_pnl = float(realized_pnl)
        self._positions = {}
        for raw in positions or []:
            pos = Position(
                symbol=str(raw["symbol"]).upper(),
                side=str(raw["side"]),
                quantity=float(raw["quantity"]),
                entry_price=float(raw["entry_price"]),
                leverage=int(raw["leverage"]),
                margin=float(raw["margin"]),
                notional=float(raw["notional"]),
                liquidation_price=float(raw["liquidation_price"]),
                entry_fees=float(raw.get("entry_fees", 0.0)),
                stop_loss=_opt_float(raw.get("stop_loss")),
                take_profit=_opt_float(raw.get("take_profit")),
                peak_pnl_pct=float(raw.get("peak_pnl_pct", 0.0)),
                opened_at=_opt_dt(raw.get("opened_at")),
            )
            self._positions[pos.key] = pos


def _opt_float(value) -> Optional[float]:
    return None if value is None else float(value)


def _opt_dt(value) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(str(value))
