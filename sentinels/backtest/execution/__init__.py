"""Turning decisions into positions.

account.py  cash, margin, weighted-average entries, proportional closes,
            liquidation levels
fills.py    what price an order actually executes at
broker.py   the only component that places orders against the account
"""

from sentinels.backtest.execution.account import (
    Account,
    AccountError,
    Fill,
    InsufficientMargin,
    InvalidOrder,
    NoSuchPosition,
    Position,
    apply_slippage,
    liquidation_price,
)
from sentinels.backtest.execution.broker import Broker, ProtectiveExit, protective_exit_for
from sentinels.backtest.execution.fills import ExecutionPrice, execution_price

__all__ = [
    "Account",
    "AccountError",
    "Broker",
    "ExecutionPrice",
    "Fill",
    "InsufficientMargin",
    "InvalidOrder",
    "NoSuchPosition",
    "Position",
    "ProtectiveExit",
    "apply_slippage",
    "execution_price",
    "liquidation_price",
    "protective_exit_for",
]
