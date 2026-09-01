"""Backtest configuration objects.

RiskControlConfig is the single source of truth for every hard limit in the
simulator: the system prompt renders these numbers for the model, validate.py
enforces them on the model's output, and runner.py sizes orders from them.
There is deliberately no second copy of these constants anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Anchored on the `sentinels` package rather than counted in parent hops, so
# moving a module deeper in the tree cannot silently break relative data paths.
PROJECT_ROOT = Path(__import__("sentinels").__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones resolve against the repo root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


# Execution price policies (see fills.py)
FILL_NEXT_OPEN = "next_open"
FILL_BAR_VWAP = "bar_vwap"
FILL_MID = "mid"
FILL_MARK = "mark"
FILL_POLICIES = (FILL_NEXT_OPEN, FILL_BAR_VWAP, FILL_MID, FILL_MARK)

PROMPT_VARIANTS = ("basic", "aggressive", "conservative", "scalping")

BTC_ETH = ("BTCUSDT", "ETHUSDT")

TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
}


def tf_seconds(timeframe: str) -> int:
    tf = (timeframe or "").strip().lower()
    if tf not in TF_SECONDS:
        raise ValueError(f"unsupported timeframe '{timeframe}'")
    return TF_SECONDS[tf]


def is_btc_eth(symbol: str) -> bool:
    return (symbol or "").upper() in BTC_ETH


@dataclass
class RiskControlConfig:
    """Hard risk limits for a run.

    Every number here reaches three consumers and must stay identical across
    them, which is why they live in one dataclass:
      1. sentinels/engine/system_prompt.py  - what the model is told
      2. sentinels/backtest/validate.py     - what the code enforces
      3. sentinels/backtest/runner.py       - how orders are sized
    """

    max_positions: int = 3
    btc_eth_max_leverage: int = 5
    altcoin_max_leverage: int = 5
    btc_eth_max_position_value_ratio: float = 5.0
    altcoin_max_position_value_ratio: float = 1.0
    max_margin_usage: float = 0.9
    min_position_size: float = 12.0
    min_position_size_btc_eth: float = 60.0
    min_risk_reward_ratio: float = 3.0
    min_confidence: int = 75

    def max_leverage_for(self, symbol: str) -> int:
        return self.btc_eth_max_leverage if is_btc_eth(symbol) else self.altcoin_max_leverage

    def position_value_ratio_for(self, symbol: str) -> float:
        if is_btc_eth(symbol):
            return self.btc_eth_max_position_value_ratio
        return self.altcoin_max_position_value_ratio

    def max_position_value_for(self, symbol: str, equity: float) -> float:
        return equity * self.position_value_ratio_for(symbol)

    def min_position_size_for(self, symbol: str) -> float:
        return self.min_position_size_btc_eth if is_btc_eth(symbol) else self.min_position_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_positions": self.max_positions,
            "btc_eth_max_leverage": self.btc_eth_max_leverage,
            "altcoin_max_leverage": self.altcoin_max_leverage,
            "btc_eth_max_position_value_ratio": self.btc_eth_max_position_value_ratio,
            "altcoin_max_position_value_ratio": self.altcoin_max_position_value_ratio,
            "max_margin_usage": self.max_margin_usage,
            "min_position_size": self.min_position_size,
            "min_position_size_btc_eth": self.min_position_size_btc_eth,
            "min_risk_reward_ratio": self.min_risk_reward_ratio,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RiskControlConfig":
        cfg = cls()
        if not data:
            return cfg
        known = cfg.to_dict().keys()
        return replace(cfg, **{k: v for k, v in data.items() if k in known and v is not None})


@dataclass
class BacktestConfig:
    """Everything one simulation run needs to be reproducible."""

    run_id: str
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["15m", "4h"])
    decision_timeframe: str = "15m"
    decision_cadence_nbars: int = 20
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    initial_balance: float = 1000.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    fill_policy: str = FILL_NEXT_OPEN
    prompt_variant: str = "basic"
    custom_prompt: str = ""
    cache_ai: bool = True
    replay_only: bool = False
    lookback_bars: int = 200
    checkpoint_interval_bars: int = 20
    max_decisions: Optional[int] = None
    ai_max_retries: int = 3
    ai_retry_base_delay: float = 0.5
    flatten_at_end: bool = True
    risk: RiskControlConfig = field(default_factory=RiskControlConfig)
    ohlcv_path: str = "examples/sample_data/ohlcv_15m.csv"
    output_dir: str = "output/mini_backtest"

    def validate(self) -> "BacktestConfig":
        """Normalize fields in place and reject impossible configurations."""
        self.run_id = (self.run_id or "").strip()
        if not self.run_id:
            raise ValueError("run_id cannot be empty")

        if not self.symbols:
            raise ValueError("at least one symbol is required")
        self.symbols = [s.strip().upper() for s in self.symbols if s and s.strip()]

        if not self.timeframes:
            self.timeframes = ["15m", "4h"]
        self.timeframes = sorted({tf.strip().lower() for tf in self.timeframes}, key=tf_seconds)
        for tf in self.timeframes:
            tf_seconds(tf)

        self.decision_timeframe = (self.decision_timeframe or self.timeframes[0]).strip().lower()
        tf_seconds(self.decision_timeframe)
        if self.decision_timeframe not in self.timeframes:
            self.timeframes = sorted(
                set(self.timeframes) | {self.decision_timeframe}, key=tf_seconds
            )

        if self.decision_cadence_nbars <= 0:
            self.decision_cadence_nbars = 20

        if self.start_ts is None or self.end_ts is None:
            raise ValueError("start_ts/end_ts are required")
        self.start_ts = _as_utc(self.start_ts)
        self.end_ts = _as_utc(self.end_ts)
        if self.end_ts <= self.start_ts:
            raise ValueError("end_ts must be after start_ts")

        if self.initial_balance <= 0:
            self.initial_balance = 1000.0
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("fee_bps/slippage_bps must be >= 0")

        self.fill_policy = (self.fill_policy or FILL_NEXT_OPEN).strip().lower()
        if self.fill_policy not in FILL_POLICIES:
            raise ValueError(
                f"unsupported fill_policy '{self.fill_policy}' (want one of {FILL_POLICIES})"
            )

        self.prompt_variant = (self.prompt_variant or "basic").strip().lower()
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"unknown prompt_variant '{self.prompt_variant}' (want one of {PROMPT_VARIANTS})"
            )

        if self.lookback_bars <= 0:
            self.lookback_bars = 200
        if self.checkpoint_interval_bars <= 0:
            self.checkpoint_interval_bars = 20
        if self.ai_max_retries <= 0:
            self.ai_max_retries = 1
        return self

    @property
    def decision_tf_seconds(self) -> int:
        return tf_seconds(self.decision_timeframe)

    @property
    def longest_timeframe(self) -> str:
        return max(self.timeframes, key=tf_seconds)

    def lookback_start(self) -> datetime:
        """Earliest bar the feed must load so indicators are warm at start_ts."""
        longest = tf_seconds(self.longest_timeframe)
        return self.start_ts - timedelta(seconds=longest * self.lookback_bars)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "decision_timeframe": self.decision_timeframe,
            "decision_cadence_nbars": self.decision_cadence_nbars,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "initial_balance": self.initial_balance,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "fill_policy": self.fill_policy,
            "prompt_variant": self.prompt_variant,
            "cache_ai": self.cache_ai,
            "replay_only": self.replay_only,
            "lookback_bars": self.lookback_bars,
            "checkpoint_interval_bars": self.checkpoint_interval_bars,
            "max_decisions": self.max_decisions,
            "ai_max_retries": self.ai_max_retries,
            "flatten_at_end": self.flatten_at_end,
            "risk": self.risk.to_dict(),
            "ohlcv_path": self.ohlcv_path,
            "output_dir": self.output_dir,
            "custom_prompt_chars": len(self.custom_prompt or ""),
        }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
