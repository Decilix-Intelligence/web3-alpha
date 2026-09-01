"""Multi-symbol, multi-timeframe data feed.

Contract (this is the no-look-ahead guarantee):
  * A bar's timestamp in the CSV is its OPEN time; close_ts = open_ts + tf.
  * slice_up_to(symbol, tf, ts) returns only bars with close_ts <= ts.
  * Indicators are computed on that slice, so a bar that has not closed at ts
    can never reach the model or the fill logic.
  * Higher timeframes are resampled from the decision timeframe, and a partial
    higher-tf bucket is dropped rather than closed early.
"""
from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from sentinels.backtest.data import indicators as ind
from sentinels.backtest.core.config import BacktestConfig, resolve_path, tf_seconds

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    open_ts: datetime
    close_ts: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TimeframeSnapshot:
    """Indicator + recent-kline view of one symbol on one timeframe at a point in time."""

    timeframe: str
    bars: List[Bar]
    close: float
    ema20: Optional[float]
    ema50: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    rsi7: Optional[float]
    rsi14: Optional[float]
    atr14: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "timeframe": self.timeframe,
            "bars": len(self.bars),
            "close": self.close,
            "ema20": self.ema20,
            "ema50": self.ema50,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_hist": self.macd_hist,
            "rsi7": self.rsi7,
            "rsi14": self.rsi14,
            "atr14": self.atr14,
        }


@dataclass
class SymbolSnapshot:
    symbol: str
    current_price: float
    by_tf: Dict[str, TimeframeSnapshot]

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "timeframes": {tf: snap.to_dict() for tf, snap in sorted(self.by_tf.items())},
        }


class DataFeed:
    def __init__(self, cfg: BacktestConfig, frame: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self.primary_tf = cfg.decision_timeframe
        self.timeframes = list(cfg.timeframes)
        self._series: Dict[str, Dict[str, List[Bar]]] = {}
        self._close_index: Dict[str, Dict[str, List[datetime]]] = {}
        self.decision_times: List[datetime] = []
        self._load(frame if frame is not None else self._read_csv())

    # ---------- loading ----------

    def _read_csv(self) -> pd.DataFrame:
        path = resolve_path(self.cfg.ohlcv_path)
        if not path.exists():
            raise FileNotFoundError(f"OHLCV file not found: {path}")
        return pd.read_csv(path)

    def _load(self, frame: pd.DataFrame) -> None:
        df = frame.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "symbol" not in df.columns:
            raise ValueError("OHLCV data needs a 'symbol' column")
        df["symbol"] = df["symbol"].astype(str).str.upper()

        lookback_start = self.cfg.lookback_start()
        # one extra decision bar past end_ts so next_open fills stay in range
        load_end = self.cfg.end_ts + timedelta(seconds=self.cfg.decision_tf_seconds)

        for symbol in self.cfg.symbols:
            rows = df[df["symbol"] == symbol].sort_values("timestamp")
            if rows.empty:
                raise ValueError(f"no OHLCV rows for {symbol} in {self.cfg.ohlcv_path}")
            base = self._to_bars(rows, symbol, self.primary_tf)
            base = [b for b in base if lookback_start <= b.open_ts <= load_end]
            if not base:
                raise ValueError(
                    f"{symbol}: no {self.primary_tf} bars inside "
                    f"[{lookback_start.isoformat()}, {load_end.isoformat()}]"
                )
            per_tf: Dict[str, List[Bar]] = {self.primary_tf: base}
            for tf in self.timeframes:
                if tf == self.primary_tf:
                    continue
                per_tf[tf] = resample(base, tf)
            self._series[symbol] = per_tf
            self._close_index[symbol] = {
                tf: [b.close_ts for b in bars] for tf, bars in per_tf.items()
            }
            self._warn_short_warmup(symbol, per_tf)

        first = self.cfg.symbols[0]
        self.decision_times = [
            b.close_ts
            for b in self._series[first][self.primary_tf]
            if self.cfg.start_ts <= b.close_ts <= self.cfg.end_ts
        ]
        if not self.decision_times:
            raise ValueError("no decision bars in the configured [start_ts, end_ts] range")

        for symbol in self.cfg.symbols[1:]:
            have = set(self._close_index[symbol][self.primary_tf])
            missing = [ts for ts in self.decision_times if ts not in have]
            if missing:
                raise ValueError(
                    f"{symbol} is missing {len(missing)} {self.primary_tf} bars covered by "
                    f"{first} (first gap {missing[0].isoformat()})"
                )

    def _warn_short_warmup(self, symbol: str, per_tf: Dict[str, List[Bar]]) -> None:
        for tf, bars in per_tf.items():
            warm = sum(1 for b in bars if b.close_ts <= self.cfg.start_ts)
            if warm < self.cfg.lookback_bars:
                logger.warning(
                    "%s %s: only %d warmup bars before start_ts (lookback_bars=%d); "
                    "indicators are still valid, just seeded on a shorter history",
                    symbol,
                    tf,
                    warm,
                    self.cfg.lookback_bars,
                )

    @staticmethod
    def _to_bars(rows: pd.DataFrame, symbol: str, timeframe: str) -> List[Bar]:
        delta = timedelta(seconds=tf_seconds(timeframe))
        bars: List[Bar] = []
        for row in rows.itertuples(index=False):
            open_ts = _utc(row.timestamp)
            bars.append(
                Bar(
                    open_ts=open_ts,
                    close_ts=open_ts + delta,
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(getattr(row, "volume", 0.0) or 0.0),
                )
            )
        return bars

    # ---------- time-progressive access ----------

    def decision_bar_count(self) -> int:
        return len(self.decision_times)

    def decision_timestamp(self, index: int) -> Optional[datetime]:
        if index < 0 or index >= len(self.decision_times):
            return None
        return self.decision_times[index]

    def slice_up_to(self, symbol: str, timeframe: str, ts: datetime) -> List[Bar]:
        """Bars whose close_ts <= ts. This is the only door data comes through."""
        closes = self._close_index.get(symbol, {}).get(timeframe)
        if not closes:
            return []
        idx = bisect_right(closes, ts)
        if idx <= 0:
            return []
        return self._series[symbol][timeframe][:idx]

    def build_market_snapshot(self, ts: datetime) -> Dict[str, SymbolSnapshot]:
        out: Dict[str, SymbolSnapshot] = {}
        for symbol in self.cfg.symbols:
            by_tf: Dict[str, TimeframeSnapshot] = {}
            for tf in self.timeframes:
                bars = self.slice_up_to(symbol, tf, ts)
                if not bars:
                    continue
                by_tf[tf] = _snapshot_from(bars, tf)
            if self.primary_tf not in by_tf:
                raise ValueError(f"no {self.primary_tf} data for {symbol} at {ts.isoformat()}")
            out[symbol] = SymbolSnapshot(
                symbol=symbol,
                current_price=by_tf[self.primary_tf].close,
                by_tf=by_tf,
            )
        return out

    def price_map(self, ts: datetime) -> Dict[str, float]:
        return {s: snap.current_price for s, snap in self.build_market_snapshot(ts).items()}

    def decision_bar(self, symbol: str, ts: datetime) -> Tuple[Optional[Bar], Optional[Bar]]:
        """(bar closing at ts, next bar) on the decision timeframe."""
        closes = self._close_index.get(symbol, {}).get(self.primary_tf)
        if not closes:
            return None, None
        idx = bisect_right(closes, ts) - 1
        if idx < 0 or closes[idx] != ts:
            return None, None
        bars = self._series[symbol][self.primary_tf]
        nxt = bars[idx + 1] if idx + 1 < len(bars) else None
        return bars[idx], nxt


def _snapshot_from(bars: Sequence[Bar], timeframe: str) -> TimeframeSnapshot:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    macd_line, macd_signal, macd_hist = ind.macd(closes)
    return TimeframeSnapshot(
        timeframe=timeframe,
        bars=list(bars),
        close=closes[-1],
        ema20=ind.ema(closes, 20),
        ema50=ind.ema(closes, 50),
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        rsi7=ind.rsi(closes, 7),
        rsi14=ind.rsi(closes, 14),
        atr14=ind.atr(highs, lows, closes, 14),
    )


def resample(bars: Sequence[Bar], target_tf: str) -> List[Bar]:
    """Aggregate lower-tf bars into target_tf. Incomplete trailing bucket is dropped."""
    if not bars:
        return []
    src = tf_seconds(bars[0].timeframe)
    dst = tf_seconds(target_tf)
    if dst < src or dst % src != 0:
        raise ValueError(f"cannot resample {bars[0].timeframe} -> {target_tf}")
    per_bucket = dst // src

    buckets: Dict[int, List[Bar]] = {}
    for bar in bars:
        key = int(bar.open_ts.timestamp()) // dst
        buckets.setdefault(key, []).append(bar)

    out: List[Bar] = []
    for key in sorted(buckets):
        group = buckets[key]
        if len(group) != per_bucket:
            # partial bucket: closing it early would invent a bar that never existed
            continue
        open_ts = datetime.fromtimestamp(key * dst, tz=timezone.utc)
        out.append(
            Bar(
                open_ts=open_ts,
                close_ts=open_ts + timedelta(seconds=dst),
                symbol=group[0].symbol,
                timeframe=target_tf,
                open=group[0].open,
                high=max(b.high for b in group),
                low=min(b.low for b in group),
                close=group[-1].close,
                volume=sum(b.volume for b in group),
            )
        )
    return out


def _utc(value) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()
