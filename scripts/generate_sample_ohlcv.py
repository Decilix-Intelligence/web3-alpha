#!/usr/bin/env python3
"""Generate the deterministic synthetic OHLCV shipped in examples/sample_data.

The sample data is SYNTHETIC. It is built to exercise the simulator — trends,
chop, a sharp reversal, correlated symbols — not to resemble any real market.
Nothing measured on it is evidence about a trading strategy.

It is generated rather than downloaded so the repository stays self-contained
and every run is byte-for-byte reproducible from the seed below.

    python scripts/generate_sample_ohlcv.py
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

SEED = 20260115
BAR_SECONDS = 900  # 15m

# (bars, drift per bar, volatility per bar, label) — 15 days ending 2026-01-16.
# The last 96 bars are 2026-01-15, the day the sample backtest runs on.
REGIMES: List[Tuple[int, float, float, str]] = [
    (384, 0.00018, 0.0022, "grind higher"),
    (192, -0.00035, 0.0038, "pullback"),
    (288, 0.00002, 0.0016, "range"),
    (288, 0.00040, 0.0030, "trend up"),
    (192, -0.00012, 0.0026, "distribution"),
    (40, 0.00085, 0.0032, "morning rally"),
    (28, -0.00160, 0.0058, "sharp reversal"),
    (28, 0.00030, 0.0030, "stabilise"),
]

SYMBOLS = {
    # symbol: (start price, beta to the shared factor, idiosyncratic vol)
    "BTCUSDT": (39800.0, 1.00, 0.0006),
    "ETHUSDT": (2180.0, 1.35, 0.0011),
}


def build(end: datetime) -> Dict[str, List[dict]]:
    total = sum(n for n, _, _, _ in REGIMES)
    start = end - timedelta(seconds=BAR_SECONDS * total)

    rng = random.Random(SEED)

    # one shared market factor drives both symbols, so cross-symbol structure is real
    factor: List[Tuple[float, float]] = []
    for bars, drift, vol, _ in REGIMES:
        for _ in range(bars):
            factor.append((drift, rng.gauss(0.0, vol)))

    series: Dict[str, List[dict]] = {}
    for symbol, (price, beta, idio_vol) in SYMBOLS.items():
        sym_rng = random.Random(SEED + sum(ord(c) for c in symbol))
        rows: List[dict] = []
        for i, (drift, shock) in enumerate(factor):
            ts = start + timedelta(seconds=BAR_SECONDS * i)
            ret = beta * (drift + shock) + sym_rng.gauss(0.0, idio_vol)
            open_px = price
            close_px = open_px * (1.0 + ret)
            # intrabar extremes: a wick beyond the body on each side
            span = abs(close_px - open_px)
            wick = max(span * sym_rng.uniform(0.3, 1.4), open_px * sym_rng.uniform(0.0004, 0.0022))
            high = max(open_px, close_px) + wick * sym_rng.uniform(0.2, 1.0)
            low = min(open_px, close_px) - wick * sym_rng.uniform(0.2, 1.0)
            volume = round(sym_rng.uniform(120, 640) * (1.0 + abs(ret) * 90), 2)
            rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "symbol": symbol,
                    "open": round(open_px, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_px, 2),
                    "volume": volume,
                }
            )
            price = close_px
        series[symbol] = rows
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="examples/sample_data/ohlcv_15m.csv",
        help="output CSV path (relative to the repo root)",
    )
    parser.add_argument(
        "--end", default="2026-01-16", help="exclusive end date of the series (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    series = build(end)

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [row for symbol in sorted(series) for row in series[symbol]]
    rows.sort(key=lambda r: (r["timestamp"], r["symbol"]))

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")
    for symbol, bars in sorted(series.items()):
        print(
            f"  {symbol}: {len(bars)} bars  {bars[0]['timestamp']} -> {bars[-1]['timestamp']}  "
            f"{bars[0]['close']:.2f} -> {bars[-1]['close']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
