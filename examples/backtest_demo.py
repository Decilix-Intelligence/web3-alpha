"""Small offline sentiment-factor plumbing demo using synthetic inputs.

This file demonstrates alignment and lagging only. Use the backtesting engine in
``scripts/run_mini_backtest.py`` for research experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_polarity(text: str) -> float:
    positive = sum(token in text.lower() for token in ("improves", "inflow", "rally"))
    negative = sum(token in text.lower() for token in ("outflow", "crash", "decline"))
    return float((positive - negative) / max(positive + negative, 1))


def main() -> None:
    articles = pd.read_csv("examples/sample_data/articles.csv")
    daily_sentiment = float(articles["title"].map(synthetic_polarity).mean())

    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.02, size=30))
    position = 1 if daily_sentiment > 0.2 else -1 if daily_sentiment < -0.2 else 0
    # Lag the signal to prevent using a return observed by the signal itself.
    strategy_returns = pd.Series(position, index=returns.index).shift(1).fillna(0) * returns
    equity = (1 + strategy_returns).cumprod()

    print(f"synthetic sentiment: {daily_sentiment:+.4f}")
    print(f"position: {position:+d}")
    print(f"final synthetic equity: {equity.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
