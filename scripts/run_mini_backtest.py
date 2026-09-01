#!/usr/bin/env python3
"""Headless LLM backtest: news sentiment -> system prompt -> bar loop -> PnL.

    python scripts/run_mini_backtest.py --date 2026-01-15

No frontend, no exchange, no server. Needs an OpenAI-compatible LLM key unless
--replay-only is used, in which case every decision comes from the run's cache.

The shipped sample OHLCV is SYNTHETIC (see scripts/generate_sample_ohlcv.py).
It exercises the simulator; it is not evidence about any trading strategy.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def looks_like_placeholder(value: str) -> bool:
    v = (value or "").strip()
    return (not v) or v.startswith("${") or v in {"YOUR_KEY", "changeme"}


def resolve_llm(config, required: bool):
    """Build the LLM client. Returns None when the run will not need one."""
    api_key = config.get_llm_api_key()
    if looks_like_placeholder(api_key):
        if not required:
            return None
        raise SystemExit(
            "No LLM API key found. Export LLM_API_KEY (or OPENAI_API_KEY / "
            "DEEPSEEK_API_KEY, plus LLM_BASE_URL and LLM_MODEL as needed), or run "
            "with --replay-only to reuse a cached run."
        )
    from sentinels.llm import create_llm

    return create_llm(
        provider=config.get_llm_provider() or "openai",
        api_key=api_key,
        base_url=config.get_llm_base_url(),
        model=config.get_llm_model(),
        temperature=float(config.get_llm_temperature()),
        max_tokens=max(int(config.get_llm_max_tokens()), 1200),
    )


def build_config(args, config, custom_prompt: str):
    from sentinels.backtest.core.config import BacktestConfig, RiskControlConfig

    day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = day
    end = day + timedelta(days=args.days)

    risk = RiskControlConfig.from_dict(config.get("backtest.risk", {}))
    leverage = config.get("backtest.leverage", {}) or {}
    if leverage.get("btc_eth"):
        risk.btc_eth_max_leverage = int(leverage["btc_eth"])
    if leverage.get("altcoin"):
        risk.altcoin_max_leverage = int(leverage["altcoin"])

    return BacktestConfig(
        run_id=args.run_id or f"mini_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        symbols=args.symbols or config.get("backtest.symbols", ["BTCUSDT", "ETHUSDT"]),
        timeframes=config.get("backtest.timeframes", ["15m", "4h"]),
        decision_timeframe=config.get("backtest.decision_timeframe", "15m"),
        decision_cadence_nbars=args.cadence
        or int(config.get("backtest.decision_cadence_nbars", 20)),
        start_ts=start,
        end_ts=end,
        initial_balance=args.balance or float(config.get("backtest.initial_balance", 1000.0)),
        fee_bps=(
            args.fee_bps if args.fee_bps is not None else float(config.get("backtest.fee_bps", 5.0))
        ),
        slippage_bps=(
            args.slippage_bps
            if args.slippage_bps is not None
            else float(config.get("backtest.slippage_bps", 2.0))
        ),
        fill_policy=args.fill_policy or config.get("backtest.fill_policy", "next_open"),
        prompt_variant=args.prompt_variant or config.get("backtest.prompt_variant", "basic"),
        custom_prompt=custom_prompt,
        cache_ai=not args.no_cache and bool(config.get("backtest.cache_ai", True)),
        replay_only=args.replay_only,
        lookback_bars=int(config.get("backtest.lookback_bars", 200)),
        max_decisions=args.max_decisions,
        ohlcv_path=args.ohlcv
        or config.get("backtest.ohlcv_path", "examples/sample_data/ohlcv_15m.csv"),
        output_dir=str(ROOT / "output" / "mini_backtest"),
        risk=risk,
    )


def load_news_memo(config, args) -> str:
    """Run the sentiment pipeline for --date and format it as the strategy memo."""
    from sentinels.engine.formatter import EnginePromptFormatter
    from sentinels.pipelines.daily import DailySentimentPipeline

    pipeline = DailySentimentPipeline(config=config)
    ctx = pipeline.run(target_date=args.date, user_id=args.user_id)
    memo = ctx.get("prompt_text") or ""
    if memo:
        return memo
    metrics = ctx.get("sentiment_metrics")
    if metrics is None:
        raise SystemExit("the sentiment stage produced no metrics and no prompt")
    return EnginePromptFormatter().format_for_engine(metrics, args.date)


def run_comparison(cfg, llm, args) -> int:
    """Score the LLM agent and the rule-based baselines on identical bars."""
    from sentinels.backtest.agents import BuyAndHoldAgent, EmaCrossAgent, LLMAgent
    from sentinels.backtest.agents.client import DecisionClient
    from sentinels.backtest.storage.cache import AICache
    from sentinels.backtest.manager import BacktestManager

    manager = BacktestManager(cfg)
    client = DecisionClient(
        llm,
        # The manager binds the LLM to the ``<run-id>__llm`` child store. This
        # placeholder cache is deliberately pathless so merely setting up a
        # comparison cannot create or write an orphan parent-run cache.
        AICache(None),
        variant=cfg.prompt_variant,
        max_retries=cfg.ai_max_retries,
        retry_base_delay=cfg.ai_retry_base_delay,
        cache_ai=cfg.cache_ai,
        replay_only=cfg.replay_only,
    )

    report = manager.compare(
        {
            "llm": LLMAgent(client, cfg),
            "buy_and_hold": BuyAndHoldAgent(cfg.risk),
            "ema_cross": EmaCrossAgent(cfg.risk),
        }
    )

    print()
    print("=" * 72)
    print(
        f"  {cfg.start_ts:%Y-%m-%d} | {', '.join(cfg.symbols)} | {cfg.decision_timeframe} "
        f"| fill {cfg.fill_policy}"
    )
    print("=" * 72)
    print(report.to_table())
    print()
    print("  Same bars, same account rules, same metrics — only the agent differs.")
    print("  Synthetic sample data: this compares implementations, not strategies.")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="News2Alpha mini LLM backtest (headless)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--date", default="2026-01-15", help="first day of the run (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="days to simulate from --date")
    parser.add_argument("--user-id", type=int, default=None, help="whose recommended news to read")
    parser.add_argument(
        "--articles", default=None, help="article CSV/Parquet produced by the bridge"
    )
    parser.add_argument(
        "--recommendations",
        default=None,
        help="daily recommendations CSV produced by the bridge",
    )
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--ohlcv", default=None, help="OHLCV CSV path")
    parser.add_argument("--cadence", type=int, default=None, help="consult the model every N bars")
    parser.add_argument("--max-decisions", type=int, default=None, help="cap on model calls")
    parser.add_argument("--balance", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument(
        "--fill-policy", default=None, choices=["next_open", "bar_vwap", "mid", "mark"]
    )
    parser.add_argument(
        "--prompt-variant",
        default=None,
        choices=["basic", "aggressive", "conservative", "scalping"],
    )
    parser.add_argument("--run-id", default=None, help="name this run (also used by --resume)")
    parser.add_argument(
        "--resume", action="store_true", help="continue --run-id from its checkpoint"
    )
    parser.add_argument(
        "--replay-only", action="store_true", help="cache only; never call the model"
    )
    parser.add_argument("--no-cache", action="store_true", help="do not read or write the AI cache")
    parser.add_argument("--no-news", action="store_true", help="skip sentiment, run with no memo")
    parser.add_argument(
        "--agent",
        default="llm",
        choices=["llm", "buy_and_hold", "ema_cross"],
        help="which strategy to run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run the LLM agent and both baselines over the same bars and print a table",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    if args.resume and not args.run_id:
        raise SystemExit("--resume needs --run-id to say which run to continue")

    os.environ.setdefault("FINBERT_TRANSLATE", "0")
    if args.articles:
        os.environ["NEWS_ARTICLES_PATH"] = str(Path(args.articles).resolve())
    if args.recommendations:
        os.environ["NEWS_RECOMMENDATIONS_PATH"] = str(Path(args.recommendations).resolve())

    from sentinels.backtest.agents import build_baseline
    from sentinels.backtest.storage.store import RunStore
    from sentinels.backtest.runner import BacktestRunner
    from sentinels.config import get_config

    config = get_config()
    needs_model = args.agent == "llm" or args.compare
    llm = resolve_llm(config, required=needs_model and not args.replay_only)

    memo = "" if args.no_news else load_news_memo(config, args)
    cfg = build_config(args, config, memo)

    if args.compare:
        return run_comparison(cfg, llm, args)

    if args.agent == "llm":
        agent = None  # the runner builds the default LLM agent from `llm`
    else:
        agent = build_baseline(args.agent, cfg.risk)
        llm = None

    runner = BacktestRunner(cfg, llm, store=RunStore(cfg.output_dir, cfg.run_id), agent=agent)
    metrics = runner.run(resume=args.resume)

    out = runner.store.root
    print()
    print("=" * 72)
    print(f"  Run            {cfg.run_id}   ({'RESUMED' if args.resume else 'fresh'})")
    print(f"  Window         {cfg.start_ts:%Y-%m-%d %H:%M} -> {cfg.end_ts:%Y-%m-%d %H:%M} UTC")
    print(f"  Symbols        {', '.join(cfg.symbols)}   timeframes {', '.join(cfg.timeframes)}")
    print(f"  Fill policy    {cfg.fill_policy}   fee {cfg.fee_bps}bps   slip {cfg.slippage_bps}bps")
    print(f"  Agent          {args.agent}")
    print(f"  Prompt         {cfg.prompt_variant}   news memo {len(memo)} chars")
    print(
        f"  Bars / cycles  {runner.feed.decision_bar_count()} bars, "
        f"{metrics['DecisionCycles']} decisions, {metrics['AICalls']} model calls"
    )
    print("-" * 72)
    print(f"  TotalReturnPct {metrics['TotalReturnPct']:+.4f}")
    print(f"  MaxDrawdownPct {metrics['MaxDrawdownPct']:.4f}")
    print(
        f"  SharpeRatio    {metrics['SharpeRatio']:.4f}"
        + (f"   ({metrics['SharpeNote']})" if metrics.get("SharpeNote") else "")
    )
    print(f"  WinRate        {metrics['WinRate']:.2f}%")
    print(f"  ProfitFactor   {metrics['ProfitFactor']:.4f}")
    print(f"  TotalTrades    {metrics['TotalTrades']}")
    print(f"  Liquidated     {metrics['Liquidated']}")
    print(f"  Final equity   {metrics['FinalEquity']:.2f} from {metrics['InitialBalance']:.2f}")
    print("-" * 72)
    print(f"  Output         {out}")
    print("=" * 72)
    print()
    print("  Synthetic sample data: these numbers exercise the engine, they are")
    print("  not evidence about a trading strategy.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
