"""Headless LLM backtesting engine.

Subpackages, in dependency order — each one may import from those above it and
never from those below, which is what keeps the layering checkable:

    core/        config, risk limits, the records that reach disk, the protocols
    data/        the feed and its indicators, with no look-ahead
    execution/   account, fill policies, the broker that places orders
    decisions/   the context an agent sees, the action schema, the validator
    agents/      the strategies: LLM and rule-based baselines
    analysis/    metrics and equity-curve reshaping
    storage/     run layout on disk, decision cache, run discovery

    runner.py    the loop — decides when things happen
    manager.py   runs several agents over one shared feed

Three things are pluggable; their contracts are in `core.protocols`. Everything
else is concrete on purpose: indirection you cannot name a second implementation
for is a cost, not a design.

    Agent       state in, decisions out    LLMAgent, BuyAndHoldAgent, EmaCrossAgent
    Storage     where records land         RunStore, InMemoryStore
    LLMClient   a chat endpoint            anything with .complete()

`Agent` is the one that matters for evaluation. Because every agent runs through
the identical loop, account, validator and metrics, their numbers are comparable
by construction:

    manager = BacktestManager(cfg)
    print(manager.compare({
        "llm":          LLMAgent(client, cfg),
        "buy_and_hold": BuyAndHoldAgent(cfg.risk),
        "ema_cross":    EmaCrossAgent(cfg.risk),
    }).to_table())
"""

from sentinels.backtest.agents import (
    BASELINE_AGENTS,
    BuyAndHoldAgent,
    DecisionClient,
    DecisionResult,
    EmaCrossAgent,
    LLMAgent,
    build_baseline,
)
from sentinels.backtest.analysis import compute_metrics, max_drawdown_pct, sharpe_ratio
from sentinels.backtest.core import (
    Agent,
    AgentDecision,
    BacktestConfig,
    Checkpoint,
    EquityPoint,
    LLMClient,
    RiskControlConfig,
    RunMetadata,
    RunState,
    RunSummary,
    Storage,
)
from sentinels.backtest.data import Bar, DataFeed
from sentinels.backtest.decisions import (
    Decision,
    DecisionContext,
    PortfolioState,
    build_context,
    parse_response,
    validate,
    validate_all,
)
from sentinels.backtest.execution import Account, Broker, Fill, Position
from sentinels.backtest.manager import BacktestManager, ComparisonReport, ComparisonRow
from sentinels.backtest.runner import BacktestRunner
from sentinels.backtest.storage import (
    AICache,
    CacheMiss,
    InMemoryStore,
    RunRecord,
    RunStore,
    list_runs,
    load_run,
)

__all__ = [
    "Account",
    "Agent",
    "AgentDecision",
    "AICache",
    "BASELINE_AGENTS",
    "BacktestConfig",
    "BacktestManager",
    "BacktestRunner",
    "Bar",
    "Broker",
    "BuyAndHoldAgent",
    "CacheMiss",
    "Checkpoint",
    "ComparisonReport",
    "ComparisonRow",
    "DataFeed",
    "Decision",
    "DecisionClient",
    "DecisionContext",
    "DecisionResult",
    "EmaCrossAgent",
    "EquityPoint",
    "Fill",
    "InMemoryStore",
    "LLMAgent",
    "LLMClient",
    "PortfolioState",
    "Position",
    "RiskControlConfig",
    "RunMetadata",
    "RunRecord",
    "RunState",
    "RunStore",
    "RunSummary",
    "Storage",
    "build_baseline",
    "build_context",
    "compute_metrics",
    "list_runs",
    "load_run",
    "max_drawdown_pct",
    "parse_response",
    "sharpe_ratio",
    "validate",
    "validate_all",
]
