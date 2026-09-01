# W3Alpha

Anonymous research artifact for **W3: A News-to-Alpha Benchmark for Utility-Driven Web3 Recommendation and Trading Backtesting**, submitted to WSDM.

This repository joins the original recommendation benchmark and the News2Alpha sentiment/backtesting engine into one reproducible pipeline:

```text
Web3 articles + anonymous interactions
              │
              ▼
 recommendation benchmark ──► per-user Top-K news
                                      │
                                      ▼
                             sentiment factors + memo
                                      │
                                      ▼
                         trading agent / shared backtester
                                      │
                                      ▼
                       ranking metrics + trading utility
```

> **Dataset availability:** 数据集将在论文被接收后公开。
>
> **The dataset will be released after the paper is accepted.**

The repository currently contains source code, synthetic examples, tests, benchmark protocols, and aggregate result tables only. It contains no private dataset, identity mapping, credentials, model checkpoints, or personal author information.

## Repository layout

```text
benchmark/                    recommendation baselines and evaluators
sentinels/                    sentiment, LLM adapters, and backtesting engine
w3alpha/                      bridge between recommendation and alpha stages
scripts/
  run_integrated_pipeline.py  W3 Top-K -> News2Alpha input/memo
  run_daily_pipeline.py       news -> sentiment factors -> memo
  run_mini_backtest.py        memo -> agent -> fills and PnL
  check_anonymity.py          repository PII/secret guard
config/                       example runtime configuration
examples/sample_data/         synthetic news and OHLCV fixtures only
tests/                        offline unit and integration tests
docs/                         data contract and reproduction notes
```

## 1. Install

Python 3.10 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

Optional components are installed separately:

```bash
# recommendation baselines and data conversion
python3 -m pip install -e ".[benchmark]"

# FinBERT sentiment inference; the first model load may download weights
python3 -m pip install -e ".[sentiment]"

# OpenAI-compatible LLM agent
python3 -m pip install -e ".[llm]"

# every optional component plus test tools
python3 -m pip install -e ".[all,dev]"
```

Copy the example configuration only when you need to customize defaults:

```bash
cp config/config.example.yaml config/config.yaml
cp config/backtest_config.example.yaml config/backtest_config.yaml
```

Both destination files are ignored by Git. Secrets are read from environment variables; do not write real keys into tracked files.

## 2. Verify the code-only release

The following path is offline and needs neither the research dataset nor an API key:

```bash
pytest
python3 scripts/run_mini_backtest.py \
  --date 2026-01-15 \
  --agent ema_cross \
  --no-news
python3 scripts/check_anonymity.py --root .
```

The included OHLCV file is deterministic synthetic data. Its returns validate the simulator plumbing, not a trading strategy.

## 3. Run the integrated pipeline

After the dataset is released, place the anonymous WebRec split under `data/webrec_v1/`:

```text
data/webrec_v1/
  interactions.csv
  item_features.csv
  train.csv
  valid.csv
  test.csv
```

Generate leakage-free popularity recommendations from the training split and export the exact files consumed by News2Alpha:

```bash
python3 scripts/run_integrated_pipeline.py \
  --data-dir data/webrec_v1 \
  --user-id 0 \
  --date 2026-01-15 \
  --top-k 30 \
  --output-dir output/integrated \
  --skip-sentiment
```

The bridge exports only anonymous contiguous `user_idx` and `item_idx` values. It does not export source database identifiers or use validation/test labels as recommendations.

To continue through FinBERT and produce the strategy memo, install the `sentiment` extra and omit `--skip-sentiment`:

```bash
FINBERT_TRANSLATE=0 python3 scripts/run_integrated_pipeline.py \
  --data-dir data/webrec_v1 \
  --user-id 0 \
  --date 2026-01-15 \
  --top-k 30 \
  --output-dir output/integrated
```

`FINBERT_TRANSLATE=0` keeps titles local. Translation is opt-in because enabling it can send article text to the configured LLM endpoint.

Run the downstream simulator with the generated recommendation files:

```bash
python3 scripts/run_mini_backtest.py \
  --date 2026-01-15 \
  --user-id 0 \
  --articles output/integrated/articles.csv \
  --recommendations output/integrated/daily_recommendations.csv \
  --agent ema_cross
```

For the LLM agent, export credentials in the current shell and change `--agent` to `llm`:

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY=YOUR_KEY
export LLM_MODEL=YOUR_MODEL
```

An OpenAI-compatible gateway may additionally require `LLM_BASE_URL`. Model decisions and prompts are written under the ignored `output/` directory.

## 4. Reproduce recommendation benchmarks

```bash
python3 benchmark/run_benchmark.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/results

python3 benchmark/export_recbole_dataset.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/recbole_data/webrec
```

RecBole, text-modality, and Prompt4NR tracks have additional dependencies and external model requirements. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [benchmark/README.md](benchmark/README.md). The classic full-ranking and LLM impression-ranking tracks use different candidate sets and must be reported separately.

## Data and privacy contract

- Public release identifiers are contiguous, anonymous indices. No reversible user/item mapping is published.
- Raw database exports, local data, prompts, model replies, logs, caches, and checkpoints are excluded from Git.
- Do not attempt to re-identify users or join anonymous identifiers to external identity sources.
- Run `python3 scripts/check_anonymity.py --root .` before every public update.

More details are in [docs/DATA.md](docs/DATA.md).

## Reproducibility notes

- Recommendation models are fit on `train.csv`; evaluation labels are not used to create bridge recommendations.
- Synthetic market data is generated from a fixed seed.
- Trading decisions can be cached and replayed, but a fresh `run_id` should be used whenever the model, memo, risk configuration, or input data changes.
- Real API keys and local configuration must remain outside Git.

## Licenses

The W3Alpha benchmark is distributed under the terms in [LICENSE](LICENSE). The incorporated News2Alpha engine retains its MIT notice in [LICENSES/NEWS2ALPHA-MIT.txt](LICENSES/NEWS2ALPHA-MIT.txt).
