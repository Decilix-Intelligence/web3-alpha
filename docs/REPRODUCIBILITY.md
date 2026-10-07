# Reproducibility

This repository provides source code, synthetic fixtures, and aggregate result tables. The research dataset is not bundled with the code.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[all,dev]"
pytest
```

The base and test suites do not make network calls. FinBERT may download model weights on first use. LLM access is opt-in and requires explicit environment variables.

## Data placement

Prepare the anonymous benchmark files at `data/webrec_v1/` as described in [DATA.md](DATA.md).

## Integrated news-to-alpha path

```bash
python3 scripts/run_integrated_pipeline.py \
  --data-dir data/webrec_v1 \
  --user-id 0 \
  --date 2026-01-15 \
  --top-k 30 \
  --output-dir output/integrated

python3 scripts/run_mini_backtest.py \
  --date 2026-01-15 \
  --user-id 0 \
  --articles output/integrated/articles.csv \
  --recommendations output/integrated/daily_recommendations.csv \
  --agent ema_cross
```

Set `FINBERT_TRANSLATE=0` to guarantee that article text is not sent to an external translation model. Use a unique `--run-id` when any input, prompt, model, or risk configuration changes.

## Classic recommendation track

```bash
python3 benchmark/run_benchmark.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/results

python3 benchmark/export_recbole_dataset.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/recbole_data/webrec

python3 benchmark/run_recbole_benchmarks.py \
  --data-dir data/webrec_v1 \
  --recbole-data-dir benchmark/recbole_data/webrec \
  --output-dir benchmark/results \
  --models BPR NCF GRU4Rec LightGCN SASRec
```

## Text-modality track

```bash
python3 benchmark/run_text_moRec_benchmark.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/results \
  --model-name TextMoRec-BERT-seq100 \
  --text-encoder bert \
  --bert-model /path/to/local-bert-model \
  --epochs 30 \
  --max-seq-len 100 \
  --hidden-size 256
```

## LLM recommendation track

The Prompt4NR runner requires a separate upstream checkout and model weights supplied by the evaluator:

```bash
python3 benchmark/run_prompt4nr_benchmark.py \
  --prompt4nr-root /path/to/Prompt4NR \
  --model-name /path/to/local-model \
  --prompt4nr-data-dir benchmark/newsrec/artifacts/webrec/prompt4nr_data \
  --skip-export
```

Classic full-ranking and LLM impression-ranking use different candidate sets; their result tables must remain separate.

## Release checks

```bash
python3 scripts/check_anonymity.py --root .
python3 -m compileall -q benchmark sentinels w3alpha scripts
pytest
```
