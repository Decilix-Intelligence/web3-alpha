# Benchmark

This directory contains the W3Alpha benchmark runners.

## Tracks

Classic full-ranking:

- Popularity, UserCF, ItemCF
- BPR, NCF, GRU4Rec, LightGCN, SASRec through RecBole
- TextMoRec-BERT, a title-and-synopsis text-modality sequential baseline
- Metrics: AUC, MRR, Precision/Recall/NDCG at 10, 20, and 50

LLM impression-ranking:

- Prompt4NR
- Metrics: AUC, MRR, NDCG@5, NDCG@10

The two tracks use different candidate sets and should be reported separately.

## Data

This repository contains code and aggregate results. Prepare anonymous dataset
files under `data/webrec_v1/` as described in `../docs/DATA.md`.

Then run:

```bash
python3 benchmark/run_benchmark.py --data-dir data/webrec_v1 --output-dir benchmark/results
```

For RecBole:

```bash
python3 benchmark/export_recbole_dataset.py --data-dir data/webrec_v1 --output-dir benchmark/recbole_data/webrec
python3 benchmark/run_recbole_benchmarks.py --data-dir data/webrec_v1 --recbole-data-dir benchmark/recbole_data/webrec --output-dir benchmark/results
```

For the text-modality sequential baseline:

```bash
python3 benchmark/run_text_moRec_benchmark.py \
  --data-dir data/webrec_v1 \
  --output-dir benchmark/results \
  --text-encoder bert --bert-model /path/to/bert-base-chinese --max-seq-len 100
```

For LLM-track scripts, pass external model paths explicitly. The repository does not include model checkpoints or local deployment paths.

## Results

Reported public result tables are kept in `benchmark/results/`.
