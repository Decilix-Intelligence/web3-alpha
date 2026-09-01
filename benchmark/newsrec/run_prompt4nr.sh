#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: run_prompt4nr.sh <prompt4nr_root> <data_dir> <output_dir> <gpu>"
  exit 1
fi

PROMPT4NR_ROOT="$1"
DATA_DIR="$2"
OUTPUT_DIR="$3"
GPU="$4"
mkdir -p "${OUTPUT_DIR}"

(
  cd "${PROMPT4NR_ROOT}/Discrete-Relevance"
  export CUDA_VISIBLE_DEVICES="${GPU}"
  python main-multigpu.py \
    --data_path "${DATA_DIR}" \
    --epochs 4 \
    --batch_size 8 \
    --test_batch_size 64 \
    --wd 1e-3 \
    --max_tokens 500 \
    --log True \
    --model_save True \
    > "${OUTPUT_DIR}/train.log" 2>&1

  python predict.py \
    --data_path "${DATA_DIR}" \
    --test_batch_size 64 \
    --max_tokens 500 \
    --model_file ./temp/BestModel.pt \
    --log True \
    > "${OUTPUT_DIR}/test.log" 2>&1
)
