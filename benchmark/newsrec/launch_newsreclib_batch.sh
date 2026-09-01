#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: launch_newsreclib_batch.sh <newsreclib_root> <data_dir> <output_dir>"
  exit 1
fi

NEWSRECLIB_ROOT="$1"
DATA_DIR="$2"
OUTPUT_DIR="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${OUTPUT_DIR}"
VALID_TIME_SPLIT="$(python3 - <<PY
import json
from pathlib import Path
data_dir = Path(r"${DATA_DIR}")
meta = json.loads(data_dir.parent.joinpath("metadata.json").read_text(encoding="utf-8"))
print(meta["valid_time_split"])
PY
)"
TRAIN_DIR="${DATA_DIR}/MINDsmall_train"
CATEG_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/categ2index.tsv") - 1 ))
SUBCATEG_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/subcateg2index.tsv") - 1 ))
SENT_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/sentiment2index.tsv") - 1 ))
USER_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/uid2index.tsv") - 1 ))

COMMON_ARGS=(
  "data=mind_rec"
  "trainer.devices=1"
  "trainer.accelerator=gpu"
  "data.valid_time_split=${VALID_TIME_SPLIT}"
  "data.dataset_attributes=[title,abstract,category,subcategory,title_entities,abstract_entities,category_class,subcategory_class,sentiment_class,sentiment_score]"
  "data.id2index_filenames={word2index:word2index.tsv,entity2index:entity2index.tsv,categ2index:categ2index.tsv,subcateg2index:subcateg2index.tsv,sentiment2index:sentiment2index.tsv,uid2index:uid2index.tsv}"
  "+data.sentiment_annotator._target_=benchmark.newsrec.dummy_sentiment.DummySentimentAnnotator"
  "logger=[]"
)

declare -a GPUS=(4 5 6 7)
declare -a EXPERIMENTS=(
  "dkn_mindsmall_pretrainedemb_celoss_bertsent"
  "npa_mindsmall_pretrainedemb_celoss_bertsent"
  "naml_mindsmall_pretrainedemb_celoss_bertsent"
  "nrms_mindsmall_pretrainedemb_celoss_bertsent"
  "lsturini_mindsmall_pretrainedemb_celoss_bertsent"
)

launch_job() {
  local exp="$1"
  local gpu="$2"
  local log_path="${OUTPUT_DIR}/${exp}.log"
  local -a extra_args=()
  mkdir -p "${OUTPUT_DIR}/${exp}"

  case "${exp}" in
    naml_*|lstur*)
      extra_args+=(
        "model.num_categ_classes=${CATEG_CLASSES}"
        "model.num_sent_classes=${SENT_CLASSES}"
      )
      ;;
  esac

  case "${exp}" in
    npa_*|lstur*)
      extra_args+=("model.num_users=${USER_CLASSES}")
      ;;
  esac

  (
    cd "${NEWSRECLIB_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${NEWSRECLIB_ROOT}:${REPO_ROOT}" python newsreclib/train.py \
      "experiment=${exp}" \
      "paths.data_dir=${DATA_DIR}/" \
      "paths.output_dir=${OUTPUT_DIR}/${exp}" \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      > "${log_path}" 2>&1
  ) &
  LAUNCHED_PID="$!"
}

declare -a RUNNING_PIDS=()
declare -a RUNNING_GPUS=()

for idx in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$idx]}"
  slot=$((idx % ${#GPUS[@]}))
  if (( idx >= ${#GPUS[@]} )); then
    wait "${RUNNING_PIDS[$slot]}"
  fi
  gpu="${GPUS[$slot]}"
  launch_job "${exp}" "${gpu}"
  RUNNING_PIDS[$slot]="${LAUNCHED_PID}"
  RUNNING_GPUS[$slot]="${gpu}"
  echo "launched ${exp} on GPU ${gpu} (pid ${RUNNING_PIDS[$slot]})"
done

for pid in "${RUNNING_PIDS[@]}"; do
  wait "${pid}"
done
