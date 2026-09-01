#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: run_newsreclib_experiment.sh <newsreclib_root> <data_dir> <output_dir> <experiment> <gpu> [extra hydra args...]"
  exit 1
fi

NEWSRECLIB_ROOT="$1"
DATA_DIR="$2"
OUTPUT_DIR="$3"
EXPERIMENT="$4"
GPU="$5"
shift 5

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_DIR="${DATA_DIR}/MINDsmall_train"

CATEG_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/categ2index.tsv") - 1 ))
SENT_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/sentiment2index.tsv") - 1 ))
USER_CLASSES=$(( $(wc -l < "${TRAIN_DIR}/uid2index.tsv") - 1 ))

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  "experiment=${EXPERIMENT}"
  "paths.data_dir=${DATA_DIR}/"
  "paths.output_dir=${OUTPUT_DIR}"
  "data=mind_rec"
  "trainer.devices=1"
  "trainer.accelerator=gpu"
  "data.dataset_attributes=[title,abstract,category,subcategory,title_entities,abstract_entities,category_class,subcategory_class,sentiment_class,sentiment_score]"
  "data.id2index_filenames={word2index:word2index.tsv,entity2index:entity2index.tsv,categ2index:categ2index.tsv,subcateg2index:subcateg2index.tsv,sentiment2index:sentiment2index.tsv,uid2index:uid2index.tsv}"
  "+data.sentiment_annotator._target_=benchmark.newsrec.dummy_sentiment.DummySentimentAnnotator"
  "logger=[]"
)

case "${EXPERIMENT}" in
  naml_*|lstur*)
    COMMON_ARGS+=(
      "model.num_categ_classes=${CATEG_CLASSES}"
      "model.num_sent_classes=${SENT_CLASSES}"
    )
    ;;
esac

case "${EXPERIMENT}" in
  npa_*|lstur*)
    COMMON_ARGS+=("model.num_users=${USER_CLASSES}")
    ;;
esac

(
  cd "${NEWSRECLIB_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${NEWSRECLIB_ROOT}:${REPO_ROOT}" python newsreclib/train.py \
    "${COMMON_ARGS[@]}" \
    "$@"
)
