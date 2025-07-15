#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=`realpath "${SCRIPT_PATH}"/../`


PYTHONPATH="${PIPELINE_PATH}"/../ python3 "${PIPELINE_PATH}"/model_dev/src/llm/train.py \
    --data_set_path "${PIPELINE_PATH}"/data/rus_tox/ \
    --experiment_path "${PIPELINE_PATH}"/model_dev/results_models/ \
    --config_id llm_cls_5 \
    --wandb
