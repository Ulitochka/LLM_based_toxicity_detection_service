#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=`realpath "${SCRIPT_PATH}"/../`

PYTHONPATH="${PIPELINE_PATH}"/../ python3 "${PIPELINE_PATH}"/models_optimization/quantization/llm_quant.py \
    --quant_result_path "${PIPELINE_PATH}"/models_optimization/quantization/results/ \
    --model_path "${PIPELINE_PATH}"/2025_07_02-04_48_077cd82859_google_gemma-2-2b-it_checkpoint-400_merge_fp16/ \
    --config_id quant_default \
    --data_set_path "${PIPELINE_PATH}"/llm_exps/data/rus_tox/
