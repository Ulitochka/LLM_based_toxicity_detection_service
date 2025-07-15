#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=`realpath "${SCRIPT_PATH}"/../`

PYTHONPATH="${PIPELINE_PATH}"/../ python3 "${PIPELINE_PATH}"/models_optimization/quantization/llm_merge.py \
    --quant_result_path "${PIPELINE_PATH}"/models_optimization/quantization/results/ \
    --model_path "${PIPELINE_PATH}"/2025_07_02-04_48_077cd82859_google_gemma-2-2b-it/checkpoint-400/ \
    --config_id merge_default
