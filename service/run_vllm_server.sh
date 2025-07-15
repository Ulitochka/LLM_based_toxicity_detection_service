#!/usr/bin/env bash
set -e

DEFAULT_MODEL_PATH="use_model_from_config"
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=$(realpath "${SCRIPT_PATH}/../")

PYTHONPATH="${PIPELINE_PATH}/../" python3 "${PIPELINE_PATH}/service/server/launch_vllm_server.py" \
    --model_path "$MODEL_PATH" \
    --log_dir "${PIPELINE_PATH}/service/log/" \
    --config_id vllm_server
