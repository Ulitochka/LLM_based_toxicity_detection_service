#!/usr/bin/env bash
set -e

# Проверка, что передан аргумент
if [ -z "$1" ]; then
  echo "Ошибка: путь к модели (model_path) не указан."
  exit 1
fi

if [ -z "$2" ]; then
  echo "Ошибка: путь к данным не указан."
  exit 1
fi

MODEL_PATH=$1
DATA_PATH=$2

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=$(realpath "${SCRIPT_PATH}/../")

PYTHONPATH="${PIPELINE_PATH}" python3 "${PIPELINE_PATH}/service/benchmarks/benchmark_runner.py" \
    --url 'localhost' \
    --model_path "$MODEL_PATH" \
    --server_port 8000 \
    --log_dir "${PIPELINE_PATH}/service/log/" \
    --n_threads 16 \
    --required_rps 10 \
    --data_set_path "$DATA_PATH" \
    --stream \
    --config_id vllm_server
