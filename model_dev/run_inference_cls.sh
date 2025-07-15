#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
PIPELINE_PATH=$(realpath "${SCRIPT_PATH}/../")

BASE_RESULTS_PATH="${PIPELINE_PATH}/model_dev/results_models"
DATA_PATH="${PIPELINE_PATH}/data/rus_tox/"

# Аргументы
MODEL_NAME="$1"              # имя модели (или пусто — все модели)
CHECKPOINT_NAME="$2"         # имя чекпойнта (или пусто — все чекпойнты)
CONFIG_ID="${3:-llm_cls_5}"  # config_id, по умолчанию

# Список моделей
if [ -n "$MODEL_NAME" ]; then
    MODEL_DIR="${BASE_RESULTS_PATH}/${MODEL_NAME}"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Указанный каталог модели не существует: $MODEL_DIR"
        exit 1
    fi
    MODEL_DIRS=("$MODEL_DIR")
else
    MODEL_DIRS=("${BASE_RESULTS_PATH}"/*)
fi

# Перебор моделей и чекпойнтов
for model_dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        echo "▶ Обрабатываем модель: $model_name"

        # Если передан конкретный чекпойнт
        if [ -n "$CHECKPOINT_NAME" ]; then
            checkpoint_dir="${model_dir}/${CHECKPOINT_NAME}"
            if [ -d "$checkpoint_dir" ]; then
                echo "➡ Начинаем инференс: $model_name — $CHECKPOINT_NAME"
                PYTHONPATH="${PIPELINE_PATH}/../" python3 "${PIPELINE_PATH}/model_dev/src/llm/train/inference.py" \
                    --data_set_path "$DATA_PATH" \
                    --experiment_path "$checkpoint_dir" \
                    --config_id "$CONFIG_ID"  
            else
                echo "⚠ Чекпойнт не найден: $checkpoint_dir"
            fi
        else
            # Если чекпойнт не указан — обработать все
            for checkpoint_dir in "$model_dir"/checkpoint-*; do
                if [ -d "$checkpoint_dir" ]; then
                    checkpoint_name=$(basename "$checkpoint_dir")
                    echo "➡ Начинаем инференс: $model_name — $checkpoint_name"
                    PYTHONPATH="${PIPELINE_PATH}/../" python3 "${PIPELINE_PATH}/model_dev/src/llm/train/inference.py" \
                        --data_set_path "$DATA_PATH" \
                        --experiment_path "$checkpoint_dir" \
                        --config_id "$CONFIG_ID" 
                fi
            done
        fi
    fi
done