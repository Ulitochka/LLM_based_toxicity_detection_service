#!/usr/bin/env bash
set -e

IMAGE_NAME="vllm_service"
TAG="latest"

DOCKERFILE="Dockerfile"

BUILD_CONTEXT="../"

echo "Собираем Docker-образ: ${IMAGE_NAME}:${TAG}"
DOCKER_BUILDKIT=1 docker build --build-arg ENABLE_FLASH_ATTN=true --progress=plain -f "$DOCKERFILE" -t "${IMAGE_NAME}:${TAG}" "$BUILD_CONTEXT"

echo "Сборка завершена: ${IMAGE_NAME}:${TAG}"
