# ----------- STAGE 1: BUILD flash-attn with nvcc (optional) -----------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS flashattn-builder

ARG ENABLE_FLASH_ATTN

WORKDIR /build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    build-essential git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir --progress-bar=on torch==2.6.0 numpy packaging

RUN if [ "$ENABLE_FLASH_ATTN" = "true" ]; then \
    echo "=== Building FlashAttention ===" && \
    git clone --depth 1 --branch v2.7.3 https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    python3 setup.py bdist_wheel && \
    rm -rf build *.egg-info __pycache__; \
else \
    echo "=== Skipping FlashAttention build ==="; \
fi

# ----------- STAGE 2: FINAL RUNTIME IMAGE -----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG ENABLE_FLASH_ATTN=false
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /service

RUN apt update && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3 python3-venv wget git && \
    python3 -m venv venv && \
    venv/bin/pip install --no-cache-dir -U pip setuptools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/service/venv/bin:$PATH"

COPY --from=flashattn-builder /build/flash-attention/dist /tmp/dist

RUN if [ "$ENABLE_FLASH_ATTN" = "true" ] && ls /tmp/dist/flash_attn-2.7.3*.whl >/dev/null 2>&1; then \
    pip install --no-deps /tmp/dist/flash_attn-2.7.3*.whl && rm -rf /tmp/dist; \
else \
    echo "flash-attn installation skipped."; \
fi

COPY requirements-service.txt /service/
RUN pip install --progress-bar=on --default-timeout=5000 --no-cache-dir -r requirements-service.txt && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/log/dpkg.log && \
    rm -rf /root/.cache /service/venv/pip-cache

COPY api /service/api
COPY configs /service/configs
COPY data_tools /service/data_tools
COPY log /service/log
COPY server /service/server
COPY run_vllm_server.sh /service/run_vllm_server.sh
COPY stop_vllm_server.sh /service/stop_vllm_server.sh

EXPOSE 8000
ENV LANG=ru_RU.UTF-8

ENTRYPOINT ["/bin/bash", "run_vllm_server.sh"]
