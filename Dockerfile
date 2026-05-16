ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG ATTENTION_BACKEND=auto
ARG BOOTSTRAP_FLASH_ATTN=0
ENV ATTENTION_BACKEND=${ATTENTION_BACKEND}
ENV BOOTSTRAP_FLASH_ATTN=${BOOTSTRAP_FLASH_ATTN}
ENV FLASH_ATTN_FALLBACK_BACKEND=sdpa
ENV FLASH_ATTN_PACKAGE=flash-attn
ENV FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
ENV MAX_JOBS=4
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app
ENV HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
ENV HIDREAM_HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
ENV OUTPUT_PREFIX=hidream-o1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY handler.py /app/handler.py
COPY src /app/src
COPY scripts /app/scripts
RUN chmod +x /app/scripts/entrypoint.sh

LABEL org.opencontainers.image.source="https://github.com/JJ-Ju/hidream-runpod-serverless"
LABEL org.opencontainers.image.description="RunPod Serverless worker for HiDream O1 image generation"
LABEL org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["python", "-u", "/app/handler.py"]
