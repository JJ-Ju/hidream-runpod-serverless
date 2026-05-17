ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.8
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG ATTENTION_BACKEND=auto
ARG BOOTSTRAP_FLASH_ATTN=1
ENV HIDREAM_PYTHON_VERSION=${PYTHON_VERSION}
ENV HIDREAM_CUDA_VERSION=${CUDA_VERSION}
ENV HIDREAM_TORCH_VERSION=${TORCH_VERSION}
ENV HIDREAM_TORCHVISION_VERSION=${TORCHVISION_VERSION}
ENV ATTENTION_BACKEND=${ATTENTION_BACKEND}
ENV BOOTSTRAP_FLASH_ATTN=${BOOTSTRAP_FLASH_ATTN}
ENV HIDREAM_BOOTSTRAP_DEPS=1
ENV HIDREAM_VOLUME_ROOT=/runpod-volume
ENV HIDREAM_DEPENDENCY_FALLBACK_ROOT=/tmp/hidream-runtime
ENV FLASH_ATTN_FALLBACK_BACKEND=sdpa
ENV FLASH_ATTN_PACKAGE=flash-attn
ENV MAX_JOBS=4
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app
ENV HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
ENV OUTPUT_PREFIX=hidream-o1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

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
