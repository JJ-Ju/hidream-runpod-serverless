ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG ATTENTION_BACKEND=sdpa
ENV ATTENTION_BACKEND=${ATTENTION_BACKEND}
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
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt \
    && if [ "${ATTENTION_BACKEND}" = "flash" ]; then \
        python -m pip install --no-cache-dir packaging ninja \
        && python -m pip install --no-cache-dir --no-build-isolation flash-attn; \
    fi

COPY pyproject.toml /app/pyproject.toml
COPY handler.py /app/handler.py
COPY src /app/src

LABEL org.opencontainers.image.source="https://github.com/JJ-Ju/hidream-runpod-serverless"
LABEL org.opencontainers.image.description="RunPod Serverless worker for HiDream O1 image generation"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["python", "-u", "/app/handler.py"]
