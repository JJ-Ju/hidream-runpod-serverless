# HiDream O1 RunPod Serverless

RunPod Serverless worker for
[`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image).
The worker uses cached Hugging Face model weights on RunPod, generates images
through the upstream HiDream inference code, uploads outputs to S3-compatible
object storage, and returns image URLs.

## Images

Stable no-flash attention image:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>-sdpa
```

Experimental flash-attn image:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>-flash
```

Use immutable `sha-*` or `vX.Y.Z-*` tags for RunPod production endpoints.
The flash image uses `ATTENTION_BACKEND=auto`. At container startup it detects
the live GPU, CUDA, PyTorch, Python ABI, and platform. If flash-attn is already
installed, available from `/runpod-volume/flash-attn-cache`, or can be built for
that exact runtime, the worker starts with flash attention. Otherwise it falls
back to SDPA and still serves requests. If pip must compile from source, seed the
cache from a CUDA devel Pod attached to the same network volume.

The container uses the official PyTorch
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` image, pinned by digest, so
Python 3.12, CUDA 12.8, cuDNN 9, and PyTorch 2.10.0 come from a known upstream
base. App dependencies are installed into a hash-keyed virtual environment under
`/runpod-volume/hidream-runtime/envs` when a writable network volume is present.
The venv inherits the base PyTorch stack instead of reinstalling it. A fresh
volume pays the app dependency install cost once; later workers reuse the cached
environment. Without a writable volume, the worker falls back to an ephemeral
cache under `/tmp/hidream-runtime`.

## Required RunPod Environment

```text
HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
HIDREAM_HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
S3_ENDPOINT_URL=https://<s3-compatible-endpoint>
S3_REGION=auto
S3_BUCKET=<bucket-name>
S3_ACCESS_KEY_ID=<access-key>
S3_SECRET_ACCESS_KEY=<secret-key>
```

Optional:

```text
S3_PUBLIC_BASE_URL=https://<public-bucket-or-cdn-base-url>
ALLOW_BASE64_OUTPUT=1
OUTPUT_PREFIX=hidream-o1
BOOTSTRAP_FLASH_ATTN=1
HIDREAM_BOOTSTRAP_DEPS=1
FLASH_ATTN_PACKAGE=flash-attn
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
FLASH_ATTN_FALLBACK_BACKEND=sdpa
MAX_JOBS=4
```

Reference images must be public image URLs or small base64/data URI payloads.
Local file paths and private-network URLs are rejected.

## Ports

No container ports need to be exposed for RunPod Serverless. This image starts
`handler.py`, which calls `runpod.serverless.start(...)` and receives jobs
through RunPod's serverless queue/control plane. Do not configure an HTTP server,
container port, or `EXPOSE` directive unless you add a separate debugging server.

## Local Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Upstream Runtime

The worker vendors the minimal HiDream runtime files from
`HiDream-ai/HiDream-O1-Image` commit
`210c9bff329467cf32dd3e15f970c029f6f7213f`. The upstream MIT license is
preserved in `src/hidream_o1/vendor/UPSTREAM_LICENSE`.

## RunPod

See [`docs/RUNPOD_SERVERLESS.md`](docs/RUNPOD_SERVERLESS.md) for endpoint
settings, example payloads, and GPU smoke tests.
