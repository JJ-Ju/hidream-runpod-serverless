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
The flash image bootstraps `flash-attn` at container startup with
`BOOTSTRAP_FLASH_ATTN=1`. It detects the live GPU, CUDA, PyTorch, Python ABI,
and platform, then caches the built wheel under `/runpod-volume/flash-attn-cache`
when a RunPod network volume is attached. The first cold start per unique runtime
can still take several minutes; later starts can reuse the cached wheel.

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
FLASH_ATTN_PACKAGE=flash-attn
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
MAX_JOBS=4
```

Reference images must be public image URLs or small base64/data URI payloads.
Local file paths and private-network URLs are rejected.

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
