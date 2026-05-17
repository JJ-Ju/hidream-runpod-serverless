# HiDream O1 RunPod Serverless

RunPod Serverless worker for
[`HiDream-ai/HiDream-O1-Image`](https://huggingface.co/HiDream-ai/HiDream-O1-Image).
The worker uses cached Hugging Face model weights on RunPod, generates images
through the upstream HiDream inference code, and returns direct base64 image
bytes, S3-compatible object URLs, or both.

## Images

Dynamic attention image:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>
```

Use immutable `sha-*` or `vX.Y.Z` tags for RunPod production endpoints.
The image uses `ATTENTION_BACKEND=auto`. At container startup it detects
the live GPU, CUDA, PyTorch, Python ABI, and platform. If flash-attn is already
installed, available from `/runpod-volume/flash-attn-cache`, or can be built for
that exact runtime, the worker starts with flash attention. Otherwise it falls
back to SDPA and still serves requests. To force SDPA for debugging, set
`ATTENTION_BACKEND=sdpa` on the RunPod endpoint. If pip must compile flash-attn
from source, seed the cache from a CUDA devel Pod attached to the same network
volume.

The container uses the official PyTorch
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` image, pinned by digest, so
Python 3.12, CUDA 12.8, cuDNN 9, and PyTorch 2.10.0 come from a known upstream
base. App dependencies are installed into a hash-keyed virtual environment under
`/runpod-volume/hidream-runtime/envs` when a writable network volume is present.
The venv inherits the base PyTorch stack instead of reinstalling it. A fresh
volume pays the app dependency install cost once; later workers reuse the cached
environment. Without a writable volume, the worker falls back to an ephemeral
cache under `/tmp/hidream-runtime`.

## RunPod Quickstart

1. Use the dynamic image:

   ```text
   ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>
   ```

2. Create a RunPod Serverless endpoint with a custom container image. Use a
   queue endpoint. Do not configure an exposed port.

3. Attach a RunPod network volume if you want dependency, model, pip, Torch
   extension, and flash-attn caches to survive cold starts. Serverless mounts the
   volume at `/runpod-volume`.

4. Configure Hugging Face model caching for:

   ```text
   HiDream-ai/HiDream-O1-Image
   ```

   The cached model must appear inside the worker at:

   ```text
   /runpod-volume/huggingface-cache/hub/models--HiDream-ai--HiDream-O1-Image/snapshots/<snapshot-hash>
   ```

   If the worker reports `Cached model not found`, the endpoint Model field has
   not prepared this model on the selected worker host, or `HIDREAM_HF_CACHE_ROOT`
   points at the wrong location. Keep the endpoint Model field set to
   `HiDream-ai/HiDream-O1-Image`, or set `HIDREAM_MODEL_PATH` to the exact local
   snapshot/model directory.

5. Set the required environment variables from the section below. S3-compatible
   storage is only required when jobs request `output_delivery=url` or
   `output_delivery=both`.

6. Submit jobs to the RunPod endpoint with JSON shaped like:

   ```json
   {
     "input": {
       "prompt": "A cinematic portrait of an astronaut botanist in a glass greenhouse on Mars",
       "width": 2048,
       "height": 2048,
       "seed": 32,
       "output_format": "png",
       "output_delivery": "base64"
     }
   }
   ```

   For a 3D pipeline that needs to process an existing image before the next
   stage, submit an edit job with a source image and direct output:

   ```json
   {
     "input": {
       "prompt": "Convert this render into a clean front-facing asset image with crisp silhouette edges and neutral studio lighting.",
       "input_image": {
         "base64": "<base64-encoded-source-image>",
         "mime_type": "image/png"
       },
       "keep_original_aspect": true,
       "output_format": "png",
       "output_delivery": "base64"
     }
   }
   ```

7. Leave `ATTENTION_BACKEND=auto` for normal use. The worker will use flash-attn
   when it is available for the live hardware and fall back to SDPA otherwise.
   Set `ATTENTION_BACKEND=sdpa` only when you want to disable flash-attn
   explicitly.

## Supported Job Modes

This is not prompt-only. The same endpoint supports the O1 reference-image
workflows used by image preprocessing and 3D generation pipelines:

- Text-to-image: send `prompt` without image inputs. Response mode is
  `text_to_image`.
- Single-image editing: send one direct source image payload with `input_image`,
  `init_image`, `image`, `ref_image`, or canonical `ref_images: [{...}]`.
  Response mode is `edit`.
- Multi-reference personalization: send two or more entries in `input_images`
  or `ref_images`.
  Response mode is `reference`.
- Layout-conditioned reference generation: send `input_images` or `ref_images`
  plus one
  normalized `[x1, x2, y1, y2]` box per image in `layout_bboxes`. Response mode
  is `layout_reference`.

Image inputs do not need to be hosted anywhere. Use direct payload objects:
`{"base64": "...", "mime_type": "image/png"}` or
`{"data_uri": "data:image/png;base64,..."}`. Plain base64 strings and public or
presigned URLs are also accepted for compatibility, but direct payloads are the
recommended path for chained 3D workflows. Local file paths and private-network
URLs are rejected before GPU work. Set `DEFAULT_OUTPUT_DELIVERY=base64` on the
endpoint or pass `output_delivery=base64` per job so the next stage can consume
the generated image bytes directly.

## Required RunPod Environment

Use one of these profiles in the RunPod environment variable editor.

### Direct Input And Direct Output

Use this for a pipeline that sends source images directly in the request and
expects generated image bytes directly in the response. No input bucket, output
bucket, or exposed port is required.

```text
HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
HIDREAM_HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
DEFAULT_OUTPUT_DELIVERY=base64
ATTENTION_BACKEND=auto
HIDREAM_BOOTSTRAP_DEPS=1
BOOTSTRAP_FLASH_ATTN=1
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
FLASH_ATTN_FALLBACK_BACKEND=sdpa
WORK_DIR=/tmp/hidream-o1
```

Submit direct image input in the job payload:

```json
{
  "input": {
    "prompt": "Clean this source image for a downstream 3D reconstruction stage.",
    "input_image": {
      "base64": "<base64-encoded-source-image>",
      "mime_type": "image/png"
    },
    "output_delivery": "base64"
  }
}
```

### Direct Input And S3 Output

Use this when the source image still arrives directly in the request, but the
generated output should be uploaded to S3-compatible storage and returned as
`image_url`, `bucket`, and `key`.

```text
HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
HIDREAM_HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
DEFAULT_OUTPUT_DELIVERY=url
ATTENTION_BACKEND=auto
HIDREAM_BOOTSTRAP_DEPS=1
BOOTSTRAP_FLASH_ATTN=1
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
FLASH_ATTN_FALLBACK_BACKEND=sdpa
WORK_DIR=/tmp/hidream-o1
OUTPUT_PREFIX=hidream-o1
S3_ENDPOINT_URL=https://<s3-compatible-endpoint>
S3_REGION=auto
S3_BUCKET=<bucket-name>
S3_ACCESS_KEY_ID=<access-key>
S3_SECRET_ACCESS_KEY=<secret-key>
S3_PUBLIC_BASE_URL=https://<public-bucket-or-cdn-base-url>
```

`S3_PUBLIC_BASE_URL` is optional. If it is omitted, the worker returns a
presigned URL for the uploaded object.

Submit the same direct image input shape and either omit `output_delivery` or
set it to `url`:

```json
{
  "input": {
    "prompt": "Clean this source image for a downstream 3D reconstruction stage.",
    "input_image": {
      "base64": "<base64-encoded-source-image>",
      "mime_type": "image/png"
    },
    "output_delivery": "url"
  }
}
```

### Reference

```text
HIDREAM_MODEL_ID=HiDream-ai/HiDream-O1-Image
HIDREAM_HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub
```

Required for URL outputs:

```text
S3_ENDPOINT_URL=https://<s3-compatible-endpoint>
S3_REGION=auto
S3_BUCKET=<bucket-name>
S3_ACCESS_KEY_ID=<access-key>
S3_SECRET_ACCESS_KEY=<secret-key>
```

Optional:

```text
S3_PUBLIC_BASE_URL=https://<public-bucket-or-cdn-base-url>
DEFAULT_OUTPUT_DELIVERY=base64
OUTPUT_PREFIX=hidream-o1
BOOTSTRAP_FLASH_ATTN=1
HIDREAM_BOOTSTRAP_DEPS=1
FLASH_ATTN_PACKAGE=flash-attn
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
FLASH_ATTN_FALLBACK_BACKEND=sdpa
MAX_JOBS=4
ATTENTION_BACKEND=auto
```

Reference images can be direct base64/data URI payloads or public/presigned
URLs. Direct payload objects are preferred for pipeline input:
`{"base64": "...", "mime_type": "image/png"}`. Local file paths and
private-network URLs are rejected.

The official PyTorch runtime image does not include the CUDA compiler toolchain.
With `ATTENTION_BACKEND=auto`, the worker uses a cached flash-attn wheel if one
is already present under `FLASH_ATTN_CACHE_DIR`; otherwise it starts with SDPA.
To build flash-attn wheels, use a CUDA devel Pod attached to the same network
volume, or leave the endpoint on SDPA.

## Output Delivery

Each job can choose how the generated image is returned:

- `output_delivery=url`: upload to S3-compatible storage and return `image_url`,
  `bucket`, and `key`.
- `output_delivery=base64`: return `image_base64`, `content_type`, and
  `image_size_bytes` directly in the RunPod response.
- `output_delivery=both`: return both the uploaded URL metadata and direct
  base64 bytes.

Set `DEFAULT_OUTPUT_DELIVERY=base64` on the endpoint to make direct outputs the
default for pipeline workflows. Per-job `output_delivery` overrides that default.

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

## GUI And Pipeline Clients

See [`docs/CLIENT_API.md`](docs/CLIENT_API.md) for the client-facing API
contract, RunPod `/runsync` and `/run` request shapes, TypeScript types, direct
image payload helpers, response handling, and GUI error-state guidance.
