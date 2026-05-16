# RunPod Serverless Deployment

## Image

Use the dynamic attention image:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>
```

Prefer immutable `sha-*` or `vX.Y.Z` tags for production endpoints. Use `latest`
only for quick testing on the default branch.

The image does not compile `flash-attn` during GitHub Actions because the Docker
build runner has no RunPod GPU attached. Instead it starts with
`ATTENTION_BACKEND=auto`, detects the live RunPod GPU/runtime at container
startup, and writes the selected backend into the environment before launching
the worker. If flash-attn is unavailable or cannot be built for the detected
hardware, the worker falls back to SDPA and still serves requests. Set
`ATTENTION_BACKEND=sdpa` only when you want to force the fallback path.

The image uses the official PyTorch
`pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` base, pinned by digest. That
base provides Python 3.12, CUDA 12.8, cuDNN 9, and PyTorch 2.10.0. The production
app dependency environment is installed at startup into a versioned virtual
environment under `/runpod-volume/hidream-runtime/envs` when a network volume is
attached and writable. The venv is created with `--system-site-packages` so it
inherits the base PyTorch/CUDA stack instead of reinstalling it. The cache key
includes Python, CUDA, Torch, TorchVision, platform, and dependency file hashes.
A fresh volume pays the one-time app dependency install cost; later workers
attached to the same volume activate the cached environment. If `/runpod-volume`
is unavailable, the same logic falls back to `/tmp/hidream-runtime`, which is
ephemeral.

Attach a RunPod network volume for persistent runtime caches. Serverless workers mount that
volume at `/runpod-volume`, and the worker caches built flash-attn wheels under
`/runpod-volume/flash-attn-cache`. The cache key includes the detected GPU
compute capability, CUDA version, PyTorch version, Python ABI, platform, and
`FLASH_ATTN_PACKAGE`. The first flash-enabled cold start per unique runtime may
take several minutes. If pip must compile from source, seed this cache from a
CUDA devel Pod attached to the same network volume, then serverless workers can
install the cached wheel without carrying a devel image.

## RunPod Endpoint Settings

- Endpoint type: Queue.
- Container image: an immutable GHCR tag from this repository.
- Model cache: configure `HiDream-ai/HiDream-O1-Image` as the Hugging Face model.
- GPU: choose a CUDA-capable worker with enough VRAM for the full O1 checkpoint.
- Timeout: start high enough for 2048px generation and tune after GPU smoke
  testing.

## Environment Variables

Required:

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
HIDREAM_MODEL_PATH=/explicit/local/model/path
S3_PUBLIC_BASE_URL=https://<public-bucket-or-cdn-base-url>
ALLOW_BASE64_OUTPUT=1
OUTPUT_PREFIX=hidream-o1
HIDREAM_BOOTSTRAP_DEPS=1
HIDREAM_DEPENDENCY_FALLBACK_ROOT=/tmp/hidream-runtime
BOOTSTRAP_FLASH_ATTN=1
FLASH_ATTN_PACKAGE=flash-attn
FLASH_ATTN_CACHE_DIR=/runpod-volume/flash-attn-cache
FLASH_ATTN_FALLBACK_BACKEND=sdpa
MAX_JOBS=4
ATTENTION_BACKEND=auto
```

`HIDREAM_MODEL_PATH` bypasses cache resolution. Leave
`HIDREAM_RUNTIME_CACHE_ROOT` unset unless you intentionally want to override the
auto-selected persistent/ephemeral dependency cache path. `S3_PUBLIC_BASE_URL`
returns deterministic public object URLs; without it, the worker creates
presigned URLs.

## Example Text-To-Image Job

```json
{
  "input": {
    "prompt": "A cinematic poster of a red biplane over rolling green fields with crisp readable text: ADVENTURE IN THE FRIENDLY SKIES",
    "width": 2048,
    "height": 2048,
    "seed": 32,
    "output_format": "png"
  }
}
```

## Example Edit Job

```json
{
  "input": {
    "prompt": "Remove the sunglasses while preserving the person's identity and lighting.",
    "ref_images": ["https://example.com/input/person.png"],
    "keep_original_aspect": true,
    "output_format": "webp"
  }
}
```

Reference images must be public image URLs or small base64/data URI payloads.
The worker rejects local file paths, private-network URLs, non-image content, and
reference payloads larger than 20 MB.

## Example Multi-Reference Job

```json
{
  "input": {
    "prompt": "Place the referenced character in a neon-lit market at night, preserving their face and outfit.",
    "ref_images": [
      "https://example.com/ref/front.png",
      "https://example.com/ref/side.png"
    ],
    "seed": 42
  }
}
```

## Example Layout Job

```json
{
  "input": {
    "prompt": "Two city council members pose on a sunlit terrace, warm approachable mood.",
    "ref_images": [
      "https://example.com/ref/person-a.png",
      "https://example.com/ref/person-b.png"
    ],
    "layout_bboxes": [
      [0.20507812, 0.43945312, 0.48828125, 0.7421875],
      [0.57617188, 0.80078125, 0.08789062, 0.34179688]
    ]
  }
}
```

The layout box order follows upstream HiDream: `[x1, x2, y1, y2]`.

## Manual GPU Smoke Test

1. Deploy the dynamic image with cached `HiDream-ai/HiDream-O1-Image` configured.
2. Submit the text-to-image example and confirm `image_url`, dimensions, seed,
   mode, model ID, and attention backend are returned.
3. Submit the edit example and confirm a single reference image is downloaded
   and `mode` is `edit`.
4. Submit the multi-reference example and confirm `mode` is `reference`.
5. Submit the layout example and confirm `mode` is `layout_reference`.
6. Record GPU type, cold-start time, generation time, peak VRAM, and any image
   quality concerns in the release notes.
