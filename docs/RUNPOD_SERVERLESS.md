# RunPod Serverless Deployment

## Image

Use the stable no-flash image first:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>-sdpa
```

The experimental flash-attn image is published separately:

```text
ghcr.io/jj-ju/hidream-runpod-serverless:<sha-or-release>-flash
```

Prefer immutable `sha-*` or `vX.Y.Z-*` tags for production endpoints. Use
`latest` only for the default SDPA channel and `flash` only for experimental
flash-attn testing.

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
```

`HIDREAM_MODEL_PATH` bypasses cache resolution. `S3_PUBLIC_BASE_URL` returns
deterministic public object URLs; without it, the worker creates presigned URLs.

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

1. Deploy the SDPA image with cached `HiDream-ai/HiDream-O1-Image` configured.
2. Submit the text-to-image example and confirm `image_url`, dimensions, seed,
   mode, model ID, and attention backend are returned.
3. Submit the edit example and confirm a single reference image is downloaded
   and `mode` is `edit`.
4. Submit the multi-reference example and confirm `mode` is `reference`.
5. Submit the layout example and confirm `mode` is `layout_reference`.
6. Record GPU type, cold-start time, generation time, peak VRAM, and any image
   quality concerns in the release notes.
