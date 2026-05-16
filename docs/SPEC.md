# HiDream O1 RunPod Serverless Spec

## Purpose

This repository builds a RunPod Serverless worker image for
`HiDream-ai/HiDream-O1-Image`. The worker loads cached model weights, accepts
RunPod queue jobs, generates images with upstream HiDream inference code, stores
outputs in S3-compatible object storage, and returns URL metadata.

## Public Contract

Input is read from `job["input"]`.

Required:

- `prompt`: non-empty string.

Optional:

- `width`, `height`: positive integers up to `2048`, default `2048`.
- `seed`: integer, default `32`.
- `model_type`: `full` or `dev`, default `full`.
- `ref_images`: up to 10 public image URLs or small base64/data URI images.
- `layout_bboxes`: array/object accepted by upstream HiDream layout handling.
- `keep_original_aspect`: boolean, default `false`.
- `editing_scheduler`: `flow_match` or `flash`, default `flow_match`.
- `guidance_scale`, `shift`, `noise_scale_start`, `noise_scale_end`,
  `noise_clip_std`: numeric scheduler knobs.
- `output_format`: `png`, `webp`, or `jpeg`, default `png`.

Response returns:

- `image_url`, `bucket`, `key`, `seed`, `width`, `height`, `mode`, `model_id`,
  `attention_backend`.

## Modes

- `text_to_image`: no reference images.
- `edit`: exactly one reference image and no layout boxes.
- `reference`: two or more reference images and no layout boxes.
- `layout_reference`: reference images with `layout_bboxes`.

## Acceptance Criteria

- The worker validates bad inputs with clear `error` responses before GPU work.
- Reference images reject local paths, private-network URLs, non-image content, and
  payloads larger than 20 MB.
- The worker resolves a local cached Hugging Face snapshot from RunPod cache
  conventions unless `HIDREAM_MODEL_PATH` is set.
- The worker supports all upstream O1 modes through a single RunPod endpoint.
- Outputs are uploaded through S3-compatible configuration and returned as URLs.
- Missing object-storage configuration returns a clear error unless
  `ALLOW_BASE64_OUTPUT=1`.
- Docker builds provide stable `sdpa` and experimental `flash` variants. The
  flash image uses `ATTENTION_BACKEND=auto`, detects live RunPod hardware at
  startup, uses flash-attn when available or cacheable, falls back to SDPA when
  not, and caches matching wheels on the attached network volume under
  `/runpod-volume/flash-attn-cache`.
- Tests cover validation, mode detection, cache resolution, storage URL behavior,
  output key generation, and handler delegation without requiring CUDA/model
  weights.
