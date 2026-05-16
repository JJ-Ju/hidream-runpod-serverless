# HiDream O1 RunPod Serverless Spec

## Purpose

This repository builds a RunPod Serverless worker image for
`HiDream-ai/HiDream-O1-Image`. The worker loads cached model weights, accepts
RunPod queue jobs, generates images with upstream HiDream inference code, stores
returns direct image bytes, S3-compatible URL metadata, or both.

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
- `output_delivery`: `url`, `base64`, or `both`, default `url`.

Response returns:

- Always: `seed`, `width`, `height`, `mode`, `model_id`, `attention_backend`,
  `output_delivery`, `content_type`, `image_size_bytes`.
- For `output_delivery=url`: `image_url`, `bucket`, `key`.
- For `output_delivery=base64`: `image_base64`.
- For `output_delivery=both`: URL metadata plus `image_base64`.

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
- Outputs can be returned directly as base64, uploaded through S3-compatible
  storage and returned as URLs, or both.
- Missing object-storage configuration returns a clear error only when a job
  requests `output_delivery=url` or `output_delivery=both`.
- Docker builds publish one dynamic image. The image uses
  `ATTENTION_BACKEND=auto`, detects live RunPod hardware at startup, uses
  flash-attn when available or cacheable, falls back to SDPA when not, and caches
  matching wheels on the attached network volume under
  `/runpod-volume/flash-attn-cache`. `ATTENTION_BACKEND=sdpa` remains available
  as an endpoint-level override for debugging or conservative rollout.
- The container uses the official
  `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` base, pinned by digest, for
  Python 3.12, CUDA 12.8, cuDNN 9, and PyTorch 2.10.0. Runtime app dependencies
  are installed into a versioned virtual environment on the attached network
  volume when present. The venv inherits the base PyTorch stack with
  `--system-site-packages`. The environment cache key includes the locked
  Python/CUDA/Torch/TorchVision versions, platform, and dependency requirement
  hashes so incompatible dependency sets do not share an environment.
- If `/runpod-volume` is not writable, the worker falls back to an ephemeral
  dependency cache under `/tmp` and still starts.
- Tests cover validation, mode detection, cache resolution, storage URL behavior,
  output key generation, and handler delegation without requiring CUDA/model
  weights.
