# HiDream O1 RunPod Serverless Roadmap

## Goal

Create a production-ready RunPod Serverless worker image for
`HiDream-ai/HiDream-O1-Image` that can be built in GitHub Actions, published to
GitHub Container Registry, and linked from a RunPod endpoint.

## Phase 1: Serverless Worker Template

Outcome: a working worker repository that accepts RunPod jobs, loads a cached
HiDream checkpoint, generates images for every upstream O1 mode, stores outputs
in S3-compatible object storage, and returns stable metadata.

Deliverables:

- Runtime package with request validation, model/cache resolution, mode
  detection, image generation orchestration, and output upload.
- Thin `handler.py` for `runpod.serverless.start`.
- Stable SDPA/no-flash Docker image and experimental flash-attn Docker image.
- GHCR publishing workflow with immutable and channel tags.
- RunPod deployment guide with endpoint environment settings and smoke tests.

## Phase 2: Production Hardening

Outcome: lower operational risk once the first endpoint is exercised on real
RunPod GPUs.

Deliverables:

- GPU smoke-test notes for latency, VRAM, and output quality.
- Recommended GPU pool and worker timeout settings.
- Optional prompt-refiner integration once the base worker is stable.
- Optional signed or presigned URL policy if public object URLs are not desired.

## Phase 3: Developer Experience

Outcome: make iteration and endpoint maintenance less fiddly.

Deliverables:

- RunPod API deployment/update script if manual console setup becomes tedious.
- More examples for edit, multi-reference personalization, skeleton, and layout
  conditioning.
- Release checklist for `vX.Y.Z-sdpa` and `vX.Y.Z-flash` image tags.
