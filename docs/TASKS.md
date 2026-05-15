# HiDream O1 RunPod Serverless Tasks

## Workstream A: Specification And Docs

- [x] Define serverless worker goals and phased roadmap.
- [x] Define implementation task tracker.
- [x] Document RunPod deployment settings and manual smoke tests.
- [x] Record acceptance criteria before runtime implementation.

## Workstream B: Worker Runtime

- [ ] Add request schema and validation for all O1 modes.
- [ ] Add mode detection for text-to-image, edit, reference personalization, and
  layout-conditioned reference generation.
- [ ] Add cached Hugging Face snapshot path resolution.
- [ ] Add singleton model loading around the upstream HiDream processor/model.
- [ ] Add image input preparation for URLs and base64 references.
- [ ] Add S3-compatible output upload and URL response metadata.
- [ ] Add local-only base64 fallback behind an explicit environment flag.
- [ ] Add thin RunPod `handler.py` entrypoint.

## Workstream C: Upstream HiDream Integration

- [ ] Vendor pinned upstream HiDream inference/model files from commit
  `210c9bff329467cf32dd3e15f970c029f6f7213f`.
- [ ] Preserve upstream MIT license and attribution.
- [ ] Add attention backend configuration for `sdpa` and `flash`.
- [ ] Keep PyTorch dependency away from unsupported `2.9.x`.

## Workstream D: Container And CI

- [ ] Add CUDA/PyTorch Dockerfile with `ATTENTION_BACKEND=sdpa|flash`.
- [ ] Add requirements for runtime and test dependencies.
- [ ] Add `.dockerignore`.
- [ ] Add example RunPod payloads.
- [ ] Add GitHub Actions workflow to test and build/publish both GHCR variants.

## Workstream E: Verification

- [ ] Unit-test validation, mode detection, cache path resolution, storage key
  generation, S3 URL behavior, and handler delegation.
- [ ] Run import/compile checks without CUDA/model weights.
- [ ] Run local unit tests.
- [ ] Review final diff for spec coverage and accidental tracked artifacts.
- [ ] Commit and push via `github.com-agent`.
