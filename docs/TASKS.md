# HiDream O1 RunPod Serverless Tasks

## Workstream A: Specification And Docs

- [x] Define serverless worker goals and phased roadmap.
- [x] Define implementation task tracker.
- [x] Document RunPod deployment settings and manual smoke tests.
- [x] Record acceptance criteria before runtime implementation.

## Workstream B: Worker Runtime

- [x] Add request schema and validation for all O1 modes.
- [x] Add mode detection for text-to-image, edit, reference personalization, and
  layout-conditioned reference generation.
- [x] Add cached Hugging Face snapshot path resolution.
- [x] Add singleton model loading around the upstream HiDream processor/model.
- [x] Add image input preparation for URLs and base64 references.
- [x] Add S3-compatible output upload and URL response metadata.
- [x] Add local-only base64 fallback behind an explicit environment flag.
- [x] Add thin RunPod `handler.py` entrypoint.

## Workstream C: Upstream HiDream Integration

- [x] Vendor pinned upstream HiDream inference/model files from commit
  `210c9bff329467cf32dd3e15f970c029f6f7213f`.
- [x] Preserve upstream MIT license and attribution.
- [x] Add attention backend configuration for `sdpa` and `flash`.
- [x] Keep PyTorch dependency away from unsupported `2.9.x`.

## Workstream D: Container And CI

- [x] Add CUDA/PyTorch Dockerfile with `ATTENTION_BACKEND=sdpa|flash`.
- [x] Add requirements for runtime and test dependencies.
- [x] Add `.dockerignore`.
- [x] Add example RunPod payloads.
- [x] Add GitHub Actions workflow to test and build/publish both GHCR variants.

## Workstream E: Verification

- [x] Unit-test validation, mode detection, cache path resolution, storage key
  generation, S3 URL behavior, and handler delegation.
- [x] Run import/compile checks without CUDA/model weights.
- [x] Run local unit tests.
- [x] Review final diff for spec coverage and accidental tracked artifacts.
- [x] Commit and push via `github.com-agent`.
