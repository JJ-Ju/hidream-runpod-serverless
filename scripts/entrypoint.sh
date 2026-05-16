#!/usr/bin/env bash
set -euo pipefail

if [[ "${HIDREAM_BOOTSTRAP_DEPS:-1}" == "1" ]]; then
  echo "Bootstrapping locked HiDream dependency environment"
  runtime_env="$(mktemp)"
  python -m hidream_o1.dependency_bootstrap --write-env "${runtime_env}"
  source "${runtime_env}"
  rm -f "${runtime_env}"
  echo "Using HIDREAM_DEPENDENCY_CACHE_KEY=${HIDREAM_DEPENDENCY_CACHE_KEY}"
fi

if [[ "${ATTENTION_BACKEND:-auto}" == "auto" ]] || [[ "${ATTENTION_BACKEND:-auto}" == "flash" && "${BOOTSTRAP_FLASH_ATTN:-0}" == "1" ]]; then
  echo "Bootstrapping flash-attn for the detected RunPod GPU using ${FLASH_ATTN_CACHE_DIR:-/runpod-volume/flash-attn-cache}"
  attention_env="$(mktemp)"
  python -m hidream_o1.flash_bootstrap --requested "${ATTENTION_BACKEND:-auto}" --write-env "${attention_env}"
  source "${attention_env}"
  rm -f "${attention_env}"
fi

echo "Starting worker with ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}"
exec "$@"
