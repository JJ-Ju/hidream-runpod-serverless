#!/usr/bin/env bash
set -euo pipefail

if [[ "${ATTENTION_BACKEND:-auto}" == "auto" || ( "${ATTENTION_BACKEND:-auto}" == "flash" && "${BOOTSTRAP_FLASH_ATTN:-0}" == "1" ) ]]; then
  echo "Bootstrapping flash-attn for the detected RunPod GPU using ${FLASH_ATTN_CACHE_DIR:-/runpod-volume/flash-attn-cache}"
  attention_env="$(mktemp)"
  python -m hidream_o1.flash_bootstrap --requested "${ATTENTION_BACKEND:-auto}" --write-env "${attention_env}"
  source "${attention_env}"
  rm -f "${attention_env}"
fi

echo "Starting worker with ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}"
exec "$@"
