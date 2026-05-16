#!/usr/bin/env bash
set -euo pipefail

if [[ "${ATTENTION_BACKEND:-sdpa}" == "flash" && "${BOOTSTRAP_FLASH_ATTN:-0}" == "1" ]]; then
  echo "Bootstrapping flash-attn for the detected RunPod GPU using ${FLASH_ATTN_CACHE_DIR:-/runpod-volume/flash-attn-cache}"
  python -m hidream_o1.flash_bootstrap
fi

exec "$@"
