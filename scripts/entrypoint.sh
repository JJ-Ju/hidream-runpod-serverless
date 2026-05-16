#!/usr/bin/env bash
set -euo pipefail

if [[ "${ATTENTION_BACKEND:-sdpa}" == "flash" && "${BOOTSTRAP_FLASH_ATTN:-0}" == "1" ]]; then
  if python - <<'PY'
try:
    import flash_attn  # noqa: F401
except Exception:
    try:
        import flash_attn_interface  # noqa: F401
    except Exception:
        raise SystemExit(1)
PY
  then
    echo "flash-attn is already available"
  else
    echo "Installing flash-attn at container startup. This can add several minutes to cold starts."
    python -m pip install --no-cache-dir packaging ninja
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install --no-cache-dir --no-build-isolation "${FLASH_ATTN_PACKAGE:-flash-attn}"
  fi
fi

exec "$@"
