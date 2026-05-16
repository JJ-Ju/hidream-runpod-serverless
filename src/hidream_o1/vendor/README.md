# Vendored HiDream O1 Runtime

This directory vendors the minimal runtime files from
`HiDream-ai/HiDream-O1-Image` commit
`210c9bff329467cf32dd3e15f970c029f6f7213f`.

The upstream source and model are MIT licensed. See `UPSTREAM_LICENSE`.

Local change:

- `models/pipeline.py` exposes `use_flash_attn` as a `generate_image(...)`
  parameter so the Docker image can ship stable SDPA and experimental flash-attn
  variants.
