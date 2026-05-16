from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_id: str, cache_root: str | Path, explicit_path: str | Path | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_dir():
            raise RuntimeError(f"Explicit model path does not exist: {path}")
        return path

    if "/" not in model_id:
        raise ValueError(f"model_id must use 'org/name' format: {model_id}")

    org, name = model_id.split("/", 1)
    model_root = Path(cache_root) / f"models--{org}--{name}"
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if refs_main.is_file():
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if candidate.is_dir():
            return candidate

    if snapshots_dir.is_dir():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[0]

    raise RuntimeError(f"Cached model not found: {model_id} under {cache_root}")
