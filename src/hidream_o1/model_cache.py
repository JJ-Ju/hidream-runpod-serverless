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

    expected = model_root / "snapshots" / "<snapshot-hash>"
    visible = _visible_cache_entries(Path(cache_root))
    raise RuntimeError(
        f"Cached model not found: {model_id} under {cache_root}. "
        f"Expected a Hugging Face cache snapshot at {expected}. "
        "In RunPod, set the endpoint Model field to HiDream-ai/HiDream-O1-Image "
        "and wait for the cached model to be prepared, or set HIDREAM_MODEL_PATH "
        "to the exact local snapshot/model directory. "
        f"Visible cache entries: {visible}"
    )


def _visible_cache_entries(cache_root: Path, limit: int = 8) -> str:
    if not cache_root.exists():
        return "<cache root does not exist>"
    if not cache_root.is_dir():
        return "<cache root is not a directory>"
    entries = sorted(path.name for path in cache_root.iterdir())[:limit]
    if not entries:
        return "<empty>"
    return ", ".join(entries)
