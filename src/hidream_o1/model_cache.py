from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def resolve_model_path(model_id: str, cache_root: str | Path, explicit_path: str | Path | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_dir():
            raise RuntimeError(f"Explicit model path does not exist: {path}")
        return path

    model_id = normalize_model_id(model_id)
    cache_name = hf_cache_name(model_id)
    searched_roots = candidate_cache_roots(cache_root)

    for root in searched_roots:
        snapshot = resolve_snapshot_from_root(model_id, root)
        if snapshot is not None:
            return snapshot

    expected = Path(searched_roots[0]) / cache_name / "snapshots" / "<snapshot-hash>"
    visible = "; ".join(f"{root}: {_visible_cache_entries(root)}" for root in searched_roots)
    raise RuntimeError(
        f"Cached model not found: {model_id}. "
        f"Expected a Hugging Face cache snapshot like {expected}. "
        "In RunPod, set the endpoint Model field to HiDream-ai/HiDream-O1-Image "
        "and wait for the cached model to be prepared, or set HIDREAM_MODEL_PATH "
        "to the exact local snapshot/model directory. "
        f"Searched cache roots: {', '.join(str(root) for root in searched_roots)}. "
        f"Visible cache entries: {visible}"
    )


def normalize_model_id(model_id: str) -> str:
    value = str(model_id).strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("huggingface.co"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    elif value.lower().startswith("huggingface.co/"):
        parts = [part for part in value.split("/", 1)[1].split("/") if part]
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    value = value.strip("/")
    if "/" not in value:
        raise ValueError(f"model_id must use 'org/name' format: {model_id}")
    org, name = value.split("/", 1)
    if not org or not name:
        raise ValueError(f"model_id must use 'org/name' format: {model_id}")
    return f"{org}/{name}"


def hf_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def candidate_cache_roots(cache_root: str | Path) -> list[Path]:
    candidates = [
        Path(cache_root),
        Path(os.environ.get("HIDREAM_HF_CACHE_ROOT", "")),
        Path(os.environ.get("HF_HUB_CACHE", "")),
        Path(os.environ.get("TRANSFORMERS_CACHE", "")),
    ]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.extend(
        [
            Path("/runpod-volume/huggingface-cache/hub"),
            Path("/runpod-volume/huggingface/hub"),
            Path("/root/.cache/huggingface/hub"),
            Path("/tmp/huggingface-cache/hub"),
        ]
    )

    result = []
    seen = set()
    for candidate in candidates:
        if not str(candidate):
            continue
        resolved = candidate.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def resolve_snapshot_from_root(model_id: str, cache_root: str | Path) -> Path | None:
    root = Path(cache_root)
    if _looks_like_model_dir(root):
        return root

    cache_name = hf_cache_name(model_id)
    model_roots = []
    if root.name == cache_name:
        model_roots.append(root)
    if _is_dir(root):
        exact = root / cache_name
        model_roots.append(exact)
        lower_cache_name = cache_name.lower()
        try:
            model_roots.extend(
                path for path in root.iterdir() if path.name.lower() == lower_cache_name
            )
        except OSError:
            pass

    for model_root in _dedupe_paths(model_roots):
        snapshot = _snapshot_from_model_root(model_root)
        if snapshot is not None:
            return snapshot
    return None


def _snapshot_from_model_root(model_root: Path) -> Path | None:
    refs_main = model_root / "refs" / "main"
    snapshots_dir = model_root / "snapshots"

    if _is_file(refs_main):
        snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / snapshot_hash
        if _is_dir(candidate):
            return candidate

    if _is_dir(snapshots_dir):
        try:
            snapshots = sorted(path for path in snapshots_dir.iterdir() if _is_dir(path))
        except OSError:
            snapshots = []
        if snapshots:
            return snapshots[0]

    return None


def _looks_like_model_dir(path: Path) -> bool:
    if not _is_dir(path):
        return False
    markers = {
        "config.json",
        "model_index.json",
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    }
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    if any(child.name in markers for child in children):
        return True
    return any(child.suffix in {".safetensors", ".bin"} for child in children)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _visible_cache_entries(cache_root: Path, limit: int = 8) -> str:
    if not _exists(cache_root):
        return "<cache root does not exist>"
    if not _is_dir(cache_root):
        return "<cache root is not a directory>"
    try:
        entries = sorted(path.name for path in cache_root.iterdir())[:limit]
    except OSError:
        return "<cache root is not readable>"
    if not entries:
        return "<empty>"
    return ", ".join(entries)


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False
