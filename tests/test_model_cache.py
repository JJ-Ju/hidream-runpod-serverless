import pytest

from hidream_o1.model_cache import resolve_model_path


def test_resolve_model_path_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "model"
    explicit.mkdir()

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "cache", explicit) == explicit


def test_resolve_model_path_uses_refs_main_snapshot(tmp_path):
    cache_root = tmp_path / "hub"
    model_root = cache_root / "models--HiDream-ai--HiDream-O1-Image"
    snapshot = model_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (model_root / "refs").mkdir()
    (model_root / "refs" / "main").write_text("abc123\n")

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", cache_root, None) == snapshot


def test_resolve_model_path_falls_back_to_first_snapshot(tmp_path):
    snapshots = tmp_path / "hub" / "models--HiDream-ai--HiDream-O1-Image" / "snapshots"
    (snapshots / "b").mkdir(parents=True)
    (snapshots / "a").mkdir()

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "hub", None) == snapshots / "a"


def test_resolve_model_path_errors_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="HIDREAM_MODEL_PATH"):
        resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "hub", None)


def test_resolve_model_path_error_lists_visible_cache_entries(tmp_path):
    cache_root = tmp_path / "hub"
    (cache_root / "models--Other--Model").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="models--Other--Model"):
        resolve_model_path("HiDream-ai/HiDream-O1-Image", cache_root, None)
