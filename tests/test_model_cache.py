from pathlib import Path

import pytest

from hidream_o1.model_cache import normalize_model_id, resolve_model_path


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


def test_resolve_model_path_accepts_hugging_face_model_url(tmp_path):
    cache_root = tmp_path / "hub"
    model_root = cache_root / "models--HiDream-ai--HiDream-O1-Image"
    snapshot = model_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)

    assert (
        resolve_model_path("https://huggingface.co/HiDream-ai/HiDream-O1-Image", cache_root, None)
        == snapshot
    )


def test_resolve_model_path_accepts_cache_root_that_is_snapshot_dir(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", snapshot, None) == snapshot


def test_resolve_model_path_falls_back_to_first_snapshot(tmp_path):
    snapshots = tmp_path / "hub" / "models--HiDream-ai--HiDream-O1-Image" / "snapshots"
    (snapshots / "b").mkdir(parents=True)
    (snapshots / "a").mkdir()

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "hub", None) == snapshots / "a"


def test_resolve_model_path_finds_case_insensitive_cache_dir(tmp_path):
    snapshots = tmp_path / "hub" / "models--hidream-ai--hidream-o1-image" / "snapshots"
    (snapshots / "abc").mkdir(parents=True)

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "hub", None) == snapshots / "abc"


def test_resolve_model_path_checks_hf_hub_cache_env(monkeypatch, tmp_path):
    configured_root = tmp_path / "wrong"
    env_root = tmp_path / "env-hub"
    snapshots = env_root / "models--HiDream-ai--HiDream-O1-Image" / "snapshots"
    (snapshots / "abc").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(env_root))

    assert resolve_model_path("HiDream-ai/HiDream-O1-Image", configured_root, None) == snapshots / "abc"


def test_resolve_model_path_errors_when_missing(tmp_path):
    with pytest.raises(RuntimeError, match="HIDREAM_MODEL_PATH"):
        resolve_model_path("HiDream-ai/HiDream-O1-Image", tmp_path / "hub", None)


def test_resolve_model_path_error_lists_visible_cache_entries(tmp_path):
    cache_root = tmp_path / "hub"
    (cache_root / "models--Other--Model").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="models--Other--Model"):
        resolve_model_path("HiDream-ai/HiDream-O1-Image", cache_root, None)


def test_resolve_model_path_skips_unreadable_cache_roots(monkeypatch, tmp_path):
    cache_root = tmp_path / "hub"

    original_is_dir = Path.is_dir

    def fake_is_dir(path):
        if str(path) == "/root/.cache/huggingface/hub":
            raise PermissionError("permission denied")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    with pytest.raises(RuntimeError, match="HIDREAM_MODEL_PATH"):
        resolve_model_path("HiDream-ai/HiDream-O1-Image", cache_root, None)


def test_normalize_model_id_handles_hugging_face_urls():
    assert normalize_model_id("https://huggingface.co/HiDream-ai/HiDream-O1-Image") == "HiDream-ai/HiDream-O1-Image"
    assert normalize_model_id("huggingface.co/HiDream-ai/HiDream-O1-Image/tree/main") == "HiDream-ai/HiDream-O1-Image"
