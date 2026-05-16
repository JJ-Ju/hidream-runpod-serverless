from pathlib import Path

import pytest

from hidream_o1.dependency_bootstrap import (
    LOCKED_CUDA_VERSION,
    LOCKED_PYTHON_VERSION,
    LOCKED_TORCH_VERSION,
    LOCKED_TORCHVISION_VERSION,
    RuntimeDependencyConfig,
    choose_volume_root,
    install_requirements,
    write_env_file,
)


def make_requirements(tmp_path: Path) -> tuple[Path, ...]:
    app_requirements = tmp_path / "requirements.txt"
    app_requirements.write_text(
        "transformers==4.57.1\n"
        "diffusers>=0.35\n",
        encoding="utf-8",
    )
    return (app_requirements,)


def test_locked_runtime_versions_are_explicit():
    assert LOCKED_PYTHON_VERSION == "3.12"
    assert LOCKED_CUDA_VERSION == "12.8"
    assert LOCKED_TORCH_VERSION == "2.10.0"
    assert LOCKED_TORCHVISION_VERSION == "0.25.0"


def test_dependency_cache_key_includes_runtime_and_requirement_hash(tmp_path):
    requirements = make_requirements(tmp_path)
    config = RuntimeDependencyConfig(
        app_dir=tmp_path,
        volume_root=Path("/runpod-volume"),
        cache_root=Path("/runpod-volume/hidream-runtime"),
        requirement_files=requirements,
        machine="x86_64",
    )

    assert config.cache_key.startswith(
        "py3.12-cu12.8-torch2.10.0-tv0.25.0-linux-x86_64-"
    )
    assert config.venv_dir == Path("/runpod-volume/hidream-runtime/envs") / config.cache_key
    assert config.ready_marker == config.venv_dir / ".hidream-runtime-ready"

    original_key = config.cache_key
    requirements[0].write_text(
        requirements[0].read_text(encoding="utf-8") + "accelerate>=1.1\n",
        encoding="utf-8",
    )
    changed = RuntimeDependencyConfig(
        app_dir=tmp_path,
        volume_root=Path("/runpod-volume"),
        cache_root=Path("/runpod-volume/hidream-runtime"),
        requirement_files=requirements,
        machine="x86_64",
    )

    assert changed.cache_key != original_key


def test_choose_volume_root_prefers_writable_runpod_volume(tmp_path):
    preferred = tmp_path / "runpod-volume"
    fallback = tmp_path / "fallback"
    preferred.mkdir()

    chosen = choose_volume_root(preferred, fallback)

    assert chosen == preferred
    assert chosen.is_dir()


def test_choose_volume_root_falls_back_when_preferred_is_absent(tmp_path):
    preferred = tmp_path / "missing-runpod-volume"
    fallback = tmp_path / "fallback"

    chosen = choose_volume_root(preferred, fallback)

    assert chosen == fallback
    assert not preferred.exists()
    assert fallback.is_dir()


def test_choose_volume_root_falls_back_when_preferred_is_unwritable(tmp_path):
    preferred = tmp_path / "runpod-volume"
    fallback = tmp_path / "fallback"

    chosen = choose_volume_root(
        preferred,
        fallback,
        is_writable_dir_fn=lambda path: path == fallback,
    )

    assert chosen == fallback


def test_write_env_file_exports_volume_cached_runtime(tmp_path):
    requirements = make_requirements(tmp_path)
    config = RuntimeDependencyConfig(
        app_dir=Path("/app"),
        volume_root=Path("/runpod-volume"),
        cache_root=Path("/runpod-volume/hidream-runtime"),
        requirement_files=requirements,
        machine="x86_64",
    )
    env_file = tmp_path / "runtime.env"

    write_env_file(env_file, config)

    text = env_file.read_text(encoding="utf-8")
    assert f"export HIDREAM_DEPENDENCY_CACHE_KEY={config.cache_key}" in text
    assert f"export VIRTUAL_ENV={config.venv_dir.as_posix()}" in text
    assert f"export PATH={config.venv_dir.as_posix()}/bin:${{PATH}}" in text
    assert "export PYTHONPATH=/app/src:/app" in text
    assert (
        'export HIDREAM_HF_CACHE_ROOT="${HIDREAM_HF_CACHE_ROOT:-'
        '/runpod-volume/huggingface-cache/hub}"'
    ) in text
    assert (
        'export FLASH_ATTN_CACHE_DIR="${FLASH_ATTN_CACHE_DIR:-'
        '/runpod-volume/flash-attn-cache}"'
    ) in text
    assert (
        'export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-'
        '/runpod-volume/hidream-runtime/torch-extensions}"'
    ) in text


def test_runtime_config_requires_existing_requirement_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        RuntimeDependencyConfig(
            app_dir=tmp_path,
            volume_root=Path("/runpod-volume"),
            cache_root=Path("/runpod-volume/hidream-runtime"),
            requirement_files=(tmp_path / "missing.txt",),
        ).requirements_hash


def test_install_requirements_inherits_base_pytorch_site_packages(tmp_path, monkeypatch):
    requirements = make_requirements(tmp_path)
    config = RuntimeDependencyConfig(
        app_dir=tmp_path,
        volume_root=tmp_path / "runpod-volume",
        cache_root=tmp_path / "hidream-runtime",
        requirement_files=requirements,
        machine="x86_64",
    )
    calls = []

    def fake_check_call(command, env=None):
        calls.append(command)
        if command[:3] == [os_sys_executable, "-m", "venv"]:
            config.venv_bin_dir.mkdir(parents=True)
            config.python_executable.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    os_sys_executable = "python"
    monkeypatch.setattr("hidream_o1.dependency_bootstrap.sys.executable", os_sys_executable)
    monkeypatch.setattr("hidream_o1.dependency_bootstrap.subprocess.check_call", fake_check_call)

    install_requirements(config)

    assert calls[0] == [
        os_sys_executable,
        "-m",
        "venv",
        "--system-site-packages",
        str(config.venv_dir),
    ]
    assert all("requirements-torch-cu128.txt" not in " ".join(map(str, call)) for call in calls)
