from pathlib import Path

from hidream_o1.flash_bootstrap import (
    FlashRuntime,
    choose_attention_backend,
    cached_wheel,
    write_env_file,
    safe_package_name,
)


def test_flash_runtime_cache_key_includes_gpu_and_abi():
    runtime = FlashRuntime(
        package="flash-attn",
        cache_root=Path("/runpod-volume/flash-attn-cache"),
        python_tag="cp310",
        torch_tag="torch2.10.0",
        cuda_tag="cu128",
        gpu_arch="sm90",
        machine="x86_64",
    )

    assert runtime.cache_key == "flash-attn/cp310-torch2.10.0-cu128-sm90-x86_64"
    assert runtime.cache_dir == Path("/runpod-volume/flash-attn-cache/flash-attn/cp310-torch2.10.0-cu128-sm90-x86_64")


def test_cached_wheel_picks_existing_wheel(tmp_path):
    runtime = FlashRuntime(
        package="flash-attn",
        cache_root=tmp_path,
        python_tag="cp310",
        torch_tag="torch2.10.0",
        cuda_tag="cu128",
        gpu_arch="sm90",
        machine="x86_64",
    )
    runtime.cache_dir.mkdir(parents=True)
    wheel = runtime.cache_dir / "flash_attn-2.8.0.whl"
    wheel.write_bytes(b"fake wheel")

    assert cached_wheel(runtime) == wheel


def test_safe_package_name_sanitizes_pip_spec():
    assert safe_package_name("flash-attn==2.8.3") == "flash-attn-2.8.3"
    assert safe_package_name("git+https://example.com/pkg") == "git-https-example.com-pkg"


def test_auto_backend_falls_back_to_sdpa_without_runtime():
    result = choose_attention_backend(
        requested="auto",
        detect_runtime_fn=lambda: (_ for _ in ()).throw(RuntimeError("no CUDA")),
        flash_available_fn=lambda: False,
    )

    assert result.backend == "sdpa"
    assert "no CUDA" in result.reason


def test_auto_backend_uses_flash_when_flash_is_already_available():
    result = choose_attention_backend(
        requested="auto",
        detect_runtime_fn=lambda: (_ for _ in ()).throw(AssertionError("should not detect")),
        flash_available_fn=lambda: True,
    )

    assert result.backend == "flash"
    assert result.reason == "flash-attn is already importable"


def test_auto_backend_installs_cached_wheel_and_selects_flash(tmp_path):
    runtime = FlashRuntime(
        package="flash-attn",
        cache_root=tmp_path,
        python_tag="cp310",
        torch_tag="torch2.10.0",
        cuda_tag="cu128",
        gpu_arch="sm90",
        machine="x86_64",
    )
    runtime.cache_dir.mkdir(parents=True)
    wheel = runtime.cache_dir / "flash_attn-2.8.0.whl"
    wheel.write_bytes(b"fake wheel")
    installed = []

    result = choose_attention_backend(
        requested="auto",
        detect_runtime_fn=lambda: runtime,
        flash_available_fn=lambda: False,
        install_wheel_fn=lambda path: installed.append(path),
    )

    assert result.backend == "flash"
    assert installed == [wheel]
    assert "cached" in result.reason


def test_auto_backend_falls_back_when_build_fails(tmp_path):
    runtime = FlashRuntime(
        package="flash-attn",
        cache_root=tmp_path,
        python_tag="cp310",
        torch_tag="torch2.10.0",
        cuda_tag="cu128",
        gpu_arch="sm90",
        machine="x86_64",
    )

    result = choose_attention_backend(
        requested="auto",
        detect_runtime_fn=lambda: runtime,
        flash_available_fn=lambda: False,
        build_and_cache_wheel_fn=lambda runtime: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )

    assert result.backend == "sdpa"
    assert "compile failed" in result.reason


def test_auto_backend_uses_configured_fallback_when_build_fails(tmp_path):
    runtime = FlashRuntime(
        package="flash-attn",
        cache_root=tmp_path,
        python_tag="cp310",
        torch_tag="torch2.10.0",
        cuda_tag="cu128",
        gpu_arch="sm90",
        machine="x86_64",
    )

    result = choose_attention_backend(
        requested="auto",
        detect_runtime_fn=lambda: runtime,
        flash_available_fn=lambda: False,
        build_and_cache_wheel_fn=lambda runtime: (_ for _ in ()).throw(RuntimeError("compile failed")),
        fallback_backend="sdpa",
    )

    assert result.backend == "sdpa"


def test_write_env_file_exports_selected_backend(tmp_path):
    env_file = tmp_path / "attention.env"

    write_env_file(env_file, "flash")

    assert env_file.read_text(encoding="utf-8") == "export ATTENTION_BACKEND=flash\n"
