from pathlib import Path

from hidream_o1.flash_bootstrap import (
    FlashRuntime,
    cached_wheel,
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
