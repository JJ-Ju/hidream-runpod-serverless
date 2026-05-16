from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_FLASH_ATTN_CACHE_DIR = "/runpod-volume/flash-attn-cache"


@dataclass(frozen=True)
class FlashRuntime:
    package: str
    cache_root: Path
    python_tag: str
    torch_tag: str
    cuda_tag: str
    gpu_arch: str
    machine: str

    @property
    def package_key(self) -> str:
        return safe_package_name(self.package)

    @property
    def cache_key(self) -> str:
        return (
            f"{self.package_key}/"
            f"{self.python_tag}-{self.torch_tag}-{self.cuda_tag}-{self.gpu_arch}-{self.machine}"
        )

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / self.cache_key

    @property
    def torch_cuda_arch_list(self) -> str:
        if self.gpu_arch.startswith("sm") and len(self.gpu_arch) >= 4:
            return f"{self.gpu_arch[2]}.{self.gpu_arch[3:]}"
        return ""


@dataclass(frozen=True)
class AttentionChoice:
    backend: str
    reason: str
    runtime: FlashRuntime | None = None


def safe_package_name(package: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", package).strip("-")


def detect_runtime(
    package: str | None = None,
    cache_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> FlashRuntime:
    env = env or os.environ
    package = package or env.get("FLASH_ATTN_PACKAGE", "flash-attn")
    cache_root = Path(cache_root or env.get("FLASH_ATTN_CACHE_DIR", DEFAULT_FLASH_ATTN_CACHE_DIR))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Cannot bootstrap flash-attn because CUDA is not available")

    major, minor = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda or "unknown"
    return FlashRuntime(
        package=package,
        cache_root=cache_root,
        python_tag=f"cp{sys.version_info.major}{sys.version_info.minor}",
        torch_tag=f"torch{torch.__version__.split('+', 1)[0]}",
        cuda_tag=f"cu{cuda_version.replace('.', '')}",
        gpu_arch=f"sm{major}{minor}",
        machine=platform.machine().lower() or "unknown",
    )


def flash_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        try:
            import flash_attn_interface  # noqa: F401

            return True
        except Exception:
            return False


def cached_wheel(runtime: FlashRuntime) -> Path | None:
    if not runtime.cache_dir.is_dir():
        return None
    wheels = sorted(runtime.cache_dir.glob("*.whl"))
    return wheels[0] if wheels else None


def install_wheel(wheel: Path) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", str(wheel)]
    )


def build_and_cache_wheel(runtime: FlashRuntime) -> Path:
    runtime.cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flash-attn-wheel-") as tmp:
        wheel_dir = Path(tmp)
        env = os.environ.copy()
        env.setdefault("MAX_JOBS", "4")
        if runtime.torch_cuda_arch_list:
            env.setdefault("TORCH_CUDA_ARCH_LIST", runtime.torch_cuda_arch_list)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-cache-dir",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                runtime.package,
            ],
            env=env,
        )
        wheels = sorted(wheel_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"flash-attn wheel build did not produce a wheel for {runtime.package}")
        destination = runtime.cache_dir / wheels[0].name
        shutil.copy2(wheels[0], destination)
        return destination


def choose_attention_backend(
    requested: str = "auto",
    detect_runtime_fn: Callable[[], FlashRuntime] = detect_runtime,
    flash_available_fn: Callable[[], bool] = flash_available,
    cached_wheel_fn: Callable[[FlashRuntime], Path | None] = cached_wheel,
    install_wheel_fn: Callable[[Path], None] = install_wheel,
    build_and_cache_wheel_fn: Callable[[FlashRuntime], Path] = build_and_cache_wheel,
    fallback_backend: str = "sdpa",
) -> AttentionChoice:
    requested = requested.lower()
    if requested == "sdpa":
        return AttentionChoice("sdpa", "ATTENTION_BACKEND explicitly set to sdpa")
    if requested not in {"auto", "flash"}:
        raise ValueError("ATTENTION_BACKEND must be one of: auto, flash, sdpa")
    if flash_available_fn():
        return AttentionChoice("flash", "flash-attn is already importable")

    try:
        runtime = detect_runtime_fn()
    except Exception as exc:
        if requested == "flash":
            raise
        return AttentionChoice(fallback_backend, f"flash-attn runtime detection failed: {exc}")

    wheel = cached_wheel_fn(runtime)
    if wheel is None:
        try:
            wheel = build_and_cache_wheel_fn(runtime)
        except Exception as exc:
            if requested == "flash":
                raise
            return AttentionChoice(
                fallback_backend,
                f"flash-attn build/cache failed for {runtime.cache_key}: {exc}",
                runtime,
            )
        reason = f"built and cached flash-attn wheel for {runtime.cache_key}"
    else:
        reason = f"installed cached flash-attn wheel for {runtime.cache_key}"

    try:
        install_wheel_fn(wheel)
    except Exception as exc:
        if requested == "flash":
            raise
        return AttentionChoice(
            fallback_backend,
            f"flash-attn wheel install failed for {runtime.cache_key}: {exc}",
            runtime,
        )
    return AttentionChoice("flash", reason, runtime)


def write_env_file(path: str | Path, backend: str) -> None:
    Path(path).write_text(f"export ATTENTION_BACKEND={backend}\n", encoding="utf-8")


def bootstrap(requested: str | None = None, write_env: str | Path | None = None) -> AttentionChoice:
    requested = requested or os.environ.get("ATTENTION_BACKEND", "auto")
    choice = choose_attention_backend(
        requested=requested,
        fallback_backend=os.environ.get("FLASH_ATTN_FALLBACK_BACKEND", "sdpa"),
    )
    print(f"Selected ATTENTION_BACKEND={choice.backend}: {choice.reason}", flush=True)
    if write_env:
        write_env_file(write_env, choice.backend)
    return choice


def bootstrap_flash_only() -> None:
    if flash_available():
        print("flash-attn is already available", flush=True)
        return

    runtime = detect_runtime()
    print(f"flash-attn runtime key: {runtime.cache_key}", flush=True)
    wheel = cached_wheel(runtime)
    if wheel is None:
        print(
            "No cached flash-attn wheel found; building one on this RunPod worker. "
            "The wheel will be reused from the network volume on later starts.",
            flush=True,
        )
        wheel = build_and_cache_wheel(runtime)
    else:
        print(f"Using cached flash-attn wheel: {wheel}", flush=True)
    install_wheel(wheel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--requested", default=os.environ.get("ATTENTION_BACKEND", "auto"))
    parser.add_argument("--write-env")
    args = parser.parse_args()
    bootstrap(requested=args.requested, write_env=args.write_env)
