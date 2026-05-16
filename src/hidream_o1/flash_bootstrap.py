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


def bootstrap() -> None:
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
    bootstrap()
