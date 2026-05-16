from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


LOCKED_PYTHON_VERSION = "3.12"
LOCKED_CUDA_VERSION = "12.8"
LOCKED_TORCH_VERSION = "2.10.0"
LOCKED_TORCHVISION_VERSION = "0.25.0"

DEFAULT_VOLUME_ROOT = "/runpod-volume"
DEFAULT_RUNTIME_CACHE_DIR = "/runpod-volume/hidream-runtime"
DEFAULT_FALLBACK_CACHE_DIR = "/tmp/hidream-runtime"
READY_MARKER = ".hidream-runtime-ready"


@dataclass(frozen=True)
class RuntimeDependencyConfig:
    app_dir: Path
    volume_root: Path
    cache_root: Path
    requirement_files: tuple[Path, ...]
    python_version: str = LOCKED_PYTHON_VERSION
    cuda_version: str = LOCKED_CUDA_VERSION
    torch_version: str = LOCKED_TORCH_VERSION
    torchvision_version: str = LOCKED_TORCHVISION_VERSION
    system: str = "linux"
    machine: str = "x86_64"

    def __post_init__(self) -> None:
        object.__setattr__(self, "app_dir", Path(self.app_dir))
        object.__setattr__(self, "volume_root", Path(self.volume_root))
        object.__setattr__(self, "cache_root", Path(self.cache_root))
        object.__setattr__(
            self,
            "requirement_files",
            tuple(Path(requirement) for requirement in self.requirement_files),
        )

    @property
    def requirements_hash(self) -> str:
        digest = hashlib.sha256()
        for requirement in self.requirement_files:
            if not requirement.is_file():
                raise FileNotFoundError(f"Dependency requirement file not found: {requirement}")
            digest.update(requirement.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(requirement.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    @property
    def cache_key(self) -> str:
        return (
            f"py{self.python_version}"
            f"-cu{self.cuda_version}"
            f"-torch{self.torch_version}"
            f"-tv{self.torchvision_version}"
            f"-{self.system.lower()}-{self.machine.lower()}"
            f"-{self.requirements_hash}"
        )

    @property
    def envs_dir(self) -> Path:
        return self.cache_root / "envs"

    @property
    def venv_dir(self) -> Path:
        return self.envs_dir / self.cache_key

    @property
    def venv_bin_dir(self) -> Path:
        return self.venv_dir / "bin"

    @property
    def python_executable(self) -> Path:
        return self.venv_bin_dir / "python"

    @property
    def ready_marker(self) -> Path:
        return self.venv_dir / READY_MARKER

    @property
    def lock_path(self) -> Path:
        return self.envs_dir / f"{self.cache_key}.lock"

    @property
    def pip_cache_dir(self) -> Path:
        return self.cache_root / "pip-cache"

    @property
    def torch_cache_dir(self) -> Path:
        return self.cache_root / "torch-cache"

    @property
    def torch_extensions_dir(self) -> Path:
        return self.cache_root / "torch-extensions"

    @property
    def xdg_cache_home(self) -> Path:
        return self.cache_root / "xdg-cache"


def posix(path: Path) -> str:
    return path.as_posix()


def is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".hidream-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def choose_volume_root(
    preferred: str | Path = DEFAULT_VOLUME_ROOT,
    fallback: str | Path = DEFAULT_FALLBACK_CACHE_DIR,
    is_writable_dir_fn: Callable[[Path], bool] = is_writable_dir,
) -> Path:
    preferred_path = Path(preferred)
    if preferred_path.is_dir() and is_writable_dir_fn(preferred_path):
        return preferred_path

    fallback_path = Path(fallback)
    if is_writable_dir_fn(fallback_path):
        return fallback_path

    raise RuntimeError(f"No writable dependency cache root: {preferred_path} or {fallback_path}")


def build_config(env: dict[str, str] | None = None, app_dir: str | Path = "/app") -> RuntimeDependencyConfig:
    env = env or os.environ
    app_dir = Path(env.get("HIDREAM_APP_DIR", str(app_dir)))
    preferred_volume_root = Path(env.get("HIDREAM_VOLUME_ROOT", DEFAULT_VOLUME_ROOT))
    fallback_root = Path(env.get("HIDREAM_DEPENDENCY_FALLBACK_ROOT", DEFAULT_FALLBACK_CACHE_DIR))
    volume_root = choose_volume_root(preferred_volume_root, fallback_root)

    explicit_cache_root = env.get("HIDREAM_RUNTIME_CACHE_ROOT")
    if explicit_cache_root:
        cache_root = Path(explicit_cache_root)
    elif volume_root == preferred_volume_root:
        cache_root = volume_root / "hidream-runtime"
    else:
        cache_root = volume_root

    detected_system = env.get("HIDREAM_RUNTIME_SYSTEM", platform.system().lower() or "linux")
    detected_machine = env.get("HIDREAM_RUNTIME_MACHINE", platform.machine().lower() or "unknown")
    default_requirements = str(app_dir / "requirements.txt")
    requirement_files = tuple(
        Path(part)
        for part in env.get(
            "HIDREAM_REQUIREMENTS_FILES",
            default_requirements,
        ).split(os.pathsep)
        if part
    )

    return RuntimeDependencyConfig(
        app_dir=app_dir,
        volume_root=volume_root,
        cache_root=cache_root,
        requirement_files=requirement_files,
        python_version=env.get("HIDREAM_PYTHON_VERSION", LOCKED_PYTHON_VERSION),
        cuda_version=env.get("HIDREAM_CUDA_VERSION", LOCKED_CUDA_VERSION),
        torch_version=env.get("HIDREAM_TORCH_VERSION", LOCKED_TORCH_VERSION),
        torchvision_version=env.get("HIDREAM_TORCHVISION_VERSION", LOCKED_TORCHVISION_VERSION),
        system=detected_system,
        machine=detected_machine,
    )


def environment_ready(config: RuntimeDependencyConfig) -> bool:
    return config.ready_marker.is_file() and config.python_executable.is_file()


@contextmanager
def build_lock(lock_path: Path, timeout_seconds: int = 1800) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for dependency build lock: {lock_path}")
            time.sleep(5)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def install_requirements(config: RuntimeDependencyConfig) -> None:
    config.cache_root.mkdir(parents=True, exist_ok=True)
    config.pip_cache_dir.mkdir(parents=True, exist_ok=True)
    config.torch_cache_dir.mkdir(parents=True, exist_ok=True)
    config.torch_extensions_dir.mkdir(parents=True, exist_ok=True)
    config.xdg_cache_home.mkdir(parents=True, exist_ok=True)

    if config.venv_dir.exists() and not environment_ready(config):
        shutil.rmtree(config.venv_dir)

    subprocess.check_call(
        [sys.executable, "-m", "venv", "--system-site-packages", str(config.venv_dir)]
    )

    env = os.environ.copy()
    env.setdefault("PIP_CACHE_DIR", str(config.pip_cache_dir))
    env.setdefault("TORCH_HOME", str(config.torch_cache_dir))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(config.torch_extensions_dir))
    env.setdefault("XDG_CACHE_HOME", str(config.xdg_cache_home))
    pip_cmd = [str(config.python_executable), "-m", "pip"]

    subprocess.check_call(
        [*pip_cmd, "install", "--upgrade", "pip", "setuptools", "wheel"],
        env=env,
    )
    for requirement in config.requirement_files:
        subprocess.check_call(
            [*pip_cmd, "install", "--cache-dir", str(config.pip_cache_dir), "-r", str(requirement)],
            env=env,
        )

    config.ready_marker.write_text(
        f"cache_key={config.cache_key}\n"
        f"python={config.python_version}\n"
        f"cuda={config.cuda_version}\n"
        f"torch={config.torch_version}\n"
        f"torchvision={config.torchvision_version}\n",
        encoding="utf-8",
    )


def ensure_environment(config: RuntimeDependencyConfig) -> None:
    if environment_ready(config):
        print(f"Using cached HiDream dependency environment: {config.venv_dir}", flush=True)
        return

    with build_lock(config.lock_path):
        if environment_ready(config):
            print(f"Using cached HiDream dependency environment: {config.venv_dir}", flush=True)
            return
        print(f"Building HiDream dependency environment: {config.venv_dir}", flush=True)
        install_requirements(config)


def default_export(name: str, value: Path | str) -> str:
    if isinstance(value, Path):
        value = posix(value)
    return f'export {name}="${{{name}:-{value}}}"'


def write_env_file(path: str | Path, config: RuntimeDependencyConfig) -> None:
    app_dir = posix(config.app_dir)
    volume_root = config.volume_root
    lines = [
        f"export HIDREAM_DEPENDENCY_CACHE_KEY={config.cache_key}",
        f"export HIDREAM_RUNTIME_CACHE_ROOT={posix(config.cache_root)}",
        f"export VIRTUAL_ENV={posix(config.venv_dir)}",
        f"export PATH={posix(config.venv_bin_dir)}:${{PATH}}",
        f"export PYTHONPATH={app_dir}/src:{app_dir}",
        "export PYTHONNOUSERSITE=1",
        default_export("HF_HOME", volume_root / "huggingface"),
        default_export("HF_HUB_CACHE", volume_root / "huggingface-cache" / "hub"),
        default_export("TRANSFORMERS_CACHE", volume_root / "huggingface-cache" / "hub"),
        default_export("HIDREAM_HF_CACHE_ROOT", volume_root / "huggingface-cache" / "hub"),
        default_export("PIP_CACHE_DIR", config.pip_cache_dir),
        default_export("TORCH_HOME", config.torch_cache_dir),
        default_export("TORCH_EXTENSIONS_DIR", config.torch_extensions_dir),
        default_export("XDG_CACHE_HOME", config.xdg_cache_home),
        default_export("FLASH_ATTN_CACHE_DIR", volume_root / "flash-attn-cache"),
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap(write_env: str | Path | None = None) -> RuntimeDependencyConfig:
    config = build_config()
    ensure_environment(config)
    if write_env:
        write_env_file(write_env, config)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-env")
    args = parser.parse_args()
    bootstrap(write_env=args.write_env)
