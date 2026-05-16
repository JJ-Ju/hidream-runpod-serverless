from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_bootstraps_flash_at_runtime_not_build_time():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")

    assert "ENTRYPOINT" in dockerfile
    assert "BOOTSTRAP_FLASH_ATTN" in dockerfile
    assert "ATTENTION_BACKEND=auto" in dockerfile
    assert "BOOTSTRAP_FLASH_ATTN=1" in dockerfile
    assert "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime" in dockerfile
    assert "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04" not in dockerfile
    assert "PYTHON_VERSION=3.12" in dockerfile
    assert "TORCH_VERSION=2.10.0" in dockerfile
    assert "TORCHVISION_VERSION=0.25.0" in dockerfile
    assert "python -m hidream_o1.dependency_bootstrap" in entrypoint
    assert "pip install --no-cache-dir --no-build-isolation flash-attn" not in dockerfile
    assert "python -m hidream_o1.flash_bootstrap" in entrypoint
    assert "--write-env" in entrypoint
    assert "source" in entrypoint
    assert "FLASH_ATTN_CACHE_DIR:-/runpod-volume/flash-attn-cache" in entrypoint


def test_github_workflow_publishes_one_dynamic_image():
    workflow = (ROOT / ".github" / "workflows" / "publish-image.yml").read_text(encoding="utf-8")

    assert "ATTENTION_BACKEND=auto" in workflow
    assert "BOOTSTRAP_FLASH_ATTN=1" in workflow
    assert "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime" in workflow
    assert "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04" not in workflow
    assert "variant: sdpa" not in workflow
    assert "variant: flash" not in workflow
    assert "suffix=-sdpa" not in workflow
    assert "suffix=-flash" not in workflow
