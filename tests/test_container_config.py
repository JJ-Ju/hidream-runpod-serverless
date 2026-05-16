from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_bootstraps_flash_at_runtime_not_build_time():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")

    assert "ENTRYPOINT" in dockerfile
    assert "BOOTSTRAP_FLASH_ATTN" in dockerfile
    assert "pip install --no-cache-dir --no-build-isolation flash-attn" not in dockerfile
    assert "pip install --no-cache-dir --no-build-isolation" in entrypoint
    assert "FLASH_ATTN_PACKAGE:-flash-attn" in entrypoint


def test_github_workflow_enables_flash_bootstrap_for_flash_variant():
    workflow = (ROOT / ".github" / "workflows" / "publish-image.yml").read_text(encoding="utf-8")

    assert "bootstrap_flash_attn: 1" in workflow
    assert "BOOTSTRAP_FLASH_ATTN=${{ matrix.bootstrap_flash_attn }}" in workflow
