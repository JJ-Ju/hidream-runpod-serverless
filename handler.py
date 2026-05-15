from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hidream_o1.runner import HiDreamRunner
from hidream_o1.service import GenerationService
from hidream_o1.storage import S3Storage, StorageConfig


_SERVICE: GenerationService | None = None


def get_service() -> GenerationService:
    global _SERVICE
    if _SERVICE is None:
        runner = HiDreamRunner.from_env()
        storage_config = StorageConfig.from_env(os.environ)
        storage = S3Storage(storage_config) if storage_config else None
        _SERVICE = GenerationService(
            runner=runner,
            storage=storage,
            work_dir=Path(os.environ.get("WORK_DIR", "/tmp/hidream-o1")),
            output_prefix=os.environ.get("OUTPUT_PREFIX", "hidream-o1"),
            model_id=runner.model_id,
            attention_backend=runner.attention_backend,
            allow_base64_output=os.environ.get("ALLOW_BASE64_OUTPUT", "").lower()
            in {"1", "true", "yes"},
        )
    return _SERVICE


def handler(job):
    return get_service().handle_job(job)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
