from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from hidream_o1.image_inputs import prepare_reference_images
from hidream_o1.schemas import GenerationRequest, RequestValidationError
from hidream_o1.storage import S3Storage, build_object_key


CONTENT_TYPES = {
    "png": "image/png",
    "webp": "image/webp",
    "jpeg": "image/jpeg",
}

PIL_FORMATS = {
    "png": "PNG",
    "webp": "WEBP",
    "jpeg": "JPEG",
}


class GenerationService:
    def __init__(
        self,
        runner,
        storage: S3Storage | None,
        work_dir: str | Path,
        output_prefix: str,
        model_id: str,
        attention_backend: str,
        allow_base64_output: bool = False,
    ):
        self.runner = runner
        self.storage = storage
        self.work_dir = Path(work_dir)
        self.output_prefix = output_prefix
        self.model_id = model_id
        self.attention_backend = attention_backend
        self.allow_base64_output = allow_base64_output

    def handle_job(self, job: dict[str, Any]) -> dict[str, Any]:
        try:
            job_id = str(job.get("id") or f"local-{uuid4().hex[:12]}")
            request = GenerationRequest.from_input(job.get("input", {}))
            if self.storage is None and not self.allow_base64_output:
                return {
                    "error": (
                        "S3 output storage is not configured. Set S3_ENDPOINT_URL, "
                        "S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_REGION, "
                        "or set ALLOW_BASE64_OUTPUT=1 for local-only testing."
                    )
                }

            job_dir = self.work_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            ref_image_paths = prepare_reference_images(request.ref_images, job_dir)
            image = self.runner.generate(request, ref_image_paths)
            image_bytes = _image_to_bytes(image, request.output_format)

            response = {
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "mode": request.mode.value,
                "model_id": self.model_id,
                "attention_backend": self.attention_backend,
            }

            if self.storage is not None:
                key = build_object_key(job_id, request.output_format, self.output_prefix)
                result = self.storage.upload_bytes(
                    key,
                    image_bytes,
                    CONTENT_TYPES[request.output_format],
                )
                response.update(
                    {
                        "image_url": result.image_url,
                        "bucket": result.bucket,
                        "key": result.key,
                    }
                )
                return response

            response.update(
                {
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    "content_type": CONTENT_TYPES[request.output_format],
                }
            )
            return response
        except RequestValidationError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}


def _image_to_bytes(image, output_format: str) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=PIL_FORMATS[output_format])
    return buffer.getvalue()
