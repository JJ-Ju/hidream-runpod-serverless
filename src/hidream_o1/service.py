from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from hidream_o1.image_inputs import prepare_reference_images
from hidream_o1.progress import ProgressCallback, ProgressReporter
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

SUPPORTED_OUTPUT_DELIVERIES = {"url", "base64", "both"}


class GenerationService:
    def __init__(
        self,
        runner,
        storage: S3Storage | None,
        work_dir: str | Path,
        output_prefix: str,
        model_id: str,
        attention_backend: str,
        default_output_delivery: str = "url",
    ):
        self.runner = runner
        self.storage = storage
        self.work_dir = Path(work_dir)
        self.output_prefix = output_prefix
        self.model_id = model_id
        self.attention_backend = attention_backend
        self.default_output_delivery = _validate_output_delivery(default_output_delivery)

    def handle_job(self, job: dict[str, Any], progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
        progress = ProgressReporter(progress_callback)
        try:
            progress.emit("validating", 2, "Validating request")
            job_id = str(job.get("id") or f"local-{uuid4().hex[:12]}")
            job_input = job.get("input", {})
            request = GenerationRequest.from_input(job_input)
            output_delivery = _requested_output_delivery(job_input, self.default_output_delivery)
            wants_url = output_delivery in {"url", "both"}
            wants_base64 = output_delivery in {"base64", "both"}
            if self.storage is None and wants_url:
                return {
                    "error": (
                        "S3 output storage is not configured. Set S3_ENDPOINT_URL, "
                        "S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_REGION, "
                        "or request output_delivery=base64 for direct inline image output."
                    )
                }
            job_dir = self.work_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            progress.emit("preparing_inputs", 8, "Preparing input images")
            ref_image_paths = prepare_reference_images(request.ref_images, job_dir)
            progress.emit("loading_model", 15, "Loading model and processor")
            image = self.runner.generate(request, ref_image_paths, progress_callback=progress.generation_step)
            progress.emit("encoding_output", 92, "Encoding generated image")
            image_bytes = _image_to_bytes(image, request.output_format)

            response = {
                "seed": request.seed,
                "width": request.width,
                "height": request.height,
                "mode": request.mode.value,
                "model_id": self.model_id,
                "attention_backend": self.attention_backend,
                "output_delivery": output_delivery,
                "content_type": CONTENT_TYPES[request.output_format],
                "image_size_bytes": len(image_bytes),
            }

            if wants_url:
                progress.emit("uploading_output", 96, "Uploading generated image")
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

            if wants_base64:
                progress.emit("preparing_response", 98, "Preparing direct image response")
                response.update(
                    {
                        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    }
                )
            progress.emit("complete", 100, "Generation complete")
            return response
        except RequestValidationError as exc:
            progress.emit("failed", 100, str(exc))
            return {"error": str(exc)}
        except (RuntimeError, ValueError) as exc:
            progress.emit("failed", 100, str(exc))
            return {"error": str(exc)}


def _image_to_bytes(image, output_format: str) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=PIL_FORMATS[output_format])
    return buffer.getvalue()


def _validate_output_delivery(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_OUTPUT_DELIVERIES:
        raise ValueError(f"DEFAULT_OUTPUT_DELIVERY must be one of {sorted(SUPPORTED_OUTPUT_DELIVERIES)}")
    return normalized


def _requested_output_delivery(job_input: Any, default_output_delivery: str) -> str:
    if not isinstance(job_input, dict) or "output_delivery" not in job_input:
        return default_output_delivery
    return _validate_output_delivery(str(job_input["output_delivery"]))
