from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RequestValidationError(ValueError):
    """Raised when a RunPod job input cannot be accepted."""


class GenerationMode(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    EDIT = "edit"
    REFERENCE = "reference"
    LAYOUT_REFERENCE = "layout_reference"


SUPPORTED_OUTPUT_FORMATS = {"png", "webp", "jpeg"}
SUPPORTED_OUTPUT_DELIVERIES = {"url", "base64", "both"}
SUPPORTED_MODEL_TYPES = {"full", "dev"}
SUPPORTED_EDITING_SCHEDULERS = {"flow_match", "flash"}
MAX_DIMENSION = 2048
MAX_REFERENCE_IMAGES = 10
MAX_PROMPT_CHARS = 12000
SINGLE_REFERENCE_ALIASES = ("ref_image", "input_image", "init_image", "image")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    width: int = 2048
    height: int = 2048
    seed: int = 32
    model_type: str = "full"
    ref_images: list[Any] = field(default_factory=list)
    layout_bboxes: Any | None = None
    keep_original_aspect: bool = False
    editing_scheduler: str = "flow_match"
    guidance_scale: float = 5.0
    shift: float | None = None
    noise_scale_start: float = 7.5
    noise_scale_end: float = 7.5
    noise_clip_std: float = 2.5
    output_format: str = "png"
    output_delivery: str = "url"

    @classmethod
    def from_input(cls, data: dict[str, Any]) -> "GenerationRequest":
        if not isinstance(data, dict):
            raise RequestValidationError("input must be an object")

        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestValidationError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise RequestValidationError(f"prompt must be {MAX_PROMPT_CHARS} characters or fewer")

        ref_images = _reference_images(data)
        if len(ref_images) > MAX_REFERENCE_IMAGES:
            raise RequestValidationError(f"ref_images cannot contain more than {MAX_REFERENCE_IMAGES} entries")
        output_format = _string_choice(
            data.get("output_format", "png"),
            SUPPORTED_OUTPUT_FORMATS,
            "output_format",
        )
        output_delivery = _string_choice(
            data.get("output_delivery", "url"),
            SUPPORTED_OUTPUT_DELIVERIES,
            "output_delivery",
        )
        model_type = _string_choice(
            data.get("model_type", "full"),
            SUPPORTED_MODEL_TYPES,
            "model_type",
        )
        editing_scheduler = _string_choice(
            data.get("editing_scheduler", "flow_match"),
            SUPPORTED_EDITING_SCHEDULERS,
            "editing_scheduler",
        )
        layout_bboxes = data.get("layout_bboxes")
        if layout_bboxes is not None and not ref_images:
            raise RequestValidationError("layout_bboxes requires at least one ref_images entry")
        layout_bboxes = _validate_layout_bboxes(layout_bboxes, len(ref_images))

        return cls(
            prompt=prompt.strip(),
            width=_dimension(data.get("width", 2048), "width"),
            height=_dimension(data.get("height", 2048), "height"),
            seed=_int_value(data.get("seed", 32), "seed"),
            model_type=model_type,
            ref_images=ref_images,
            layout_bboxes=layout_bboxes,
            keep_original_aspect=bool(data.get("keep_original_aspect", False)),
            editing_scheduler=editing_scheduler,
            guidance_scale=_bounded_float(data.get("guidance_scale", 5.0), "guidance_scale", 0.0, 20.0),
            shift=_optional_bounded_float(data.get("shift"), "shift", 0.0, 10.0),
            noise_scale_start=_bounded_float(data.get("noise_scale_start", 7.5), "noise_scale_start", 0.0, 20.0),
            noise_scale_end=_bounded_float(data.get("noise_scale_end", 7.5), "noise_scale_end", 0.0, 20.0),
            noise_clip_std=_bounded_float(data.get("noise_clip_std", 2.5), "noise_clip_std", 0.0, 20.0),
            output_format=output_format,
            output_delivery=output_delivery,
        )

    @property
    def mode(self) -> GenerationMode:
        if self.layout_bboxes is not None:
            return GenerationMode.LAYOUT_REFERENCE
        if not self.ref_images:
            return GenerationMode.TEXT_TO_IMAGE
        if len(self.ref_images) == 1:
            return GenerationMode.EDIT
        return GenerationMode.REFERENCE

    def generation_kwargs(self) -> dict[str, Any]:
        if self.model_type == "full":
            return {
                "num_inference_steps": 50,
                "guidance_scale": self.guidance_scale,
                "shift": self.shift if self.shift is not None else 3.0,
                "timesteps_list": None,
                "scheduler_name": "default",
                "noise_scale_start": self.noise_scale_start,
                "noise_scale_end": self.noise_scale_end,
                "noise_clip_std": self.noise_clip_std,
            }

        is_editing = self.mode == GenerationMode.EDIT
        scheduler_name = self.editing_scheduler if is_editing else "flash"
        return {
            "num_inference_steps": 28,
            "guidance_scale": 0.0,
            "shift": self.shift if self.shift is not None else 1.0,
            "timesteps_list": "DEFAULT_TIMESTEPS",
            "scheduler_name": scheduler_name,
            "noise_scale_start": self.noise_scale_start,
            "noise_scale_end": self.noise_scale_end,
            "noise_clip_std": self.noise_clip_std,
        }


def _string_choice(value: Any, choices: set[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise RequestValidationError(f"{field_name} must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise RequestValidationError(f"{field_name} must be one of {sorted(choices)}")
    return normalized


def _reference_image_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestValidationError(f"{field_name} must be an array of image references")
    result = []
    for item in value:
        result.append(_reference_image_source(item, field_name))
    return result


def _reference_images(data: dict[str, Any]) -> list[Any]:
    ref_images = data.get("ref_images")
    input_images = data.get("input_images")
    single_refs = [
        (field_name, data[field_name])
        for field_name in SINGLE_REFERENCE_ALIASES
        if field_name in data and data[field_name] is not None
    ]
    multi_refs = [field_name for field_name, value in (("ref_images", ref_images), ("input_images", input_images)) if value is not None]
    if len(multi_refs) > 1:
        raise RequestValidationError("use ref_images or input_images, not both")
    if multi_refs and single_refs:
        raise RequestValidationError("use input_images/ref_images or one single-image alias, not both")
    if len(single_refs) > 1:
        raise RequestValidationError(
            f"use only one single-image alias: {', '.join(SINGLE_REFERENCE_ALIASES)}"
        )
    if ref_images is not None:
        return _reference_image_list(ref_images, "ref_images")
    if input_images is not None:
        return _reference_image_list(input_images, "input_images")
    if not single_refs:
        return []

    field_name, value = single_refs[0]
    return [_reference_image_source(value, field_name)]


def _reference_image_source(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        if not value.strip():
            raise RequestValidationError(f"{field_name} must contain non-empty image references")
        return value.strip()
    if not isinstance(value, dict):
        raise RequestValidationError(
            f"{field_name} entries must be strings or objects with url, data_uri, or base64"
        )

    source_keys = [
        key
        for key in ("url", "image_url", "data_uri", "base64", "image_base64")
        if value.get(key) is not None
    ]
    if len(source_keys) != 1:
        raise RequestValidationError(
            f"{field_name} image objects must contain exactly one of url, image_url, data_uri, base64, image_base64"
        )
    source_value = value[source_keys[0]]
    if not isinstance(source_value, str) or not source_value.strip():
        raise RequestValidationError(f"{field_name}.{source_keys[0]} must be a non-empty string")

    normalized = {source_keys[0]: source_value.strip()}
    for metadata_key in ("mime_type", "content_type"):
        metadata_value = value.get(metadata_key)
        if metadata_value is not None:
            if not isinstance(metadata_value, str) or not metadata_value.strip():
                raise RequestValidationError(f"{field_name}.{metadata_key} must be a non-empty string")
            normalized[metadata_key] = metadata_value.strip()
    return normalized


def _positive_int(value: Any, field_name: str) -> int:
    number = _int_value(value, field_name)
    if number <= 0:
        raise RequestValidationError(f"{field_name} must be greater than 0")
    return number


def _dimension(value: Any, field_name: str) -> int:
    number = _positive_int(value, field_name)
    if number > MAX_DIMENSION:
        raise RequestValidationError(f"{field_name} must be no greater than {MAX_DIMENSION}")
    return number


def _int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise RequestValidationError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{field_name} must be an integer") from exc


def _float_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise RequestValidationError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{field_name} must be a number") from exc


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _float_value(value, field_name)


def _bounded_float(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    number = _float_value(value, field_name)
    if number < minimum or number > maximum:
        raise RequestValidationError(f"{field_name} must be between {minimum:g} and {maximum:g}")
    return number


def _optional_bounded_float(
    value: Any,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    return _bounded_float(value, field_name, minimum, maximum)


def _validate_layout_bboxes(value: Any, ref_count: int) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        boxes = value.get("bboxes") or value.get("layout_bboxes")
    else:
        boxes = value
    if not isinstance(boxes, list) or not boxes:
        raise RequestValidationError("layout_bboxes must be a non-empty array")
    if len(boxes) != ref_count:
        raise RequestValidationError("layout_bboxes must contain one box per ref_images entry")

    normalized = []
    for box in boxes:
        if isinstance(box, dict):
            box = box.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise RequestValidationError("layout_bboxes entries must be [x1, x2, y1, y2]")
        try:
            x1, x2, y1, y2 = [float(item) for item in box]
        except (TypeError, ValueError) as exc:
            raise RequestValidationError("layout_bboxes entries must contain numbers") from exc
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise RequestValidationError("layout_bboxes coordinates must be normalized x1 < x2 and y1 < y2")
        normalized.append([x1, x2, y1, y2])

    return normalized
