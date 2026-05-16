import pytest

from hidream_o1.schemas import GenerationMode, GenerationRequest, RequestValidationError


def test_request_defaults_to_full_text_to_image():
    request = GenerationRequest.from_input({"prompt": "A red biplane poster"})

    assert request.prompt == "A red biplane poster"
    assert request.width == 2048
    assert request.height == 2048
    assert request.seed == 32
    assert request.model_type == "full"
    assert request.output_format == "png"
    assert request.output_delivery == "url"
    assert request.mode == GenerationMode.TEXT_TO_IMAGE


def test_request_detects_edit_reference_and_layout_modes():
    edit = GenerationRequest.from_input(
        {"prompt": "remove sunglasses", "ref_images": ["https://example.com/a.png"]}
    )
    reference = GenerationRequest.from_input(
        {
            "prompt": "new scene",
            "ref_images": ["https://example.com/a.png", "https://example.com/b.png"],
        }
    )
    layout = GenerationRequest.from_input(
        {
            "prompt": "pose together",
            "ref_images": ["https://example.com/a.png", "https://example.com/b.png"],
            "layout_bboxes": [[0.1, 0.4, 0.2, 0.8], [0.5, 0.9, 0.2, 0.8]],
        }
    )

    assert edit.mode == GenerationMode.EDIT
    assert reference.mode == GenerationMode.REFERENCE
    assert layout.mode == GenerationMode.LAYOUT_REFERENCE


def test_request_accepts_single_image_edit_aliases():
    for field_name in ("ref_image", "input_image", "init_image", "image"):
        request = GenerationRequest.from_input(
            {
                "prompt": "clean up this source render",
                field_name: "https://example.com/source.png",
            }
        )

        assert request.ref_images == ["https://example.com/source.png"]
        assert request.mode == GenerationMode.EDIT


def test_request_rejects_invalid_prompt_and_format():
    with pytest.raises(RequestValidationError, match="prompt"):
        GenerationRequest.from_input({"prompt": "   "})

    with pytest.raises(RequestValidationError, match="output_format"):
        GenerationRequest.from_input({"prompt": "x", "output_format": "tiff"})

    with pytest.raises(RequestValidationError, match="output_delivery"):
        GenerationRequest.from_input({"prompt": "x", "output_delivery": "disk"})


def test_request_accepts_direct_output_delivery_modes():
    base64_request = GenerationRequest.from_input(
        {"prompt": "x", "output_delivery": "base64"}
    )
    both_request = GenerationRequest.from_input(
        {"prompt": "x", "output_delivery": "both"}
    )

    assert base64_request.output_delivery == "base64"
    assert both_request.output_delivery == "both"


def test_request_rejects_bad_reference_images():
    with pytest.raises(RequestValidationError, match="ref_images"):
        GenerationRequest.from_input({"prompt": "x", "ref_images": "https://example.com/a.png"})

    with pytest.raises(RequestValidationError, match="ref_images"):
        GenerationRequest.from_input({"prompt": "x", "ref_images": [""]})

    with pytest.raises(RequestValidationError, match="single-image alias"):
        GenerationRequest.from_input(
            {
                "prompt": "x",
                "ref_images": ["https://example.com/a.png"],
                "input_image": "https://example.com/b.png",
            }
        )

    with pytest.raises(RequestValidationError, match="input_image"):
        GenerationRequest.from_input({"prompt": "x", "input_image": ""})


def test_request_rejects_resource_exhaustion_values():
    with pytest.raises(RequestValidationError, match="width"):
        GenerationRequest.from_input({"prompt": "x", "width": 4096})

    with pytest.raises(RequestValidationError, match="height"):
        GenerationRequest.from_input({"prompt": "x", "height": 4096})

    with pytest.raises(RequestValidationError, match="ref_images"):
        GenerationRequest.from_input(
            {
                "prompt": "x",
                "ref_images": [f"https://example.com/{index}.png" for index in range(11)],
            }
        )

    with pytest.raises(RequestValidationError, match="guidance_scale"):
        GenerationRequest.from_input({"prompt": "x", "guidance_scale": 100})


def test_request_validates_layout_bboxes_before_gpu_work():
    base = {
        "prompt": "x",
        "ref_images": ["https://example.com/a.png", "https://example.com/b.png"],
    }

    with pytest.raises(RequestValidationError, match="layout_bboxes"):
        GenerationRequest.from_input({**base, "layout_bboxes": [[0.1, 0.2, 0.3]]})

    with pytest.raises(RequestValidationError, match="layout_bboxes"):
        GenerationRequest.from_input({**base, "layout_bboxes": [[0.4, 0.1, 0.2, 0.8]]})

    with pytest.raises(RequestValidationError, match="layout_bboxes"):
        GenerationRequest.from_input({**base, "layout_bboxes": [[0.1, 0.4, 0.2, 0.8]]})


def test_generation_kwargs_follow_full_and_dev_recipes():
    full = GenerationRequest.from_input({"prompt": "x", "guidance_scale": 4.5, "shift": 2.0})
    dev_edit = GenerationRequest.from_input(
        {
            "prompt": "x",
            "model_type": "dev",
            "ref_images": ["https://example.com/a.png"],
            "editing_scheduler": "flow_match",
        }
    )

    assert full.generation_kwargs()["num_inference_steps"] == 50
    assert full.generation_kwargs()["guidance_scale"] == 4.5
    assert full.generation_kwargs()["shift"] == 2.0
    assert full.generation_kwargs()["scheduler_name"] == "default"
    assert dev_edit.generation_kwargs()["num_inference_steps"] == 28
    assert dev_edit.generation_kwargs()["guidance_scale"] == 0.0
    assert dev_edit.generation_kwargs()["shift"] == 1.0
    assert dev_edit.generation_kwargs()["scheduler_name"] == "flow_match"
