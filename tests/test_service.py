from hidream_o1.service import GenerationService
from hidream_o1.storage import StorageResult


class FakeImage:
    def save(self, buffer, format):
        buffer.write(f"fake-{format}".encode("ascii"))


class FakeRunner:
    def __init__(self):
        self.requests = []

    def generate(self, request, ref_image_paths):
        self.requests.append((request, ref_image_paths))
        return FakeImage()


class FakeStorage:
    def __init__(self):
        self.uploads = []

    def upload_bytes(self, key, data, content_type):
        self.uploads.append((key, data, content_type))
        return StorageResult(
            image_url=f"https://cdn.example.com/{key}",
            bucket="images",
            key=key,
        )


def test_service_generates_uploads_and_returns_metadata(tmp_path):
    runner = FakeRunner()
    storage = FakeStorage()
    service = GenerationService(
        runner=runner,
        storage=storage,
        work_dir=tmp_path,
        output_prefix="hidream-o1",
        model_id="HiDream-ai/HiDream-O1-Image",
        attention_backend="sdpa",
    )

    result = service.handle_job({"id": "job-1", "input": {"prompt": "A red biplane"}})

    assert result["image_url"].startswith("https://cdn.example.com/hidream-o1/job-1-")
    assert result["bucket"] == "images"
    assert result["seed"] == 32
    assert result["width"] == 2048
    assert result["height"] == 2048
    assert result["mode"] == "text_to_image"
    assert result["model_id"] == "HiDream-ai/HiDream-O1-Image"
    assert result["attention_backend"] == "sdpa"
    assert storage.uploads[0][2] == "image/png"
    assert runner.requests[0][0].prompt == "A red biplane"


def test_service_returns_clear_error_when_storage_is_missing(tmp_path):
    service = GenerationService(
        runner=FakeRunner(),
        storage=None,
        work_dir=tmp_path,
        output_prefix="hidream-o1",
        model_id="HiDream-ai/HiDream-O1-Image",
        attention_backend="sdpa",
        allow_base64_output=False,
    )

    result = service.handle_job({"id": "job-1", "input": {"prompt": "A red biplane"}})

    assert "error" in result
    assert "S3" in result["error"]


def test_service_supports_explicit_base64_fallback(tmp_path):
    service = GenerationService(
        runner=FakeRunner(),
        storage=None,
        work_dir=tmp_path,
        output_prefix="hidream-o1",
        model_id="HiDream-ai/HiDream-O1-Image",
        attention_backend="sdpa",
        allow_base64_output=True,
    )

    result = service.handle_job({"id": "job-1", "input": {"prompt": "A red biplane"}})

    assert result["image_base64"]
    assert result["mode"] == "text_to_image"
