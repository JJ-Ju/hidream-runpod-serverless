import base64
from pathlib import Path

import pytest
import requests

from hidream_o1.image_inputs import MAX_REFERENCE_BYTES, prepare_reference_images


def test_reference_images_reject_existing_local_paths(tmp_path):
    local_file = tmp_path / "secret.png"
    local_file.write_bytes(b"not really an image")

    with pytest.raises(ValueError, match="Local reference paths"):
        prepare_reference_images([str(local_file)], tmp_path)


def test_reference_images_reject_private_network_urls(tmp_path):
    with pytest.raises(ValueError, match="private or local network"):
        prepare_reference_images(["http://127.0.0.1/image.png"], tmp_path)


def test_reference_images_decode_small_base64_payloads(tmp_path):
    encoded = base64.b64encode(b"fake-image").decode("ascii")

    [path] = prepare_reference_images([encoded], tmp_path)

    assert path.endswith(".png")
    assert (tmp_path / "references").exists()


def test_reference_images_decode_direct_payload_objects(tmp_path):
    encoded = base64.b64encode(b"fake-image").decode("ascii")

    [path] = prepare_reference_images(
        [{"base64": encoded, "mime_type": "image/webp"}],
        tmp_path,
    )

    assert path.endswith(".webp")
    assert Path(path).read_bytes() == b"fake-image"


def test_reference_images_reject_oversized_downloads(monkeypatch, tmp_path):
    class FakeResponse:
        headers = {
            "content-length": str(MAX_REFERENCE_BYTES + 1),
            "content-type": "image/png",
        }

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(ValueError, match="too large"):
        prepare_reference_images(["https://8.8.8.8/image.png"], tmp_path)
