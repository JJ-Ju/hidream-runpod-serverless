from __future__ import annotations

import base64
import binascii
import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4


MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def prepare_reference_images(ref_images: list[Any], work_dir: Path) -> list[str]:
    if not ref_images:
        return []

    refs_dir = work_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, source in enumerate(ref_images):
        prepared.append(str(_prepare_single_reference(source, refs_dir, index)))
    return prepared


def _prepare_single_reference(source: Any, refs_dir: Path, index: int) -> Path:
    if isinstance(source, dict):
        return _prepare_reference_object(source, refs_dir, index)
    if not isinstance(source, str):
        raise ValueError("ref_images entries must be strings or image payload objects")

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return _download_reference(source, refs_dir, index)

    if parsed.scheme == "data":
        header, encoded = source.split(",", 1)
        if "image/" not in header:
            raise ValueError("data URI ref_images entries must be image payloads")
        extension = _extension_from_data_uri_header(header)
        return _write_base64(encoded, refs_dir / f"ref-{index}-{uuid4().hex[:8]}{extension}")

    local = Path(source)
    if local.exists():
        raise ValueError("Local reference paths are not accepted by the public worker")

    return _write_base64(source, refs_dir / f"ref-{index}-{uuid4().hex[:8]}.png")


def _prepare_reference_object(source: dict[str, Any], refs_dir: Path, index: int) -> Path:
    url = source.get("url") or source.get("image_url")
    if url:
        if not isinstance(url, str):
            raise ValueError("ref_images object URLs must be strings")
        return _download_reference(url, refs_dir, index)

    data_uri = source.get("data_uri")
    if data_uri:
        if not isinstance(data_uri, str):
            raise ValueError("ref_images object data_uri values must be strings")
        return _prepare_single_reference(data_uri, refs_dir, index)

    encoded = source.get("base64") or source.get("image_base64")
    if not encoded:
        raise ValueError("ref_images objects must contain url, image_url, data_uri, base64, or image_base64")
    if not isinstance(encoded, str):
        raise ValueError("ref_images object base64 values must be strings")
    content_type = source.get("mime_type") or source.get("content_type") or "image/png"
    if not isinstance(content_type, str):
        raise ValueError("ref_images object content types must be strings")
    extension = _extension_from_content_type(content_type)
    return _write_base64(encoded, refs_dir / f"ref-{index}-{uuid4().hex[:8]}{extension}")


def _download_reference(url: str, refs_dir: Path, index: int) -> Path:
    _reject_unsafe_url(url)
    import requests

    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if content_type and not content_type.lower().startswith("image/"):
        raise ValueError("ref_images URLs must return image content")
    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > MAX_REFERENCE_BYTES:
        raise ValueError(f"ref_images downloads are too large; max is {MAX_REFERENCE_BYTES} bytes")

    data = bytearray()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) > MAX_REFERENCE_BYTES:
            raise ValueError(f"ref_images downloads are too large; max is {MAX_REFERENCE_BYTES} bytes")

    extension = _extension_from_content_type(content_type)
    path = refs_dir / f"ref-{index}-{uuid4().hex[:8]}{extension}"
    path.write_bytes(bytes(data))
    return path


def _write_base64(encoded: str, path: Path) -> Path:
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ref_images entries must be image URLs, data URIs, or base64 image data") from exc
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError(f"ref_images base64 payloads are too large; max is {MAX_REFERENCE_BYTES} bytes")
    path.write_bytes(data)
    return path


def _reject_unsafe_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("ref_images URLs must include a hostname")
    if hostname.lower() == "localhost" or hostname.lower().endswith(".localhost"):
        raise ValueError("ref_images URLs cannot target private or local network hosts")

    addresses = _resolve_host_addresses(hostname)
    for address in addresses:
        if _is_private_or_local(address):
            raise ValueError("ref_images URLs cannot target private or local network hosts")


def _resolve_host_addresses(hostname: str) -> list[ipaddress._BaseAddress]:
    try:
        return [ipaddress.ip_address(hostname)]
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve ref_images host: {hostname}") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _is_private_or_local(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )


def _extension_from_content_type(content_type: str) -> str:
    if "webp" in content_type:
        return ".webp"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    return ".png"


def _extension_from_data_uri_header(header: str) -> str:
    if "webp" in header:
        return ".webp"
    if "jpeg" in header or "jpg" in header:
        return ".jpg"
    return ".png"
