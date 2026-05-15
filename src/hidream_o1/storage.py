from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping
from uuid import uuid4


@dataclass(frozen=True)
class StorageResult:
    image_url: str
    bucket: str
    key: str


@dataclass(frozen=True)
class StorageConfig:
    endpoint_url: str
    region: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    public_base_url: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "StorageConfig | None":
        required = {
            "S3_ENDPOINT_URL": env.get("S3_ENDPOINT_URL"),
            "S3_REGION": env.get("S3_REGION", "auto"),
            "S3_BUCKET": env.get("S3_BUCKET"),
            "S3_ACCESS_KEY_ID": env.get("S3_ACCESS_KEY_ID"),
            "S3_SECRET_ACCESS_KEY": env.get("S3_SECRET_ACCESS_KEY"),
        }
        if any(not value for value in required.values()):
            return None
        return cls(
            endpoint_url=str(required["S3_ENDPOINT_URL"]),
            region=str(required["S3_REGION"]),
            bucket=str(required["S3_BUCKET"]),
            access_key_id=str(required["S3_ACCESS_KEY_ID"]),
            secret_access_key=str(required["S3_SECRET_ACCESS_KEY"]),
            public_base_url=env.get("S3_PUBLIC_BASE_URL"),
        )

    def public_url(self, key: str) -> str | None:
        if not self.public_base_url:
            return None
        return f"{self.public_base_url.rstrip('/')}/{key.lstrip('/')}"


class S3Storage:
    def __init__(self, config: StorageConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
            )
        return self._client

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> StorageResult:
        self.client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        image_url = self.config.public_url(key)
        if image_url is None:
            image_url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket, "Key": key},
                ExpiresIn=7 * 24 * 60 * 60,
            )
        return StorageResult(image_url=image_url, bucket=self.config.bucket, key=key)


def build_object_key(job_id: str, output_format: str, prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:12]
    safe_job_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in job_id)
    return f"{prefix.strip('/')}/{safe_job_id}-{timestamp}-{suffix}.{output_format}"
