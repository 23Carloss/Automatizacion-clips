"""Temporary Cloudflare R2 storage for Instagram media ingestion."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from config import Settings


@dataclass(frozen=True, slots=True)
class RemoteVideo:
    key: str
    url: str


class R2VideoStorage:
    """Upload masters to R2 and expose a temporary HTTPS download URL."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.r2_configured:
            raise ValueError(
                "R2 requires R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
                "R2_SECRET_ACCESS_KEY and R2_BUCKET."
            )
        self.settings = settings
        self._client_override = client
        self._client_instance: Any | None = None

    def _client(self):
        if self._client_override is not None:
            return self._client_override
        if self._client_instance is None:
            import boto3

            self._client_instance = boto3.client(
                service_name="s3",
                endpoint_url=(
                    f"https://{self.settings.r2_account_id}.r2.cloudflarestorage.com"
                ),
                aws_access_key_id=self.settings.r2_access_key_id,
                aws_secret_access_key=self.settings.r2_secret_access_key,
                region_name="auto",
            )
        return self._client_instance

    def upload(self, clip_path: Path) -> RemoteVideo:
        if not clip_path.is_file():
            raise FileNotFoundError(f"Local clip does not exist: {clip_path}")
        key = self._object_key(clip_path.name)
        self._client().upload_file(
            str(clip_path),
            self.settings.r2_bucket,
            key,
            ExtraArgs={"ContentType": "video/mp4", "CacheControl": "no-store"},
        )
        return RemoteVideo(key=key, url=self._download_url(key))

    def delete(self, key: str) -> None:
        self._client().delete_object(Bucket=self.settings.r2_bucket, Key=key)

    def _object_key(self, filename: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        prefix = self.settings.r2_key_prefix.strip("/")
        return f"{prefix}/{safe_name}" if prefix else safe_name

    def _download_url(self, key: str) -> str:
        if self.settings.r2_public_base_url:
            return f"{self.settings.r2_public_base_url.rstrip('/')}/{quote(key, safe='/')}"
        return self._client().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.r2_bucket, "Key": key},
            ExpiresIn=self.settings.r2_presigned_url_ttl_seconds,
        )
