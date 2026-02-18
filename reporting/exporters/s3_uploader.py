"""S3 uploader — upload reports to Amazon S3 (or compatible)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)

try:
    import boto3  # type: ignore[import-untyped]
    from botocore.exceptions import ClientError  # type: ignore[import-untyped]

    _HAS_BOTO3 = True
except ImportError:  # pragma: no cover
    _HAS_BOTO3 = False


class S3Uploader:
    """Upload report artefacts to an S3-compatible bucket.

    If ``boto3`` is not installed the uploader logs a warning and
    silently no-ops.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix (folder) inside the bucket.
        region: AWS region (defaults to ``us-east-1``).
        endpoint_url: Optional S3-compatible endpoint (e.g. MinIO).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "reports",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = None

        if _HAS_BOTO3:
            kwargs: dict = {"region_name": region}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("s3", **kwargs)

    @property
    def available(self) -> bool:
        """Whether the boto3 SDK is importable."""
        return _HAS_BOTO3

    def upload_file(
        self,
        local_path: str | Path,
        key: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Upload a single file to S3.

        Args:
            local_path: Absolute or relative path to the local file.
            key: Object key.  Defaults to ``<prefix>/<filename>``.
            content_type: MIME type.  Guessed from extension if omitted.

        Returns:
            The S3 URI (``s3://bucket/key``) on success, ``None``
            on failure or if boto3 is unavailable.
        """
        if not _HAS_BOTO3 or self._client is None:
            _logger.warning("boto3 not available; skipping S3 upload")
            return None

        path = Path(local_path)
        if not path.is_file():
            _logger.error("File not found: %s", path)
            return None

        obj_key = key or f"{self._prefix}/{path.name}"
        extra: dict = {}
        ct = content_type or self._guess_content_type(path.suffix)
        if ct:
            extra["ContentType"] = ct

        try:
            self._client.upload_file(
                str(path),
                self._bucket,
                obj_key,
                ExtraArgs=extra or None,
            )
            s3_uri = f"s3://{self._bucket}/{obj_key}"
            _logger.info("Uploaded %s → %s", path.name, s3_uri)
            return s3_uri
        except Exception:
            _logger.exception("S3 upload failed for %s", path.name)
            return None

    @staticmethod
    def _guess_content_type(suffix: str) -> Optional[str]:
        return {
            ".html": "text/html",
            ".json": "application/json",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(suffix.lower())
