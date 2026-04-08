"""
MedGraphRAG — MinIO Object Storage Client
Manages upload / download of original PDF files.
"""

from minio import Minio
from minio.error import S3Error
from pathlib import Path
from loguru import logger
import io

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_BUCKET, MINIO_SECURE,
)


class MinioClient:
    """Wrapper around MinIO S3-compatible object storage."""

    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        self._ensure_bucket()
        logger.info(f"MinIO client ready — bucket '{MINIO_BUCKET}'")

    def _ensure_bucket(self):
        if not self.client.bucket_exists(MINIO_BUCKET):
            self.client.make_bucket(MINIO_BUCKET)
            logger.info(f"Created MinIO bucket '{MINIO_BUCKET}'")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_pdf(self, local_path: str, object_name: str | None = None) -> str:
        """
        Upload a PDF file to MinIO.

        Returns the object path used as the key.
        """
        path = Path(local_path)
        if object_name is None:
            object_name = f"pdfs/{path.name}"

        self.client.fput_object(
            MINIO_BUCKET,
            object_name,
            str(path),
            content_type="application/pdf",
        )
        logger.info(f"Uploaded {path.name} → {object_name}")
        return object_name

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to MinIO."""
        self.client.put_object(
            MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_name

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download_pdf(self, object_name: str, local_path: str) -> Path:
        """Download a PDF from MinIO to a local path."""
        self.client.fget_object(MINIO_BUCKET, object_name, local_path)
        logger.info(f"Downloaded {object_name} → {local_path}")
        return Path(local_path)

    def get_bytes(self, object_name: str) -> bytes:
        """Get raw bytes of an object."""
        response = self.client.get_object(MINIO_BUCKET, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    # ------------------------------------------------------------------
    # List & Delete
    # ------------------------------------------------------------------
    def list_objects(self, prefix: str = "pdfs/") -> list[str]:
        """List all object keys under a prefix."""
        objects = self.client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def delete_object(self, object_name: str) -> None:
        self.client.remove_object(MINIO_BUCKET, object_name)
        logger.info(f"Deleted {object_name}")
