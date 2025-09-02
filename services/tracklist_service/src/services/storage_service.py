"""
Storage service for CUE files with filesystem and S3 support.

This module provides file storage capabilities for CUE files with support for
both local filesystem and S3 cloud storage, including versioning and retrieval.
"""

import hashlib
import json
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Storage configuration model."""

    primary: str = Field("filesystem", description="Primary storage type")
    filesystem: dict[str, Any] = Field(
        default_factory=lambda: {
            "base_path": "/data/cue_files/",
            "structure": "{year}/{month}/{audio_file_id}/{format}.cue",
            "permissions": "644",
        }
    )
    s3: dict[str, Any] = Field(
        default_factory=lambda: {
            "bucket": "tracktion-cue-files",
            "prefix": "cue_files/",
            "structure": "{year}/{month}/{audio_file_id}/{format}.cue",
            "acl": "private",
            "storage_class": "STANDARD_IA",
        }
    )
    backup: bool = Field(True, description="Keep previous versions")
    max_versions: int = Field(5, description="Maximum versions to retain")


class StorageResult(BaseModel):
    """Result of storage operations."""

    success: bool = Field(description="Whether operation was successful")
    file_path: str | None = Field(None, description="Stored file path")
    checksum: str | None = Field(None, description="SHA256 checksum of file")
    file_size: int | None = Field(None, description="File size in bytes")
    version: int | None = Field(None, description="File version number")
    error: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store(self, content: str, file_path: str, metadata: dict[str, Any] | None = None) -> StorageResult:
        """Store content to the specified path."""

    @abstractmethod
    def retrieve(self, file_path: str) -> tuple[bool, str | None, str | None]:
        """Retrieve content from the specified path."""

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete file from the specified path."""

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists at the specified path."""

    @abstractmethod
    def list_versions(self, file_path: str) -> list[dict[str, Any]]:
        """List all versions of a file."""


class FilesystemBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, config: dict[str, Any]):
        """Initialize filesystem backend with configuration."""
        self.base_path = Path(config.get("base_path", "/data/cue_files/"))
        self.permissions = config.get("permissions", "644")
        self.structure = config.get("structure", "{year}/{month}/{tracklist_id}/{format}.cue")

        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Filesystem backend initialized with base path: {self.base_path}")

    def store(self, content: str, file_path: str, metadata: dict[str, Any] | None = None) -> StorageResult:
        """
        Store content to filesystem.

        Args:
            content: File content to store
            file_path: Relative path for file
            metadata: Optional metadata

        Returns:
            StorageResult with operation details
        """
        try:
            # Construct full path
            full_path = self.base_path / file_path

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle versioning if file exists
            version = 1
            if full_path.exists():
                version = self._create_backup(full_path)

            # Write content
            full_path.write_text(content, encoding="utf-8")

            # Set permissions
            full_path.chmod(int(self.permissions, 8))

            # Calculate checksum
            checksum = hashlib.sha256(content.encode()).hexdigest()

            # Get file size
            file_size = len(content.encode())

            logger.info(f"Stored file: {full_path} (size: {file_size}, checksum: {checksum[:8]}...)")

            return StorageResult(
                success=True,
                file_path=str(full_path),
                checksum=checksum,
                file_size=file_size,
                version=version,
                error=None,
                metadata=metadata or {},
            )

        except Exception as e:
            logger.error(f"Failed to store file {file_path}: {e}", exc_info=True)
            return StorageResult(
                success=False,
                file_path="",
                checksum="",
                file_size=0,
                version=0,
                error=str(e),
                metadata={},
            )

    def retrieve(self, file_path: str) -> tuple[bool, str | None, str | None]:
        """
        Retrieve content from filesystem.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (success, content, error)
        """
        try:
            # Handle both absolute and relative paths
            full_path = Path(file_path) if Path(file_path).is_absolute() else self.base_path / file_path

            if not full_path.exists():
                return False, None, f"File not found: {full_path}"

            content = full_path.read_text(encoding="utf-8")
            logger.debug(f"Retrieved file: {full_path} ({len(content)} bytes)")

            return True, content, None

        except Exception as e:
            logger.error(f"Failed to retrieve file {file_path}: {e}", exc_info=True)
            return False, None, str(e)

    def delete(self, file_path: str) -> bool:
        """
        Delete file from filesystem.

        Args:
            file_path: Path to file

        Returns:
            Success status
        """
        try:
            full_path = self.base_path / file_path

            if full_path.exists():
                # Create backup before deletion
                self._create_backup(full_path, suffix=".deleted")
                full_path.unlink()
                logger.info(f"Deleted file: {full_path}")
                return True

            logger.warning(f"File not found for deletion: {full_path}")
            return False

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}", exc_info=True)
            return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        # Handle both absolute and relative paths
        if Path(file_path).is_absolute():
            return Path(file_path).exists()
        return (self.base_path / file_path).exists()

    def list_versions(self, file_path: str) -> list[dict[str, Any]]:
        """List all versions of a file."""
        try:
            full_path = self.base_path / file_path
            versions = []

            # Current version
            if full_path.exists():
                stat = full_path.stat()
                versions.append(
                    {
                        "version": "current",
                        "path": str(full_path),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    }
                )

            # Backup versions
            for backup in full_path.parent.glob(f"{full_path.name}.v*"):
                stat = backup.stat()
                version_num = backup.suffix.split(".v")[1]
                versions.append(
                    {
                        "version": version_num,
                        "path": str(backup),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    }
                )

            return sorted(versions, key=lambda x: x["version"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list versions for {file_path}: {e}", exc_info=True)
            return []

    def _create_backup(self, file_path: Path, suffix: str = ".v") -> int:
        """Create backup of existing file."""
        if not file_path.exists():
            return 1

        # Find next version number
        version = 1
        existing_versions = list(file_path.parent.glob(f"{file_path.name}{suffix}*"))
        if existing_versions:
            version_numbers = []
            for v in existing_versions:
                try:
                    version_numbers.append(int(v.suffix.split(suffix)[1]))
                except (ValueError, IndexError):
                    continue
            if version_numbers:
                version = max(version_numbers) + 1

        # Create backup
        backup_path = file_path.with_suffix(f"{file_path.suffix}{suffix}{version}")
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")

        return version


class S3Backend(StorageBackend):
    """AWS S3 storage backend with retry logic and error handling."""

    def __init__(self, config: dict[str, Any]):
        """Initialize S3 backend with configuration."""
        self.bucket = config.get("bucket", "tracktion-cue-files")
        self.prefix = config.get("prefix", "cue_files/")
        self.acl = config.get("acl", "private")
        self.storage_class = config.get("storage_class", "STANDARD_IA")
        self.region = config.get("region", "us-east-1")

        # Initialize S3 client with configuration
        try:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.region,
                # AWS credentials will be picked up from environment variables,
                # IAM roles, or AWS credentials file
            )

            # Test connection by checking if bucket exists
            self._verify_bucket_access()
            logger.info(f"S3 backend initialized for bucket: {self.bucket}")

        except NoCredentialsError as e:
            logger.error(
                "AWS credentials not found. Configure credentials via environment variables, "
                "IAM roles, or AWS credentials file."
            )
            raise RuntimeError("AWS credentials not configured") from e
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise RuntimeError(f"S3 initialization failed: {e}") from e

    def _verify_bucket_access(self) -> None:
        """Verify bucket exists and we have access."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.debug(f"Verified access to S3 bucket: {self.bucket}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise RuntimeError(f"S3 bucket '{self.bucket}' does not exist") from e
            if error_code == "403":
                raise RuntimeError(f"Access denied to S3 bucket '{self.bucket}'. Check bucket permissions.") from e
            raise RuntimeError(f"Failed to access S3 bucket '{self.bucket}': {e}") from e

    def _get_s3_key(self, file_path: str) -> str:
        """Generate S3 key from file path."""
        # Remove leading slash if present and combine with prefix
        clean_path = file_path.lstrip("/")
        return f"{self.prefix}{clean_path}"

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def store(self, content: str, file_path: str, metadata: dict[str, Any] | None = None) -> StorageResult:
        """Store content to S3 with retry logic."""
        try:
            s3_key = self._get_s3_key(file_path)
            content_bytes = content.encode("utf-8")

            # Calculate checksum
            checksum = hashlib.sha256(content_bytes).hexdigest()
            file_size = len(content_bytes)

            # Prepare S3 metadata
            s3_metadata = {
                "checksum": checksum,
                "content-type": "text/plain; charset=utf-8",
                "original-filename": file_path.split("/")[-1],
            }

            # Add custom metadata if provided
            if metadata:
                for key, value in metadata.items():
                    # S3 metadata keys must be lowercase and cannot contain underscores
                    clean_key = str(key).lower().replace("_", "-")
                    s3_metadata[clean_key] = str(value)

            # Check if file already exists to determine version
            version = 1
            try:
                existing_versions = self._list_object_versions(s3_key)
                if existing_versions:
                    version = len(existing_versions) + 1
            except Exception as e:
                logger.warning(f"Could not determine version for {s3_key}: {e}")

            # Upload to S3
            upload_args = {
                "Bucket": self.bucket,
                "Key": s3_key,
                "Body": content_bytes,
                "ContentType": "text/plain; charset=utf-8",
                "Metadata": s3_metadata,
                "StorageClass": self.storage_class,
            }

            # Add ACL if specified and not None
            if self.acl and self.acl.lower() != "none":
                upload_args["ACL"] = self.acl

            self.s3_client.put_object(**upload_args)

            # Generate S3 URL
            s3_url = f"s3://{self.bucket}/{s3_key}"

            logger.info(f"Stored file to S3: {s3_url} (size: {file_size}, checksum: {checksum[:8]}...)")

            return StorageResult(
                success=True,
                file_path=s3_url,
                checksum=checksum,
                file_size=file_size,
                version=version,
                error=None,
                metadata=metadata or {},
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "NoSuchBucket":
                error_msg = f"S3 bucket '{self.bucket}' does not exist"
            elif error_code == "AccessDenied":
                error_msg = f"Access denied to S3 bucket '{self.bucket}'. Check IAM permissions."
            else:
                error_msg = f"S3 error ({error_code}): {error_message}"

            logger.error(f"Failed to store file to S3 {file_path}: {error_msg}", exc_info=True)
            return StorageResult(
                success=False,
                file_path="",
                checksum="",
                file_size=0,
                version=0,
                error=error_msg,
                metadata={},
            )

        except Exception as e:
            logger.error(f"Failed to store file to S3 {file_path}: {e}", exc_info=True)
            return StorageResult(
                success=False,
                file_path="",
                checksum="",
                file_size=0,
                version=0,
                error=str(e),
                metadata={},
            )

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def retrieve(self, file_path: str) -> tuple[bool, str | None, str | None]:
        """Retrieve content from S3 with retry logic."""
        try:
            # Handle both S3 URLs and relative paths
            if file_path.startswith("s3://"):
                # Extract key from S3 URL: s3://bucket/key
                parts = file_path[5:].split("/", 1)
                if len(parts) != 2 or parts[0] != self.bucket:
                    return False, None, f"Invalid S3 URL or wrong bucket: {file_path}"
                s3_key = parts[1]
            else:
                s3_key = self._get_s3_key(file_path)

            # Get object from S3
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)

            # Read content
            content = response["Body"].read().decode("utf-8")

            logger.debug(f"Retrieved file from S3: s3://{self.bucket}/{s3_key} ({len(content)} bytes)")
            return True, content, None

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "NoSuchKey":
                error_msg = f"File not found in S3: {file_path}"
            elif error_code == "NoSuchBucket":
                error_msg = f"S3 bucket '{self.bucket}' does not exist"
            elif error_code == "AccessDenied":
                error_msg = f"Access denied to S3 object: {file_path}"
            else:
                error_msg = f"S3 error ({error_code}): {e.response['Error']['Message']}"

            logger.error(f"Failed to retrieve file from S3 {file_path}: {error_msg}")
            return False, None, error_msg

        except Exception as e:
            logger.error(f"Failed to retrieve file from S3 {file_path}: {e}", exc_info=True)
            return False, None, str(e)

    @retry(
        retry=retry_if_exception_type((ClientError, BotoCoreError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def delete(self, file_path: str) -> bool:
        """Delete file from S3 with retry logic."""
        try:
            # Handle both S3 URLs and relative paths
            if file_path.startswith("s3://"):
                parts = file_path[5:].split("/", 1)
                if len(parts) != 2 or parts[0] != self.bucket:
                    logger.error(f"Invalid S3 URL or wrong bucket: {file_path}")
                    return False
                s3_key = parts[1]
            else:
                s3_key = self._get_s3_key(file_path)

            # Check if file exists before deletion
            if not self._object_exists(s3_key):
                logger.warning(f"File not found for deletion: s3://{self.bucket}/{s3_key}")
                return False

            # Delete from S3
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)

            logger.info(f"Deleted file from S3: s3://{self.bucket}/{s3_key}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            logger.error(f"Failed to delete file from S3 {file_path}: {error_code} - {e.response['Error']['Message']}")
            return False

        except Exception as e:
            logger.error(f"Failed to delete file from S3 {file_path}: {e}", exc_info=True)
            return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            # Handle both S3 URLs and relative paths
            if file_path.startswith("s3://"):
                parts = file_path[5:].split("/", 1)
                if len(parts) != 2 or parts[0] != self.bucket:
                    return False
                s3_key = parts[1]
            else:
                s3_key = self._get_s3_key(file_path)

            return self._object_exists(s3_key)

        except Exception as e:
            logger.error(f"Error checking if file exists in S3 {file_path}: {e}")
            return False

    def _object_exists(self, s3_key: str) -> bool:
        """Check if S3 object exists."""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def list_versions(self, file_path: str) -> list[dict[str, Any]]:
        """List all versions of a file in S3."""
        try:
            # Handle both S3 URLs and relative paths
            if file_path.startswith("s3://"):
                parts = file_path[5:].split("/", 1)
                if len(parts) != 2 or parts[0] != self.bucket:
                    return []
                s3_key = parts[1]
            else:
                s3_key = self._get_s3_key(file_path)

            return self._list_object_versions(s3_key)

        except Exception as e:
            logger.error(f"Failed to list versions for S3 file {file_path}: {e}", exc_info=True)
            return []

    def _list_object_versions(self, s3_key: str) -> list[dict[str, Any]]:
        """List versions of an S3 object."""
        try:
            # List object versions
            response = self.s3_client.list_object_versions(Bucket=self.bucket, Prefix=s3_key)

            # Use list comprehension as suggested by ruff
            versions = [
                {
                    "version": version["VersionId"] if version.get("VersionId") != "null" else "current",
                    "path": f"s3://{self.bucket}/{s3_key}",
                    "size": version["Size"],
                    "modified": version["LastModified"].isoformat(),
                    "storage_class": version.get("StorageClass", "STANDARD"),
                    "etag": version["ETag"].strip('"'),
                }
                for version in response.get("Versions", [])
                if version["Key"] == s3_key
            ]

            # Sort by modification date, most recent first
            return sorted(versions, key=lambda x: x["modified"], reverse=True)

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                logger.error(f"S3 bucket '{self.bucket}' does not exist")
            else:
                logger.error(f"Failed to list object versions: {e}")
            return []


class StorageService:
    """Main storage service managing different backends."""

    def __init__(self, config: StorageConfig | None = None):
        """Initialize storage service with configuration."""
        self.config = config or StorageConfig(
            primary="filesystem",
            backup=True,
            max_versions=5,
        )

        # Initialize backends
        self.backends: dict[str, StorageBackend] = {}

        # Initialize filesystem backend
        self.backends["filesystem"] = FilesystemBackend(self.config.filesystem)

        # Initialize S3 backend if configured
        if self.config.primary == "s3":
            self.backends["s3"] = S3Backend(self.config.s3)

        # Set primary backend
        self.primary_backend = self.backends.get(self.config.primary, self.backends["filesystem"])

        logger.info(f"Storage service initialized with primary backend: {self.config.primary}")

    def store_cue_file(
        self, file_path: str, content: str, metadata: dict[str, Any] | None = None
    ) -> tuple[bool, str | None, str | None]:
        """
        Store CUE file content.

        Args:
            file_path: Relative path for file
            content: CUE file content
            metadata: Optional metadata

        Returns:
            Tuple of (success, stored_path, error)
        """
        try:
            result = self.primary_backend.store(content, file_path, metadata)

            if result.success:
                # Store metadata
                self._store_metadata(
                    file_path,
                    {
                        "checksum": result.checksum,
                        "size": result.file_size,
                        "version": result.version,
                        "stored_at": datetime.now(UTC).isoformat(),
                        **(metadata or {}),
                    },
                )

                return True, result.file_path, None
            return False, "", result.error

        except Exception as e:
            logger.error(f"Failed to store CUE file {file_path}: {e}", exc_info=True)
            return False, "", str(e)

    def retrieve_cue_file(self, file_path: str) -> tuple[bool, str | None, str | None]:
        """
        Retrieve CUE file content.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (success, content, error)
        """
        return self.primary_backend.retrieve(file_path)

    def delete_cue_file(self, file_path: str) -> bool:
        """
        Delete CUE file.

        Args:
            file_path: Path to file

        Returns:
            Success status
        """
        return self.primary_backend.delete(file_path)

    def get_file_info(self, file_path: str) -> dict[str, Any] | None:
        """
        Get file information.

        Args:
            file_path: Path to file

        Returns:
            File information dictionary or None
        """
        try:
            if not self.primary_backend.exists(file_path):
                return None

            # Get metadata
            metadata = self._get_metadata(file_path)

            # Get version info
            versions = self.primary_backend.list_versions(file_path)

            return {
                "exists": True,
                "metadata": metadata,
                "versions": versions,
                "version_count": len(versions),
            }

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}", exc_info=True)
            return None

    def _store_metadata(self, file_path: str, metadata: dict[str, Any]) -> None:
        """Store metadata for a file."""
        try:
            metadata_path = f"{file_path}.metadata.json"
            metadata_content = json.dumps(metadata, indent=2)
            self.primary_backend.store(metadata_content, metadata_path)
        except Exception as e:
            logger.warning(f"Failed to store metadata for {file_path}: {e}")

    def _get_metadata(self, file_path: str) -> dict[str, Any] | None:
        """Get metadata for a file."""
        try:
            metadata_path = f"{file_path}.metadata.json"
            success, content, _ = self.primary_backend.retrieve(metadata_path)
            if success and content:
                return json.loads(content)  # type: ignore[no-any-return]
        except Exception as e:
            logger.debug(f"No metadata found for {file_path}: {e}")
        return None

    def list_cue_files(self, tracklist_id: UUID | None = None, format_type: str | None = None) -> list[str]:
        """
        List CUE files matching criteria.

        Args:
            tracklist_id: Filter by tracklist ID
            format_type: Filter by format

        Returns:
            List of file paths
        """
        # This would need proper implementation based on backend
        # For now, return empty list
        return []

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "backend": self.config.primary,
            "backup_enabled": self.config.backup,
            "max_versions": self.config.max_versions,
        }


class StorageServiceSingleton:
    """Singleton wrapper for StorageService."""

    _instance: StorageService | None = None

    @classmethod
    def get_instance(cls, config: StorageConfig | None = None) -> StorageService:
        """Get the singleton StorageService instance."""
        if cls._instance is None:
            cls._instance = StorageService(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


def get_storage_service(config: StorageConfig | None = None) -> StorageService:
    """Get or create storage service singleton instance."""
    return StorageServiceSingleton.get_instance(config)
