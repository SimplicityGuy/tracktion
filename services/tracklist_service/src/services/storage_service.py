"""
Storage service for CUE files with filesystem and S3 support.

This module provides file storage capabilities for CUE files with support for
both local filesystem and S3 cloud storage, including versioning and retrieval.
"""

import hashlib
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Storage configuration model."""

    primary: str = Field("filesystem", description="Primary storage type")
    filesystem: Dict[str, Any] = Field(
        default_factory=lambda: {
            "base_path": "/data/cue_files/",
            "structure": "{year}/{month}/{audio_file_id}/{format}.cue",
            "permissions": "644",
        }
    )
    s3: Dict[str, Any] = Field(
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
    file_path: Optional[str] = Field(None, description="Stored file path")
    checksum: Optional[str] = Field(None, description="SHA256 checksum of file")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    version: Optional[int] = Field(None, description="File version number")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store(self, content: str, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> StorageResult:
        """Store content to the specified path."""
        pass

    @abstractmethod
    def retrieve(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Retrieve content from the specified path."""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete file from the specified path."""
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists at the specified path."""
        pass

    @abstractmethod
    def list_versions(self, file_path: str) -> List[Dict[str, Any]]:
        """List all versions of a file."""
        pass


class FilesystemBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize filesystem backend with configuration."""
        self.base_path = Path(config.get("base_path", "/data/cue_files/"))
        self.permissions = config.get("permissions", "644")
        self.structure = config.get("structure", "{year}/{month}/{tracklist_id}/{format}.cue")

        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Filesystem backend initialized with base path: {self.base_path}")

    def store(self, content: str, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> StorageResult:
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
            os.chmod(full_path, int(self.permissions, 8))

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

    def retrieve(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Retrieve content from filesystem.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (success, content, error)
        """
        try:
            # Handle both absolute and relative paths
            if Path(file_path).is_absolute():
                full_path = Path(file_path)
            else:
                full_path = self.base_path / file_path

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
        else:
            return (self.base_path / file_path).exists()

    def list_versions(self, file_path: str) -> List[Dict[str, Any]]:
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
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
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
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
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
    """AWS S3 storage backend."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize S3 backend with configuration."""
        self.bucket = config.get("bucket", "tracktion-cue-files")
        self.prefix = config.get("prefix", "cue_files/")
        self.acl = config.get("acl", "private")
        self.storage_class = config.get("storage_class", "STANDARD_IA")

        # S3 client would be initialized here
        self.s3_client = None  # Placeholder
        logger.info(f"S3 backend initialized for bucket: {self.bucket}")

    def store(self, content: str, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> StorageResult:
        """Store content to S3."""
        # Placeholder implementation
        logger.warning("S3 storage not implemented, using placeholder")
        return StorageResult(
            success=False,
            file_path="",
            checksum="",
            file_size=0,
            version=0,
            error="S3 backend not implemented",
            metadata={},
        )

    def retrieve(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Retrieve content from S3."""
        # Placeholder implementation
        return False, None, "S3 backend not implemented"

    def delete(self, file_path: str) -> bool:
        """Delete file from S3."""
        # Placeholder implementation
        return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        # Placeholder implementation
        return False

    def list_versions(self, file_path: str) -> List[Dict[str, Any]]:
        """List all versions of a file in S3."""
        # Placeholder implementation
        return []


class StorageService:
    """Main storage service managing different backends."""

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize storage service with configuration."""
        self.config = config or StorageConfig(
            primary="filesystem",
            backup=True,
            max_versions=5,
        )

        # Initialize backends
        self.backends: Dict[str, StorageBackend] = {}

        # Initialize filesystem backend
        self.backends["filesystem"] = FilesystemBackend(self.config.filesystem)

        # Initialize S3 backend if configured
        if self.config.primary == "s3":
            self.backends["s3"] = S3Backend(self.config.s3)

        # Set primary backend
        self.primary_backend = self.backends.get(self.config.primary, self.backends["filesystem"])

        logger.info(f"Storage service initialized with primary backend: {self.config.primary}")

    def store_cue_file(
        self, file_path: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
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
                        "stored_at": datetime.utcnow().isoformat(),
                        **(metadata or {}),
                    },
                )

                return True, result.file_path, None
            else:
                return False, "", result.error

        except Exception as e:
            logger.error(f"Failed to store CUE file {file_path}: {e}", exc_info=True)
            return False, "", str(e)

    def retrieve_cue_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
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

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
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

            return {"exists": True, "metadata": metadata, "versions": versions, "version_count": len(versions)}

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}", exc_info=True)
            return None

    def _store_metadata(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a file."""
        try:
            metadata_path = f"{file_path}.metadata.json"
            metadata_content = json.dumps(metadata, indent=2)
            self.primary_backend.store(metadata_content, metadata_path)
        except Exception as e:
            logger.warning(f"Failed to store metadata for {file_path}: {e}")

    def _get_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file."""
        try:
            metadata_path = f"{file_path}.metadata.json"
            success, content, _ = self.primary_backend.retrieve(metadata_path)
            if success and content:
                return json.loads(content)  # type: ignore[no-any-return]
        except Exception as e:
            logger.debug(f"No metadata found for {file_path}: {e}")
        return None

    def list_cue_files(self, tracklist_id: Optional[UUID] = None, format_type: Optional[str] = None) -> List[str]:
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

    def get_storage_stats(self) -> Dict[str, Any]:
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


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service(config: Optional[StorageConfig] = None) -> StorageService:
    """Get or create storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService(config)
    return _storage_service
