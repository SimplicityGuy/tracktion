"""
Storage service for CUE files with filesystem and S3 support.

This module provides file storage capabilities for CUE files with support for
both local filesystem and S3 cloud storage, including versioning and retrieval.
"""

import os
import hashlib
import shutil
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage configuration model."""

    primary: str = Field("filesystem", description="Primary storage type")
    filesystem: Dict = Field(
        default_factory=lambda: {
            "base_path": "/data/cue_files/",
            "structure": "{year}/{month}/{audio_file_id}/{format}.cue",
            "permissions": "644",
        }
    )
    s3: Dict = Field(
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
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store(self, content: str, file_path: str, metadata: Optional[Dict] = None) -> StorageResult:
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
    def list_versions(self, file_path: str) -> List[Dict]:
        """List all versions of a file."""
        pass


class FilesystemBackend(StorageBackend):
    """Filesystem storage backend."""

    def __init__(self, config: Dict):
        self.base_path = Path(config.get("base_path", "/data/cue_files/"))
        self.structure = config.get("structure", "{year}/{month}/{audio_file_id}/{format}.cue")
        self.permissions = config.get("permissions", "644")

        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store(self, content: str, file_path: str, metadata: Optional[Dict] = None) -> StorageResult:
        """Store content to filesystem."""
        try:
            full_path = self.base_path / file_path.lstrip("/")

            # Create directory structure
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle versioning
            version = 1
            if full_path.exists():
                version = self._get_next_version(full_path)
                if version > 1:
                    # Backup existing file
                    backup_path = full_path.with_suffix(f".v{version - 1}.cue")
                    shutil.copy2(full_path, backup_path)

            # Write content
            full_path.write_text(content, encoding="utf-8")

            # Set permissions
            os.chmod(full_path, int(self.permissions, 8))

            # Calculate checksum and size
            checksum = self._calculate_checksum(content)
            file_size = len(content.encode("utf-8"))

            # Store metadata if provided
            if metadata:
                metadata_path = full_path.with_suffix(".metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

            return StorageResult(
                success=True,
                file_path=str(full_path),
                checksum=checksum,
                file_size=file_size,
                version=version,
                error=None,
            )

        except Exception as e:
            return StorageResult(
                success=False,
                file_path=None,
                checksum=None,
                file_size=None,
                version=None,
                error=f"Failed to store file: {str(e)}",
            )

    def retrieve(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Retrieve content from filesystem."""
        try:
            full_path = self.base_path / file_path.lstrip("/")

            if not full_path.exists():
                return False, None, f"File not found: {file_path}"

            content = full_path.read_text(encoding="utf-8")
            return True, content, None

        except Exception as e:
            return False, None, f"Failed to retrieve file: {str(e)}"

    def delete(self, file_path: str) -> bool:
        """Delete file from filesystem."""
        try:
            full_path = self.base_path / file_path.lstrip("/")

            if full_path.exists():
                full_path.unlink()

                # Also delete metadata file if it exists
                metadata_path = full_path.with_suffix(".metadata.json")
                if metadata_path.exists():
                    metadata_path.unlink()

                return True
            return False

        except Exception:
            return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists on filesystem."""
        full_path = self.base_path / file_path.lstrip("/")
        return full_path.exists()

    def list_versions(self, file_path: str) -> List[Dict]:
        """List all versions of a file."""
        try:
            full_path = self.base_path / file_path.lstrip("/")
            versions = []

            # Current version
            if full_path.exists():
                stat = full_path.stat()
                versions.append(
                    {
                        "version": 1,
                        "path": str(full_path),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_current": True,
                    }
                )

            # Previous versions
            for i in range(2, 10):  # Check up to version 10
                version_path = full_path.with_suffix(f".v{i}.cue")
                if version_path.exists():
                    stat = version_path.stat()
                    versions.append(
                        {
                            "version": i,
                            "path": str(version_path),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "is_current": False,
                        }
                    )

            return sorted(versions, key=lambda x: x["version"], reverse=True)

        except Exception:
            return []

    def _get_next_version(self, file_path: Path) -> int:
        """Get the next version number for a file."""
        version = 1
        for i in range(2, 100):  # Max 100 versions
            version_path = file_path.with_suffix(f".v{i}.cue")
            if not version_path.exists():
                return i
        return version

    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def cleanup_old_versions(self, file_path: str, max_versions: int = 5) -> None:
        """Clean up old versions beyond the maximum."""
        try:
            versions = self.list_versions(file_path)

            # Remove versions beyond max_versions
            if len(versions) > max_versions:
                for version_info in versions[max_versions:]:
                    if not version_info["is_current"]:
                        Path(version_info["path"]).unlink()

        except Exception:
            pass  # Fail silently for cleanup


class S3Backend(StorageBackend):
    """S3 storage backend (placeholder implementation)."""

    def __init__(self, config: Dict):
        self.bucket = config.get("bucket", "tracktion-cue-files")
        self.prefix = config.get("prefix", "cue_files/")
        self.acl = config.get("acl", "private")
        self.storage_class = config.get("storage_class", "STANDARD_IA")

        # NOTE: This is a placeholder implementation
        # In a real implementation, you would initialize boto3 client here
        self._client = None

    def store(self, content: str, file_path: str, metadata: Optional[Dict] = None) -> StorageResult:
        """Store content to S3 (placeholder)."""
        # This is a placeholder implementation
        # In a real implementation, you would use boto3 to upload to S3
        return StorageResult(
            success=False,
            file_path=None,
            checksum=None,
            file_size=None,
            version=None,
            error="S3 backend not implemented in this demo version",
        )

    def retrieve(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Retrieve content from S3 (placeholder)."""
        return False, None, "S3 backend not implemented in this demo version"

    def delete(self, file_path: str) -> bool:
        """Delete file from S3 (placeholder)."""
        return False

    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3 (placeholder)."""
        return False

    def list_versions(self, file_path: str) -> List[Dict]:
        """List all versions of a file in S3 (placeholder)."""
        return []


class StorageService:
    """Main storage service for CUE files."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig(primary="filesystem", backup=True, max_versions=5)
        self.primary_backend = self._create_backend(self.config.primary)
        self.backup_enabled = self.config.backup
        self.max_versions = self.config.max_versions

    def _create_backend(self, backend_type: str) -> StorageBackend:
        """Create a storage backend based on type."""
        if backend_type == "filesystem":
            return FilesystemBackend(self.config.filesystem)
        elif backend_type == "s3":
            return S3Backend(self.config.s3)
        else:
            raise ValueError(f"Unsupported storage backend: {backend_type}")

    def generate_file_path(
        self, audio_file_id: UUID, cue_format: str, year: Optional[int] = None, month: Optional[int] = None
    ) -> str:
        """Generate file path based on configuration structure."""
        now = datetime.utcnow()
        year = year or now.year
        month = month or now.month

        structure_template = self.config.filesystem.get("structure", "{year}/{month}/{audio_file_id}/{format}.cue")
        return str(
            structure_template.format(
                year=year, month=f"{month:02d}", audio_file_id=str(audio_file_id), format=cue_format
            )
        )

    def store_cue_file(
        self, content: str, audio_file_id: UUID, cue_format: str, metadata: Optional[Dict] = None
    ) -> StorageResult:
        """Store a CUE file."""
        file_path = self.generate_file_path(audio_file_id, cue_format)

        # Add storage metadata
        storage_metadata = {
            "audio_file_id": str(audio_file_id),
            "format": cue_format,
            "stored_at": datetime.utcnow().isoformat(),
            "content_type": "application/x-cue",
        }

        if metadata:
            storage_metadata.update(metadata)

        result = self.primary_backend.store(content, file_path, storage_metadata)

        # Update result with the relative file path for consistency
        if result.success:
            result.file_path = file_path

            # Clean up old versions if successful
            if self.backup_enabled and hasattr(self.primary_backend, "cleanup_old_versions"):
                self.primary_backend.cleanup_old_versions(file_path, self.max_versions)

        return result

    def retrieve_cue_file(self, file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Retrieve a CUE file."""
        return self.primary_backend.retrieve(file_path)

    def delete_cue_file(self, file_path: str) -> bool:
        """Delete a CUE file."""
        return self.primary_backend.delete(file_path)

    def file_exists(self, file_path: str) -> bool:
        """Check if CUE file exists."""
        return self.primary_backend.exists(file_path)

    def list_file_versions(self, file_path: str) -> List[Dict]:
        """List all versions of a CUE file."""
        return self.primary_backend.list_versions(file_path)

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """Get information about a stored file."""
        try:
            if not self.file_exists(file_path):
                return None

            versions = self.list_file_versions(file_path)
            if not versions:
                return None

            current_version = next((v for v in versions if v["is_current"]), versions[0])

            return {
                "path": file_path,
                "current_version": current_version["version"],
                "size": current_version["size"],
                "modified": current_version["modified"],
                "total_versions": len(versions),
            }

        except Exception:
            return None

    def cleanup_orphaned_files(self, active_file_paths: List[str]) -> int:
        """Clean up orphaned CUE files not in the active list."""
        # This is a placeholder for a cleanup operation
        # In a real implementation, you would scan the storage
        # and remove files not in the active_file_paths list
        return 0

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        # This is a placeholder for storage statistics
        # In a real implementation, you would calculate actual usage
        return {
            "backend": self.config.primary,
            "backup_enabled": self.backup_enabled,
            "max_versions": self.max_versions,
            "total_files": 0,
            "total_size": 0,
            "total_versions": 0,
        }
