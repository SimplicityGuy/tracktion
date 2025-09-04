"""
Unit tests for storage service.
"""

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from services.tracklist_service.src.services.storage_service import (
    FilesystemBackend,
    StorageConfig,
    StorageResult,
    StorageService,
)


class TestStorageConfig:
    """Test StorageConfig model."""

    def test_default_config(self):
        """Test default storage configuration."""
        config = StorageConfig()

        assert config.primary == "filesystem"
        assert config.backup is True
        assert config.max_versions == 5
        assert "base_path" in config.filesystem

    def test_custom_config(self):
        """Test custom storage configuration."""
        config = StorageConfig(primary="filesystem", backup=False, max_versions=10)

        assert config.primary == "filesystem"
        assert config.backup is False
        assert config.max_versions == 10


class TestFilesystemBackend:
    """Test FilesystemBackend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def backend(self, temp_dir):
        """Create filesystem backend with temporary directory."""
        config = {"base_path": temp_dir}
        return FilesystemBackend(config)

    def test_store_file(self, backend):
        """Test storing a file."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO\nTITLE "Track 1"'
        file_path = "test/sample.cue"

        result = backend.store(content, file_path)

        assert result.success is True
        assert result.file_path is not None
        assert result.checksum is not None
        assert result.file_size == len(content.encode("utf-8"))
        assert result.version == 1

    def test_store_file_with_metadata(self, backend):
        """Test storing a file with metadata."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        file_path = "test/metadata.cue"
        metadata = {"format": "standard", "tracks": 1}

        result = backend.store(content, file_path, metadata)

        assert result.success is True
        # Check metadata file was created
        metadata_path = Path(result.file_path).with_suffix(".metadata.json")
        assert metadata_path.exists()

    def test_retrieve_file(self, backend):
        """Test retrieving a file."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        file_path = "test/retrieve.cue"

        # Store first
        backend.store(content, file_path)

        # Retrieve
        success, retrieved_content, error = backend.retrieve(file_path)

        assert success is True
        assert retrieved_content == content
        assert error is None

    def test_retrieve_nonexistent_file(self, backend):
        """Test retrieving a non-existent file."""
        success, content, error = backend.retrieve("nonexistent.cue")

        assert success is False
        assert content is None
        assert "not found" in error.lower()

    def test_delete_file(self, backend):
        """Test deleting a file."""
        content = 'FILE "mix.wav" WAVE'
        file_path = "test/delete.cue"

        # Store first
        store_result = backend.store(content, file_path)
        assert store_result.success is True
        assert backend.exists(file_path) is True

        # Delete
        deleted = backend.delete(file_path)

        assert deleted is True
        assert backend.exists(file_path) is False

    def test_delete_nonexistent_file(self, backend):
        """Test deleting a non-existent file."""
        deleted = backend.delete("nonexistent.cue")
        assert deleted is False

    def test_file_exists(self, backend):
        """Test checking if file exists."""
        content = 'FILE "mix.wav" WAVE'
        file_path = "test/exists.cue"

        assert backend.exists(file_path) is False

        backend.store(content, file_path)

        assert backend.exists(file_path) is True

    def test_versioning(self, backend):
        """Test file versioning."""
        file_path = "test/versioned.cue"
        content1 = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        content2 = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO\nTRACK 02 AUDIO'

        # Store first version
        result1 = backend.store(content1, file_path)
        assert result1.version == 1

        # Store second version
        result2 = backend.store(content2, file_path)
        assert result2.version == 2

        # Check current content is second version
        success, content, error = backend.retrieve(file_path)
        assert success is True
        assert content == content2

    def test_list_versions(self, backend):
        """Test listing file versions."""
        file_path = "test/versions.cue"
        content1 = "VERSION 1"
        content2 = "VERSION 2"

        # Store two versions
        backend.store(content1, file_path)
        backend.store(content2, file_path)

        versions = backend.list_versions(file_path)

        assert len(versions) >= 1  # At least current version
        assert any(v["is_current"] for v in versions)

    def test_cleanup_old_versions(self, backend):
        """Test cleaning up old versions."""
        file_path = "test/cleanup.cue"

        # Store multiple versions
        for i in range(7):  # More than max_versions
            content = f"VERSION {i + 1}"
            backend.store(content, file_path)

        # Cleanup with max_versions=3
        backend.cleanup_old_versions(file_path, max_versions=3)

        versions = backend.list_versions(file_path)
        # Should have at most 3 versions now
        assert len(versions) <= 3


class TestStorageService:
    """Test StorageService."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage_service(self, temp_dir):
        """Create storage service with temporary directory."""
        config = StorageConfig()
        config.filesystem["base_path"] = temp_dir
        return StorageService(config)

    def test_initialization(self, storage_service):
        """Test storage service initialization."""
        assert storage_service.config is not None
        assert storage_service.primary_backend is not None
        assert storage_service.config.backup is True
        assert storage_service.config.max_versions == 5

    def test_generate_file_path(self, storage_service):
        """Test file path generation."""
        audio_file_id = uuid4()
        cue_format = "standard"

        file_path = storage_service.generate_file_path(audio_file_id, cue_format)

        assert str(audio_file_id) in file_path
        assert cue_format in file_path
        assert file_path.endswith(".cue")

    def test_generate_file_path_with_date(self, storage_service):
        """Test file path generation with specific date."""
        audio_file_id = uuid4()
        cue_format = "cdj"

        file_path = storage_service.generate_file_path(audio_file_id, cue_format, year=2024, month=3)

        assert "2024" in file_path
        assert "03" in file_path

    def test_store_cue_file(self, storage_service):
        """Test storing a CUE file."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        audio_file_id = uuid4()
        cue_format = "standard"

        result = storage_service.store_cue_file(content, audio_file_id, cue_format)

        assert result.success is True
        assert result.file_path is not None
        assert result.checksum is not None

    def test_store_cue_file_with_metadata(self, storage_service):
        """Test storing a CUE file with metadata."""
        content = 'FILE "mix.wav" WAVE'
        audio_file_id = uuid4()
        cue_format = "traktor"
        metadata = {"bpm_included": True}

        result = storage_service.store_cue_file(content, audio_file_id, cue_format, metadata)

        assert result.success is True

    def test_retrieve_cue_file(self, storage_service):
        """Test retrieving a CUE file."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        audio_file_id = uuid4()
        cue_format = "standard"

        # Store first
        result = storage_service.store_cue_file(content, audio_file_id, cue_format)

        # Retrieve
        success, retrieved_content, error = storage_service.retrieve_cue_file(result.file_path)

        assert success is True
        assert retrieved_content == content
        assert error is None

    def test_delete_cue_file(self, storage_service):
        """Test deleting a CUE file."""
        content = 'FILE "mix.wav" WAVE'
        audio_file_id = uuid4()
        cue_format = "serato"

        # Store first
        result = storage_service.store_cue_file(content, audio_file_id, cue_format)

        # Delete
        deleted = storage_service.delete_cue_file(result.file_path)

        assert deleted is True
        assert storage_service.file_exists(result.file_path) is False

    def test_file_exists(self, storage_service):
        """Test checking if CUE file exists."""
        content = 'FILE "mix.wav" WAVE'
        audio_file_id = uuid4()
        cue_format = "rekordbox"

        file_path = storage_service.generate_file_path(audio_file_id, cue_format)

        assert storage_service.file_exists(file_path) is False

        storage_service.store_cue_file(content, audio_file_id, cue_format)

        assert storage_service.file_exists(file_path) is True

    def test_list_file_versions(self, storage_service):
        """Test listing file versions."""
        content = 'FILE "mix.wav" WAVE'
        audio_file_id = uuid4()
        cue_format = "kodi"

        # Store file
        storage_service.store_cue_file(content, audio_file_id, cue_format)
        file_path = storage_service.generate_file_path(audio_file_id, cue_format)

        versions = storage_service.list_file_versions(file_path)

        assert len(versions) >= 1
        assert any(v["is_current"] for v in versions)

    def test_get_file_info(self, storage_service):
        """Test getting file information."""
        content = 'FILE "mix.wav" WAVE\nTRACK 01 AUDIO'
        audio_file_id = uuid4()
        cue_format = "standard"

        # Store file
        storage_service.store_cue_file(content, audio_file_id, cue_format)
        file_path = storage_service.generate_file_path(audio_file_id, cue_format)

        info = storage_service.get_file_info(file_path)

        assert info is not None
        assert "path" in info
        assert "current_version" in info
        assert "size" in info
        assert "total_versions" in info

    def test_get_file_info_nonexistent(self, storage_service):
        """Test getting info for non-existent file."""
        info = storage_service.get_file_info("nonexistent.cue")
        assert info is None

    def test_get_storage_stats(self, storage_service):
        """Test getting storage statistics."""
        stats = storage_service.get_storage_stats()

        assert "backend" in stats
        assert "backup_enabled" in stats
        assert "max_versions" in stats
        assert stats["backend"] == "filesystem"

    def test_unsupported_backend(self, temp_dir):
        """Test creating service with unsupported backend falls back to filesystem."""
        config = StorageConfig(primary="unsupported")
        config.filesystem["base_path"] = temp_dir

        # Should not raise, should fallback to filesystem
        service = StorageService(config)
        assert service.config.primary == "filesystem"


class TestStorageResult:
    """Test StorageResult model."""

    def test_successful_result(self):
        """Test successful storage result."""
        result = StorageResult(
            success=True,
            file_path="/data/test.cue",
            checksum="abc123",
            file_size=1024,
            version=1,
        )

        assert result.success is True
        assert result.file_path == "/data/test.cue"
        assert result.error is None

    def test_failed_result(self):
        """Test failed storage result."""
        result = StorageResult(success=False, error="Permission denied")

        assert result.success is False
        assert result.error == "Permission denied"
        assert result.file_path is None


class TestIntegration:
    """Integration tests for storage service."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_workflow(self, temp_dir):
        """Test complete storage workflow."""
        # Setup
        config = StorageConfig()
        config.filesystem["base_path"] = temp_dir
        service = StorageService(config)

        audio_file_id = uuid4()
        cue_format = "standard"
        content = """FILE "mix.wav" WAVE
TRACK 01 AUDIO
  TITLE "Track 1"
  PERFORMER "Artist 1"
  INDEX 01 00:00:00
TRACK 02 AUDIO
  TITLE "Track 2"
  PERFORMER "Artist 2"
  INDEX 01 03:45:12"""

        # Store
        result = service.store_cue_file(content, audio_file_id, cue_format)
        assert result.success is True

        # Verify exists
        assert service.file_exists(result.file_path) is True

        # Retrieve
        success, retrieved, error = service.retrieve_cue_file(result.file_path)
        assert success is True
        assert retrieved == content

        # Get info
        info = service.get_file_info(result.file_path)
        assert info is not None
        assert info["current_version"] == 1

        # Update (new version)
        updated_content = content + "\nTRACK 03 AUDIO"
        result2 = service.store_cue_file(updated_content, audio_file_id, cue_format)
        assert result2.success is True
        assert result2.version == 2

        # List versions
        versions = service.list_file_versions(result.file_path)
        assert len(versions) >= 1

        # Delete
        deleted = service.delete_cue_file(result.file_path)
        assert deleted is True
        assert service.file_exists(result.file_path) is False
