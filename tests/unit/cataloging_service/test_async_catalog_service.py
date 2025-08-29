"""Unit tests for async catalog service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from services.cataloging_service.src.async_catalog_service import AsyncCatalogService
from shared.core_types.src.models import Metadata, Recording, Tracklist


@pytest.fixture
async def mock_db_manager():
    """Create mock database manager."""
    return MagicMock()


@pytest.fixture
async def mock_repositories():
    """Create mock repositories."""
    recording_repo = MagicMock()
    recording_repo.create = AsyncMock()
    recording_repo.get_by_file_path = AsyncMock()
    recording_repo.get_by_id = AsyncMock()
    recording_repo.update = AsyncMock()
    recording_repo.delete = AsyncMock()
    recording_repo.batch_create = AsyncMock()
    recording_repo.get_all = AsyncMock()

    metadata_repo = MagicMock()
    metadata_repo.create = AsyncMock()
    metadata_repo.get_by_key = AsyncMock()
    metadata_repo.update = AsyncMock()
    metadata_repo.batch_create = AsyncMock()
    metadata_repo.get_by_recording = AsyncMock()

    tracklist_repo = MagicMock()
    tracklist_repo.create = AsyncMock()
    tracklist_repo.get_by_recording = AsyncMock()
    tracklist_repo.update = AsyncMock()

    batch_ops = MagicMock()

    return recording_repo, metadata_repo, tracklist_repo, batch_ops


@pytest.fixture
async def catalog_service(mock_db_manager, mock_repositories):
    """Create catalog service with mocked dependencies."""
    recording_repo, metadata_repo, tracklist_repo, batch_ops = mock_repositories

    service = AsyncCatalogService(mock_db_manager)
    service.recording_repo = recording_repo
    service.metadata_repo = metadata_repo
    service.tracklist_repo = tracklist_repo
    service.batch_ops = batch_ops

    return service


class TestAsyncCatalogService:
    """Test async catalog service operations."""

    @pytest.mark.asyncio
    async def test_catalog_new_file(self, catalog_service):
        """Test cataloging a new file."""
        # Setup
        recording_id = uuid4()
        recording = Recording(
            id=recording_id,
            file_path="/test/file.mp3",
            file_name="file.mp3",
            sha256_hash="testhash",
            xxh128_hash="testxxhash",
            created_at=datetime.utcnow(),
        )

        catalog_service.recording_repo.get_by_file_path.return_value = None
        catalog_service.recording_repo.create.return_value = recording

        # Execute
        result = await catalog_service.catalog_file(
            file_path="/test/file.mp3",
            file_name="file.mp3",
            sha256_hash="testhash",
            xxh128_hash="testxxhash",
            metadata={"genre": "electronic", "bpm": "128"},
        )

        # Verify
        assert result == recording
        catalog_service.recording_repo.get_by_file_path.assert_called_once_with("/test/file.mp3")
        catalog_service.recording_repo.create.assert_called_once()
        catalog_service.metadata_repo.batch_create.assert_called_once_with(
            recording_id=recording_id, metadata_dict={"genre": "electronic", "bpm": "128"}
        )

    @pytest.mark.asyncio
    async def test_catalog_existing_file(self, catalog_service):
        """Test cataloging an existing file."""
        # Setup
        existing_recording = Recording(
            id=uuid4(), file_path="/test/file.mp3", file_name="file.mp3", created_at=datetime.utcnow()
        )

        catalog_service.recording_repo.get_by_file_path.return_value = existing_recording

        # Execute
        result = await catalog_service.catalog_file(file_path="/test/file.mp3", file_name="file.mp3")

        # Verify
        assert result == existing_recording
        catalog_service.recording_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_file_metadata(self, catalog_service):
        """Test updating file metadata."""
        # Setup
        recording_id = uuid4()
        existing_metadata = Metadata(id=1, recording_id=recording_id, key="genre", value="house")

        catalog_service.metadata_repo.get_by_key.return_value = existing_metadata
        catalog_service.metadata_repo.update.return_value = Metadata(
            id=1, recording_id=recording_id, key="genre", value="techno"
        )

        # Execute
        result = await catalog_service.update_file_metadata(recording_id=recording_id, metadata={"genre": "techno"})

        # Verify
        assert len(result) == 1
        assert result[0].value == "techno"
        catalog_service.metadata_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_tracklist(self, catalog_service):
        """Test adding a tracklist."""
        # Setup
        recording_id = uuid4()
        tracks = [
            {"position": 1, "name": "Track 1", "artist": "Artist A"},
            {"position": 2, "name": "Track 2", "artist": "Artist B"},
        ]

        catalog_service.tracklist_repo.get_by_recording.return_value = None
        catalog_service.tracklist_repo.create.return_value = Tracklist(
            id=1, recording_id=recording_id, source="manual", tracks={"tracks": tracks}
        )

        # Execute
        result = await catalog_service.add_tracklist(recording_id=recording_id, source="manual", tracks=tracks)

        # Verify
        assert result.source == "manual"
        catalog_service.tracklist_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_deleted(self, catalog_service):
        """Test handling file deletion."""
        # Setup
        recording = Recording(id=uuid4(), file_path="/test/file.mp3", file_name="file.mp3")

        catalog_service.recording_repo.get_by_file_path.return_value = recording
        catalog_service.recording_repo.delete.return_value = True

        # Execute
        result = await catalog_service.handle_file_deleted("/test/file.mp3")

        # Verify
        assert result is True
        catalog_service.recording_repo.delete.assert_called_once_with(recording.id)

    @pytest.mark.asyncio
    async def test_handle_file_moved(self, catalog_service):
        """Test handling file move/rename."""
        # Setup
        recording = Recording(id=uuid4(), file_path="/test/old.mp3", file_name="old.mp3")
        updated_recording = Recording(id=recording.id, file_path="/test/new.mp3", file_name="new.mp3")

        catalog_service.recording_repo.get_by_file_path.return_value = recording
        catalog_service.recording_repo.update.return_value = updated_recording

        # Execute
        result = await catalog_service.handle_file_moved(
            old_path="/test/old.mp3", new_path="/test/new.mp3", new_name="new.mp3"
        )

        # Verify
        assert result == updated_recording
        catalog_service.recording_repo.update.assert_called_once_with(
            recording_id=recording.id, file_path="/test/new.mp3", file_name="new.mp3"
        )

    @pytest.mark.asyncio
    async def test_batch_catalog_files(self, catalog_service):
        """Test batch cataloging of files."""
        # Setup
        files_data = [
            {
                "file_path": f"/test/file_{i}.mp3",
                "file_name": f"file_{i}.mp3",
                "sha256_hash": f"hash_{i}",
                "xxh128_hash": f"xxhash_{i}",
            }
            for i in range(5)
        ]

        catalog_service.recording_repo.get_by_file_path.return_value = None
        catalog_service.recording_repo.batch_create.return_value = [
            Recording(id=uuid4(), file_path=data["file_path"], file_name=data["file_name"]) for data in files_data
        ]

        # Execute
        result = await catalog_service.batch_catalog_files(files_data)

        # Verify
        assert len(result) == 5
        assert catalog_service.recording_repo.get_by_file_path.call_count == 5
        catalog_service.recording_repo.batch_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_recordings(self, catalog_service):
        """Test searching for recordings."""
        # Setup
        recordings = [
            Recording(id=uuid4(), file_path=f"/test/file_{i}.mp3", file_name=f"file_{i}.mp3") for i in range(5)
        ]

        catalog_service.recording_repo.get_all.return_value = recordings

        # Execute
        result = await catalog_service.search_recordings(file_name="file_2", limit=10)

        # Verify
        assert len(result) == 1
        assert result[0].file_name == "file_2.mp3"

    @pytest.mark.asyncio
    async def test_get_recording_details(self, catalog_service):
        """Test getting detailed recording information."""
        # Setup
        recording_id = uuid4()
        recording = Recording(
            id=recording_id,
            file_path="/test/file.mp3",
            file_name="file.mp3",
            sha256_hash="hash",
            xxh128_hash="xxhash",
            created_at=datetime.utcnow(),
        )

        metadata = [
            Metadata(id=1, recording_id=recording_id, key="genre", value="electronic"),
            Metadata(id=2, recording_id=recording_id, key="bpm", value="128"),
        ]

        tracklist = Tracklist(
            id=1,
            recording_id=recording_id,
            source="manual",
            tracks={"tracks": [{"position": 1, "name": "Track 1"}]},
            cue_file_path="/test/file.cue",
        )

        catalog_service.recording_repo.get_by_id.return_value = recording
        catalog_service.metadata_repo.get_by_recording.return_value = metadata
        catalog_service.tracklist_repo.get_by_recording.return_value = tracklist

        # Execute
        result = await catalog_service.get_recording_details(recording_id)

        # Verify
        assert result is not None
        assert result["recording"]["id"] == str(recording_id)
        assert result["metadata"]["genre"] == "electronic"
        assert result["metadata"]["bpm"] == "128"
        assert result["tracklist"]["source"] == "manual"


class TestAsyncMessageConsumer:
    """Test async message consumer."""

    @pytest.mark.asyncio
    async def test_process_created_event(self):
        """Test processing file created event."""
        from services.cataloging_service.src.message_consumer import CatalogingMessageConsumer

        consumer = CatalogingMessageConsumer()
        consumer.catalog_service = MagicMock()
        consumer.catalog_service.catalog_file = AsyncMock()

        # Create mock message
        message = MagicMock()
        message.body = b"""{
            "event_type": "created",
            "file_path": "/test/file.mp3",
            "sha256_hash": "hash",
            "xxh128_hash": "xxhash",
            "size_bytes": "1024",
            "metadata": {"genre": "electronic"}
        }"""
        message.process = MagicMock()
        message.process.__aenter__ = AsyncMock(return_value=None)
        message.process.__aexit__ = AsyncMock(return_value=None)

        # Process message
        await consumer.process_message(message)

        # Verify
        consumer.catalog_service.catalog_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_deleted_event(self):
        """Test processing file deleted event."""
        from services.cataloging_service.src.message_consumer import CatalogingMessageConsumer

        consumer = CatalogingMessageConsumer()
        consumer.catalog_service = MagicMock()
        consumer.catalog_service.handle_file_deleted = AsyncMock()

        # Create mock message
        message = MagicMock()
        message.body = b"""{
            "event_type": "deleted",
            "file_path": "/test/file.mp3"
        }"""
        message.process = MagicMock()
        message.process.__aenter__ = AsyncMock(return_value=None)
        message.process.__aexit__ = AsyncMock(return_value=None)

        # Process message
        await consumer.process_message(message)

        # Verify
        consumer.catalog_service.handle_file_deleted.assert_called_once_with("/test/file.mp3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
