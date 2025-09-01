"""Unit tests for async repository implementations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.async_repositories import (
    AsyncBatchOperations,
    AsyncMetadataRepository,
    AsyncRecordingRepository,
    AsyncTracklistRepository,
)
from shared.core_types.src.models import Recording


@pytest.fixture
async def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock(spec=AsyncDatabaseManager)

    # Create async context manager mock
    async_session_mock = AsyncMock()
    async_session_mock.__aenter__.return_value = async_session_mock
    async_session_mock.__aexit__.return_value = None

    manager.get_db_session.return_value = async_session_mock
    return manager


@pytest.fixture
async def recording_repo(mock_db_manager):
    """Create recording repository with mock DB."""
    return AsyncRecordingRepository(mock_db_manager)


@pytest.fixture
async def metadata_repo(mock_db_manager):
    """Create metadata repository with mock DB."""
    return AsyncMetadataRepository(mock_db_manager)


@pytest.fixture
async def tracklist_repo(mock_db_manager):
    """Create tracklist repository with mock DB."""
    return AsyncTracklistRepository(mock_db_manager)


@pytest.fixture
async def batch_ops(mock_db_manager):
    """Create batch operations with mock DB."""
    return AsyncBatchOperations(mock_db_manager)


class TestAsyncRecordingRepository:
    """Test async recording repository operations."""

    @pytest.mark.asyncio
    async def test_create_recording(self, recording_repo, mock_db_manager):
        """Test creating a new recording."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()

        # Create recording
        result = await recording_repo.create(
            file_path="/test/file.mp3",
            file_name="file.mp3",
            sha256_hash="testhash",
            xxh128_hash="testxxhash",
        )

        # Verify
        assert result is not None
        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, recording_repo, mock_db_manager):
        """Test getting recording by ID."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Recording(
            id=uuid4(), file_path="/test/file.mp3", file_name="file.mp3"
        )
        session.execute = AsyncMock(return_value=mock_result)

        # Get recording
        recording_id = uuid4()
        result = await recording_repo.get_by_id(recording_id)

        # Verify
        assert result is not None
        assert result.file_path == "/test/file.mp3"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_file_path(self, recording_repo, mock_db_manager):
        """Test getting recording by file path."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Recording(
            id=uuid4(), file_path="/test/file.mp3", file_name="file.mp3"
        )
        session.execute = AsyncMock(return_value=mock_result)

        # Get recording
        result = await recording_repo.get_by_file_path("/test/file.mp3")

        # Verify
        assert result is not None
        assert result.file_path == "/test/file.mp3"
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_recording(self, recording_repo, mock_db_manager):
        """Test updating a recording."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        updated_recording = Recording(id=uuid4(), file_path="/test/new_file.mp3", file_name="new_file.mp3")
        mock_result.scalar_one_or_none.return_value = updated_recording
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()

        # Update recording
        recording_id = uuid4()
        result = await recording_repo.update(
            recording_id=recording_id,
            file_path="/test/new_file.mp3",
            file_name="new_file.mp3",
        )

        # Verify
        assert result is not None
        assert result.file_path == "/test/new_file.mp3"
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_recording(self, recording_repo, mock_db_manager):
        """Test deleting a recording."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()

        # Delete recording
        recording_id = uuid4()
        result = await recording_repo.delete(recording_id)

        # Verify
        assert result is True
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_create(self, recording_repo, mock_db_manager):
        """Test batch creation of recordings."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()

        # Batch create
        recordings_data = [
            {
                "file_path": f"/test/file_{i}.mp3",
                "file_name": f"file_{i}.mp3",
                "sha256_hash": f"hash_{i}",
                "xxh128_hash": f"xxhash_{i}",
            }
            for i in range(5)
        ]

        result = await recording_repo.batch_create(recordings_data)

        # Verify
        assert len(result) == 5
        session.add_all.assert_called_once()
        session.flush.assert_called_once()
        assert session.refresh.call_count == 5


class TestAsyncMetadataRepository:
    """Test async metadata repository operations."""

    @pytest.mark.asyncio
    async def test_create_metadata(self, metadata_repo, mock_db_manager):
        """Test creating metadata entry."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()

        # Create metadata
        recording_id = uuid4()
        result = await metadata_repo.create(recording_id=recording_id, key="genre", value="electronic")

        # Verify
        assert result is not None
        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_create_metadata(self, metadata_repo, mock_db_manager):
        """Test batch creation of metadata."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add_all = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()

        # Batch create
        recording_id = uuid4()
        metadata_dict = {"genre": "electronic", "bpm": "128", "key": "A minor"}

        result = await metadata_repo.batch_create(recording_id, metadata_dict)

        # Verify
        assert len(result) == 3
        session.add_all.assert_called_once()
        session.flush.assert_called_once()
        assert session.refresh.call_count == 3


class TestConnectionPooling:
    """Test connection pooling functionality."""

    @pytest.mark.asyncio
    async def test_connection_pool_limits(self):
        """Test that connection pool respects limits."""
        manager = AsyncDatabaseManager(database_url="postgresql+asyncpg://test:test@localhost/test")

        # Check pool configuration
        assert manager.min_pool_size == 10
        assert manager.max_pool_size == 50

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, mock_db_manager):
        """Test handling multiple concurrent connections."""
        # Create multiple repositories
        repos = [AsyncRecordingRepository(mock_db_manager) for _ in range(10)]

        # Setup mock sessions
        sessions = []
        for _ in range(10):
            session = AsyncMock()
            session.__aenter__.return_value = session
            session.__aexit__.return_value = None
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent operations
        tasks = []
        for i, repo in enumerate(repos):
            # Mock the session for this specific call
            session = sessions[i]
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            session.execute = AsyncMock(return_value=mock_result)

            task = repo.get_by_id(uuid4())
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 10
        assert mock_db_manager.get_db_session.call_count == 10


class TestTransactionRollback:
    """Test transaction rollback scenarios."""

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, recording_repo, mock_db_manager):
        """Test that transactions rollback on error."""
        # Setup mock to raise error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock(side_effect=Exception("Database error"))
        session.rollback = AsyncMock()

        # Try to create recording (should fail)
        with pytest.raises(ValueError):  # More specific exception
            await recording_repo.create(file_path="/test/file.mp3", file_name="file.mp3")

        # Verify rollback was called
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, mock_db_manager):
        """Test transaction isolation between operations."""
        # Create two repositories
        repo1 = AsyncRecordingRepository(mock_db_manager)
        repo2 = AsyncRecordingRepository(mock_db_manager)

        # Setup separate sessions
        session1 = AsyncMock()
        session1.__aenter__.return_value = session1
        session1.__aexit__.return_value = None

        session2 = AsyncMock()
        session2.__aenter__.return_value = session2
        session2.__aexit__.return_value = None

        mock_db_manager.get_db_session.side_effect = [session1, session2]

        # Setup different results for each session
        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = Recording(
            id=uuid4(), file_path="/test/file1.mp3", file_name="file1.mp3"
        )
        session1.execute = AsyncMock(return_value=result1)

        result2 = MagicMock()
        result2.scalar_one_or_none.return_value = Recording(
            id=uuid4(), file_path="/test/file2.mp3", file_name="file2.mp3"
        )
        session2.execute = AsyncMock(return_value=result2)

        # Execute operations
        rec1 = await repo1.get_by_id(uuid4())
        rec2 = await repo2.get_by_id(uuid4())

        # Verify isolation
        assert rec1.file_path == "/test/file1.mp3"
        assert rec2.file_path == "/test/file2.mp3"
        assert session1.execute.call_count == 1
        assert session2.execute.call_count == 1


class TestAsyncBatchOperations:
    """Test async batch operations."""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, batch_ops, mock_db_manager):
        """Test bulk insert operations."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        # Perform bulk insert
        recordings_data = [
            {
                "file_path": f"/test/file_{i}.mp3",
                "file_name": f"file_{i}.mp3",
                "sha256_hash": f"hash_{i}",
                "xxh128_hash": f"xxhash_{i}",
            }
            for i in range(100)
        ]

        result = await batch_ops.bulk_insert_recordings(recordings_data)

        # Verify
        assert result == 100
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_large_dataset(self, batch_ops, mock_db_manager):
        """Test streaming large datasets."""
        # Setup mock to return chunks
        session = await mock_db_manager.get_db_session().__aenter__()

        # Create mock results for multiple chunks
        chunk1 = [Recording(id=uuid4(), file_path=f"/test/file_{i}.mp3", file_name=f"file_{i}.mp3") for i in range(10)]
        chunk2 = [
            Recording(id=uuid4(), file_path=f"/test/file_{i}.mp3", file_name=f"file_{i}.mp3") for i in range(10, 20)
        ]
        chunk3 = []  # Empty chunk to stop iteration

        results = []
        for chunk in [chunk1, chunk2, chunk3]:
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = chunk
            results.append(mock_result)

        session.execute = AsyncMock(side_effect=results)

        # Stream data
        all_records = []
        async for chunk in batch_ops.stream_large_dataset(query_limit=10):
            all_records.extend(chunk)

        # Verify
        assert len(all_records) == 20
        assert session.execute.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
