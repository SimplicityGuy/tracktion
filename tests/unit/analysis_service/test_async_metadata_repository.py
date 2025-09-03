"""Comprehensive unit tests for AsyncMetadataRepository class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from services.analysis_service.src.repositories import AsyncMetadataRepository
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import Metadata, Recording


class TestAsyncMetadataRepository:
    """Test async metadata repository operations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        manager = MagicMock(spec=AsyncDatabaseManager)

        # Create async context manager mock
        async_session_mock = AsyncMock()
        async_session_mock.__aenter__ = AsyncMock(return_value=async_session_mock)
        async_session_mock.__aexit__ = AsyncMock(return_value=None)

        manager.get_db_session.return_value = async_session_mock
        return manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        """Create repository with mock DB."""
        return AsyncMetadataRepository(mock_db_manager)

    @pytest.fixture
    def sample_recording_id(self):
        """Create a sample recording ID for testing."""
        return uuid4()

    @pytest.fixture
    def sample_metadata_items(self, sample_recording_id):
        """Create sample metadata items for testing."""
        return [
            Metadata(id=uuid4(), recording_id=sample_recording_id, key="title", value="Sample Track"),
            Metadata(id=uuid4(), recording_id=sample_recording_id, key="artist", value="DJ Example"),
            Metadata(id=uuid4(), recording_id=sample_recording_id, key="album", value="Live Set 2024"),
            Metadata(id=uuid4(), recording_id=sample_recording_id, key="genre", value="Electronic"),
            Metadata(id=uuid4(), recording_id=sample_recording_id, key="bpm", value="128.5"),
        ]

    @pytest.fixture
    def sample_metadata_dict(self):
        """Create sample metadata dictionary for batch creation."""
        return {
            "title": "Test Mix",
            "artist": "DJ Test",
            "album": "Club Nights 2024",
            "genre": "House",
            "year": "2024",
            "bpm": "130.0",
            "key": "C major",
            "energy": "0.85",
            "danceability": "0.92",
            "format": "MP3",
            "bitrate": "320",
            "sample_rate": "44100",
            "channels": "2",
            "duration": "3600.5",
            "file_size": "123456789",
            "encoding": "UTF-8",
            "comment": "Recorded live at Club XYZ",
            "track_number": "1",
            "disc_number": "1",
            "compilation": "false",
            "custom_tag_1": "custom_value_1",
            "unicode_tag": "Test éñcödîñg 测试",
        }

    @pytest.fixture
    def sample_recording(self, sample_recording_id):
        """Create a sample recording for testing."""
        return Recording(
            id=sample_recording_id,
            file_path="/test/sample.mp3",
            file_name="sample.mp3",
            processing_status="completed",
        )

    # Basic CRUD Operations Tests

    @pytest.mark.asyncio
    async def test_get_by_recording_id(self, repository, mock_db_manager, sample_metadata_items):
        """Test getting all metadata for a recording."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = sample_metadata_items
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get metadata
        recording_id = sample_metadata_items[0].recording_id
        result = await repository.get_by_recording_id(recording_id)

        # Verify
        assert len(result) == 5
        assert result[0].key == "title"
        assert result[0].value == "Sample Track"
        assert result[1].key == "artist"
        assert result[1].value == "DJ Example"
        assert all(metadata.recording_id == recording_id for metadata in result)
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_recording_id_empty(self, repository, mock_db_manager):
        """Test getting metadata for recording with no metadata."""
        # Setup mock to return empty list
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get metadata
        result = await repository.get_by_recording_id(uuid4())

        # Verify
        assert result == []
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_key(self, repository, mock_db_manager, sample_metadata_items):
        """Test getting specific metadata by key."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        title_metadata = sample_metadata_items[0]  # "title" metadata
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = title_metadata
        session.execute = AsyncMock(return_value=mock_result)

        # Get specific metadata
        result = await repository.get_by_key(title_metadata.recording_id, "title")

        # Verify
        assert result is not None
        assert result.key == "title"
        assert result.value == "Sample Track"
        assert result.recording_id == title_metadata.recording_id
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_key_not_found(self, repository, mock_db_manager):
        """Test getting non-existent metadata by key."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Get non-existent metadata
        result = await repository.get_by_key(uuid4(), "nonexistent_key")

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_key_case_sensitive(self, repository, mock_db_manager, sample_metadata_items):
        """Test that key lookup is case sensitive."""
        # Setup mock to return None for different case
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Get metadata with different case
        recording_id = sample_metadata_items[0].recording_id
        result = await repository.get_by_key(recording_id, "TITLE")  # uppercase

        # Verify case sensitivity
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_single_item(self, repository, mock_db_manager, sample_recording_id):
        """Test creating a single metadata item via batch method."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create single item
        metadata_dict = {"title": "Single Track"}
        result = await repository.create_batch(sample_recording_id, metadata_dict)

        # Verify
        assert len(result) == 1
        assert result[0].recording_id == sample_recording_id
        assert result[0].key == "title"
        assert result[0].value == "Single Track"

        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_multiple_items(
        self, repository, mock_db_manager, sample_recording_id, sample_metadata_dict
    ):
        """Test creating multiple metadata items in batch."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create batch
        result = await repository.create_batch(sample_recording_id, sample_metadata_dict)

        # Verify
        assert len(result) == len(sample_metadata_dict)
        assert all(metadata.recording_id == sample_recording_id for metadata in result)

        # Check specific items
        keys_created = {metadata.key for metadata in result}
        expected_keys = set(sample_metadata_dict.keys())
        assert keys_created == expected_keys

        # Verify specific values are preserved as strings
        title_metadata = next(m for m in result if m.key == "title")
        assert title_metadata.value == "Test Mix"

        bpm_metadata = next(m for m in result if m.key == "bpm")
        assert bpm_metadata.value == "130.0"

        year_metadata = next(m for m in result if m.key == "year")
        assert year_metadata.value == "2024"  # Should be converted to string

        # Verify database operations
        assert session.add.call_count == len(sample_metadata_dict)
        session.flush.assert_called_once()
        assert session.refresh.call_count == len(sample_metadata_dict)
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_empty_dict(self, repository, mock_db_manager, sample_recording_id):
        """Test creating batch with empty metadata dictionary."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create with empty dict
        result = await repository.create_batch(sample_recording_id, {})

        # Verify
        assert result == []

        # Verify no database operations were performed
        session.add.assert_not_called()
        session.flush.assert_called_once()  # Still called for consistency
        session.refresh.assert_not_called()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_with_unicode_values(self, repository, mock_db_manager, sample_recording_id):
        """Test creating batch with Unicode characters in values."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create batch with Unicode
        unicode_metadata = {
            "title": "Café del Mar",
            "artist": "José González",
            "album": "Música Electrónica",
            "comment": "测试中文字符",
            "emoji_tag": "🎵🎶🎧",
            "special_chars": "äöüß€£¥",
        }

        result = await repository.create_batch(sample_recording_id, unicode_metadata)

        # Verify Unicode handling
        assert len(result) == 6

        title_metadata = next(m for m in result if m.key == "title")
        assert title_metadata.value == "Café del Mar"

        artist_metadata = next(m for m in result if m.key == "artist")
        assert artist_metadata.value == "José González"

        comment_metadata = next(m for m in result if m.key == "comment")
        assert comment_metadata.value == "测试中文字符"

        emoji_metadata = next(m for m in result if m.key == "emoji_tag")
        assert emoji_metadata.value == "🎵🎶🎧"

    @pytest.mark.asyncio
    async def test_create_batch_with_none_values(self, repository, mock_db_manager, sample_recording_id):
        """Test creating batch with None values (should be converted to string)."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create batch with None values
        metadata_with_none = {
            "title": "Test Track",
            "album": None,
            "year": None,
            "comment": "",
        }

        result = await repository.create_batch(sample_recording_id, metadata_with_none)

        # Verify None handling
        assert len(result) == 4

        album_metadata = next(m for m in result if m.key == "album")
        assert album_metadata.value == "None"  # str(None)

        year_metadata = next(m for m in result if m.key == "year")
        assert year_metadata.value == "None"

        comment_metadata = next(m for m in result if m.key == "comment")
        assert comment_metadata.value == ""

    @pytest.mark.asyncio
    async def test_create_batch_with_numeric_values(self, repository, mock_db_manager, sample_recording_id):
        """Test creating batch with various numeric value types."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create batch with numeric values
        numeric_metadata = {
            "bpm": 128.5,
            "year": 2024,
            "track_number": 1,
            "bitrate": 320,
            "sample_rate": 44100,
            "duration": 3600.75,
            "file_size": 123456789,
            "confidence": 0.95,
        }

        result = await repository.create_batch(sample_recording_id, numeric_metadata)

        # Verify numeric conversion to strings
        assert len(result) == 8

        bpm_metadata = next(m for m in result if m.key == "bpm")
        assert bpm_metadata.value == "128.5"

        year_metadata = next(m for m in result if m.key == "year")
        assert year_metadata.value == "2024"

        confidence_metadata = next(m for m in result if m.key == "confidence")
        assert confidence_metadata.value == "0.95"

    @pytest.mark.asyncio
    async def test_create_batch_with_boolean_values(self, repository, mock_db_manager, sample_recording_id):
        """Test creating batch with boolean values."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create batch with boolean values
        boolean_metadata = {
            "is_compilation": True,
            "has_lyrics": False,
            "is_live_recording": True,
            "explicit": False,
        }

        result = await repository.create_batch(sample_recording_id, boolean_metadata)

        # Verify boolean conversion to strings
        assert len(result) == 4

        compilation_metadata = next(m for m in result if m.key == "is_compilation")
        assert compilation_metadata.value == "True"

        lyrics_metadata = next(m for m in result if m.key == "has_lyrics")
        assert lyrics_metadata.value == "False"

    @pytest.mark.asyncio
    async def test_update_by_key(self, repository, mock_db_manager, sample_recording_id):
        """Test updating metadata value by key."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        updated_metadata = Metadata(id=uuid4(), recording_id=sample_recording_id, key="title", value="Updated Title")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = updated_metadata
        session.execute = AsyncMock(return_value=mock_result)

        # Update metadata
        result = await repository.update_by_key(sample_recording_id, "title", "Updated Title")

        # Verify
        assert result is not None
        assert result.key == "title"
        assert result.value == "Updated Title"
        assert result.recording_id == sample_recording_id
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_by_key_not_found(self, repository, mock_db_manager):
        """Test updating non-existent metadata by key."""
        # Setup mock to return None
        session = await mock_db_manager.get_db_session().__aenter__()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_result)

        # Update non-existent metadata
        result = await repository.update_by_key(uuid4(), "nonexistent_key", "New Value")

        # Verify
        assert result is None
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_by_key_unicode_value(self, repository, mock_db_manager, sample_recording_id):
        """Test updating metadata with Unicode value."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        updated_metadata = Metadata(
            id=uuid4(), recording_id=sample_recording_id, key="title", value="Cañón de la Música 测试"
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = updated_metadata
        session.execute = AsyncMock(return_value=mock_result)

        # Update with Unicode value
        result = await repository.update_by_key(sample_recording_id, "title", "Cañón de la Música 测试")

        # Verify Unicode handling
        assert result is not None
        assert result.value == "Cañón de la Música 测试"

    @pytest.mark.asyncio
    async def test_update_by_key_empty_value(self, repository, mock_db_manager, sample_recording_id):
        """Test updating metadata with empty string value."""
        # Setup mock
        session = await mock_db_manager.get_db_session().__aenter__()
        updated_metadata = Metadata(id=uuid4(), recording_id=sample_recording_id, key="comment", value="")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = updated_metadata
        session.execute = AsyncMock(return_value=mock_result)

        # Update with empty value
        result = await repository.update_by_key(sample_recording_id, "comment", "")

        # Verify empty value handling
        assert result is not None
        assert result.value == ""

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_get_by_recording_id_database_error(self, repository, mock_db_manager):
        """Test handling database errors during retrieval."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Database connection failed"))

        # Attempt to get (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_recording_id(uuid4())

    @pytest.mark.asyncio
    async def test_get_by_key_database_error(self, repository, mock_db_manager):
        """Test handling database errors during key lookup."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Query failed"))

        # Attempt to get by key (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.get_by_key(uuid4(), "title")

    @pytest.mark.asyncio
    async def test_create_batch_integrity_error(self, repository, mock_db_manager, sample_recording_id):
        """Test handling integrity constraint violations during batch creation."""
        # Setup mock to raise integrity error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock(side_effect=IntegrityError("Duplicate key", "", ""))
        session.rollback = AsyncMock()

        # Attempt to create (should raise exception)
        with pytest.raises(IntegrityError):
            await repository.create_batch(sample_recording_id, {"title": "Test"})

    @pytest.mark.asyncio
    async def test_create_batch_database_error(self, repository, mock_db_manager, sample_recording_id):
        """Test handling database errors during batch creation."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.commit = AsyncMock(side_effect=SQLAlchemyError("Connection lost"))

        # Attempt to create (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.create_batch(sample_recording_id, {"title": "Test"})

    @pytest.mark.asyncio
    async def test_update_by_key_database_error(self, repository, mock_db_manager):
        """Test handling database errors during update."""
        # Setup mock to raise database error
        session = await mock_db_manager.get_db_session().__aenter__()
        session.execute = AsyncMock(side_effect=SQLAlchemyError("Update failed"))

        # Attempt to update (should raise exception)
        with pytest.raises(SQLAlchemyError):
            await repository.update_by_key(uuid4(), "title", "New Value")

    # Concurrency and Transaction Tests

    @pytest.mark.asyncio
    async def test_concurrent_creates(self, mock_db_manager, sample_recording_id):
        """Test handling concurrent create operations."""
        # Create multiple repositories
        repos = [AsyncMetadataRepository(mock_db_manager) for _ in range(5)]

        # Setup separate sessions
        sessions = []
        for _i in range(5):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)
            session.add = MagicMock()
            session.flush = AsyncMock()
            session.refresh = AsyncMock()
            session.commit = AsyncMock()
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent creates
        recording_ids = [uuid4() for _ in range(5)]
        tasks = []
        for i, (repo, recording_id) in enumerate(zip(repos, recording_ids, strict=False)):
            metadata_dict = {f"key_{i}": f"value_{i}"}
            task = repo.create_batch(recording_id, metadata_dict)
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert len(result) == 1
            assert result[0].recording_id == recording_ids[i]
            assert result[0].key == f"key_{i}"
            assert result[0].value == f"value_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, mock_db_manager):
        """Test handling concurrent read operations."""
        # Create multiple repositories
        repos = [AsyncMetadataRepository(mock_db_manager) for _ in range(10)]

        # Setup sessions with different mock results
        sessions = []
        expected_results = []
        for i in range(10):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            # Create unique metadata for each session
            metadata_items = [Metadata(id=uuid4(), recording_id=uuid4(), key=f"key_{i}", value=f"value_{i}")]
            expected_results.append(metadata_items)

            mock_result = MagicMock()
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = metadata_items
            mock_result.scalars.return_value = mock_scalars
            session.execute = AsyncMock(return_value=mock_result)
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent reads
        recording_ids = [uuid4() for _ in range(10)]
        tasks = []
        for i, repo in enumerate(repos):
            task = repo.get_by_recording_id(recording_ids[i])
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 10
        for i, result in enumerate(results):
            assert len(result) == 1
            assert result[0].key == f"key_{i}"
            assert result[0].value == f"value_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, mock_db_manager):
        """Test handling concurrent update operations."""
        # Create multiple repositories
        repos = [AsyncMetadataRepository(mock_db_manager) for _ in range(5)]

        # Setup sessions with different update results
        sessions = []
        recording_ids = [uuid4() for _ in range(5)]
        for i in range(5):
            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            updated_metadata = Metadata(
                id=uuid4(), recording_id=recording_ids[i], key="title", value=f"Updated Title {i}"
            )

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = updated_metadata
            session.execute = AsyncMock(return_value=mock_result)
            sessions.append(session)

        mock_db_manager.get_db_session.side_effect = sessions

        # Execute concurrent updates
        tasks = []
        for i, (repo, recording_id) in enumerate(zip(repos, recording_ids, strict=False)):
            task = repo.update_by_key(recording_id, "title", f"Updated Title {i}")
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result is not None
            assert result.value == f"Updated Title {i}"

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, mock_db_manager):
        """Test transaction isolation between operations."""
        # Create two repositories
        repo1 = AsyncMetadataRepository(mock_db_manager)
        repo2 = AsyncMetadataRepository(mock_db_manager)

        # Setup separate sessions
        session1 = AsyncMock()
        session1.__aenter__ = AsyncMock(return_value=session1)
        session1.__aexit__ = AsyncMock(return_value=None)

        session2 = AsyncMock()
        session2.__aenter__ = AsyncMock(return_value=session2)
        session2.__aexit__ = AsyncMock(return_value=None)

        mock_db_manager.get_db_session.side_effect = [session1, session2]

        # Setup different results for each session
        metadata1 = Metadata(id=uuid4(), recording_id=uuid4(), key="title", value="Result 1")
        metadata2 = None  # Not found

        result1 = MagicMock()
        result1.scalar_one_or_none.return_value = metadata1
        session1.execute = AsyncMock(return_value=result1)

        result2 = MagicMock()
        result2.scalar_one_or_none.return_value = metadata2
        session2.execute = AsyncMock(return_value=result2)

        # Execute operations in different transactions
        recording_id1 = uuid4()
        recording_id2 = uuid4()

        get_result1 = await repo1.get_by_key(recording_id1, "title")
        get_result2 = await repo2.get_by_key(recording_id2, "title")

        # Verify isolation - each got different results
        assert get_result1 is not None
        assert get_result1.value == "Result 1"
        assert get_result2 is None
        assert session1.execute.call_count == 1
        assert session2.execute.call_count == 1

    # Data Validation and Edge Cases Tests

    @pytest.mark.asyncio
    async def test_create_batch_with_very_long_keys(self, repository, mock_db_manager, sample_recording_id):
        """Test creating metadata with very long keys."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create metadata with long keys
        long_key = "a" * 250  # Just under the 255 limit
        very_long_key = "b" * 300  # Over the 255 limit

        long_key_metadata = {
            long_key: "value for long key",
            very_long_key: "value for very long key",
        }

        result = await repository.create_batch(sample_recording_id, long_key_metadata)

        # Verify creation (database constraint would handle validation)
        assert len(result) == 2
        assert any(m.key == long_key for m in result)
        assert any(m.key == very_long_key for m in result)

    @pytest.mark.asyncio
    async def test_create_batch_with_very_long_values(self, repository, mock_db_manager, sample_recording_id):
        """Test creating metadata with very long values."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create metadata with very long value
        very_long_value = "x" * 10000  # Very long text

        long_value_metadata = {
            "description": very_long_value,
            "lyrics": "Normal length lyrics",
        }

        result = await repository.create_batch(sample_recording_id, long_value_metadata)

        # Verify creation with long value
        assert len(result) == 2
        description_metadata = next(m for m in result if m.key == "description")
        assert description_metadata.value == very_long_value

    @pytest.mark.asyncio
    async def test_create_batch_with_special_characters_in_keys(self, repository, mock_db_manager, sample_recording_id):
        """Test creating metadata with special characters in keys."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create metadata with special character keys
        special_key_metadata = {
            "key-with-dashes": "value1",
            "key_with_underscores": "value2",
            "key.with.dots": "value3",
            "key with spaces": "value4",
            "key:with:colons": "value5",
            "key/with/slashes": "value6",
            "key(with)parentheses": "value7",
            "key[with]brackets": "value8",
            "key{with}braces": "value9",
            "key@with#symbols$": "value10",
        }

        result = await repository.create_batch(sample_recording_id, special_key_metadata)

        # Verify all special character keys are handled
        assert len(result) == 10
        keys_created = {m.key for m in result}
        expected_keys = set(special_key_metadata.keys())
        assert keys_created == expected_keys

    @pytest.mark.asyncio
    async def test_get_by_recording_id_large_dataset(self, repository, mock_db_manager):
        """Test getting metadata for recording with large number of metadata items."""
        # Setup mock with large dataset
        session = await mock_db_manager.get_db_session().__aenter__()
        recording_id = uuid4()

        # Create 1000 metadata items
        large_metadata_list = []
        for i in range(1000):
            metadata = Metadata(id=uuid4(), recording_id=recording_id, key=f"key_{i:04d}", value=f"value_{i:04d}")
            large_metadata_list.append(metadata)

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = large_metadata_list
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get large dataset
        result = await repository.get_by_recording_id(recording_id)

        # Verify large dataset handling
        assert len(result) == 1000
        assert all(m.recording_id == recording_id for m in result)
        assert result[0].key == "key_0000"
        assert result[999].key == "key_0999"

    @pytest.mark.asyncio
    async def test_metadata_key_value_consistency(self, repository, mock_db_manager, sample_recording_id):
        """Test that metadata key-value pairs remain consistent through operations."""
        # Setup mock session for create
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Test data with various types that should be stringified
        test_metadata = {
            "string_key": "string_value",
            "int_key": 42,
            "float_key": 3.14159,
            "bool_key": True,
            "none_key": None,
            "empty_key": "",
            "unicode_key": "unicode_value_éñ测试",
        }

        result = await repository.create_batch(sample_recording_id, test_metadata)

        # Verify key-value consistency and type conversion
        assert len(result) == 7

        string_meta = next(m for m in result if m.key == "string_key")
        assert string_meta.value == "string_value"

        int_meta = next(m for m in result if m.key == "int_key")
        assert int_meta.value == "42"

        float_meta = next(m for m in result if m.key == "float_key")
        assert float_meta.value == "3.14159"

        bool_meta = next(m for m in result if m.key == "bool_key")
        assert bool_meta.value == "True"

        none_meta = next(m for m in result if m.key == "none_key")
        assert none_meta.value == "None"

        empty_meta = next(m for m in result if m.key == "empty_key")
        assert empty_meta.value == ""

        unicode_meta = next(m for m in result if m.key == "unicode_key")
        assert unicode_meta.value == "unicode_value_éñ测试"

    @pytest.mark.asyncio
    async def test_session_context_manager_exception_handling(self, mock_db_manager):
        """Test that session context manager properly handles exceptions."""
        repository = AsyncMetadataRepository(mock_db_manager)

        # Setup mock session that raises exception
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.execute = AsyncMock(side_effect=Exception("Connection lost"))

        mock_db_manager.get_db_session.return_value = session

        # Verify exception is propagated and context manager cleanup occurs
        with pytest.raises(Exception, match="Connection lost"):
            await repository.get_by_recording_id(uuid4())

        # Verify context manager cleanup was called
        session.__aenter__.assert_called_once()
        session.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization with database manager."""
        mock_manager = MagicMock(spec=AsyncDatabaseManager)

        repository = AsyncMetadataRepository(mock_manager)

        assert repository.db is mock_manager
        assert hasattr(repository, "db")

    # Metadata-Specific Edge Cases

    @pytest.mark.asyncio
    async def test_metadata_encoding_edge_cases(self, repository, mock_db_manager, sample_recording_id):
        """Test metadata with various encoding edge cases."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create metadata with encoding edge cases
        encoding_metadata = {
            "latin1": "café",
            "utf8": "测试",
            "emoji": "🎵🎶🎧🎤",
            "mixed_scripts": "Hello мир 世界 🌍",
            "control_chars": "line1\nline2\ttab",
            "quotes": 'single "double" quotes',
            "backslashes": "path\\to\\file",
            "url_encoded": "hello%20world",
            "html_entities": "&lt;tag&gt;",
            "sql_injection": "'; DROP TABLE users; --",
        }

        result = await repository.create_batch(sample_recording_id, encoding_metadata)

        # Verify encoding handling
        assert len(result) == 10

        # Check specific encoding cases
        emoji_meta = next(m for m in result if m.key == "emoji")
        assert emoji_meta.value == "🎵🎶🎧🎤"

        mixed_meta = next(m for m in result if m.key == "mixed_scripts")
        assert mixed_meta.value == "Hello мир 世界 🌍"

        sql_meta = next(m for m in result if m.key == "sql_injection")
        assert sql_meta.value == "'; DROP TABLE users; --"

    @pytest.mark.asyncio
    async def test_metadata_audio_specific_tags(self, repository, mock_db_manager, sample_recording_id):
        """Test metadata with audio-specific tag formats."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create audio-specific metadata
        audio_metadata = {
            # ID3v2 tags
            "TIT2": "Title",
            "TPE1": "Artist",
            "TALB": "Album",
            "TDRC": "2024",
            "TCON": "Electronic",
            "TRCK": "1/12",
            "TPE2": "Album Artist",
            "TPOS": "1/2",
            # Vorbis comments
            "TITLE": "Vorbis Title",
            "ARTIST": "Vorbis Artist",
            "ALBUMARTIST": "Vorbis Album Artist",
            "DATE": "2024-01-15",
            "TRACKNUMBER": "01",
            "DISCNUMBER": "01",
            # Technical metadata
            "BITRATE": "320000",
            "SAMPLERATE": "44100",
            "CHANNELS": "2",
            "DURATION": "210.5",
            "CODEC": "MP3",
            "VBR": "false",
            # Custom DJ/Electronic music tags
            "BPM": "128.0",
            "KEY": "Cmaj",
            "INITIALKEY": "8A",
            "ENERGY": "7",
            "DANCEABILITY": "9",
            "CAMELOT": "8A",
            "MIXKEY": "C",
            "CUE_POINTS": "[60.0, 120.0, 180.0]",
            "BEATGRID": "128,128,128",
        }

        result = await repository.create_batch(sample_recording_id, audio_metadata)

        # Verify audio tag handling
        assert len(result) == len(audio_metadata)

        # Check specific audio tags
        bpm_meta = next(m for m in result if m.key == "BPM")
        assert bpm_meta.value == "128.0"

        key_meta = next(m for m in result if m.key == "KEY")
        assert key_meta.value == "Cmaj"

        cue_meta = next(m for m in result if m.key == "CUE_POINTS")
        assert cue_meta.value == "[60.0, 120.0, 180.0]"

    @pytest.mark.asyncio
    async def test_duplicate_key_handling(self, repository, mock_db_manager, sample_recording_id):
        """Test behavior when creating metadata with duplicate keys in same batch."""
        # Setup mock session
        session = await mock_db_manager.get_db_session().__aenter__()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.commit = AsyncMock()

        # Create metadata with duplicate keys (last value should win)
        # Note: Python dict automatically handles duplicates by keeping the last value
        duplicate_metadata = {"title": "Second Title", "artist": "Artist Name"}

        result = await repository.create_batch(sample_recording_id, duplicate_metadata)

        # Verify duplicate handling (Python dict behavior)
        assert len(result) == 2  # Only 2 items due to dict key uniqueness

        # Check that the last value for duplicate key is used
        title_meta = next(m for m in result if m.key == "title")
        assert title_meta.value == "Second Title"

        artist_meta = next(m for m in result if m.key == "artist")
        assert artist_meta.value == "Artist Name"


class TestAsyncMetadataRepositoryIntegration:
    """Integration-style tests that test multiple repository methods together."""

    @pytest.fixture
    def mock_db_manager_integration(self):
        """Create a mock database manager for integration tests."""
        manager = MagicMock(spec=AsyncDatabaseManager)

        # Create async context manager mock that can be reused
        async_session_mock = AsyncMock()
        async_session_mock.__aenter__ = AsyncMock(return_value=async_session_mock)
        async_session_mock.__aexit__ = AsyncMock(return_value=None)
        async_session_mock.add = MagicMock()
        async_session_mock.flush = AsyncMock()
        async_session_mock.refresh = AsyncMock()
        async_session_mock.commit = AsyncMock()

        manager.get_db_session.return_value = async_session_mock
        return manager, async_session_mock

    @pytest.mark.asyncio
    async def test_create_then_get_workflow(self, mock_db_manager_integration):
        """Test complete workflow: create metadata batch then retrieve it."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncMetadataRepository(mock_db_manager)

        recording_id = uuid4()

        # Step 1: Create batch
        metadata_dict = {
            "title": "Test Mix",
            "artist": "DJ Test",
            "bpm": "128.0",
        }

        created_metadata = await repository.create_batch(recording_id, metadata_dict)

        # Verify creation
        assert len(created_metadata) == 3
        assert all(m.recording_id == recording_id for m in created_metadata)

        # Step 2: Setup for get operation
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = created_metadata
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Get the created metadata
        retrieved = await repository.get_by_recording_id(recording_id)

        # Verify retrieval
        assert len(retrieved) == 3
        retrieved_keys = {m.key for m in retrieved}
        assert retrieved_keys == {"title", "artist", "bpm"}

    @pytest.mark.asyncio
    async def test_create_get_update_workflow(self, mock_db_manager_integration):
        """Test workflow: create batch, get specific item, then update it."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncMetadataRepository(mock_db_manager)

        recording_id = uuid4()

        # Step 1: Create batch
        metadata_dict = {"title": "Original Title", "artist": "Original Artist"}
        await repository.create_batch(recording_id, metadata_dict)

        # Step 2: Get specific metadata by key
        title_metadata = Metadata(id=uuid4(), recording_id=recording_id, key="title", value="Original Title")

        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = title_metadata
        session.execute = AsyncMock(return_value=mock_get_result)

        retrieved = await repository.get_by_key(recording_id, "title")
        assert retrieved is not None
        assert retrieved.value == "Original Title"

        # Step 3: Update the metadata
        updated_metadata = Metadata(id=title_metadata.id, recording_id=recording_id, key="title", value="Updated Title")

        mock_update_result = MagicMock()
        mock_update_result.scalar_one_or_none.return_value = updated_metadata
        session.execute = AsyncMock(return_value=mock_update_result)

        updated = await repository.update_by_key(recording_id, "title", "Updated Title")

        # Verify update
        assert updated is not None
        assert updated.value == "Updated Title"

    @pytest.mark.asyncio
    async def test_batch_operations_with_mixed_types(self, mock_db_manager_integration):
        """Test batch operations with mixed data types and subsequent retrieval."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncMetadataRepository(mock_db_manager)

        recording_id = uuid4()

        # Create batch with mixed types
        mixed_metadata = {
            "title": "Mixed Types Test",
            "year": 2024,
            "bpm": 128.5,
            "is_favorite": True,
            "tags": None,
            "energy": 0.85,
            "unicode": "测试éñcödîñg",
            "empty": "",
        }

        created_metadata = await repository.create_batch(recording_id, mixed_metadata)

        # Verify type conversion
        assert len(created_metadata) == 8

        # Setup for retrieval
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = created_metadata
        mock_result.scalars.return_value = mock_scalars
        session.execute = AsyncMock(return_value=mock_result)

        # Retrieve all metadata
        retrieved = await repository.get_by_recording_id(recording_id)

        # Verify all types were converted to strings and preserved
        assert len(retrieved) == 8

        retrieved_dict = {m.key: m.value for m in retrieved}
        assert retrieved_dict["title"] == "Mixed Types Test"
        assert retrieved_dict["year"] == "2024"
        assert retrieved_dict["bpm"] == "128.5"
        assert retrieved_dict["is_favorite"] == "True"
        assert retrieved_dict["tags"] == "None"
        assert retrieved_dict["energy"] == "0.85"
        assert retrieved_dict["unicode"] == "测试éñcödîñg"
        assert retrieved_dict["empty"] == ""

    @pytest.mark.asyncio
    async def test_multiple_recordings_isolation(self, mock_db_manager_integration):
        """Test that metadata for different recordings is properly isolated."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncMetadataRepository(mock_db_manager)

        # Create metadata for two different recordings
        recording_id1 = uuid4()
        recording_id2 = uuid4()

        metadata_1 = {"title": "Recording 1", "artist": "Artist 1"}
        metadata_2 = {"title": "Recording 2", "artist": "Artist 2"}

        # Create both batches
        created_1 = await repository.create_batch(recording_id1, metadata_1)
        created_2 = await repository.create_batch(recording_id2, metadata_2)

        # Setup separate mock results for each recording
        mock_result_1 = MagicMock()
        mock_scalars_1 = MagicMock()
        mock_scalars_1.all.return_value = created_1
        mock_result_1.scalars.return_value = mock_scalars_1

        mock_result_2 = MagicMock()
        mock_scalars_2 = MagicMock()
        mock_scalars_2.all.return_value = created_2
        mock_result_2.scalars.return_value = mock_scalars_2

        # Setup session to return different results based on call
        session.execute = AsyncMock(side_effect=[mock_result_1, mock_result_2])

        # Retrieve metadata for each recording
        retrieved_1 = await repository.get_by_recording_id(recording_id1)
        retrieved_2 = await repository.get_by_recording_id(recording_id2)

        # Verify isolation
        assert len(retrieved_1) == 2
        assert len(retrieved_2) == 2

        # Verify correct metadata returned for each recording
        retrieved_dict_1 = {m.key: m.value for m in retrieved_1}
        retrieved_dict_2 = {m.key: m.value for m in retrieved_2}

        assert retrieved_dict_1["title"] == "Recording 1"
        assert retrieved_dict_1["artist"] == "Artist 1"
        assert retrieved_dict_2["title"] == "Recording 2"
        assert retrieved_dict_2["artist"] == "Artist 2"

    @pytest.mark.asyncio
    async def test_large_batch_create_and_selective_update(self, mock_db_manager_integration):
        """Test creating large batch and then updating selective items."""
        mock_db_manager, session = mock_db_manager_integration
        repository = AsyncMetadataRepository(mock_db_manager)

        recording_id = uuid4()

        # Create large metadata batch (simulating full audio file metadata)
        large_metadata = {
            # Basic tags
            "title": "Large Metadata Test",
            "artist": "Test Artist",
            "album": "Test Album",
            "year": "2024",
            "genre": "Electronic",
            # Technical metadata
            "format": "MP3",
            "bitrate": "320",
            "sample_rate": "44100",
            "channels": "2",
            "duration": "600.5",
            "file_size": "15000000",
            # DJ/Mix metadata
            "bpm": "128.0",
            "key": "C major",
            "energy": "8",
            "mood": "energetic",
            # Custom tags
            **{f"custom_tag_{i}": f"custom_value_{i}" for i in range(20)},
        }

        # Create the batch
        created_metadata = await repository.create_batch(recording_id, large_metadata)

        # Verify large batch creation
        assert len(created_metadata) == 35  # 15 standard + 20 custom

        # Setup for selective updates
        updated_bpm = Metadata(id=uuid4(), recording_id=recording_id, key="bpm", value="130.0")

        updated_mood = Metadata(id=uuid4(), recording_id=recording_id, key="mood", value="uplifting")

        # Setup mock returns for updates
        session.execute = AsyncMock(
            side_effect=[
                MagicMock(scalar_one_or_none=lambda: updated_bpm),
                MagicMock(scalar_one_or_none=lambda: updated_mood),
            ]
        )

        # Update selective items
        bpm_result = await repository.update_by_key(recording_id, "bpm", "130.0")
        mood_result = await repository.update_by_key(recording_id, "mood", "uplifting")

        # Verify selective updates
        assert bpm_result is not None
        assert bpm_result.value == "130.0"
        assert mood_result is not None
        assert mood_result.value == "uplifting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
