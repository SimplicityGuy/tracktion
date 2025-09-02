"""Comprehensive unit tests for repository implementations in the cataloging service."""

import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from services.cataloging_service.src.models.metadata import Metadata
from services.cataloging_service.src.models.recording import Recording
from services.cataloging_service.src.models.tracklist import Tracklist
from services.cataloging_service.src.repositories.base import BaseRepository
from services.cataloging_service.src.repositories.metadata import MetadataRepository
from services.cataloging_service.src.repositories.recording import RecordingRepository
from services.cataloging_service.src.repositories.tracklist import TracklistRepository


class TestBaseRepository:
    """Tests for the BaseRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def base_repository(self, mock_session):
        """Create a BaseRepository instance with Recording model."""
        return BaseRepository(Recording, mock_session)

    @pytest.fixture
    def sample_test_id(self):
        """Sample UUID for testing."""
        return uuid.uuid4()

    @pytest.mark.asyncio
    async def test_create_success(self, base_repository, mock_session):
        """Test successful record creation."""
        # Setup
        mock_instance = MagicMock(spec=Recording)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        with patch.object(Recording, "__new__", return_value=mock_instance):
            # Execute
            result = await base_repository.create(file_path="/test/path.mp3", file_name="test.mp3")

            # Verify
            assert result == mock_instance
            mock_session.add.assert_called_once_with(mock_instance)
            mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, base_repository, mock_session, sample_test_id):
        """Test get_by_id when record exists."""
        # Setup
        mock_instance = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_instance
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await base_repository.get_by_id(sample_test_id)

        # Verify
        assert result == mock_instance
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, base_repository, mock_session, sample_test_id):
        """Test get_by_id when record doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await base_repository.get_by_id(sample_test_id)

        # Verify
        assert result is None
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_all_with_pagination(self, base_repository, mock_session):
        """Test get_all with pagination parameters."""
        # Setup
        mock_instances = [MagicMock(spec=Recording) for _ in range(3)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_instances
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await base_repository.get_all(limit=10, offset=20)

        # Verify
        assert result == mock_instances
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_existing_record(self, base_repository, mock_session, sample_test_id):
        """Test updating an existing record."""
        # Setup
        mock_instance = MagicMock(spec=Recording)
        base_repository.get_by_id = AsyncMock(return_value=mock_instance)
        mock_session.flush = AsyncMock()

        # Execute
        result = await base_repository.update(sample_test_id, file_name="updated_name.mp3")

        # Verify
        assert result == mock_instance
        assert mock_instance.file_name == "updated_name.mp3"
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_nonexistent_record(self, base_repository, mock_session, sample_test_id):
        """Test updating a non-existent record."""
        # Setup
        base_repository.get_by_id = AsyncMock(return_value=None)

        # Execute
        result = await base_repository.update(sample_test_id, file_name="updated_name.mp3")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing_record(self, base_repository, mock_session, sample_test_id):
        """Test deleting an existing record."""
        # Setup
        mock_instance = MagicMock(spec=Recording)
        base_repository.get_by_id = AsyncMock(return_value=mock_instance)
        mock_session.delete = AsyncMock()
        mock_session.flush = AsyncMock()

        # Execute
        result = await base_repository.delete(sample_test_id)

        # Verify
        assert result is True
        mock_session.delete.assert_awaited_once_with(mock_instance)
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_record(self, base_repository, mock_session, sample_test_id):
        """Test deleting a non-existent record."""
        # Setup
        base_repository.get_by_id = AsyncMock(return_value=None)

        # Execute
        result = await base_repository.delete(sample_test_id)

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_count_records(self, base_repository, mock_session):
        """Test counting total records."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await base_repository.count()

        # Verify
        assert result == 42
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_records_none_result(self, base_repository, mock_session):
        """Test counting when result is None."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await base_repository.count()

        # Verify
        assert result == 0


class TestRecordingRepository:
    """Tests for the RecordingRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def recording_repository(self, mock_session):
        """Create a RecordingRepository instance."""
        return RecordingRepository(mock_session)

    @pytest.fixture
    def sample_recording_id(self):
        """Sample recording ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_recording(self, sample_recording_id):
        """Sample recording instance."""
        return Recording(
            id=sample_recording_id,
            file_path="/music/test.mp3",
            file_name="test.mp3",
            sha256_hash="abc123" * 10,
            xxh128_hash="def456" * 5,
            created_at=datetime.now(UTC),
        )

    @pytest.mark.asyncio
    async def test_get_by_file_path_found(self, recording_repository, mock_session):
        """Test get_by_file_path when recording exists."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_by_file_path("/music/test.mp3")

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_file_path_not_found(self, recording_repository, mock_session):
        """Test get_by_file_path when recording doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_by_file_path("/music/nonexistent.mp3")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_sha256_hash_found(self, recording_repository, mock_session):
        """Test get_by_sha256_hash when recording exists."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_by_sha256_hash("abc123" * 10)

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_xxh128_hash_found(self, recording_repository, mock_session):
        """Test get_by_xxh128_hash when recording exists."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_by_xxh128_hash("def456" * 5)

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_with_metadata(self, recording_repository, mock_session, sample_recording_id):
        """Test get_with_metadata loads metadata relationship."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_with_metadata(sample_recording_id)

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()
        # Verify that the query used selectinload for metadata_items

    @pytest.mark.asyncio
    async def test_get_with_tracklists(self, recording_repository, mock_session, sample_recording_id):
        """Test get_with_tracklists loads tracklists relationship."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_with_tracklists(sample_recording_id)

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_with_all_relations(self, recording_repository, mock_session, sample_recording_id):
        """Test get_with_all_relations loads all relationships."""
        # Setup
        mock_recording = MagicMock(spec=Recording)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.get_with_all_relations(sample_recording_id)

        # Verify
        assert result == mock_recording
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_by_file_name(self, recording_repository, mock_session):
        """Test search_by_file_name with pattern matching."""
        # Setup
        mock_recordings = [MagicMock(spec=Recording) for _ in range(3)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_recordings
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await recording_repository.search_by_file_name("test", limit=50)

        # Verify
        assert result == mock_recordings
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inheritance_from_base_repository(self, recording_repository):
        """Test that RecordingRepository inherits from BaseRepository."""
        assert isinstance(recording_repository, BaseRepository)
        assert recording_repository.model == Recording

    @pytest.mark.asyncio
    async def test_database_error_handling(self, recording_repository, mock_session):
        """Test proper handling of database errors."""
        # Setup
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        # Execute & Verify
        with pytest.raises(SQLAlchemyError):
            await recording_repository.get_by_file_path("/music/test.mp3")


class TestMetadataRepository:
    """Tests for the MetadataRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def metadata_repository(self, mock_session):
        """Create a MetadataRepository instance."""
        return MetadataRepository(mock_session)

    @pytest.fixture
    def sample_recording_id(self):
        """Sample recording ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_metadata(self, sample_recording_id):
        """Sample metadata instance."""
        return Metadata(id=uuid.uuid4(), recording_id=sample_recording_id, key="artist", value="Test Artist")

    @pytest.mark.asyncio
    async def test_get_by_recording_id(self, metadata_repository, mock_session, sample_recording_id):
        """Test get_by_recording_id returns all metadata for a recording."""
        # Setup
        mock_metadata = [MagicMock(spec=Metadata) for _ in range(3)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_metadata
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await metadata_repository.get_by_recording_id(sample_recording_id)

        # Verify
        assert result == mock_metadata
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_key_found(self, metadata_repository, mock_session, sample_recording_id):
        """Test get_by_key when metadata exists."""
        # Setup
        mock_metadata = MagicMock(spec=Metadata)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_metadata
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await metadata_repository.get_by_key(sample_recording_id, "artist")

        # Verify
        assert result == mock_metadata
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_key_not_found(self, metadata_repository, mock_session, sample_recording_id):
        """Test get_by_key when metadata doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await metadata_repository.get_by_key(sample_recording_id, "nonexistent")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_new_metadata(self, metadata_repository, mock_session, sample_recording_id):
        """Test upsert creating new metadata."""
        # Setup
        metadata_repository.get_by_key = AsyncMock(return_value=None)
        mock_metadata = MagicMock(spec=Metadata)
        metadata_repository.create = AsyncMock(return_value=mock_metadata)

        # Execute
        result = await metadata_repository.upsert(sample_recording_id, "genre", "Electronic")

        # Verify
        assert result == mock_metadata
        metadata_repository.create.assert_awaited_once_with(
            recording_id=sample_recording_id, key="genre", value="Electronic"
        )

    @pytest.mark.asyncio
    async def test_upsert_existing_metadata(self, metadata_repository, mock_session, sample_recording_id):
        """Test upsert updating existing metadata."""
        # Setup
        mock_existing = MagicMock(spec=Metadata)
        mock_existing.value = "Old Value"
        metadata_repository.get_by_key = AsyncMock(return_value=mock_existing)
        mock_session.flush = AsyncMock()

        # Execute
        result = await metadata_repository.upsert(sample_recording_id, "genre", "New Electronic")

        # Verify
        assert result == mock_existing
        assert mock_existing.value == "New Electronic"
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bulk_create(self, metadata_repository, mock_session, sample_recording_id):
        """Test bulk_create for multiple metadata entries."""
        # Setup
        metadata_dict = {"artist": "Test Artist", "album": "Test Album", "genre": "Electronic"}
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        # Execute
        result = await metadata_repository.bulk_create(sample_recording_id, metadata_dict)

        # Verify
        assert len(result) == 3
        assert mock_session.add.call_count == 3
        mock_session.flush.assert_awaited_once()

        # Verify each metadata entry
        for metadata_entry in result:
            assert isinstance(metadata_entry, Metadata)
            assert metadata_entry.recording_id == sample_recording_id

    @pytest.mark.asyncio
    async def test_delete_by_recording_id(self, metadata_repository, mock_session, sample_recording_id):
        """Test delete_by_recording_id removes all metadata for a recording."""
        # Setup
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await metadata_repository.delete_by_recording_id(sample_recording_id)

        # Verify
        assert result == 5
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_by_key_value(self, metadata_repository, mock_session):
        """Test search_by_key_value with pattern matching."""
        # Setup
        mock_metadata = [MagicMock(spec=Metadata) for _ in range(2)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_metadata
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await metadata_repository.search_by_key_value("artist", "Electronic", limit=50)

        # Verify
        assert result == mock_metadata
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_inheritance_from_base_repository(self, metadata_repository):
        """Test that MetadataRepository inherits from BaseRepository."""
        assert isinstance(metadata_repository, BaseRepository)
        assert metadata_repository.model == Metadata

    @pytest.mark.asyncio
    async def test_bulk_create_empty_dict(self, metadata_repository, mock_session, sample_recording_id):
        """Test bulk_create with empty metadata dictionary."""
        # Setup
        mock_session.flush = AsyncMock()

        # Execute
        result = await metadata_repository.bulk_create(sample_recording_id, {})

        # Verify
        assert result == []
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_database_error_handling(self, metadata_repository, mock_session, sample_recording_id):
        """Test proper handling of database errors."""
        # Setup
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        # Execute & Verify
        with pytest.raises(SQLAlchemyError):
            await metadata_repository.get_by_recording_id(sample_recording_id)


class TestTracklistRepository:
    """Tests for the TracklistRepository class."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def tracklist_repository(self, mock_session):
        """Create a TracklistRepository instance."""
        return TracklistRepository(mock_session)

    @pytest.fixture
    def sample_recording_id(self):
        """Sample recording ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_tracks(self):
        """Sample tracks data."""
        return [
            {"track_number": 1, "title": "Track 1", "artist": "Artist 1", "duration": 180.5},
            {"track_number": 2, "title": "Track 2", "artist": "Artist 2", "duration": 210.3},
        ]

    @pytest.fixture
    def sample_tracklist(self, sample_recording_id, sample_tracks):
        """Sample tracklist instance."""
        return Tracklist(
            id=uuid.uuid4(),
            recording_id=sample_recording_id,
            source="cue",
            cue_file_path="/music/test.cue",
            tracks=sample_tracks,
        )

    @pytest.mark.asyncio
    async def test_get_by_recording_id(self, tracklist_repository, mock_session, sample_recording_id):
        """Test get_by_recording_id returns all tracklists for a recording."""
        # Setup
        mock_tracklists = [MagicMock(spec=Tracklist) for _ in range(2)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_tracklists
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await tracklist_repository.get_by_recording_id(sample_recording_id)

        # Verify
        assert result == mock_tracklists
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_source_found(self, tracklist_repository, mock_session, sample_recording_id):
        """Test get_by_source when tracklist exists."""
        # Setup
        mock_tracklist = MagicMock(spec=Tracklist)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_tracklist
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await tracklist_repository.get_by_source(sample_recording_id, "cue")

        # Verify
        assert result == mock_tracklist
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_source_not_found(self, tracklist_repository, mock_session, sample_recording_id):
        """Test get_by_source when tracklist doesn't exist."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await tracklist_repository.get_by_source(sample_recording_id, "nonexistent")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_new_tracklist(self, tracklist_repository, mock_session, sample_recording_id, sample_tracks):
        """Test upsert creating new tracklist."""
        # Setup
        tracklist_repository.get_by_source = AsyncMock(return_value=None)
        mock_tracklist = MagicMock(spec=Tracklist)
        tracklist_repository.create = AsyncMock(return_value=mock_tracklist)

        # Execute
        result = await tracklist_repository.upsert(
            sample_recording_id, "metadata", sample_tracks, cue_file_path="/music/test.cue"
        )

        # Verify
        assert result == mock_tracklist
        tracklist_repository.create.assert_awaited_once_with(
            recording_id=sample_recording_id, source="metadata", tracks=sample_tracks, cue_file_path="/music/test.cue"
        )

    @pytest.mark.asyncio
    async def test_upsert_existing_tracklist(
        self, tracklist_repository, mock_session, sample_recording_id, sample_tracks
    ):
        """Test upsert updating existing tracklist."""
        # Setup
        mock_existing = MagicMock(spec=Tracklist)
        mock_existing.tracks = []
        mock_existing.cue_file_path = None
        tracklist_repository.get_by_source = AsyncMock(return_value=mock_existing)
        mock_session.flush = AsyncMock()

        # Execute
        result = await tracklist_repository.upsert(
            sample_recording_id, "metadata", sample_tracks, cue_file_path="/music/new.cue"
        )

        # Verify
        assert result == mock_existing
        assert mock_existing.tracks == sample_tracks
        assert mock_existing.cue_file_path == "/music/new.cue"
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_existing_tracklist_without_cue_path(
        self, tracklist_repository, mock_session, sample_recording_id, sample_tracks
    ):
        """Test upsert updating existing tracklist without changing cue_file_path."""
        # Setup
        mock_existing = MagicMock(spec=Tracklist)
        mock_existing.tracks = []
        mock_existing.cue_file_path = "/music/original.cue"
        tracklist_repository.get_by_source = AsyncMock(return_value=mock_existing)
        mock_session.flush = AsyncMock()

        # Execute
        result = await tracklist_repository.upsert(sample_recording_id, "metadata", sample_tracks, cue_file_path=None)

        # Verify
        assert result == mock_existing
        assert mock_existing.tracks == sample_tracks
        assert mock_existing.cue_file_path == "/music/original.cue"  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_delete_by_recording_id(self, tracklist_repository, mock_session, sample_recording_id):
        """Test delete_by_recording_id removes all tracklists for a recording."""
        # Setup
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await tracklist_repository.delete_by_recording_id(sample_recording_id)

        # Verify
        assert result == 3
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_by_track_title(self, tracklist_repository, mock_session):
        """Test search_by_track_title using JSONB query."""
        # Setup
        mock_tracklists = [MagicMock(spec=Tracklist) for _ in range(2)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_tracklists
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Execute
        result = await tracklist_repository.search_by_track_title("Electronic", limit=25)

        # Verify
        assert result == mock_tracklists
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_tracks_by_recording_id(self, tracklist_repository, sample_recording_id, sample_tracks):
        """Test get_tracks_by_recording_id combines all tracks from all tracklists."""
        # Setup
        mock_tracklist1 = MagicMock(spec=Tracklist)
        mock_tracklist1.tracks = sample_tracks[:1]  # First track
        mock_tracklist2 = MagicMock(spec=Tracklist)
        mock_tracklist2.tracks = sample_tracks[1:]  # Second track

        tracklist_repository.get_by_recording_id = AsyncMock(return_value=[mock_tracklist1, mock_tracklist2])

        # Execute
        result = await tracklist_repository.get_tracks_by_recording_id(sample_recording_id)

        # Verify
        assert len(result) == 2
        assert result == sample_tracks

    @pytest.mark.asyncio
    async def test_get_tracks_by_recording_id_no_tracklists(self, tracklist_repository, sample_recording_id):
        """Test get_tracks_by_recording_id when no tracklists exist."""
        # Setup
        tracklist_repository.get_by_recording_id = AsyncMock(return_value=[])

        # Execute
        result = await tracklist_repository.get_tracks_by_recording_id(sample_recording_id)

        # Verify
        assert result == []

    @pytest.mark.asyncio
    async def test_inheritance_from_base_repository(self, tracklist_repository):
        """Test that TracklistRepository inherits from BaseRepository."""
        assert isinstance(tracklist_repository, BaseRepository)
        assert tracklist_repository.model == Tracklist

    @pytest.mark.asyncio
    async def test_database_error_handling(self, tracklist_repository, mock_session, sample_recording_id):
        """Test proper handling of database errors."""
        # Setup
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        # Execute & Verify
        with pytest.raises(SQLAlchemyError):
            await tracklist_repository.get_by_recording_id(sample_recording_id)


class TestRepositoryIntegration:
    """Integration tests for repository interactions."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def all_repositories(self, mock_session):
        """All repository instances."""
        return {
            "recording": RecordingRepository(mock_session),
            "metadata": MetadataRepository(mock_session),
            "tracklist": TracklistRepository(mock_session),
        }

    @pytest.mark.asyncio
    async def test_repository_initialization(self, all_repositories):
        """Test that all repositories are properly initialized."""
        recording_repo = all_repositories["recording"]
        metadata_repo = all_repositories["metadata"]
        tracklist_repo = all_repositories["tracklist"]

        assert recording_repo.model == Recording
        assert metadata_repo.model == Metadata
        assert tracklist_repo.model == Tracklist

        # All should inherit from BaseRepository
        assert isinstance(recording_repo, BaseRepository)
        assert isinstance(metadata_repo, BaseRepository)
        assert isinstance(tracklist_repo, BaseRepository)

    @pytest.mark.asyncio
    async def test_session_sharing(self, mock_session):
        """Test that repositories can share the same session."""
        recording_repo = RecordingRepository(mock_session)
        metadata_repo = MetadataRepository(mock_session)

        assert recording_repo.session is metadata_repo.session

    @pytest.mark.asyncio
    async def test_cascade_operations_simulation(self, all_repositories, mock_session):
        """Test simulated cascade operations between repositories."""
        recording_id = uuid.uuid4()

        # Mock successful deletion
        all_repositories["metadata"].delete_by_recording_id = AsyncMock(return_value=5)
        all_repositories["tracklist"].delete_by_recording_id = AsyncMock(return_value=2)
        all_repositories["recording"].delete = AsyncMock(return_value=True)

        # Simulate cascade delete
        metadata_deleted = await all_repositories["metadata"].delete_by_recording_id(recording_id)
        tracklist_deleted = await all_repositories["tracklist"].delete_by_recording_id(recording_id)
        recording_deleted = await all_repositories["recording"].delete(recording_id)

        assert metadata_deleted == 5
        assert tracklist_deleted == 2
        assert recording_deleted is True

    @pytest.mark.asyncio
    async def test_transaction_simulation(self, all_repositories, mock_session):
        """Test simulated transaction behavior across repositories."""
        recording_id = uuid.uuid4()

        # Setup transaction-like behavior
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        try:
            # Simulate creating recording with metadata and tracklist
            recording_repo = all_repositories["recording"]
            metadata_repo = all_repositories["metadata"]
            tracklist_repo = all_repositories["tracklist"]

            # Mock successful operations
            recording_repo.create = AsyncMock(return_value=MagicMock())
            metadata_repo.bulk_create = AsyncMock(return_value=[MagicMock(), MagicMock()])
            tracklist_repo.create = AsyncMock(return_value=MagicMock())

            # Execute operations
            await recording_repo.create(file_path="/music/test.mp3", file_name="test.mp3")
            await metadata_repo.bulk_create(recording_id, {"artist": "Test", "album": "Test Album"})
            await tracklist_repo.create(recording_id=recording_id, source="metadata", tracks=[])

            # Verify operations were called
            recording_repo.create.assert_awaited_once()
            metadata_repo.bulk_create.assert_awaited_once()
            tracklist_repo.create.assert_awaited_once()

        except Exception:
            await mock_session.rollback()
            raise


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling scenarios."""

    @pytest.fixture
    def mock_session(self):
        """Mock async session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_large_pagination_parameters(self, mock_session):
        """Test repositories handle large pagination parameters."""
        base_repo = BaseRepository(Recording, mock_session)

        # Setup
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        # Test very large limit
        result = await base_repo.get_all(limit=1000000, offset=999999)

        assert result == []
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_uuid_handling(self, mock_session):
        """Test repositories handle invalid UUID gracefully."""
        recording_repo = RecordingRepository(mock_session)

        # Test with invalid UUID format - this should raise an exception at the calling level
        # The repository itself doesn't validate UUIDs, that's handled by the UUID type
        invalid_id = "not-a-uuid"

        # This would normally be caught by type checking or UUID validation
        # We're testing that the repository doesn't crash on database operations
        mock_session.execute = AsyncMock(side_effect=ValueError("Invalid UUID"))

        with pytest.raises(ValueError):
            await recording_repo.get_by_id(invalid_id)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_empty_string_searches(self, mock_session):
        """Test repositories handle empty string searches."""
        recording_repo = RecordingRepository(mock_session)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await recording_repo.search_by_file_name("")

        assert result == []

    @pytest.mark.asyncio
    async def test_none_value_handling_in_updates(self, mock_session):
        """Test repositories handle None values in updates correctly."""
        base_repo = BaseRepository(Recording, mock_session)
        sample_id = uuid.uuid4()

        # Setup mock instance
        mock_instance = MagicMock(spec=Recording)
        base_repo.get_by_id = AsyncMock(return_value=mock_instance)
        mock_session.flush = AsyncMock()

        # Update with None values
        result = await base_repo.update(sample_id, file_name=None, file_path="test")

        assert result == mock_instance
        assert mock_instance.file_name is None
        assert mock_instance.file_path == "test"

    @pytest.mark.asyncio
    async def test_concurrent_operations_simulation(self, mock_session):
        """Test simulated concurrent operations on repositories."""

        metadata_repo = MetadataRepository(mock_session)
        recording_id = uuid.uuid4()

        # Mock responses for concurrent operations
        mock_session.execute = AsyncMock()
        mock_session.flush = AsyncMock()
        metadata_repo.create = AsyncMock()

        # Simulate concurrent metadata creation
        tasks = []
        for i in range(5):
            task = metadata_repo.create(recording_id=recording_id, key=f"key_{i}", value=f"value_{i}")
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (in this mock scenario)
        assert len(results) == 5
        assert metadata_repo.create.await_count == 5

    @pytest.mark.asyncio
    async def test_jsonb_operations_error_handling(self, mock_session):
        """Test JSONB operations handle errors gracefully."""
        tracklist_repo = TracklistRepository(mock_session)

        # Simulate PostgreSQL JSONB error
        mock_session.execute = AsyncMock(side_effect=SQLAlchemyError("JSONB operation failed"))

        with pytest.raises(SQLAlchemyError):
            await tracklist_repo.search_by_track_title("test")

    @pytest.mark.asyncio
    async def test_relationship_loading_edge_cases(self, mock_session):
        """Test relationship loading with edge cases."""
        recording_repo = RecordingRepository(mock_session)
        recording_id = uuid.uuid4()

        # Test when recording exists but has no relationships
        mock_recording = MagicMock(spec=Recording)
        mock_recording.metadata_items = []
        mock_recording.tracklists = []
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_recording
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await recording_repo.get_with_all_relations(recording_id)

        assert result == mock_recording
        assert result.metadata_items == []
        assert result.tracklists == []
