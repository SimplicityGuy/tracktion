"""Unit tests for repository pattern implementations."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from shared.core_types.src.models import Metadata, Recording, Tracklist
from shared.core_types.src.repositories import MetadataRepository, RecordingRepository, TracklistRepository


class TestRecordingRepository:
    """Test cases for RecordingRepository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock()
        session = MagicMock()
        db_manager.get_db_session.return_value.__enter__ = Mock(return_value=session)
        db_manager.get_db_session.return_value.__exit__ = Mock(return_value=None)
        return db_manager, session

    @pytest.fixture
    def repository(self, mock_db_manager):
        """Create a repository instance with mock database."""
        db_manager, _ = mock_db_manager
        return RecordingRepository(db_manager)

    def test_create_recording(self, repository, mock_db_manager):
        """Test creating a new recording."""
        _, session = mock_db_manager

        # Call create method
        repository.create(
            file_path="/path/to/file.wav",
            file_name="file.wav",
            sha256_hash="abc123",
            xxh128_hash="def456",
        )

        # Verify session methods were called
        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()

        # Verify the recording was created with correct attributes
        recording_arg = session.add.call_args[0][0]
        assert recording_arg.file_path == "/path/to/file.wav"
        assert recording_arg.file_name == "file.wav"
        assert recording_arg.sha256_hash == "abc123"
        assert recording_arg.xxh128_hash == "def456"

    def test_get_by_id(self, repository, mock_db_manager):
        """Test getting a recording by ID."""
        _, session = mock_db_manager
        mock_recording = Mock(spec=Recording)
        mock_recording.id = uuid4()

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_recording
        session.query.return_value = mock_query

        # Call get_by_id
        recording_id = uuid4()
        result = repository.get_by_id(recording_id)

        # Verify query was constructed correctly
        session.query.assert_called_once_with(Recording)
        assert result == mock_recording

    def test_get_by_id_not_found(self, repository, mock_db_manager):
        """Test getting a recording by ID when not found."""
        _, session = mock_db_manager

        # Setup mock query chain to return None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        session.query.return_value = mock_query

        # Call get_by_id
        recording_id = uuid4()
        result = repository.get_by_id(recording_id)

        assert result is None

    def test_get_by_file_path(self, repository, mock_db_manager):
        """Test getting a recording by file path."""
        _, session = mock_db_manager
        mock_recording = Mock(spec=Recording)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_recording
        session.query.return_value = mock_query

        # Call get_by_file_path
        result = repository.get_by_file_path("/path/to/file.wav")

        # Verify query was constructed correctly
        session.query.assert_called_once_with(Recording)
        assert result == mock_recording

    def test_get_all_with_pagination(self, repository, mock_db_manager):
        """Test getting all recordings with pagination."""
        _, session = mock_db_manager
        mock_recordings = [Mock(spec=Recording) for _ in range(3)]

        # Setup mock query chain
        mock_query = Mock()
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_recordings
        session.query.return_value = mock_query

        # Call get_all with pagination
        result = repository.get_all(limit=10, offset=5)

        # Verify pagination was applied
        mock_query.offset.assert_called_once_with(5)
        mock_query.limit.assert_called_once_with(10)
        assert result == mock_recordings

    def test_update_recording(self, repository, mock_db_manager):
        """Test updating a recording."""
        _, session = mock_db_manager
        mock_recording = Mock(spec=Recording)
        mock_recording.file_name = "old_name.wav"

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_recording
        session.query.return_value = mock_query

        # Call update
        recording_id = uuid4()
        result = repository.update(recording_id, file_name="new_name.wav")

        # Verify recording was updated
        assert mock_recording.file_name == "new_name.wav"
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        assert result == mock_recording

    def test_update_recording_not_found(self, repository, mock_db_manager):
        """Test updating a recording that doesn't exist."""
        _, session = mock_db_manager

        # Setup mock query chain to return None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        session.query.return_value = mock_query

        # Call update
        recording_id = uuid4()
        result = repository.update(recording_id, file_name="new_name.wav")

        assert result is None
        session.flush.assert_not_called()

    def test_delete_recording(self, repository, mock_db_manager):
        """Test deleting a recording."""
        _, session = mock_db_manager
        mock_recording = Mock(spec=Recording)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_recording
        session.query.return_value = mock_query

        # Call delete
        recording_id = uuid4()
        result = repository.delete(recording_id)

        # Verify recording was deleted
        session.delete.assert_called_once_with(mock_recording)
        assert result is True

    def test_delete_recording_not_found(self, repository, mock_db_manager):
        """Test deleting a recording that doesn't exist."""
        _, session = mock_db_manager

        # Setup mock query chain to return None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        session.query.return_value = mock_query

        # Call delete
        recording_id = uuid4()
        result = repository.delete(recording_id)

        session.delete.assert_not_called()
        assert result is False

    def test_bulk_create(self, repository, mock_db_manager):
        """Test bulk creating recordings."""
        _, session = mock_db_manager

        recordings_data = [
            {"file_path": "/path1.wav", "file_name": "file1.wav"},
            {"file_path": "/path2.wav", "file_name": "file2.wav"},
        ]

        # Call bulk_create
        result = repository.bulk_create(recordings_data)

        # Verify bulk_save_objects was called
        session.bulk_save_objects.assert_called_once()
        assert len(result) == 2


class TestMetadataRepository:
    """Test cases for MetadataRepository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock()
        session = MagicMock()
        db_manager.get_db_session.return_value.__enter__ = Mock(return_value=session)
        db_manager.get_db_session.return_value.__exit__ = Mock(return_value=None)
        return db_manager, session

    @pytest.fixture
    def repository(self, mock_db_manager):
        """Create a repository instance with mock database."""
        db_manager, _ = mock_db_manager
        return MetadataRepository(db_manager)

    def test_create_metadata(self, repository, mock_db_manager):
        """Test creating metadata."""
        _, session = mock_db_manager
        recording_id = uuid4()

        # Call create method
        repository.create(recording_id=recording_id, key="artist", value="Test Artist")

        # Verify session methods were called
        session.add.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()

        # Verify metadata was created with correct attributes
        metadata_arg = session.add.call_args[0][0]
        assert metadata_arg.recording_id == recording_id
        assert metadata_arg.key == "artist"
        assert metadata_arg.value == "Test Artist"

    def test_get_by_recording(self, repository, mock_db_manager):
        """Test getting metadata by recording."""
        _, session = mock_db_manager
        mock_metadata = [Mock(spec=Metadata) for _ in range(3)]

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_metadata
        session.query.return_value = mock_query

        # Call get_by_recording
        recording_id = uuid4()
        result = repository.get_by_recording(recording_id)

        session.query.assert_called_once_with(Metadata)
        assert result == mock_metadata

    def test_get_by_key(self, repository, mock_db_manager):
        """Test getting metadata by key."""
        _, session = mock_db_manager
        mock_metadata = Mock(spec=Metadata)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_metadata
        session.query.return_value = mock_query

        # Call get_by_key
        recording_id = uuid4()
        result = repository.get_by_key(recording_id, "artist")

        session.query.assert_called_once_with(Metadata)
        assert result == mock_metadata

    def test_update_metadata(self, repository, mock_db_manager):
        """Test updating metadata."""
        _, session = mock_db_manager
        mock_metadata = Mock(spec=Metadata)
        mock_metadata.value = "old_value"

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_metadata
        session.query.return_value = mock_query

        # Call update
        metadata_id = uuid4()
        result = repository.update(metadata_id, "new_value")

        assert mock_metadata.value == "new_value"
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        assert result == mock_metadata

    def test_delete_metadata(self, repository, mock_db_manager):
        """Test deleting metadata."""
        _, session = mock_db_manager
        mock_metadata = Mock(spec=Metadata)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_metadata
        session.query.return_value = mock_query

        # Call delete
        metadata_id = uuid4()
        result = repository.delete(metadata_id)

        session.delete.assert_called_once_with(mock_metadata)
        assert result is True

    def test_bulk_create_metadata(self, repository, mock_db_manager):
        """Test bulk creating metadata."""
        _, session = mock_db_manager
        recording_id = uuid4()

        metadata_items = [
            {"key": "artist", "value": "Test Artist"},
            {"key": "album", "value": "Test Album"},
        ]

        # Call bulk_create
        result = repository.bulk_create(recording_id, metadata_items)

        session.bulk_save_objects.assert_called_once()
        assert len(result) == 2


class TestTracklistRepository:
    """Test cases for TracklistRepository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock()
        session = MagicMock()
        db_manager.get_db_session.return_value.__enter__ = Mock(return_value=session)
        db_manager.get_db_session.return_value.__exit__ = Mock(return_value=None)
        return db_manager, session

    @pytest.fixture
    def repository(self, mock_db_manager):
        """Create a repository instance with mock database."""
        db_manager, _ = mock_db_manager
        return TracklistRepository(db_manager)

    def test_create_tracklist(self, repository, mock_db_manager):
        """Test creating a tracklist."""
        _, session = mock_db_manager
        recording_id = uuid4()

        tracks = [
            {"title": "Track 1", "artist": "Artist 1", "start_time": "00:00"},
            {"title": "Track 2", "artist": "Artist 2", "start_time": "03:30"},
        ]

        # Mock the validate_tracks method to return True
        with patch.object(Tracklist, "validate_tracks", return_value=True):
            # Call create method
            repository.create(
                recording_id=recording_id,
                source="manual",
                tracks=tracks,
                cue_file_path="/path/to/file.cue",
            )

            # Verify session methods were called
            session.add.assert_called_once()
            session.flush.assert_called_once()
            session.refresh.assert_called_once()

    def test_create_tracklist_invalid_tracks(self, repository, mock_db_manager):
        """Test creating a tracklist with invalid tracks."""
        _, session = mock_db_manager
        recording_id = uuid4()

        tracks = [{"invalid": "structure"}]

        # Mock the validate_tracks method to return False
        with (
            patch.object(Tracklist, "validate_tracks", return_value=False),
            pytest.raises(ValueError, match="Invalid tracks structure"),
        ):
            repository.create(recording_id=recording_id, source="manual", tracks=tracks)

    def test_get_by_recording(self, repository, mock_db_manager):
        """Test getting tracklist by recording."""
        _, session = mock_db_manager
        mock_tracklist = Mock(spec=Tracklist)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_tracklist
        session.query.return_value = mock_query

        # Call get_by_recording
        recording_id = uuid4()
        result = repository.get_by_recording(recording_id)

        session.query.assert_called_once_with(Tracklist)
        assert result == mock_tracklist

    def test_get_by_id(self, repository, mock_db_manager):
        """Test getting tracklist by ID."""
        _, session = mock_db_manager
        mock_tracklist = Mock(spec=Tracklist)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_tracklist
        session.query.return_value = mock_query

        # Call get_by_id
        tracklist_id = uuid4()
        result = repository.get_by_id(tracklist_id)

        session.query.assert_called_once_with(Tracklist)
        assert result == mock_tracklist

    def test_update_tracklist(self, repository, mock_db_manager):
        """Test updating a tracklist."""
        _, session = mock_db_manager
        mock_tracklist = Mock(spec=Tracklist)
        mock_tracklist.source = "old_source"
        mock_tracklist.validate_tracks = Mock(return_value=True)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_tracklist
        session.query.return_value = mock_query

        # Call update
        tracklist_id = uuid4()
        new_tracks = [{"title": "New Track", "artist": "New Artist", "start_time": "00:00"}]
        result = repository.update(tracklist_id, source="new_source", tracks=new_tracks)

        assert mock_tracklist.source == "new_source"
        assert mock_tracklist.tracks == new_tracks
        mock_tracklist.validate_tracks.assert_called_once()
        session.flush.assert_called_once()
        session.refresh.assert_called_once()
        assert result == mock_tracklist

    def test_update_tracklist_invalid_tracks(self, repository, mock_db_manager):
        """Test updating a tracklist with invalid tracks."""
        _, session = mock_db_manager
        mock_tracklist = Mock(spec=Tracklist)
        mock_tracklist.validate_tracks = Mock(return_value=False)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_tracklist
        session.query.return_value = mock_query

        # Call update with invalid tracks and expect ValueError
        tracklist_id = uuid4()
        with pytest.raises(ValueError, match="Invalid tracks structure"):
            repository.update(tracklist_id, tracks=[{"invalid": "structure"}])

    def test_delete_tracklist(self, repository, mock_db_manager):
        """Test deleting a tracklist."""
        _, session = mock_db_manager
        mock_tracklist = Mock(spec=Tracklist)

        # Setup mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_tracklist
        session.query.return_value = mock_query

        # Call delete
        tracklist_id = uuid4()
        result = repository.delete(tracklist_id)

        session.delete.assert_called_once_with(mock_tracklist)
        assert result is True
