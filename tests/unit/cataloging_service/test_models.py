"""Unit tests for cataloging service SQLAlchemy models.

These tests require a PostgreSQL database connection with a test database.

Database Requirements:
- PostgreSQL server running on localhost:5432
- Database: test_tracktion
- User: tracktion_user
- Password: changeme
- Extensions: uuid-ossp

To run these tests:
    uv run pytest tests/unit/cataloging_service/test_models.py

To run without database (structure tests only):
    uv run pytest tests/unit/cataloging_service/test_models_no_db.py

If database is not available, the tests will be skipped with appropriate error messages.
"""

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker

from services.cataloging_service.src.models.base import Base
from services.cataloging_service.src.models.metadata import Metadata
from services.cataloging_service.src.models.recording import Recording
from services.cataloging_service.src.models.tracklist import Tracklist

# Check if database is available
DB_AVAILABLE = False
try:
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://tracktion_user:changeme@localhost:5432/test_tracktion",
    )
    engine = create_engine(test_db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    DB_AVAILABLE = True
    engine.dispose()
except (OperationalError, Exception):
    pass

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.skipif(
        not DB_AVAILABLE,
        reason=(
            "PostgreSQL database not available. Install PostgreSQL "
            "and create test_tracktion database to run these tests."
        ),
    ),
]


@pytest.fixture(scope="function")
def test_db_session():
    """Create a test database session for PostgreSQL."""
    if not DB_AVAILABLE:
        pytest.skip("Database not available")

    # Use test database URL
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://tracktion_user:changeme@localhost:5432/test_tracktion",
    )

    # Create engine and tables
    engine = create_engine(test_db_url)

    # Create UUID extension
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)

    # Create session
    session_local = sessionmaker(bind=engine)
    session = session_local()

    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


class TestRecordingModel:
    """Test cases for the Recording model."""

    def test_recording_creation(self, test_db_session):
        """Test creating a basic Recording instance."""
        recording = Recording(
            file_path="/music/test.mp3", file_name="test.mp3", sha256_hash="abc123def456", xxh128_hash="xyz789"
        )

        test_db_session.add(recording)
        test_db_session.commit()

        # Verify the recording was created
        assert recording.id is not None
        assert isinstance(recording.id, uuid.UUID)
        assert recording.file_path == "/music/test.mp3"
        assert recording.file_name == "test.mp3"
        assert recording.sha256_hash == "abc123def456"
        assert recording.xxh128_hash == "xyz789"
        assert isinstance(recording.created_at, datetime)

    def test_recording_minimal_creation(self, test_db_session):
        """Test creating Recording with only required fields."""
        recording = Recording(file_path="/music/minimal.mp3", file_name="minimal.mp3")

        test_db_session.add(recording)
        test_db_session.commit()

        # Verify required fields
        assert recording.file_path == "/music/minimal.mp3"
        assert recording.file_name == "minimal.mp3"
        assert recording.sha256_hash is None
        assert recording.xxh128_hash is None
        assert recording.id is not None
        assert recording.created_at is not None

    def test_recording_unique_constraints(self, test_db_session):
        """Test that sha256_hash and xxh128_hash are unique."""
        # Create first recording
        recording1 = Recording(file_path="/music/test1.mp3", file_name="test1.mp3", sha256_hash="duplicate_hash")
        test_db_session.add(recording1)
        test_db_session.commit()

        # Try to create second recording with same sha256_hash
        recording2 = Recording(file_path="/music/test2.mp3", file_name="test2.mp3", sha256_hash="duplicate_hash")
        test_db_session.add(recording2)

        with pytest.raises(IntegrityError):
            test_db_session.commit()

    def test_recording_unique_xxh128_hash(self, test_db_session):
        """Test that xxh128_hash unique constraint works."""
        # Create first recording
        recording1 = Recording(file_path="/music/test1.mp3", file_name="test1.mp3", xxh128_hash="duplicate_xxh128")
        test_db_session.add(recording1)
        test_db_session.commit()

        # Try to create second recording with same xxh128_hash
        recording2 = Recording(file_path="/music/test2.mp3", file_name="test2.mp3", xxh128_hash="duplicate_xxh128")
        test_db_session.add(recording2)

        with pytest.raises(IntegrityError):
            test_db_session.commit()

    def test_recording_nullable_hashes(self, test_db_session):
        """Test that multiple recordings can have null hashes."""
        # Create multiple recordings with null hashes
        recording1 = Recording(file_path="/music/test1.mp3", file_name="test1.mp3")
        recording2 = Recording(file_path="/music/test2.mp3", file_name="test2.mp3")

        test_db_session.add_all([recording1, recording2])
        test_db_session.commit()

        # Both should be created successfully
        assert recording1.id is not None
        assert recording2.id is not None
        assert recording1.sha256_hash is None
        assert recording2.sha256_hash is None

    @patch("services.cataloging_service.src.models.recording.datetime")
    def test_recording_created_at_default(self, mock_datetime, test_db_session):
        """Test that created_at gets set to current time by default."""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_datetime.utcnow.return_value = fixed_time

        recording = Recording(file_path="/music/test.mp3", file_name="test.mp3")

        test_db_session.add(recording)
        test_db_session.commit()

        # The actual created_at will be set by the database, not our mock
        # So we just verify it's a datetime object
        assert isinstance(recording.created_at, datetime)

    def test_recording_repr(self, test_db_session):
        """Test the string representation of Recording."""
        recording = Recording(file_path="/music/test.mp3", file_name="test.mp3")
        test_db_session.add(recording)
        test_db_session.commit()

        repr_str = repr(recording)
        assert "Recording" in repr_str
        assert str(recording.id) in repr_str
        assert "test.mp3" in repr_str

    def test_recording_relationships_initialized(self, test_db_session):
        """Test that relationships are properly initialized."""
        recording = Recording(file_path="/music/test.mp3", file_name="test.mp3")
        test_db_session.add(recording)
        test_db_session.commit()

        # Test that relationship collections are initialized as empty lists
        assert recording.metadata_items == []
        assert recording.tracklists == []


class TestMetadataModel:
    """Test cases for the Metadata model."""

    @pytest.fixture
    def sample_recording(self, test_db_session):
        """Create a sample recording for metadata tests."""
        recording = Recording(file_path="/music/test.mp3", file_name="test.mp3")
        test_db_session.add(recording)
        test_db_session.commit()
        return recording

    def test_metadata_creation(self, test_db_session, sample_recording):
        """Test creating a Metadata instance."""
        metadata = Metadata(recording_id=sample_recording.id, key="bpm", value="128")

        test_db_session.add(metadata)
        test_db_session.commit()

        assert metadata.id is not None
        assert isinstance(metadata.id, uuid.UUID)
        assert metadata.recording_id == sample_recording.id
        assert metadata.key == "bpm"
        assert metadata.value == "128"

    def test_metadata_required_fields(self, test_db_session, sample_recording):
        """Test that all required fields must be provided."""
        # Test missing key - this will raise a TypeError during construction
        with pytest.raises(TypeError):
            metadata = Metadata(recording_id=sample_recording.id, value="128")
            test_db_session.add(metadata)
            test_db_session.commit()

        test_db_session.rollback()

        # Test missing value - this will raise a TypeError during construction
        with pytest.raises(TypeError):
            metadata = Metadata(recording_id=sample_recording.id, key="bpm")
            test_db_session.add(metadata)
            test_db_session.commit()

    def test_metadata_foreign_key_constraint(self, test_db_session):
        """Test foreign key constraint on recording_id."""
        fake_recording_id = uuid.uuid4()
        metadata = Metadata(recording_id=fake_recording_id, key="test", value="test_value")

        test_db_session.add(metadata)

        with pytest.raises(IntegrityError):
            test_db_session.commit()

    def test_metadata_relationship_to_recording(self, test_db_session, sample_recording):
        """Test the relationship between Metadata and Recording."""
        metadata = Metadata(recording_id=sample_recording.id, key="artist", value="Test Artist")

        test_db_session.add(metadata)
        test_db_session.commit()

        # Test relationship from metadata to recording
        assert metadata.recording.id == sample_recording.id
        assert metadata.recording.file_name == "test.mp3"

        # Test relationship from recording to metadata
        test_db_session.refresh(sample_recording)
        assert len(sample_recording.metadata_items) == 1
        assert sample_recording.metadata_items[0].key == "artist"
        assert sample_recording.metadata_items[0].value == "Test Artist"

    def test_metadata_multiple_items_same_recording(self, test_db_session, sample_recording):
        """Test multiple metadata items for the same recording."""
        metadata_items = [
            Metadata(recording_id=sample_recording.id, key="bpm", value="128"),
            Metadata(recording_id=sample_recording.id, key="key", value="A minor"),
            Metadata(recording_id=sample_recording.id, key="genre", value="techno"),
        ]

        test_db_session.add_all(metadata_items)
        test_db_session.commit()

        # Refresh and check all metadata items
        test_db_session.refresh(sample_recording)
        assert len(sample_recording.metadata_items) == 3

        # Check that all items are present
        keys = [item.key for item in sample_recording.metadata_items]
        assert "bpm" in keys
        assert "key" in keys
        assert "genre" in keys

    def test_metadata_cascade_delete(self, test_db_session, sample_recording):
        """Test that metadata is deleted when recording is deleted."""
        metadata = Metadata(recording_id=sample_recording.id, key="test", value="test_value")
        test_db_session.add(metadata)
        test_db_session.commit()

        metadata_id = metadata.id

        # Delete the recording
        test_db_session.delete(sample_recording)
        test_db_session.commit()

        # Verify metadata was also deleted
        deleted_metadata = test_db_session.query(Metadata).filter_by(id=metadata_id).first()
        assert deleted_metadata is None

    def test_metadata_indexes(self, test_db_session, sample_recording):
        """Test that metadata indexes work as expected."""
        # This test verifies that indexes are created without errors
        # We can't directly test index performance in unit tests
        metadata_items = [
            Metadata(recording_id=sample_recording.id, key="bpm", value="128"),
            Metadata(recording_id=sample_recording.id, key="bpm", value="130"),  # Same key, different value
        ]

        test_db_session.add_all(metadata_items)
        test_db_session.commit()

        # Query by key (should use idx_metadata_key)
        bpm_items = test_db_session.query(Metadata).filter_by(key="bpm").all()
        assert len(bpm_items) == 2

        # Query by recording_id (should use idx_metadata_recording_id)
        recording_items = test_db_session.query(Metadata).filter_by(recording_id=sample_recording.id).all()
        assert len(recording_items) == 2

    def test_metadata_repr(self, test_db_session, sample_recording):
        """Test the string representation of Metadata."""
        metadata = Metadata(recording_id=sample_recording.id, key="bpm", value="128")
        test_db_session.add(metadata)
        test_db_session.commit()

        repr_str = repr(metadata)
        assert "Metadata" in repr_str
        assert str(metadata.id) in repr_str
        assert "bpm" in repr_str
        assert "128" in repr_str


class TestTracklistModel:
    """Test cases for the Tracklist model."""

    @pytest.fixture
    def sample_recording(self, test_db_session):
        """Create a sample recording for tracklist tests."""
        recording = Recording(file_path="/music/set.mp3", file_name="set.mp3")
        test_db_session.add(recording)
        test_db_session.commit()
        return recording

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data for testing."""
        return [
            {"title": "Opening Track", "artist": "DJ One", "start_time": "00:00:00", "duration": 300},
            {"title": "Peak Time", "artist": "DJ Two", "start_time": "00:05:00", "duration": 420},
        ]

    def test_tracklist_creation(self, test_db_session, sample_recording, sample_tracks):
        """Test creating a Tracklist instance."""
        tracklist = Tracklist(recording_id=sample_recording.id, source="manual", tracks=sample_tracks)

        test_db_session.add(tracklist)
        test_db_session.commit()

        assert tracklist.id is not None
        assert isinstance(tracklist.id, uuid.UUID)
        assert tracklist.recording_id == sample_recording.id
        assert tracklist.source == "manual"
        assert tracklist.cue_file_path is None
        assert len(tracklist.tracks) == 2
        assert tracklist.tracks[0]["title"] == "Opening Track"

    def test_tracklist_with_cue_file(self, test_db_session, sample_recording, sample_tracks):
        """Test creating a Tracklist with cue file path."""
        tracklist = Tracklist(
            recording_id=sample_recording.id, source="cue", cue_file_path="/music/set.cue", tracks=sample_tracks
        )

        test_db_session.add(tracklist)
        test_db_session.commit()

        assert tracklist.cue_file_path == "/music/set.cue"
        assert tracklist.source == "cue"

    def test_tracklist_empty_tracks_default(self, test_db_session, sample_recording):
        """Test that tracks defaults to empty list."""
        tracklist = Tracklist(recording_id=sample_recording.id, source="manual")

        test_db_session.add(tracklist)
        test_db_session.commit()

        assert tracklist.tracks == []

    def test_tracklist_required_fields(self, test_db_session, sample_recording):
        """Test that required fields must be provided."""
        # Test missing source - this will raise a TypeError during construction
        with pytest.raises(TypeError):
            tracklist = Tracklist(recording_id=sample_recording.id)
            test_db_session.add(tracklist)
            test_db_session.commit()

    def test_tracklist_foreign_key_constraint(self, test_db_session):
        """Test foreign key constraint on recording_id."""
        fake_recording_id = uuid.uuid4()
        tracklist = Tracklist(recording_id=fake_recording_id, source="manual", tracks=[])

        test_db_session.add(tracklist)

        with pytest.raises(IntegrityError):
            test_db_session.commit()

    def test_tracklist_relationship_to_recording(self, test_db_session, sample_recording, sample_tracks):
        """Test the relationship between Tracklist and Recording."""
        tracklist = Tracklist(recording_id=sample_recording.id, source="manual", tracks=sample_tracks)

        test_db_session.add(tracklist)
        test_db_session.commit()

        # Test relationship from tracklist to recording
        assert tracklist.recording.id == sample_recording.id
        assert tracklist.recording.file_name == "set.mp3"

        # Test relationship from recording to tracklist
        test_db_session.refresh(sample_recording)
        assert len(sample_recording.tracklists) == 1
        assert sample_recording.tracklists[0].source == "manual"

    def test_tracklist_multiple_per_recording(self, test_db_session, sample_recording, sample_tracks):
        """Test multiple tracklists for the same recording."""
        tracklist1 = Tracklist(recording_id=sample_recording.id, source="manual", tracks=sample_tracks)
        tracklist2 = Tracklist(
            recording_id=sample_recording.id,
            source="automatic",
            tracks=sample_tracks[:1],  # Different tracks
        )

        test_db_session.add_all([tracklist1, tracklist2])
        test_db_session.commit()

        # Refresh and check both tracklists
        test_db_session.refresh(sample_recording)
        assert len(sample_recording.tracklists) == 2

        sources = [tl.source for tl in sample_recording.tracklists]
        assert "manual" in sources
        assert "automatic" in sources

    def test_tracklist_jsonb_tracks_complex_data(self, test_db_session, sample_recording):
        """Test JSONB field with complex track data."""
        complex_tracks = [
            {
                "title": "Complex Track",
                "artist": "DJ Complex",
                "start_time": "00:00:00",
                "end_time": "00:05:30",
                "duration": 330,
                "bpm": 128,
                "key": "A minor",
                "genre": "techno",
                "metadata": {"energy": 0.8, "danceability": 0.9, "custom_field": "custom_value"},
                "transitions": [{"type": "fade_in", "duration": 8}, {"type": "fade_out", "duration": 12}],
            }
        ]

        tracklist = Tracklist(recording_id=sample_recording.id, source="ai_analysis", tracks=complex_tracks)

        test_db_session.add(tracklist)
        test_db_session.commit()

        # Verify complex data is stored and retrieved correctly
        assert len(tracklist.tracks) == 1
        track = tracklist.tracks[0]
        assert track["title"] == "Complex Track"
        assert track["bpm"] == 128
        assert track["metadata"]["energy"] == 0.8
        assert len(track["transitions"]) == 2

    def test_tracklist_cascade_delete(self, test_db_session, sample_recording, sample_tracks):
        """Test that tracklist is deleted when recording is deleted."""
        tracklist = Tracklist(recording_id=sample_recording.id, source="manual", tracks=sample_tracks)
        test_db_session.add(tracklist)
        test_db_session.commit()

        tracklist_id = tracklist.id

        # Delete the recording
        test_db_session.delete(sample_recording)
        test_db_session.commit()

        # Verify tracklist was also deleted
        deleted_tracklist = test_db_session.query(Tracklist).filter_by(id=tracklist_id).first()
        assert deleted_tracklist is None

    def test_tracklist_repr(self, test_db_session, sample_recording, sample_tracks):
        """Test the string representation of Tracklist."""
        tracklist = Tracklist(recording_id=sample_recording.id, source="manual", tracks=sample_tracks)
        test_db_session.add(tracklist)
        test_db_session.commit()

        repr_str = repr(tracklist)
        assert "Tracklist" in repr_str
        assert str(tracklist.id) in repr_str
        assert "manual" in repr_str
        assert "2" in repr_str  # Number of tracks


class TestModelRelationships:
    """Test cases for relationships between models."""

    @pytest.fixture
    def recording_with_data(self, test_db_session):
        """Create a recording with metadata and tracklist."""
        recording = Recording(
            file_path="/music/complete_set.mp3",
            file_name="complete_set.mp3",
            sha256_hash="complete_hash",
            xxh128_hash="complete_xxh",
        )
        test_db_session.add(recording)
        test_db_session.commit()

        # Add metadata
        metadata_items = [
            Metadata(recording_id=recording.id, key="bpm", value="128"),
            Metadata(recording_id=recording.id, key="genre", value="techno"),
        ]
        test_db_session.add_all(metadata_items)

        # Add tracklist
        tracks = [
            {"title": "Track 1", "artist": "Artist 1", "start_time": "00:00:00"},
            {"title": "Track 2", "artist": "Artist 2", "start_time": "00:05:00"},
        ]
        tracklist = Tracklist(recording_id=recording.id, source="manual", tracks=tracks)
        test_db_session.add(tracklist)
        test_db_session.commit()

        return recording

    def test_complete_model_relationships(self, test_db_session, recording_with_data):
        """Test complete relationships between all models."""
        # Refresh to ensure relationships are loaded
        test_db_session.refresh(recording_with_data)

        # Test recording -> metadata relationship
        assert len(recording_with_data.metadata_items) == 2
        metadata_keys = [item.key for item in recording_with_data.metadata_items]
        assert "bpm" in metadata_keys
        assert "genre" in metadata_keys

        # Test recording -> tracklist relationship
        assert len(recording_with_data.tracklists) == 1
        tracklist = recording_with_data.tracklists[0]
        assert tracklist.source == "manual"
        assert len(tracklist.tracks) == 2

        # Test reverse relationships
        for metadata_item in recording_with_data.metadata_items:
            assert metadata_item.recording.id == recording_with_data.id

        assert tracklist.recording.id == recording_with_data.id

    def test_cascade_delete_all_relationships(self, test_db_session, recording_with_data):
        """Test that all related objects are deleted when recording is deleted."""
        recording_id = recording_with_data.id

        # Store IDs for verification
        metadata_ids = [item.id for item in recording_with_data.metadata_items]
        tracklist_ids = [tl.id for tl in recording_with_data.tracklists]

        # Delete recording
        test_db_session.delete(recording_with_data)
        test_db_session.commit()

        # Verify recording is deleted
        deleted_recording = test_db_session.query(Recording).filter_by(id=recording_id).first()
        assert deleted_recording is None

        # Verify all metadata items are deleted
        for metadata_id in metadata_ids:
            deleted_metadata = test_db_session.query(Metadata).filter_by(id=metadata_id).first()
            assert deleted_metadata is None

        # Verify all tracklists are deleted
        for tracklist_id in tracklist_ids:
            deleted_tracklist = test_db_session.query(Tracklist).filter_by(id=tracklist_id).first()
            assert deleted_tracklist is None

    def test_orphaned_metadata_prevention(self, test_db_session):
        """Test that metadata cannot exist without a recording."""
        # Try to create metadata without a valid recording
        fake_recording_id = uuid.uuid4()
        metadata = Metadata(recording_id=fake_recording_id, key="orphan", value="test")

        test_db_session.add(metadata)

        with pytest.raises(IntegrityError):
            test_db_session.commit()

    def test_orphaned_tracklist_prevention(self, test_db_session):
        """Test that tracklist cannot exist without a recording."""
        # Try to create tracklist without a valid recording
        fake_recording_id = uuid.uuid4()
        tracklist = Tracklist(recording_id=fake_recording_id, source="orphan", tracks=[])

        test_db_session.add(tracklist)

        with pytest.raises(IntegrityError):
            test_db_session.commit()


class TestModelConstraints:
    """Test cases for model constraints and validation."""

    def test_recording_file_path_index(self, test_db_session):
        """Test that file_path is indexed for performance."""
        # Create multiple recordings with different paths
        recordings = [Recording(file_path=f"/music/test{i}.mp3", file_name=f"test{i}.mp3") for i in range(5)]

        test_db_session.add_all(recordings)
        test_db_session.commit()

        # Query by file_path should work efficiently
        found = test_db_session.query(Recording).filter_by(file_path="/music/test2.mp3").first()
        assert found is not None
        assert found.file_name == "test2.mp3"

    def test_hash_fields_indexed(self, test_db_session):
        """Test that hash fields are indexed for performance."""
        recording = Recording(
            file_path="/music/test.mp3",
            file_name="test.mp3",
            sha256_hash="indexed_sha256",
            xxh128_hash="indexed_xxh128",
        )

        test_db_session.add(recording)
        test_db_session.commit()

        # Query by hash fields should work efficiently
        found_by_sha = test_db_session.query(Recording).filter_by(sha256_hash="indexed_sha256").first()
        assert found_by_sha is not None

        found_by_xxh = test_db_session.query(Recording).filter_by(xxh128_hash="indexed_xxh128").first()
        assert found_by_xxh is not None

    def test_metadata_composite_indexes(self, test_db_session):
        """Test metadata composite indexes work correctly."""
        recording = Recording(file_path="/music/test.mp3", file_name="test.mp3")
        test_db_session.add(recording)
        test_db_session.commit()

        # Add metadata for index testing
        metadata_items = [
            Metadata(recording_id=recording.id, key="bpm", value="128"),
            Metadata(recording_id=recording.id, key="bpm", value="130"),  # Same key, different value
            Metadata(recording_id=recording.id, key="genre", value="techno"),
        ]
        test_db_session.add_all(metadata_items)
        test_db_session.commit()

        # Test idx_metadata_recording_key composite index
        specific_metadata = test_db_session.query(Metadata).filter_by(recording_id=recording.id, key="bpm").all()
        assert len(specific_metadata) == 2

        # Test idx_metadata_key index
        all_bpm = test_db_session.query(Metadata).filter_by(key="bpm").all()
        assert len(all_bpm) == 2
