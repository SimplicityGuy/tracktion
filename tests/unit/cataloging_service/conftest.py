"""Configuration and fixtures for cataloging service tests."""

import uuid
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_recording_data():
    """Sample recording data for testing."""
    return {
        "file_path": "/music/test.mp3",
        "file_name": "test.mp3",
        "sha256_hash": "abc123def456",
        "xxh128_hash": "xyz789",
    }


@pytest.fixture
def sample_metadata_data():
    """Sample metadata data for testing."""
    return [
        {"key": "bpm", "value": "128"},
        {"key": "genre", "value": "techno"},
        {"key": "key", "value": "A minor"},
        {"key": "energy", "value": "0.8"},
    ]


@pytest.fixture
def sample_track_data():
    """Sample track data for testing."""
    return [
        {
            "title": "Opening Track",
            "artist": "DJ One",
            "start_time": "00:00:00",
            "duration": 300,
            "bpm": 128,
            "key": "A minor",
        },
        {
            "title": "Peak Time",
            "artist": "DJ Two",
            "start_time": "00:05:00",
            "duration": 420,
            "bpm": 130,
            "key": "C major",
        },
        {
            "title": "Breakdown",
            "artist": "DJ Three",
            "start_time": "00:12:00",
            "duration": 360,
            "bpm": 126,
            "key": "F minor",
        },
    ]


@pytest.fixture
def complex_track_data():
    """Complex track data with nested metadata for JSONB testing."""
    return [
        {
            "title": "Complex Track",
            "artist": "DJ Complex",
            "start_time": "00:00:00",
            "end_time": "00:05:30",
            "duration": 330,
            "bpm": 128,
            "key": "A minor",
            "genre": "techno",
            "metadata": {
                "energy": 0.8,
                "danceability": 0.9,
                "valence": 0.7,
                "acousticness": 0.1,
                "custom_field": "custom_value",
            },
            "transitions": [
                {"type": "fade_in", "duration": 8, "start_volume": 0.0, "end_volume": 1.0},
                {"type": "fade_out", "duration": 12, "start_volume": 1.0, "end_volume": 0.0},
            ],
            "analysis": {
                "tempo_changes": [{"time": "00:02:30", "bpm": 130}, {"time": "00:04:00", "bpm": 126}],
                "key_changes": [{"time": "00:03:15", "key": "C major"}],
                "structure": [
                    {"section": "intro", "start": "00:00:00", "end": "00:01:00"},
                    {"section": "buildup", "start": "00:01:00", "end": "00:03:00"},
                    {"section": "drop", "start": "00:03:00", "end": "00:05:30"},
                ],
            },
        }
    ]


@pytest.fixture
def mock_recording():
    """Create a mock Recording instance for testing."""
    mock = Mock()
    mock.id = uuid.uuid4()
    mock.file_path = "/music/test.mp3"
    mock.file_name = "test.mp3"
    mock.sha256_hash = "abc123def456"
    mock.xxh128_hash = "xyz789"
    mock.created_at = datetime.now(UTC)
    mock.metadata_items = []
    mock.tracklists = []
    return mock


@pytest.fixture
def mock_metadata(mock_recording):
    """Create a mock Metadata instance for testing."""
    mock = Mock()
    mock.id = uuid.uuid4()
    mock.recording_id = mock_recording.id
    mock.key = "bpm"
    mock.value = "128"
    mock.recording = mock_recording
    return mock


@pytest.fixture
def mock_tracklist(mock_recording, sample_track_data):
    """Create a mock Tracklist instance for testing."""
    mock = Mock()
    mock.id = uuid.uuid4()
    mock.recording_id = mock_recording.id
    mock.source = "manual"
    mock.cue_file_path = None
    mock.tracks = sample_track_data
    mock.recording = mock_recording
    return mock


# Database-dependent fixtures (commented out until database is available)
# These fixtures require a working PostgreSQL database with test_tracktion database

# @pytest.fixture(scope="function")
# def test_db_session():
#     """Create a test database session for PostgreSQL."""
#     # Import here to avoid issues when DB is not available
#     from sqlalchemy import create_engine, text
#     from sqlalchemy.orm import sessionmaker
#     from services.cataloging_service.src.models.base import Base
#
#     # Use test database URL
#     test_db_url = os.getenv(
#         "TEST_DATABASE_URL",
#         "postgresql://tracktion_user:changeme@localhost:5432/test_tracktion",
#     )
#
#     # Create engine and tables
#     engine = create_engine(test_db_url)
#
#     # Create UUID extension
#     with engine.connect() as conn:
#         conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
#         conn.commit()
#
#     # Create all tables
#     Base.metadata.create_all(engine)
#
#     # Create session
#     session_local = sessionmaker(bind=engine)
#     session = session_local()
#
#     yield session
#
#     # Cleanup
#     session.close()
#     Base.metadata.drop_all(engine)
#     engine.dispose()
