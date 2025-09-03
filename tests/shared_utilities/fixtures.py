"""Common test fixtures for all services."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import aiofiles
import pytest


@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "operations_count": 0,
        "errors_count": 0,
        "concurrent_tasks": [],
    }

    def start():
        metrics["start_time"] = asyncio.get_event_loop().time()

    def end():
        metrics["end_time"] = asyncio.get_event_loop().time()

    def record_operation():
        metrics["operations_count"] += 1

    def record_error():
        metrics["errors_count"] += 1

    def get_duration():
        if metrics["start_time"] and metrics["end_time"]:
            return metrics["end_time"] - metrics["start_time"]
        return None

    def reset():
        metrics.update(
            {
                "start_time": None,
                "end_time": None,
                "operations_count": 0,
                "errors_count": 0,
                "concurrent_tasks": [],
            }
        )

    metrics.update(
        {
            "start": start,
            "end": end,
            "record_operation": record_operation,
            "record_error": record_error,
            "get_duration": get_duration,
            "reset": reset,
        }
    )

    return metrics


@pytest.fixture
def mock_message_queue():
    """Mock message queue for testing."""
    queue = AsyncMock()

    # Mock RabbitMQ connection and channel
    connection = AsyncMock()
    channel = AsyncMock()

    connection.channel.return_value = channel
    channel.is_closed = False
    channel.close = AsyncMock()
    connection.close = AsyncMock()

    queue.connection = connection
    queue.channel = channel
    queue.publish = AsyncMock()
    queue.consume = AsyncMock()
    queue.connect = AsyncMock()
    queue.disconnect = AsyncMock()
    queue.declare_queue = AsyncMock()
    queue.declare_exchange = AsyncMock()

    return queue


@pytest.fixture
async def mock_file_system(tmp_path: Path):
    """Create a mock file system for testing."""
    # Create test directory structure
    test_dir = tmp_path / "test_music"
    test_dir.mkdir()

    # Create some test files
    test_files = [
        test_dir / "song1.mp3",
        test_dir / "song2.flac",
        test_dir / "song3.wav",
        test_dir / "subfolder" / "song4.mp3",
    ]

    # Create subfolder
    (test_dir / "subfolder").mkdir()

    # Create test files asynchronously
    for file_path in test_files:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(b"test audio data")

    yield test_dir


@pytest.fixture
def mock_metadata_extractor():
    """Mock metadata extractor for async operations."""
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value={
            "duration": 180.5,
            "bitrate": 320000,
            "sample_rate": 44100,
            "channels": 2,
            "format": "mp3",
            "artist": "Test Artist",
            "title": "Test Song",
            "album": "Test Album",
            "genre": "Electronic",
            "year": 2025,
        }
    )
    return extractor


@pytest.fixture
def mock_bpm_detector():
    """Mock BPM detector for testing."""
    detector = AsyncMock()
    detector.detect_bpm = AsyncMock(
        return_value={
            "bpm": 128.5,
            "confidence": 0.85,
            "algorithm": "percival",
            "needs_review": False,
        }
    )
    return detector


@pytest.fixture
def mock_key_detector():
    """Mock key detector for testing."""
    detector = AsyncMock()
    detector.detect_key = AsyncMock(
        return_value={
            "key": "A minor",
            "confidence": 0.92,
            "algorithm": "krumhansl",
            "needs_review": False,
        }
    )
    return detector


@pytest.fixture
def mock_audio_analyzer():
    """Mock audio analyzer with multiple analysis types."""
    analyzer = AsyncMock()
    analyzer.analyze_audio = AsyncMock(
        return_value={
            "duration": 240.0,
            "bpm": 128.0,
            "key": "A minor",
            "energy": 0.8,
            "danceability": 0.9,
            "valence": 0.7,
            "loudness": -8.5,
            "tempo_stability": 0.95,
        }
    )
    return analyzer


@pytest.fixture
def mock_service_config():
    """Mock service configuration."""
    config = Mock()
    config.database_url = "postgresql://test:test@localhost:5432/test_db"
    config.redis_host = "localhost"
    config.redis_port = 6379
    config.redis_db = 0
    config.neo4j_uri = "bolt://localhost:7687"
    config.neo4j_user = "neo4j"
    config.neo4j_password = "password"
    config.rabbitmq_url = "amqp://guest:guest@localhost:5672/"
    config.log_level = "INFO"
    config.enable_debug = False
    config.max_workers = 4
    config.timeout_seconds = 30.0
    return config


@pytest.fixture
def mock_storage_service():
    """Mock storage service for testing."""
    storage = AsyncMock()
    storage.store_recording = AsyncMock(return_value=str(uuid4()))
    storage.store_metadata = AsyncMock(return_value=True)
    storage.store_tracklist = AsyncMock(return_value=str(uuid4()))
    storage.get_recording = AsyncMock(return_value=None)
    storage.get_metadata = AsyncMock(return_value=[])
    storage.get_tracklist = AsyncMock(return_value=None)
    storage.delete_recording = AsyncMock(return_value=True)
    storage.health_check = AsyncMock(return_value={"status": "healthy"})
    return storage


@pytest.fixture
def mock_notification_service():
    """Mock notification service for testing."""
    notifier = AsyncMock()
    notifier.send_notification = AsyncMock(return_value=True)
    notifier.send_discord_message = AsyncMock(return_value={"message_id": "123456"})
    notifier.send_email = AsyncMock(return_value={"email_id": "email_123"})
    notifier.health_check = AsyncMock(return_value={"status": "healthy"})
    return notifier


class CommonTestFixtures:
    """Collection of common test fixtures that can be used across services."""

    @staticmethod
    def create_test_recording(file_path: str = "/music/test.mp3"):
        """Create a test recording object."""
        return {
            "id": str(uuid4()),
            "file_path": file_path,
            "file_name": file_path.split("/")[-1],
            "sha256_hash": "abc123def456",
            "xxh128_hash": "xyz789uvw012",
            "created_at": "2025-01-01T12:00:00Z",
        }

    @staticmethod
    def create_test_metadata(recording_id: str | None = None):
        """Create test metadata items."""
        if recording_id is None:
            recording_id = str(uuid4())

        return [
            {"id": 1, "recording_id": recording_id, "key": "bpm", "value": "128"},
            {"id": 2, "recording_id": recording_id, "key": "genre", "value": "techno"},
            {"id": 3, "recording_id": recording_id, "key": "key", "value": "A minor"},
        ]

    @staticmethod
    def create_test_tracklist(recording_id: str | None = None):
        """Create a test tracklist."""
        if recording_id is None:
            recording_id = str(uuid4())

        return {
            "id": str(uuid4()),
            "recording_id": recording_id,
            "source": "manual",
            "cue_file_path": None,
            "tracks": [
                {
                    "position": 1,
                    "title": "Test Track 1",
                    "artist": "Test Artist 1",
                    "start_time": "00:00:00",
                    "duration": 300,
                    "bpm": 128,
                    "key": "A minor",
                },
                {
                    "position": 2,
                    "title": "Test Track 2",
                    "artist": "Test Artist 2",
                    "start_time": "00:05:00",
                    "duration": 420,
                    "bpm": 130,
                    "key": "C major",
                },
            ],
        }
