"""Configuration and fixtures for file_watcher async tests."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock

import aiofiles  # type: ignore
import pytest
import pytest_asyncio
from aio_pika import Channel, Connection


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop(event_loop_policy):
    """Create an event loop for async tests."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_rabbitmq_connection() -> AsyncGenerator[AsyncMock]:
    """Mock RabbitMQ connection for async tests."""
    connection = AsyncMock(spec=Connection)
    channel = AsyncMock(spec=Channel)

    connection.channel.return_value = channel
    channel.is_closed = False
    channel.close = AsyncMock()
    connection.close = AsyncMock()

    yield connection


@pytest_asyncio.fixture
async def mock_file_system(tmp_path: Path) -> AsyncGenerator[Path]:
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

    # Cleanup is handled by tmp_path fixture


@pytest.fixture
def mock_async_file_watcher():
    """Mock async file watcher."""
    watcher = AsyncMock()
    watcher.start = AsyncMock()
    watcher.stop = AsyncMock()
    watcher.process_file = AsyncMock()
    watcher.handle_event = AsyncMock()
    return watcher


@pytest_asyncio.fixture
async def async_message_publisher(mock_rabbitmq_connection):
    """Mock async message publisher."""
    publisher = AsyncMock()
    publisher.connection = mock_rabbitmq_connection
    publisher.channel = await mock_rabbitmq_connection.channel()
    publisher.publish = AsyncMock()
    publisher.connect = AsyncMock()
    publisher.disconnect = AsyncMock()
    return publisher


@pytest.fixture
def sample_file_events():
    """Sample file events for testing."""
    return [
        {
            "event_type": "created",
            "file_path": "/music/new_song.mp3",
            "timestamp": "2025-01-01T00:00:00Z",
            "instance_id": "test_watcher",
            "sha256_hash": "abc123" * 10,
            "xxh128_hash": "def456" * 5,
        },
        {
            "event_type": "modified",
            "file_path": "/music/existing_song.mp3",
            "timestamp": "2025-01-01T00:01:00Z",
            "instance_id": "test_watcher",
            "sha256_hash": "ghi789" * 10,
            "xxh128_hash": "jkl012" * 5,
        },
        {
            "event_type": "deleted",
            "file_path": "/music/old_song.mp3",
            "timestamp": "2025-01-01T00:02:00Z",
            "instance_id": "test_watcher",
        },
    ]


@pytest.fixture
def async_semaphore():
    """Semaphore for limiting concurrent operations."""
    return asyncio.Semaphore(10)  # Limit to 10 concurrent operations for testing


@pytest_asyncio.fixture
async def async_task_pool():
    """Task pool for concurrent processing."""
    tasks = []

    async def add_task(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    yield add_task

    # Cancel all pending tasks on cleanup
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for all tasks to complete
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


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
        }
    )
    return extractor


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

    metrics["start"] = start
    metrics["end"] = end
    metrics["record_operation"] = record_operation
    metrics["record_error"] = record_error
    metrics["get_duration"] = get_duration

    return metrics
