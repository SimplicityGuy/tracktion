"""Database testing helpers and mock utilities."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import fakeredis
import pytest


@pytest.fixture
def mock_redis_client():
    """Provide a fakeredis client for testing Redis operations."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_redis_strict():
    """Provide a strict fakeredis client that mimics Redis behavior more closely."""
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def mock_database_session():
    """Mock database session for testing."""
    session = Mock()
    session.execute = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.add = Mock()
    session.delete = Mock()
    session.query = Mock()
    session.get = Mock()
    session.merge = Mock()
    session.flush = Mock()

    # Mock common query results
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.filter.return_value.all.return_value = []
    session.query.return_value.all.return_value = []
    session.query.return_value.count.return_value = 0

    return session


@pytest.fixture
def mock_async_database_session():
    """Mock async database session for testing."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = Mock()  # add is typically sync
    session.delete = Mock()  # delete is typically sync
    session.get = AsyncMock()
    session.merge = AsyncMock()
    session.flush = AsyncMock()

    # Mock async query results
    result_mock = AsyncMock()
    result_mock.fetchone = AsyncMock(return_value=None)
    result_mock.fetchall = AsyncMock(return_value=[])
    result_mock.scalars = AsyncMock()
    result_mock.scalars.return_value.first = AsyncMock(return_value=None)
    result_mock.scalars.return_value.all = AsyncMock(return_value=[])

    session.execute.return_value = result_mock

    return session


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock()
    session = Mock()
    result = Mock()

    # Setup mock chain
    driver.session.return_value = session
    session.run.return_value = result
    session.close = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=None)

    # Mock result operations
    result.data.return_value = []
    result.single.return_value = None
    result.values.return_value = []

    driver.close = Mock()

    return driver


class DatabaseTestHelper:
    """Helper class for database testing operations."""

    @staticmethod
    def create_mock_query_result(data: list[dict[str, Any]]):
        """Create a mock query result with specified data."""
        result = Mock()
        result.fetchall = Mock(return_value=data)
        result.fetchone = Mock(return_value=data[0] if data else None)
        result.rowcount = len(data)
        return result

    @staticmethod
    def create_mock_async_query_result(data: list[dict[str, Any]]):
        """Create a mock async query result with specified data."""
        result = AsyncMock()
        result.fetchall = AsyncMock(return_value=data)
        result.fetchone = AsyncMock(return_value=data[0] if data else None)
        result.rowcount = len(data)

        # Mock scalars for SQLAlchemy async
        scalars_mock = AsyncMock()
        scalars_mock.all = AsyncMock(return_value=data)
        scalars_mock.first = AsyncMock(return_value=data[0] if data else None)
        result.scalars = AsyncMock(return_value=scalars_mock)

        return result

    @staticmethod
    def setup_redis_cache_mock(redis_client, cache_data: dict[str, Any] | None = None):
        """Setup Redis client mock with cache data."""
        if cache_data is None:
            cache_data = {}

        def mock_get(key):
            return cache_data.get(key)

        def mock_set(key, value, ex=None):
            cache_data[key] = value
            return True

        def mock_delete(key):
            return cache_data.pop(key, None) is not None

        redis_client.get = mock_get
        redis_client.set = mock_set
        redis_client.delete = mock_delete
        redis_client.exists = lambda key: key in cache_data
        redis_client.keys = lambda pattern: list(cache_data.keys())

        return redis_client


@pytest.fixture
def database_test_data():
    """Provide common test data for database operations."""
    return {
        "recordings": [
            {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "file_path": "/music/test1.mp3",
                "file_name": "test1.mp3",
                "sha256_hash": "abc123",
                "xxh128_hash": "def456",
            },
            {
                "id": "550e8400-e29b-41d4-a716-446655440002",
                "file_path": "/music/test2.mp3",
                "file_name": "test2.mp3",
                "sha256_hash": "ghi789",
                "xxh128_hash": "jkl012",
            },
        ],
        "metadata": [
            {"id": 1, "recording_id": "550e8400-e29b-41d4-a716-446655440001", "key": "bmp", "value": "128"},
            {"id": 2, "recording_id": "550e8400-e29b-41d4-a716-446655440001", "key": "genre", "value": "techno"},
            {"id": 3, "recording_id": "550e8400-e29b-41d4-a716-446655440002", "key": "bmp", "value": "130"},
        ],
        "tracks": [
            {
                "position": 1,
                "title": "Test Track 1",
                "artist": "Test Artist 1",
                "start_time": "00:00:00",
                "duration": 300,
                "bmp": 128,
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
