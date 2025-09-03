"""Pytest configuration and fixtures for analysis_service tests."""

import fakeredis
import pytest

from services.analysis_service.src.audio_cache import AudioCache


@pytest.fixture
def mock_redis():
    """Provide a fakeredis client for testing Redis operations."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_redis_strict():
    """Provide a strict fakeredis client that mimics Redis behavior more closely."""
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def redis_cache(mock_redis, monkeypatch):
    """Provide an AudioCache instance with fakeredis backend."""

    # Monkeypatch the Redis connection in AudioCache
    def mock_redis_init(self, *args, **kwargs):
        self.redis_client = mock_redis
        self.redis_available = True

    monkeypatch.setattr(AudioCache, "__init__", mock_redis_init)

    cache = AudioCache()
    cache.redis_client = mock_redis
    cache.redis_available = True
    return cache
