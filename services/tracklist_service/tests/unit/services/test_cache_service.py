"""
Tests for CacheService implementation.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from services.cache_service import CacheService, CacheMetrics, CacheConfig


class TestCacheConfig:
    """Test CacheConfig model."""

    def test_cache_config_defaults(self) -> None:
        """Test cache config with default values."""
        config = CacheConfig()
        assert config.redis_host == "localhost"
        assert config.redis_port == 6379
        assert config.redis_db == 0
        assert config.connection_timeout == 5
        assert config.max_connections == 10
        assert config.default_ttl == 3600
        assert config.cue_content_ttl == 7200
        assert config.format_capabilities_ttl == 86400


class TestCacheMetrics:
    """Test CacheMetrics model."""

    def test_cache_metrics_default(self) -> None:
        """Test cache metrics with default values."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0
        assert metrics.total_operations == 0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

        metrics.hits = 8
        metrics.misses = 2
        assert metrics.hit_rate == 80.0  # percentage

        metrics.hits = 0
        metrics.misses = 5
        assert metrics.hit_rate == 0.0


@pytest.fixture
def cache_config() -> CacheConfig:
    """Create a test cache configuration."""
    return CacheConfig(
        redis_host="localhost",
        redis_port=6379,
        memory_cache_max_size=100,
        memory_cache_ttl=300,
        default_ttl=300,
        cue_content_ttl=600,
        format_capabilities_ttl=1200,
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Create a mock Redis connection."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = 1
    redis_mock.keys.return_value = []
    redis_mock.close.return_value = None
    return redis_mock


class TestCacheService:
    """Test CacheService implementation."""

    @pytest.mark.asyncio
    async def test_cache_service_initialization(self, cache_config: CacheConfig) -> None:
        """Test cache service initialization."""
        service = CacheService(cache_config)
        assert service.config == cache_config
        assert service.redis_client is None
        assert service.memory_cache is not None
        assert service.metrics.hits == 0

    @pytest.mark.asyncio
    async def test_cue_content_caching_memory_fallback(self, cache_config: CacheConfig) -> None:
        """Test CUE content caching with memory fallback."""
        service = CacheService(cache_config)
        # Don't connect to Redis to test memory fallback

        tracklist_id = uuid4()
        format_name = "standard"
        content = 'FILE "audio.wav" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00'

        # Test cache miss
        result = await service.get_cue_content(tracklist_id, format_name)
        assert result is None
        assert service.metrics.misses == 1

        # Test cache set
        success = await service.set_cue_content(tracklist_id, format_name, content)
        assert success is True
        assert service.metrics.sets == 1

        # Test cache hit
        result = await service.get_cue_content(tracklist_id, format_name)
        assert result == content
        assert service.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_format_capabilities_caching(self, cache_config: CacheConfig) -> None:
        """Test format capabilities caching."""
        service = CacheService(cache_config)

        format_name = "traktor"
        capabilities = {"max_tracks": 100, "supports_bpm": True, "supports_keys": True, "supports_cue_points": True}

        # Test cache miss
        result = await service.get_format_capabilities(format_name)
        assert result is None
        assert service.metrics.misses == 1

        # Test cache set
        success = await service.set_format_capabilities(format_name, capabilities)
        assert success is True
        assert service.metrics.sets == 1

        # Test cache hit (from memory cache)
        result = await service.get_format_capabilities(format_name)
        assert result == capabilities
        assert service.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_config: CacheConfig) -> None:
        """Test cache invalidation."""
        service = CacheService(cache_config)

        tracklist_id = uuid4()

        # Add some content to cache
        await service.set_cue_content(tracklist_id, "standard", "content1")
        await service.set_cue_content(tracklist_id, "traktor", "content2")

        # Test invalidation
        count = await service.invalidate_tracklist_cache(tracklist_id)
        # Since we're using memory cache, it depends on implementation
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_config: CacheConfig) -> None:
        """Test cache statistics."""
        service = CacheService(cache_config)

        # Perform some operations
        tracklist_id = uuid4()
        await service.set_cue_content(tracklist_id, "standard", "content")
        await service.get_cue_content(tracklist_id, "standard")
        await service.get_cue_content(tracklist_id, "nonexistent")

        stats = await service.get_cache_stats()

        assert isinstance(stats, dict)
        assert "metrics" in stats
        assert "hits" in stats["metrics"]
        assert "misses" in stats["metrics"]
        assert "sets" in stats["metrics"]

    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_config: CacheConfig) -> None:
        """Test cache warming."""
        service = CacheService(cache_config)

        tracklist_ids = [uuid4(), uuid4()]
        formats = ["standard", "traktor"]

        # Test cache warming
        result = await service.warm_cache(tracklist_ids, formats)

        assert isinstance(result, dict)
        # The actual implementation depends on having tracklist data
        # So we just test that it returns a dict response

    @pytest.mark.asyncio
    async def test_cache_clearing(self, cache_config: CacheConfig) -> None:
        """Test cache clearing."""
        service = CacheService(cache_config)

        # Add some content
        tracklist_id = uuid4()
        await service.set_cue_content(tracklist_id, "standard", "content")

        # Clear cache
        count = await service.clear_cache()

        assert isinstance(count, int)

        # Verify content is gone
        result = await service.get_cue_content(tracklist_id, "standard")
        assert result is None

    @pytest.mark.asyncio
    async def test_validation_caching(self, cache_config: CacheConfig) -> None:
        """Test validation result caching."""
        service = CacheService(cache_config)

        cue_file_id = uuid4()
        validation_result = {"valid": True, "warnings": [], "metadata": {"test": "data"}}

        # Test cache miss
        result = await service.get_validation_result(cue_file_id)
        assert result is None
        assert service.metrics.misses == 1

        # Test cache set
        success = await service.set_validation_result(cue_file_id, validation_result)
        assert success is True
        assert service.metrics.sets == 1

        # Test cache hit
        result = await service.get_validation_result(cue_file_id)
        assert result == validation_result
        assert service.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_disconnect_without_connection(self, cache_config: CacheConfig) -> None:
        """Test disconnecting when no Redis connection exists."""
        service = CacheService(cache_config)

        # Should not raise exception when disconnecting without connection
        await service.disconnect()



