"""Unit tests for sync state caching service."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import redis.asyncio as redis
from redis.exceptions import RedisError

from services.tracklist_service.src.services.sync_cache_service import SyncCacheService


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock(spec=redis.Redis)
    client.ping = AsyncMock()
    client.setex = AsyncMock()
    client.get = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock()
    client.zadd = AsyncMock()
    client.expire = AsyncMock()
    client.zrangebyscore = AsyncMock()
    client.info = AsyncMock()
    client.scan_iter = AsyncMock()
    return client


@pytest.fixture
def cache_service(mock_redis_client):
    """Create sync cache service instance."""
    service = SyncCacheService(redis_client=mock_redis_client)
    service._initialized = True
    return service


@pytest.fixture
def sample_sync_state():
    """Create sample sync state data."""
    return {
        "status": "completed",
        "last_sync": datetime.now(UTC).isoformat(),
        "changes_applied": 5,
        "confidence": 0.9,
    }


@pytest.fixture
def sample_conflicts():
    """Create sample conflict data."""
    return [
        {
            "id": "conflict1",
            "type": "track_modified",
            "severity": "medium",
        },
        {
            "id": "conflict2",
            "type": "track_added",
            "severity": "low",
        },
    ]


class TestSyncCacheService:
    """Test SyncCacheService methods."""

    @pytest.mark.asyncio
    async def test_initialize_with_connection(self):
        """Test initializing with Redis connection."""
        service = SyncCacheService()

        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock()
            mock_redis_class.return_value = mock_client

            await service.initialize()

            assert service._initialized is True
            assert service.redis_client is not None

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self):
        """Test handling Redis connection failure."""
        service = SyncCacheService()

        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_redis_class.side_effect = Exception("Connection failed")

            await service.initialize()

            assert service._initialized is True
            assert service.redis_client is None  # Falls back to no caching

    @pytest.mark.asyncio
    async def test_cache_sync_state(self, cache_service, mock_redis_client, sample_sync_state):
        """Test caching synchronization state."""
        tracklist_id = uuid4()

        result = await cache_service.cache_sync_state(tracklist_id, sample_sync_state)

        assert result is True
        mock_redis_client.setex.assert_called_once()

        # Check the key and TTL
        call_args = mock_redis_client.setex.call_args
        assert f"sync:state:{tracklist_id}" in call_args[0]
        assert call_args[0][1] == cache_service.STATE_TTL

    @pytest.mark.asyncio
    async def test_cache_sync_state_error(self, cache_service, mock_redis_client, sample_sync_state):
        """Test handling Redis error when caching state."""
        tracklist_id = uuid4()
        mock_redis_client.setex.side_effect = RedisError("Redis error")

        result = await cache_service.cache_sync_state(tracklist_id, sample_sync_state)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_sync_state(self, cache_service, mock_redis_client, sample_sync_state):
        """Test retrieving cached sync state."""
        tracklist_id = uuid4()
        mock_redis_client.get.return_value = json.dumps(sample_sync_state)

        result = await cache_service.get_sync_state(tracklist_id)

        assert result == sample_sync_state
        mock_redis_client.get.assert_called_once_with(f"sync:state:{tracklist_id}")

    @pytest.mark.asyncio
    async def test_get_sync_state_not_found(self, cache_service, mock_redis_client):
        """Test retrieving non-existent sync state."""
        tracklist_id = uuid4()
        mock_redis_client.get.return_value = None

        result = await cache_service.get_sync_state(tracklist_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_sync_state(self, cache_service, mock_redis_client):
        """Test invalidating cached sync state."""
        tracklist_id = uuid4()
        mock_redis_client.delete.return_value = 1

        result = await cache_service.invalidate_sync_state(tracklist_id)

        assert result is True
        mock_redis_client.delete.assert_called_once_with(f"sync:state:{tracklist_id}")

    @pytest.mark.asyncio
    async def test_acquire_sync_lock(self, cache_service, mock_redis_client):
        """Test acquiring a distributed lock."""
        tracklist_id = uuid4()

        mock_lock = AsyncMock()
        mock_lock.acquire = AsyncMock(return_value=True)
        mock_redis_client.lock.return_value = mock_lock

        lock = await cache_service.acquire_sync_lock(tracklist_id)

        assert lock is not None
        mock_redis_client.lock.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_sync_lock_failure(self, cache_service, mock_redis_client):
        """Test failing to acquire a lock."""
        tracklist_id = uuid4()

        mock_lock = AsyncMock()
        mock_lock.acquire = AsyncMock(return_value=False)
        mock_redis_client.lock.return_value = mock_lock

        lock = await cache_service.acquire_sync_lock(tracklist_id)

        assert lock is None

    @pytest.mark.asyncio
    async def test_release_sync_lock(self, cache_service):
        """Test releasing a distributed lock."""
        mock_lock = AsyncMock()
        mock_lock.release = AsyncMock()

        result = await cache_service.release_sync_lock(mock_lock)

        assert result is True
        mock_lock.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_locked(self, cache_service, mock_redis_client):
        """Test checking if a tracklist is locked."""
        tracklist_id = uuid4()
        mock_redis_client.exists.return_value = 1

        result = await cache_service.is_locked(tracklist_id)

        assert result is True
        mock_redis_client.exists.assert_called_once_with(f"sync:lock:{tracklist_id}")

    @pytest.mark.asyncio
    async def test_cache_conflicts(self, cache_service, mock_redis_client, sample_conflicts):
        """Test caching conflicts."""
        tracklist_id = uuid4()

        result = await cache_service.cache_conflicts(tracklist_id, sample_conflicts)

        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_conflicts(self, cache_service, mock_redis_client, sample_conflicts):
        """Test retrieving cached conflicts."""
        tracklist_id = uuid4()
        conflict_data = {
            "conflicts": sample_conflicts,
            "count": len(sample_conflicts),
            "cached_at": datetime.now(UTC).isoformat(),
        }
        mock_redis_client.get.return_value = json.dumps(conflict_data)

        result = await cache_service.get_cached_conflicts(tracklist_id)

        assert result == sample_conflicts

    @pytest.mark.asyncio
    async def test_cache_version_info(self, cache_service, mock_redis_client):
        """Test caching version information."""
        version_id = uuid4()
        version_data = {
            "version_number": 5,
            "change_type": "manual",
            "change_summary": "Test changes",
        }

        result = await cache_service.cache_version_info(version_id, version_data)

        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cached_version(self, cache_service, mock_redis_client):
        """Test retrieving cached version."""
        version_id = uuid4()
        version_data = {
            "version_number": 5,
            "change_type": "manual",
            "cached_at": datetime.now(UTC).isoformat(),
        }
        mock_redis_client.get.return_value = json.dumps(version_data)

        result = await cache_service.get_cached_version(version_id)

        assert result == version_data

    @pytest.mark.asyncio
    async def test_cache_batch_progress(self, cache_service, mock_redis_client):
        """Test caching batch operation progress."""
        batch_id = uuid4()
        progress = {
            "total": 10,
            "completed": 5,
            "successful": 3,
            "failed": 2,
        }

        result = await cache_service.cache_batch_progress(batch_id, progress)

        assert result is True
        mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_sync_metric(self, cache_service, mock_redis_client):
        """Test recording a sync metric."""
        tracklist_id = uuid4()

        result = await cache_service.record_sync_metric(
            tracklist_id,
            "duration",
            120.5,
        )

        assert result is True
        mock_redis_client.zadd.assert_called_once()
        mock_redis_client.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sync_metrics(self, cache_service, mock_redis_client):
        """Test retrieving sync metrics."""
        tracklist_id = uuid4()
        metric_data = [
            json.dumps({"value": 120.5, "time": 1234567890}),
            json.dumps({"value": 95.2, "time": 1234567900}),
        ]
        mock_redis_client.zrangebyscore.return_value = metric_data

        result = await cache_service.get_sync_metrics(tracklist_id, "duration")

        assert len(result) == 2
        assert result[0]["value"] == 120.5

    @pytest.mark.asyncio
    async def test_invalidate_all_caches(self, cache_service, mock_redis_client):
        """Test invalidating all caches for a tracklist."""
        tracklist_id = uuid4()

        # Mock scan_iter to return some keys
        async def mock_scan_iter(pattern):
            if "state" in pattern:
                yield f"sync:state:{tracklist_id}"
            elif "conflict" in pattern:
                yield f"sync:conflict:{tracklist_id}"
            elif "metrics" in pattern:
                yield f"sync:metrics:{tracklist_id}:duration"

        mock_redis_client.scan_iter = mock_scan_iter
        mock_redis_client.delete.return_value = 1

        result = await cache_service.invalidate_all_caches(tracklist_id)

        assert result == 3  # 3 keys deleted

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_service, mock_redis_client):
        """Test getting cache statistics."""
        mock_redis_client.info.return_value = {
            "connected_clients": 5,
            "used_memory_human": "1.5M",
            "keyspace_hits": 1000,
            "keyspace_misses": 100,
        }

        stats = await cache_service.get_cache_stats()

        assert stats["status"] == "active"
        assert stats["connected_clients"] == 5
        assert stats["hits"] == 1000
        assert stats["hit_rate"] == 0.9090909090909091  # 1000 / (1000 + 100)

    @pytest.mark.asyncio
    async def test_no_redis_fallback(self):
        """Test service works without Redis."""
        service = SyncCacheService(redis_client=None)
        tracklist_id = uuid4()

        # All operations should return False/None without Redis
        assert await service.cache_sync_state(tracklist_id, {}) is False
        assert await service.get_sync_state(tracklist_id) is None
        assert await service.acquire_sync_lock(tracklist_id) is None
        assert await service.is_locked(tracklist_id) is False
        assert await service.get_cache_stats() == {"status": "disabled"}
