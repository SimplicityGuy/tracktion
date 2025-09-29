"""Sync state caching service using Redis for performance optimization."""

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.asyncio.lock import Lock
from redis.exceptions import LockError, RedisError
from services.tracklist_service.src.config import get_config

logger = logging.getLogger(__name__)


class SyncCacheService:
    """Service for caching synchronization state and managing distributed locks."""

    def __init__(self, redis_client: Redis | None = None):
        """Initialize sync cache service.

        Args:
            redis_client: Redis client instance
        """
        self.config = get_config()
        self.redis_client = redis_client
        self._initialized = False

        # Cache key prefixes
        self.SYNC_STATE_PREFIX = "sync:state:"
        self.SYNC_LOCK_PREFIX = "sync:lock:"
        self.CONFLICT_PREFIX = "sync:conflict:"
        self.VERSION_PREFIX = "sync:version:"
        self.BATCH_PREFIX = "sync:batch:"
        self.METRICS_PREFIX = "sync:metrics:"

        # Default TTLs
        self.STATE_TTL = 3600  # 1 hour
        self.CONFLICT_TTL = 86400  # 24 hours
        self.VERSION_TTL = 604800  # 7 days
        self.LOCK_TTL = 300  # 5 minutes
        self.METRICS_TTL = 2592000  # 30 days

    async def initialize(self) -> None:
        """Initialize Redis connection if not provided."""
        if self._initialized:
            return

        if not self.redis_client:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.cache.redis_host,
                    port=self.config.cache.redis_port,
                    db=self.config.cache.redis_db,
                    password=self.config.cache.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Connected to Redis for sync state caching")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Fall back to no caching
                self.redis_client = None

        self._initialized = True

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    # Sync State Caching
    async def cache_sync_state(
        self,
        tracklist_id: UUID,
        state: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache synchronization state for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            state: Sync state data
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"{self.SYNC_STATE_PREFIX}{tracklist_id}"

            # Add timestamp
            state["cached_at"] = datetime.now(UTC).isoformat()

            # Serialize and store
            await self.redis_client.setex(
                key,
                ttl or self.STATE_TTL,
                json.dumps(state, default=str),
            )

            logger.debug(f"Cached sync state for tracklist {tracklist_id}")
            return True

        except RedisError as e:
            logger.error(f"Failed to cache sync state: {e}")
            return False

    async def get_sync_state(
        self,
        tracklist_id: UUID,
    ) -> dict[str, Any] | None:
        """Get cached synchronization state.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Cached state or None
        """
        if not self.redis_client:
            return None

        try:
            key = f"{self.SYNC_STATE_PREFIX}{tracklist_id}"
            data = await self.redis_client.get(key)

            if data:
                state = json.loads(data)
                logger.debug(f"Retrieved cached sync state for tracklist {tracklist_id}")
                return state  # type: ignore[no-any-return]

            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get cached sync state: {e}")
            return None

    async def invalidate_sync_state(
        self,
        tracklist_id: UUID,
    ) -> bool:
        """Invalidate cached sync state.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            True if invalidated successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"{self.SYNC_STATE_PREFIX}{tracklist_id}"
            result = await self.redis_client.delete(key)

            logger.debug(f"Invalidated sync state for tracklist {tracklist_id}")
            return bool(result > 0)

        except RedisError as e:
            logger.error(f"Failed to invalidate sync state: {e}")
            return False

    # Distributed Locking
    async def acquire_sync_lock(
        self,
        tracklist_id: UUID,
        timeout: int | None = None,
        blocking: bool = True,
        blocking_timeout: int | None = None,
    ) -> Lock | None:
        """Acquire a distributed lock for synchronization.

        Args:
            tracklist_id: ID of the tracklist
            timeout: Lock timeout in seconds
            blocking: Whether to block waiting for lock
            blocking_timeout: Maximum time to wait for lock

        Returns:
            Lock object if acquired, None otherwise
        """
        if not self.redis_client:
            return None

        try:
            lock_name = f"{self.SYNC_LOCK_PREFIX}{tracklist_id}"

            lock = self.redis_client.lock(
                lock_name,
                timeout=timeout or self.LOCK_TTL,
                blocking=blocking,
                blocking_timeout=blocking_timeout or 10,
            )

            if await lock.acquire():
                logger.debug(f"Acquired sync lock for tracklist {tracklist_id}")
                return lock
            logger.warning(f"Failed to acquire sync lock for tracklist {tracklist_id}")
            return None

        except LockError as e:
            logger.error(f"Lock error: {e}")
            return None

    async def release_sync_lock(
        self,
        lock: Lock,
    ) -> bool:
        """Release a distributed lock.

        Args:
            lock: Lock object to release

        Returns:
            True if released successfully
        """
        try:
            await lock.release()
            logger.debug("Released sync lock")
            return True

        except LockError as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    async def is_locked(
        self,
        tracklist_id: UUID,
    ) -> bool:
        """Check if a tracklist is locked for sync.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            True if locked
        """
        if not self.redis_client:
            return False

        try:
            lock_name = f"{self.SYNC_LOCK_PREFIX}{tracklist_id}"
            result = await self.redis_client.exists(lock_name)
            return bool(result > 0)

        except RedisError as e:
            logger.error(f"Failed to check lock status: {e}")
            return False

    # Conflict Caching
    async def cache_conflicts(
        self,
        tracklist_id: UUID,
        conflicts: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> bool:
        """Cache detected conflicts.

        Args:
            tracklist_id: ID of the tracklist
            conflicts: List of conflicts
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"{self.CONFLICT_PREFIX}{tracklist_id}"

            data = {
                "conflicts": conflicts,
                "count": len(conflicts),
                "cached_at": datetime.now(UTC).isoformat(),
            }

            await self.redis_client.setex(
                key,
                ttl or self.CONFLICT_TTL,
                json.dumps(data, default=str),
            )

            logger.debug(f"Cached {len(conflicts)} conflicts for tracklist {tracklist_id}")
            return True

        except RedisError as e:
            logger.error(f"Failed to cache conflicts: {e}")
            return False

    async def get_cached_conflicts(
        self,
        tracklist_id: UUID,
    ) -> list[dict[str, Any]] | None:
        """Get cached conflicts.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            List of conflicts or None
        """
        if not self.redis_client:
            return None

        try:
            key = f"{self.CONFLICT_PREFIX}{tracklist_id}"
            data = await self.redis_client.get(key)

            if data:
                conflict_data = json.loads(data)
                return conflict_data.get("conflicts", [])  # type: ignore[no-any-return]

            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get cached conflicts: {e}")
            return None

    # Version Caching
    async def cache_version_info(
        self,
        version_id: UUID,
        version_data: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache version information.

        Args:
            version_id: ID of the version
            version_data: Version data
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"{self.VERSION_PREFIX}{version_id}"

            version_data["cached_at"] = datetime.now(UTC).isoformat()

            await self.redis_client.setex(
                key,
                ttl or self.VERSION_TTL,
                json.dumps(version_data, default=str),
            )

            logger.debug(f"Cached version info for {version_id}")
            return True

        except RedisError as e:
            logger.error(f"Failed to cache version info: {e}")
            return False

    async def get_cached_version(
        self,
        version_id: UUID,
    ) -> dict[str, Any] | None:
        """Get cached version information.

        Args:
            version_id: ID of the version

        Returns:
            Version data or None
        """
        if not self.redis_client:
            return None

        try:
            key = f"{self.VERSION_PREFIX}{version_id}"
            data = await self.redis_client.get(key)

            if data:
                return json.loads(data)  # type: ignore[no-any-return]

            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get cached version: {e}")
            return None

    # Batch Operation Caching
    async def cache_batch_progress(
        self,
        batch_id: UUID,
        progress: dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        """Cache batch operation progress.

        Args:
            batch_id: Batch operation ID
            progress: Progress data
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        try:
            key = f"{self.BATCH_PREFIX}{batch_id}"

            progress["updated_at"] = datetime.now(UTC).isoformat()

            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(progress, default=str),
            )

            return True

        except RedisError as e:
            logger.error(f"Failed to cache batch progress: {e}")
            return False

    async def get_batch_progress(
        self,
        batch_id: UUID,
    ) -> dict[str, Any] | None:
        """Get cached batch operation progress.

        Args:
            batch_id: Batch operation ID

        Returns:
            Progress data or None
        """
        if not self.redis_client:
            return None

        try:
            key = f"{self.BATCH_PREFIX}{batch_id}"
            data = await self.redis_client.get(key)

            if data:
                return json.loads(data)  # type: ignore[no-any-return]

            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get batch progress: {e}")
            return None

    # Performance Metrics
    async def record_sync_metric(
        self,
        tracklist_id: UUID,
        metric_type: str,
        value: Any,
    ) -> bool:
        """Record a synchronization metric.

        Args:
            tracklist_id: ID of the tracklist
            metric_type: Type of metric (duration, changes, conflicts, etc.)
            value: Metric value

        Returns:
            True if recorded successfully
        """
        if not self.redis_client:
            return False

        try:
            # Use sorted set for time-series metrics
            key = f"{self.METRICS_PREFIX}{tracklist_id}:{metric_type}"
            timestamp = datetime.now(UTC).timestamp()

            # Store as sorted set with timestamp as score
            await self.redis_client.zadd(
                key,
                {json.dumps({"value": value, "time": timestamp}): timestamp},
            )

            # Set TTL
            await self.redis_client.expire(key, self.METRICS_TTL)

            return True

        except RedisError as e:
            logger.error(f"Failed to record metric: {e}")
            return False

    async def get_sync_metrics(
        self,
        tracklist_id: UUID,
        metric_type: str,
        hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Get synchronization metrics.

        Args:
            tracklist_id: ID of the tracklist
            metric_type: Type of metric
            hours: Number of hours to look back

        Returns:
            List of metrics
        """
        if not self.redis_client:
            return []

        try:
            key = f"{self.METRICS_PREFIX}{tracklist_id}:{metric_type}"

            # Calculate time range
            end_time = datetime.now(UTC).timestamp()
            start_time = end_time - (hours * 3600)

            # Get metrics in time range
            results = await self.redis_client.zrangebyscore(
                key,
                start_time,
                end_time,
            )

            metrics = []
            for item in results:
                try:
                    metrics.append(json.loads(item))
                except json.JSONDecodeError:
                    continue

            return metrics

        except RedisError as e:
            logger.error(f"Failed to get metrics: {e}")
            return []

    # Cache Invalidation
    async def invalidate_all_caches(
        self,
        tracklist_id: UUID,
    ) -> int:
        """Invalidate all caches for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            patterns = [
                f"{self.SYNC_STATE_PREFIX}{tracklist_id}",
                f"{self.CONFLICT_PREFIX}{tracklist_id}",
                f"{self.METRICS_PREFIX}{tracklist_id}:*",
            ]

            deleted = 0
            for pattern in patterns:
                # Use scan to find matching keys
                async for key in self.redis_client.scan_iter(pattern):
                    deleted += await self.redis_client.delete(key)

            logger.info(f"Invalidated {deleted} cache entries for tracklist {tracklist_id}")
            return deleted

        except RedisError as e:
            logger.error(f"Failed to invalidate caches: {e}")
            return 0

    # Cache Monitoring
    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        if not self.redis_client:
            return {"status": "disabled"}

        try:
            info = await self.redis_client.info("stats")

            return {
                "status": "active",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1)
                ),
            }

        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
