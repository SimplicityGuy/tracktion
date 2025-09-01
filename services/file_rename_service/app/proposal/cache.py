"""Redis caching layer for rename proposals with mock fallback."""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

from services.file_rename_service.app.config import get_settings

# Handle optional dependencies properly
HAS_REDIS = False
redis = None
try:
    import redis.asyncio as redis_lib

    redis = redis_lib
    HAS_REDIS = True
except ImportError:
    pass

HAS_MODELS = False
RenameProposal = None
try:
    from services.file_rename_service.app.proposal.models import RenameProposal as ProposalModel

    RenameProposal = ProposalModel
    HAS_MODELS = True
except ImportError:
    pass

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheStats:
    """Thread-safe cache statistics tracking."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._errors = 0
        self._last_reset = datetime.now(UTC)

    async def increment_hit(self) -> None:
        """Increment cache hit counter."""
        async with self._lock:
            self._total_requests += 1
            self._cache_hits += 1

    async def increment_miss(self) -> None:
        """Increment cache miss counter."""
        async with self._lock:
            self._total_requests += 1
            self._cache_misses += 1

    async def increment_error(self) -> None:
        """Increment error counter."""
        async with self._lock:
            self._errors += 1

    async def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        async with self._lock:
            hit_rate = self._cache_hits / self._total_requests if self._total_requests > 0 else 0.0

            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": round(hit_rate, 3),
                "errors": self._errors,
                "last_reset": self._last_reset.isoformat(),
            }

    async def reset(self) -> None:
        """Reset all statistics."""
        async with self._lock:
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._errors = 0
            self._last_reset = datetime.now(UTC)


class MockRedisCache:
    """Mock Redis cache implementation for testing and development."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()
        self._stats = CacheStats()

    async def get(self, key: str) -> str | None:
        """Get value from mock cache."""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    await self._stats.increment_hit()
                    return value
                # Expired, remove from cache
                del self._cache[key]

            await self._stats.increment_miss()
            return None

    async def setex(self, key: str, ttl: int, value: str) -> None:
        """Set value in mock cache with TTL."""
        async with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)

    async def delete(self, *keys: str) -> int:
        """Delete keys from mock cache."""
        async with self._lock:
            deleted = 0
            for key in keys:
                if key in self._cache:
                    del self._cache[key]
                    deleted += 1
            return deleted

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern from mock cache."""
        async with self._lock:
            # Simple pattern matching - replace * with anything
            regex_pattern = pattern.replace("*", ".*")
            compiled_pattern = re.compile(regex_pattern)

            # Only return non-expired keys
            current_time = time.time()
            valid_keys = []
            expired_keys = []

            for key, (_, expiry) in self._cache.items():
                if current_time < expiry:
                    if compiled_pattern.match(key):
                        valid_keys.append(key)
                else:
                    expired_keys.append(key)

            # Clean up expired keys
            for key in expired_keys:
                del self._cache[key]

            return valid_keys

    async def ping(self) -> bool:
        """Mock ping method."""
        return True

    async def flushdb(self) -> None:
        """Clear all cache data."""
        async with self._lock:
            self._cache.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = await self._stats.get_stats()
        async with self._lock:
            stats["cache_size"] = len(self._cache)
            stats["memory_usage_bytes"] = sum(len(key) + len(value) for key, (value, _) in self._cache.items())
        return stats


class ProposalCache:
    """Redis caching layer for rename proposals with fallback to mock implementation."""

    def __init__(self) -> None:
        self._redis_client: Any | None = None  # redis.Redis | None when redis is available
        self._mock_cache: MockRedisCache | None = None
        self._use_mock = False
        self._stats = CacheStats()
        self._connection_pool: Any | None = None  # redis.ConnectionPool | None when redis is available

    async def _get_client(self) -> Any:
        """Get Redis client or mock cache, initializing if needed."""
        if self._use_mock:
            if self._mock_cache is None:
                self._mock_cache = MockRedisCache()
            return self._mock_cache

        if self._redis_client is None:
            try:
                # Check if redis is available
                if redis is None:
                    raise ImportError("Redis library not available")

                # Create connection pool if not exists
                if self._connection_pool is None:
                    self._connection_pool = redis.ConnectionPool.from_url(
                        settings.redis_url,
                        max_connections=settings.redis_pool_size,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                    )

                self._redis_client = redis.Redis(connection_pool=self._connection_pool)

                # Test connection
                await self._redis_client.ping()
                logger.info("Connected to Redis successfully")

            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to mock cache.")
                self._use_mock = True
                if self._mock_cache is None:
                    self._mock_cache = MockRedisCache()
                return self._mock_cache

        return self._redis_client

    def generate_cache_key(self, filename: str, template_id: str | None = None) -> str:
        """Generate cache key for filename and optional template."""
        # Create a consistent cache key from filename and template
        key_parts = [filename.lower().strip()]
        if template_id:
            key_parts.append(template_id)

        # Hash the key to ensure consistent length and avoid special characters
        key_data = "|".join(key_parts)
        hash_value = hashlib.sha256(key_data.encode()).hexdigest()[:16]

        prefix = "proposal"
        return f"{prefix}:{hash_value}:{len(filename)}"

    async def cache_proposal(
        self,
        key: str,
        proposal: Any,  # RenameProposal when available
        ttl: int = 900,
    ) -> bool:
        """Cache a rename proposal with TTL (default 15 minutes)."""
        try:
            client = await self._get_client()

            # Serialize the proposal to JSON
            proposal_data = {
                "data": proposal.model_dump(),
                "cached_at": datetime.now(UTC).isoformat(),
                "ttl": ttl,
            }

            serialized_data = json.dumps(proposal_data, ensure_ascii=False, separators=(",", ":"))

            if isinstance(client, MockRedisCache):
                await client.setex(key, ttl, serialized_data)
            else:
                await client.setex(key, ttl, serialized_data)

            logger.debug(f"Cached proposal for key: {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache proposal for key {key}: {e}")
            await self._stats.increment_error()
            return False

    async def get_cached_proposal(self, key: str) -> Any | None:
        """Retrieve a cached rename proposal."""
        try:
            client = await self._get_client()

            if isinstance(client, MockRedisCache):
                cached_data = await client.get(key)
                # Stats are handled by MockRedisCache internally
            else:
                cached_data = await client.get(key)
                if cached_data:
                    await self._stats.increment_hit()
                else:
                    await self._stats.increment_miss()

            if not cached_data:
                return None

            # Deserialize the proposal data
            try:
                proposal_data = json.loads(cached_data)
                proposal_dict = proposal_data["data"]

                # Validate RenameProposal if available
                if HAS_MODELS and RenameProposal is not None:
                    proposal = RenameProposal.model_validate(proposal_dict)
                else:
                    # If models can't be imported, return raw dict
                    proposal = proposal_dict

                logger.debug(f"Retrieved cached proposal for key: {key}")
                return proposal

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to deserialize cached proposal for key {key}: {e}")
                # Remove corrupted cache entry
                await self.invalidate_cache(key)
                await self._stats.increment_error()
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve cached proposal for key {key}: {e}")
            await self._stats.increment_error()
            return None

    async def invalidate_cache(self, pattern: str | None = None) -> int:
        """Invalidate cache entries matching pattern or specific key."""
        try:
            client = await self._get_client()

            if pattern is None:
                # If no pattern, assume it's a specific key
                logger.warning("No pattern provided for cache invalidation")
                return 0

            # If pattern doesn't contain wildcards, treat as specific key
            if "*" not in pattern:
                deleted = await client.delete(pattern)
                logger.debug(f"Invalidated cache key: {pattern}")
                return int(deleted) if deleted is not None else 0

            # Pattern matching - get keys and delete them
            keys = await client.keys(pattern)
            if keys:
                deleted = await client.delete(*keys)
                logger.debug(f"Invalidated {deleted} cache keys matching pattern: {pattern}")
                return int(deleted) if deleted is not None else 0

            return 0

        except Exception as e:
            logger.error(f"Failed to invalidate cache with pattern {pattern}: {e}")
            await self._stats.increment_error()
            return 0

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            client = await self._get_client()

            # Get basic stats
            base_stats = await self._stats.get_stats()

            # Add cache-specific stats
            cache_stats = {
                **base_stats,
                "backend": "mock" if self._use_mock else "redis",
                "connection_status": "connected" if await self._is_connected() else "disconnected",
            }

            if isinstance(client, MockRedisCache):
                mock_stats = await client.get_stats()
                cache_stats.update(
                    {
                        "cache_size": mock_stats["cache_size"],
                        "memory_usage_bytes": mock_stats["memory_usage_bytes"],
                    }
                )
            else:
                # For real Redis, we can get more detailed info
                try:
                    info = await client.info("memory")
                    cache_stats.update(
                        {
                            "redis_memory_used": info.get("used_memory", 0),
                            "redis_memory_human": info.get("used_memory_human", "0B"),
                        }
                    )
                except Exception:
                    # Redis info might not be available in all configurations
                    pass

            return cache_stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "error": str(e),
                "backend": "mock" if self._use_mock else "redis",
                "connection_status": "error",
            }

    async def _is_connected(self) -> bool:
        """Check if cache backend is connected."""
        try:
            client = await self._get_client()

            if isinstance(client, MockRedisCache):
                return await client.ping()
            result = await client.ping()
            return bool(result)
        except Exception:
            return False

    async def clear_all_cache(self) -> bool:
        """Clear all cached data (use with caution)."""
        try:
            client = await self._get_client()

            if isinstance(client, MockRedisCache):
                await client.flushdb()
            else:
                await client.flushdb()

            # Reset stats
            await self._stats.reset()

            logger.info("Cleared all cache data")
            return True

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection and cleanup resources."""
        if self._redis_client:
            try:
                await self._redis_client.aclose()
                logger.debug("Closed Redis connection")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

        if self._connection_pool:
            try:
                await self._connection_pool.aclose()
                logger.debug("Closed Redis connection pool")
            except Exception as e:
                logger.warning(f"Error closing Redis connection pool: {e}")


# Create a singleton instance for the application
class _CacheManager:
    """Singleton cache manager."""

    def __init__(self) -> None:
        self._instance: ProposalCache | None = None
        self._lock = asyncio.Lock()

    async def get_instance(self) -> ProposalCache:
        """Get the cache instance, creating it if needed."""
        if self._instance is None:
            async with self._lock:
                if self._instance is None:  # Double-checked locking
                    self._instance = ProposalCache()
        return self._instance


_cache_manager = _CacheManager()


async def get_cache() -> ProposalCache:
    """Get the global cache instance."""
    return await _cache_manager.get_instance()


# Convenience functions for common operations
async def cache_proposal_by_filename(
    filename: str,
    proposal: Any,  # RenameProposal when available
    template_id: str | None = None,
    ttl: int = 900,
) -> bool:
    """Cache a proposal using filename-based key generation."""
    cache = await get_cache()
    key = cache.generate_cache_key(filename, template_id)
    return await cache.cache_proposal(key, proposal, ttl)


async def get_cached_proposal_by_filename(filename: str, template_id: str | None = None) -> Any | None:
    """Get cached proposal using filename-based key generation."""
    cache = await get_cache()
    key = cache.generate_cache_key(filename, template_id)
    return await cache.get_cached_proposal(key)


async def invalidate_filename_cache(filename: str, template_id: str | None = None) -> int:
    """Invalidate cache for specific filename."""
    cache = await get_cache()
    key = cache.generate_cache_key(filename, template_id)
    return await cache.invalidate_cache(key)


async def invalidate_all_proposals() -> int:
    """Invalidate all proposal cache entries."""
    cache = await get_cache()
    return await cache.invalidate_cache("proposal:*")
