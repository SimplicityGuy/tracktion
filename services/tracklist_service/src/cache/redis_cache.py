"""
Redis caching implementation for the tracklist service.

Provides caching for search results to reduce scraping load.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import redis.asyncio as redis
from redis.exceptions import RedisError
from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.models.search_models import (
    CachedSearchResponse,
    CacheKey,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache manager for search results."""

    def __init__(self) -> None:
        """Initialize Redis cache connection."""
        self.config = get_config().cache
        self.enabled = self.config.enabled
        self.client: redis.Redis | None = None

        if self.enabled:
            try:
                self.client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Connection will be tested on first use
                logger.info("Redis cache initialized successfully")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enabled = False
                self.client = None

    async def get_cached_response(self, request: SearchRequest) -> SearchResponse | None:
        """Get cached search response if available.

        Args:
            request: Search request to look up

        Returns:
            Cached search response or None if not found/expired
        """
        if not self.enabled or not self.client:
            return None

        try:
            # Generate cache key
            cache_key = CacheKey(
                search_type=request.search_type,
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
                page=request.page,
                limit=request.limit,
            )

            key = cache_key.generate_key(self.config.key_prefix)

            # Get cached data
            cached_data = await self.client.get(key)

            if cached_data:
                logger.debug(f"Cache hit for key: {key}")

                # Parse cached response
                cached_response = CachedSearchResponse.model_validate_json(cached_data)

                # Check if still valid
                if cached_response.expires_at > datetime.now(UTC).replace(tzinfo=None):
                    # Update response to indicate cache hit
                    response = cached_response.response
                    response.cache_hit = True
                    response.correlation_id = request.correlation_id
                    return cast("SearchResponse | None", response)
                # Expired - delete from cache
                logger.debug(f"Cache expired for key: {key}")
                await self.client.delete(key)

            logger.debug(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    async def cache_response(
        self,
        request: SearchRequest,
        response: SearchResponse,
        ttl_hours: int | None = None,
    ) -> bool:
        """Cache a search response.

        Args:
            request: Original search request
            response: Search response to cache
            ttl_hours: Time to live in hours (uses config default if not specified)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            # Generate cache key
            cache_key = CacheKey(
                search_type=request.search_type,
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
                page=request.page,
                limit=request.limit,
            )

            key = cache_key.generate_key(self.config.key_prefix)

            # Determine TTL
            if ttl_hours is None:
                ttl_hours = self.config.search_ttl_hours

            # Create cached response
            now = datetime.now(UTC).replace(tzinfo=None)
            expires_at = now + timedelta(hours=ttl_hours)

            cached_response = CachedSearchResponse(
                response=response,
                cached_at=now,
                expires_at=expires_at,
                cache_version="1.0",
            )

            # Store in Redis with expiration
            ttl_seconds = ttl_hours * 3600
            await self.client.setex(
                key,
                ttl_seconds,
                cached_response.model_dump_json(),
            )

            logger.debug(f"Cached response for key: {key}, TTL: {ttl_hours}h")
            return True

        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False

    async def cache_failed_search(self, request: SearchRequest, error_message: str) -> bool:
        """Cache a failed search to prevent repeated failures.

        Args:
            request: Search request that failed
            error_message: Error message to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            # Generate cache key with 'failed' prefix
            cache_key = CacheKey(
                search_type=request.search_type,
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
                page=request.page,
                limit=request.limit,
            )

            key = cache_key.generate_key(f"{self.config.key_prefix}failed:")

            # Store error with shorter TTL
            failed_data = {
                "error": error_message,
                "failed_at": datetime.now(UTC).isoformat(),
                "correlation_id": (str(request.correlation_id) if request.correlation_id else None),
            }

            ttl_seconds = self.config.failed_search_ttl_minutes * 60
            await self.client.setex(
                key,
                ttl_seconds,
                json.dumps(failed_data),
            )

            logger.debug(f"Cached failed search for key: {key}")
            return True

        except Exception as e:
            logger.error(f"Error caching failed search: {e}")
            return False

    async def is_search_failed_recently(self, request: SearchRequest) -> str | None:
        """Check if a search has failed recently.

        Args:
            request: Search request to check

        Returns:
            Error message if search failed recently, None otherwise
        """
        if not self.enabled or not self.client:
            return None

        try:
            # Generate cache key with 'failed' prefix
            cache_key = CacheKey(
                search_type=request.search_type,
                query=request.query,
                start_date=request.start_date,
                end_date=request.end_date,
                page=request.page,
                limit=request.limit,
            )

            key = cache_key.generate_key(f"{self.config.key_prefix}failed:")

            # Check for failed search
            failed_data = await self.client.get(key)

            if failed_data:
                data = json.loads(failed_data)
                logger.debug(f"Found recent failed search for key: {key}")
                return str(data.get("error", "Previous search failed"))

            return None

        except Exception as e:
            logger.error(f"Error checking failed search cache: {e}")
            return None

    async def clear_cache(self, pattern: str | None = None) -> int:
        """Clear cache entries matching a pattern.

        Args:
            pattern: Redis key pattern to match (e.g., "tracklist:search:dj:*")
                    If None, clears all tracklist cache entries

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            if pattern is None:
                pattern = f"{self.config.key_prefix}*"

            # Find all matching keys
            keys = [key async for key in self.client.scan_iter(match=pattern)]

            if keys:
                # Delete all matching keys
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return int(deleted)

            return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.client:
            return {"enabled": False}

        try:
            # Get Redis info
            info = await self.client.info()

            # Count tracklist keys
            tracklist_keys = 0
            async for _key in self.client.scan_iter(match=f"{self.config.key_prefix}*"):
                tracklist_keys += 1
            failed_keys = 0
            async for _key in self.client.scan_iter(match=f"{self.config.key_prefix}failed:*"):
                failed_keys += 1

            return {
                "enabled": True,
                "connected": True,
                "total_keys": tracklist_keys,
                "failed_keys": failed_keys,
                "memory_used": info.get("used_memory_human", "N/A"),
                "hit_rate": info.get("keyspace_hit_ratio", 0),
                "evicted_keys": info.get("evicted_keys", 0),
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}

    async def get(self, key: str) -> str | None:
        """Get a value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.client:
            return None

        try:
            value = await self.client.get(key)
            return str(value) if value is not None else None
        except Exception as e:
            logger.error(f"Error getting value from cache: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set a value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            await self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error setting value in cache: {e}")
            return False

    async def delete(self, key: str) -> int:
        """Delete a key from cache.

        Args:
            key: Cache key to delete

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            return int(await self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return 0

    async def ping(self) -> bool:
        """Check if Redis is accessible.

        Returns:
            True if Redis is accessible, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            try:
                await self.client.aclose()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


class RedisCacheSingleton:
    """Singleton wrapper for RedisCache."""

    _instance: RedisCache | None = None

    def __new__(cls) -> "RedisCacheSingleton":
        """Get the singleton RedisCache instance."""
        if cls._instance is None:
            cls._instance = RedisCache()
        return cls._instance  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped RedisCache instance."""
        if self._instance is None:
            self._instance = RedisCache()
        return getattr(self._instance, name)

    @classmethod
    async def close(cls) -> None:
        """Close the singleton cache instance."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


def get_cache() -> "RedisCacheSingleton":
    """Get or create the singleton cache instance.

    Returns:
        RedisCache singleton instance
    """
    return RedisCacheSingleton()


async def close_cache() -> None:
    """Close the singleton cache instance."""
    await RedisCacheSingleton.close()
