"""
Redis caching implementation for the tracklist service.

Provides caching for search results to reduce scraping load.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import redis
from redis.exceptions import RedisError

from ..config import get_config
from ..models.search_models import (
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
        self.client: Optional[redis.Redis] = None

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
                # Test connection
                self.client.ping()
                logger.info("Redis cache connected successfully")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enabled = False
                self.client = None

    def get_cached_response(self, request: SearchRequest) -> Optional[SearchResponse]:
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
            cached_data = self.client.get(key)

            if cached_data:
                logger.debug(f"Cache hit for key: {key}")

                # Parse cached response
                cached_response = CachedSearchResponse.model_validate_json(cached_data)  # type: ignore[arg-type]

                # Check if still valid
                if cached_response.expires_at > datetime.now(timezone.utc).replace(tzinfo=None):
                    # Update response to indicate cache hit
                    response = cached_response.response
                    response.cache_hit = True
                    response.correlation_id = request.correlation_id
                    return response
                else:
                    # Expired - delete from cache
                    logger.debug(f"Cache expired for key: {key}")
                    self.client.delete(key)

            logger.debug(f"Cache miss for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def cache_response(self, request: SearchRequest, response: SearchResponse, ttl_hours: Optional[int] = None) -> bool:
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
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            expires_at = now + timedelta(hours=ttl_hours)

            cached_response = CachedSearchResponse(
                response=response,
                cached_at=now,
                expires_at=expires_at,
                cache_version="1.0",
            )

            # Store in Redis with expiration
            ttl_seconds = ttl_hours * 3600
            self.client.setex(
                key,
                ttl_seconds,
                cached_response.model_dump_json(),
            )

            logger.debug(f"Cached response for key: {key}, TTL: {ttl_hours}h")
            return True

        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False

    def cache_failed_search(self, request: SearchRequest, error_message: str) -> bool:
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
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(request.correlation_id) if request.correlation_id else None,
            }

            ttl_seconds = self.config.failed_search_ttl_minutes * 60
            self.client.setex(
                key,
                ttl_seconds,
                json.dumps(failed_data),
            )

            logger.debug(f"Cached failed search for key: {key}")
            return True

        except Exception as e:
            logger.error(f"Error caching failed search: {e}")
            return False

    def is_search_failed_recently(self, request: SearchRequest) -> Optional[str]:
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
            failed_data = self.client.get(key)

            if failed_data:
                data = json.loads(failed_data)  # type: ignore[arg-type]
                logger.debug(f"Found recent failed search for key: {key}")
                return str(data.get("error", "Previous search failed"))

            return None

        except Exception as e:
            logger.error(f"Error checking failed search cache: {e}")
            return None

    def clear_cache(self, pattern: Optional[str] = None) -> int:
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
            keys = list(self.client.scan_iter(match=pattern))

            if keys:
                # Delete all matching keys
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return int(deleted)  # type: ignore[arg-type]

            return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.client:
            return {"enabled": False}

        try:
            # Get Redis info
            info = self.client.info()

            # Count tracklist keys
            tracklist_keys = len(list(self.client.scan_iter(match=f"{self.config.key_prefix}*")))
            failed_keys = len(list(self.client.scan_iter(match=f"{self.config.key_prefix}failed:*")))

            return {
                "enabled": True,
                "connected": True,
                "total_keys": tracklist_keys,
                "failed_keys": failed_keys,
                "memory_used": info.get("used_memory_human", "N/A"),  # type: ignore[union-attr]
                "hit_rate": info.get("keyspace_hit_ratio", 0),  # type: ignore[union-attr]
                "evicted_keys": info.get("evicted_keys", 0),  # type: ignore[union-attr]
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}

    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.client:
            return None

        try:
            value = self.client.get(key)
            return value  # type: ignore[return-value]
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
            self.client.setex(key, ttl, value)
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
            return int(self.client.delete(key))  # type: ignore[arg-type]
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
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create the global cache instance.

    Returns:
        RedisCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RedisCache()

    return _cache_instance


def close_cache() -> None:
    """Close the global cache instance."""
    global _cache_instance

    if _cache_instance:
        _cache_instance.close()
        _cache_instance = None
