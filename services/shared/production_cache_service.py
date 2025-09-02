"""
Production-ready Redis caching service for cross-service usage.

This service provides a unified caching interface that can be used across
all services in the Tracktion monorepo, replacing in-memory caches.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

import redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


class ProductionCacheService:
    """Production Redis cache service with error handling and fallback."""

    # Default TTL values (in seconds)
    DEFAULT_TTL = 30 * 24 * 60 * 60  # 30 days
    SHORT_TTL = 60 * 60  # 1 hour
    MEDIUM_TTL = 7 * 24 * 60 * 60  # 7 days
    LONG_TTL = 30 * 24 * 60 * 60  # 30 days

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        default_ttl: int = DEFAULT_TTL,
        service_prefix: str = "cache",
        enabled: bool = True,
    ) -> None:
        """
        Initialize production cache service.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            default_ttl: Default TTL for cache entries in seconds
            service_prefix: Prefix for cache keys to namespace by service
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.service_prefix = service_prefix
        self.default_ttl = default_ttl
        self.redis_client: redis.Redis | None = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

        if not self.enabled:
            logger.info("Production cache disabled")
            return

        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to connect to Redis: {e!s}")
            self.redis_client = None
            self.enabled = False

    def _build_key(self, key: str) -> str:
        """Build cache key with service prefix."""
        return f"{self.service_prefix}:{key}"

    def _serialize_value(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str, ensure_ascii=False)

    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from Redis storage."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            cache_key = self._build_key(key)
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                self._stats["hits"] += 1
                logger.debug(f"Cache hit: {cache_key}")
                return self._deserialize_value(cached_data)

            self._stats["misses"] += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached value for {key}: {e!s}")
            self._stats["errors"] += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            nx: Only set if key doesn't exist

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        ttl = ttl or self.default_ttl
        cache_key = self._build_key(key)

        try:
            serialized_value = self._serialize_value(value)

            if nx:
                success = self.redis_client.set(cache_key, serialized_value, ex=ttl, nx=True)
            else:
                success = self.redis_client.setex(cache_key, ttl, serialized_value)

            if success:
                self._stats["sets"] += 1
                logger.debug(f"Cached value: {cache_key} (TTL: {ttl}s)")
                return True
            return False

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache value for {key}: {e!s}")
            self._stats["errors"] += 1
            return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successfully deleted, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            cache_key = self._build_key(key)
            result = self.redis_client.delete(cache_key)

            if result:
                self._stats["deletes"] += 1
                logger.debug(f"Deleted cache key: {cache_key}")
                return True
            return False

        except RedisError as e:
            logger.error(f"Failed to delete cache key {key}: {e!s}")
            self._stats["errors"] += 1
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            cache_key = self._build_key(key)
            return bool(self.redis_client.exists(cache_key))
        except RedisError as e:
            logger.error(f"Failed to check existence of {key}: {e!s}")
            self._stats["errors"] += 1
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if expiration was set, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            cache_key = self._build_key(key)
            return bool(self.redis_client.expire(cache_key, ttl))
        except RedisError as e:
            logger.error(f"Failed to set expiration for {key}: {e!s}")
            self._stats["errors"] += 1
            return False

    def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int | None:
        """
        Increment a numeric value in cache.

        Args:
            key: Cache key
            amount: Amount to increment by
            ttl: TTL for the key if it doesn't exist

        Returns:
            New value after increment, or None on error
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            cache_key = self._build_key(key)
            new_value = self.redis_client.incrby(cache_key, amount)

            # Set TTL if provided and key was just created
            if ttl and new_value == amount:
                self.redis_client.expire(cache_key, ttl)

            return int(new_value)

        except (RedisError, ValueError) as e:
            logger.error(f"Failed to increment {key}: {e!s}")
            self._stats["errors"] += 1
            return None

    def hash_set(self, key: str, field: str, value: Any, ttl: int | None = None) -> bool:
        """
        Set field in hash.

        Args:
            key: Hash key
            field: Field name
            value: Field value
            ttl: TTL for the hash

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            cache_key = self._build_key(key)
            serialized_value = self._serialize_value(value)
            success = self.redis_client.hset(cache_key, field, serialized_value)

            if ttl:
                self.redis_client.expire(cache_key, ttl)

            return bool(success)

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to set hash field {key}:{field}: {e!s}")
            self._stats["errors"] += 1
            return False

    def hash_get(self, key: str, field: str) -> Any | None:
        """
        Get field from hash.

        Args:
            key: Hash key
            field: Field name

        Returns:
            Field value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            cache_key = self._build_key(key)
            value = self.redis_client.hget(cache_key, field)

            if value:
                self._stats["hits"] += 1
                return self._deserialize_value(value)

            self._stats["misses"] += 1
            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get hash field {key}:{field}: {e!s}")
            self._stats["errors"] += 1
            return None

    def hash_get_all(self, key: str) -> dict[str, Any]:
        """
        Get all fields from hash.

        Args:
            key: Hash key

        Returns:
            Dictionary of all field-value pairs
        """
        if not self.enabled or not self.redis_client:
            return {}

        try:
            cache_key = self._build_key(key)
            hash_data = self.redis_client.hgetall(cache_key)

            if hash_data:
                self._stats["hits"] += 1
                return {field: self._deserialize_value(value) for field, value in hash_data.items()}

            self._stats["misses"] += 1
            return {}

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get hash {key}: {e!s}")
            self._stats["errors"] += 1
            return {}

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        try:
            full_pattern = self._build_key(pattern)
            keys_result = self.redis_client.keys(full_pattern)

            # Handle both sync and async returns
            if hasattr(keys_result, "__await__"):
                keys = []  # Skip async results
            else:
                try:
                    keys = cast("list[str]", keys_result) if keys_result else []
                except TypeError:
                    keys = []

            if keys:
                deleted = self.redis_client.delete(*keys)
                deleted_count = cast("int", deleted) if deleted is not None else 0
                self._stats["deletes"] += deleted_count
                logger.info(f"Deleted {deleted_count} keys matching pattern: {full_pattern}")
                return deleted_count

            return 0

        except RedisError as e:
            logger.error(f"Failed to clear pattern {pattern}: {e!s}")
            self._stats["errors"] += 1
            return 0

    def flush_service_cache(self) -> bool:
        """
        Flush all cache entries for this service.

        Returns:
            True if successful, False otherwise
        """
        try:
            deleted = self.clear_pattern("*")
            logger.info(f"Flushed {deleted} cache entries for service {self.service_prefix}")
            return True
        except Exception as e:
            logger.error(f"Failed to flush service cache: {e!s}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.enabled,
            "connected": self.redis_client is not None,
            "service_prefix": self.service_prefix,
            **self._stats,
        }

        if self.enabled and self.redis_client:
            try:
                # Calculate hit rate
                hits: int = stats["hits"]  # type: ignore[assignment]  # Stats dict contains mixed types but hits/misses are known to be int
                misses: int = stats["misses"]  # type: ignore[assignment]  # Stats dict contains mixed types but hits/misses are known to be int
                total_reads = hits + misses
                stats["hit_rate"] = (hits / total_reads * 100) if total_reads > 0 else 0.0

                # Get Redis info
                info = self.redis_client.info()
                stats.update(
                    {
                        "redis_memory": info.get("used_memory_human", "N/A"),
                        "redis_clients": info.get("connected_clients", 0),
                        "redis_commands": info.get("total_commands_processed", 0),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e!s}")
                stats["error"] = str(e)

        return stats

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on cache service.

        Returns:
            Health check results
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "healthy": True,
                "message": "Cache service is disabled",
            }

        if not self.redis_client:
            return {
                "status": "error",
                "healthy": False,
                "message": "Redis client not initialized",
            }

        try:
            # Test Redis connection
            start_time = datetime.now(UTC)
            self.redis_client.ping()
            response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "healthy": True,
                "redis_response_time_ms": response_time,
                "message": "Cache service is healthy",
            }

        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "message": f"Redis connection failed: {e!s}",
            }

    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self.redis_client = None


# Utility functions for common cache patterns
def generate_cache_key(*components: Any) -> str:
    """Generate a cache key from components."""
    key_parts = []
    for component in components:
        if component is None:
            key_parts.append("null")
        else:
            key_parts.append(str(component))
    return ":".join(key_parts)


def hash_key(data: str | bytes) -> str:
    """Generate a hash key from data for cache keys."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]  # First 16 chars for shorter keys
