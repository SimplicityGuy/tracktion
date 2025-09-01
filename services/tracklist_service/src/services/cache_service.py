"""
Caching service for CUE generation operations.
"""

import gzip
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

import redis.asyncio as redis
from pydantic import BaseModel

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for cache service."""

    # Redis connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_username: str | None = None
    redis_ssl: bool = False
    connection_timeout: int = 5
    socket_timeout: int = 5
    max_connections: int = 10

    # Cache settings
    default_ttl: int = 3600  # 1 hour in seconds
    cue_content_ttl: int = 7200  # 2 hours for CUE content
    validation_ttl: int = 1800  # 30 minutes for validation results
    format_capabilities_ttl: int = 86400  # 24 hours for format capabilities

    # Performance settings
    enable_compression: bool = True
    enable_metrics: bool = True
    cache_warming_enabled: bool = True
    popular_formats: list[str] = ["standard", "cdj", "traktor"]


class MemoryCache:
    """Fallback in-memory cache implementation."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize memory cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._access_order: list[str] = []  # For LRU eviction

    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()


class CacheMetrics(BaseModel):
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_operations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_reads = self.hits + self.misses
        return (self.hits / total_reads * 100) if total_reads > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_operations": self.total_operations,
            "hit_rate_percent": round(self.hit_rate, 2),
        }


class CacheService:
    """CUE generation caching service with Redis backend."""

    def __init__(self, config: CacheConfig):
        """
        Initialize cache service.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.redis_client: Redis | None = None
        self.metrics = CacheMetrics()
        self.memory_cache = MemoryCache(max_size=config.memory_cache_max_size, default_ttl=config.memory_cache_ttl)
        self._redis_available = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                username=self.config.redis_username,
                ssl=self.config.redis_ssl,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.connection_timeout,
                max_connections=self.config.max_connections,
                decode_responses=True,
            )

            # Test connection
            await self.redis_client.ping()
            self._redis_available = True
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            self._redis_available = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self._redis_available = False
            logger.info("Disconnected from Redis")

    def _generate_cache_key(self, prefix: str, *components: Any) -> str:
        """Generate standardized cache key."""
        key_parts = [prefix]
        for component in components:
            if isinstance(component, UUID):
                key_parts.append(str(component))
            else:
                key_parts.append(str(component))
        return ":".join(key_parts)

    async def _set_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis with compression if enabled."""
        if not self.redis_client:
            return False

        try:
            serialized_value = json.dumps(value)

            if self.config.enable_compression and len(serialized_value) > 1000:
                compressed = gzip.compress(serialized_value.encode("utf-8"))
                await self.redis_client.setex(f"{key}:gz", ttl, compressed)
                await self.redis_client.setex(f"{key}:meta", ttl, json.dumps({"compressed": True}))
            else:
                await self.redis_client.setex(key, ttl, serialized_value)

            return True

        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def _get_redis(self, key: str) -> Any | None:
        """Get value from Redis with decompression if needed."""
        if not self.redis_client:
            return None

        try:
            # Check if compressed version exists
            meta_data = await self.redis_client.get(f"{key}:meta")
            if meta_data:
                meta = json.loads(meta_data)
                if meta.get("compressed"):
                    compressed_data = await self.redis_client.get(f"{key}:gz")
                    if compressed_data:
                        decompressed = gzip.decompress(
                            compressed_data.encode() if isinstance(compressed_data, str) else compressed_data
                        ).decode("utf-8")
                        return json.loads(decompressed)

            # Try regular key
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)

            return None

        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            self.metrics.errors += 1
            return None

    async def _delete_redis(self, key: str) -> bool:
        """Delete key from Redis including compressed variants."""
        if not self.redis_client:
            return False

        try:
            # Delete all variants
            deleted = await self.redis_client.delete(key, f"{key}:gz", f"{key}:meta")
            return bool(deleted > 0)

        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            self.metrics.errors += 1
            return False

    async def get_cue_content(self, tracklist_id: UUID, cue_format: str) -> str | None:
        """
        Get cached CUE file content.

        Args:
            tracklist_id: Tracklist ID
            cue_format: CUE format

        Returns:
            Cached CUE content if available
        """
        key = self._generate_cache_key("cue_content", tracklist_id, cue_format)

        content = await self._get_redis(key)
        if content:
            self.metrics.hits += 1
            logger.debug(f"Cache hit for CUE content: {key}")
            return str(content)

        self.metrics.misses += 1
        logger.debug(f"Cache miss for CUE content: {key}")
        return None

    async def set_cue_content(self, tracklist_id: UUID, cue_format: str, content: str) -> bool:
        """
        Cache CUE file content.

        Args:
            tracklist_id: Tracklist ID
            cue_format: CUE format
            content: CUE file content

        Returns:
            True if cached successfully
        """
        key = self._generate_cache_key("cue_content", tracklist_id, cue_format)

        success = await self._set_redis(key, content, self.config.cue_content_ttl)
        if success:
            self.metrics.sets += 1
            logger.debug(f"Cached CUE content: {key}")
            return True

        return False

    async def get_validation_result(
        self, cue_file_id: UUID, audio_file_path: str | None = None
    ) -> dict[str, Any] | None:
        """
        Get cached validation result.

        Args:
            cue_file_id: CUE file ID
            audio_file_path: Optional audio file path for key generation

        Returns:
            Cached validation result if available
        """
        if audio_file_path:
            audio_hash = hashlib.md5(audio_file_path.encode()).hexdigest()[:8]
            key = self._generate_cache_key("validation", cue_file_id, audio_hash)
        else:
            key = self._generate_cache_key("validation", cue_file_id)

        result = await self._get_redis(key)
        if result:
            self.metrics.hits += 1
            return dict(result) if isinstance(result, dict) else result

        self.metrics.misses += 1
        return None

    async def set_validation_result(
        self,
        cue_file_id: UUID,
        result: dict[str, Any],
        audio_file_path: str | None = None,
    ) -> bool:
        """
        Cache validation result.

        Args:
            cue_file_id: CUE file ID
            result: Validation result
            audio_file_path: Optional audio file path for key generation

        Returns:
            True if cached successfully
        """
        if audio_file_path:
            audio_hash = hashlib.md5(audio_file_path.encode()).hexdigest()[:8]
            key = self._generate_cache_key("validation", cue_file_id, audio_hash)
        else:
            key = self._generate_cache_key("validation", cue_file_id)

        success = await self._set_redis(key, result, self.config.validation_ttl)
        if success:
            self.metrics.sets += 1
            return True

        return False

    async def get_format_capabilities(self, cue_format: str) -> dict[str, Any] | None:
        """
        Get cached format capabilities.

        Args:
            cue_format: CUE format

        Returns:
            Cached format capabilities if available
        """
        key = self._generate_cache_key("format_capabilities", cue_format)

        capabilities = await self._get_redis(key)
        if capabilities:
            self.metrics.hits += 1
            return dict(capabilities) if isinstance(capabilities, dict) else capabilities

        self.metrics.misses += 1
        return None

    async def set_format_capabilities(self, cue_format: str, capabilities: dict[str, Any]) -> bool:
        """
        Cache format capabilities.

        Args:
            cue_format: CUE format
            capabilities: Format capabilities

        Returns:
            True if cached successfully
        """
        key = self._generate_cache_key("format_capabilities", cue_format)

        success = await self._set_redis(key, capabilities, self.config.format_capabilities_ttl)
        if success:
            self.metrics.sets += 1
            return True

        return False

    async def invalidate_tracklist_cache(self, tracklist_id: UUID) -> int:
        """
        Invalidate all cache entries for a specific tracklist.

        Args:
            tracklist_id: Tracklist ID

        Returns:
            Number of keys invalidated
        """
        if not self.redis_client:
            logger.debug(f"Redis not available for invalidating tracklist {tracklist_id}")
            return 0

        try:
            pattern = self._generate_cache_key("cue_content", tracklist_id, "*")
            keys = await self.redis_client.keys(pattern)

            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.metrics.deletes += deleted
                logger.info(f"Invalidated {deleted} cache entries for tracklist {tracklist_id}")
                return int(deleted)

            return 0

        except Exception as e:
            logger.error(f"Cache invalidation failed for tracklist {tracklist_id}: {e}")
            self.metrics.errors += 1
            return 0

    async def warm_cache(self, tracklist_ids: list[UUID], formats: list[str] | None = None) -> dict[str, Any]:
        """
        Warm cache for popular tracklist/format combinations.

        Args:
            tracklist_ids: List of tracklist IDs to warm
            formats: List of formats to warm (defaults to popular formats)

        Returns:
            Cache warming results
        """
        if not self.config.cache_warming_enabled:
            return {"cache_warming": "disabled"}

        formats = formats or self.config.popular_formats

        results = {
            "tracklist_count": len(tracklist_ids),
            "format_count": len(formats),
            "combinations": len(tracklist_ids) * len(formats),
            "warmed": 0,
            "errors": 0,
        }

        logger.info(f"Starting cache warming for {results['combinations']} combinations")

        for tracklist_id in tracklist_ids:
            for format_name in formats:
                try:
                    # Check if already cached
                    content = await self.get_cue_content(tracklist_id, format_name)
                    if content:
                        results["warmed"] += 1
                        continue

                    # TODO: Generate content if not cached
                    # This would integrate with the CUE generation service
                    # For now, we'll just log the cache miss
                    logger.debug(f"Cache miss during warming: {tracklist_id}:{format_name}")

                except Exception as e:
                    logger.error(f"Cache warming error for {tracklist_id}:{format_name}: {e}")
                    results["errors"] += 1

        logger.info(f"Cache warming completed: {results}")
        return results

    async def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics and health information.

        Returns:
            Cache statistics
        """
        stats = {
            "cache_service": "healthy",
            "redis_available": self.redis_client is not None,
            "metrics": self.metrics.to_dict(),
            "config": {
                "compression_enabled": self.config.enable_compression,
                "cache_warming_enabled": self.config.cache_warming_enabled,
                "popular_formats": self.config.popular_formats,
            },
        }

        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["redis_info"] = {
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                }

                # Calculate Redis hit rate if available
                redis_hits = info.get("keyspace_hits", 0)
                redis_misses = info.get("keyspace_misses", 0)
                if redis_hits + redis_misses > 0:
                    stats["redis_hit_rate"] = round((redis_hits / (redis_hits + redis_misses)) * 100, 2)

            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
                stats["redis_error"] = str(e)

        return stats

    async def clear_cache(self, pattern: str | None = None) -> int:
        """
        Clear cache entries matching pattern.

        Args:
            pattern: Optional pattern to match keys (Redis only)

        Returns:
            Number of keys cleared
        """
        cleared = 0

        # Clear Redis cache
        if self.redis_client:
            try:
                if pattern:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        deleted = await self.redis_client.delete(*keys)
                        cleared += deleted
                else:
                    await self.redis_client.flushdb()
                    cleared += 1  # Count as one operation for full flush

            except Exception as e:
                logger.error(f"Redis cache clear failed: {e}")
                self.metrics.errors += 1

        logger.info(f"Cleared {cleared} cache entries")
        return cleared


# Global cache service instance
cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    if not cache_service:
        raise RuntimeError("Cache service not initialized")
    return cache_service


def initialize_cache_service(config: CacheConfig) -> CacheService:
    """Initialize the global cache service."""
    global cache_service  # noqa: PLW0603  # Global is needed for cache service initialization
    cache_service = CacheService(config)
    return cache_service
