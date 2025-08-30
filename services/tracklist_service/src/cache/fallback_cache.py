"""Fallback cache mechanism for resilient data retrieval."""

import json
import logging
from datetime import datetime, timedelta, UTC
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache retrieval strategies."""

    STRICT = "strict"  # Only return if within max_age
    FLEXIBLE = "flexible"  # Return stale data with warning
    FALLBACK = "fallback"  # Return any cached data as last resort
    PROGRESSIVE = "progressive"  # Try multiple cache layers


@dataclass
class CachedItem:
    """Cached data with metadata."""

    key: str
    data: Dict[str, Any]
    cached_at: datetime
    expires_at: Optional[datetime] = None
    quality_score: float = 1.0
    source: str = "primary"
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def age_seconds(self) -> float:
        """Get age of cached item in seconds."""
        return (datetime.now(UTC) - self.cached_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.expires_at:
            return datetime.now(UTC) > self.expires_at
        return False

    @property
    def validity_score(self) -> float:
        """Calculate validity score based on age and quality."""
        # Start with quality score
        score = self.quality_score

        # Reduce score based on age
        age_hours = self.age_seconds / 3600
        if age_hours < 1:
            age_penalty = 0.0
        elif age_hours < 24:
            age_penalty = 0.1 * (age_hours / 24)
        elif age_hours < 168:  # 1 week
            age_penalty = 0.3 * (age_hours / 168)
        else:
            age_penalty = 0.5

        # Apply expiration penalty
        if self.is_expired:
            age_penalty += 0.3

        return max(0.0, min(1.0, score - age_penalty))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "data": self.data,
            "cached_at": self.cached_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "quality_score": self.quality_score,
            "source": self.source,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedItem":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            data=data["data"],
            cached_at=datetime.fromisoformat(data["cached_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            quality_score=data.get("quality_score", 1.0),
            source=data.get("source", "primary"),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )


class FallbackCache:
    """Fallback cache with multiple layers and intelligent retrieval."""

    def __init__(
        self,
        redis_client: Optional[redis.Redis[bytes]] = None,
        default_ttl: int = 3600,
        max_fallback_age: int = 86400 * 7,  # 7 days
    ):
        """Initialize fallback cache.

        Args:
            redis_client: Redis client for cache storage
            default_ttl: Default TTL in seconds
            max_fallback_age: Maximum age for fallback data in seconds
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.max_fallback_age = max_fallback_age
        self._memory_cache: Dict[str, CachedItem] = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "fallback_hits": 0,
            "stale_hits": 0,
        }

    async def get_with_fallback(
        self,
        key: str,
        max_age: Optional[int] = None,
        strategy: CacheStrategy = CacheStrategy.FLEXIBLE,
    ) -> Optional[dict]:
        """Get cached data with fallback options.

        Args:
            key: Cache key
            max_age: Maximum acceptable age in seconds
            strategy: Cache retrieval strategy

        Returns:
            Cached data or None
        """
        # Try memory cache first
        if key in self._memory_cache:
            item = self._memory_cache[key]
            result = self._evaluate_cached_item(item, max_age, strategy)
            if result:
                self._cache_stats["hits"] += 1
                await self._update_access_stats(item)
                return result

        # Try Redis cache
        if self.redis_client:
            redis_item = await self._get_from_redis(key)
            if redis_item:
                result = self._evaluate_cached_item(redis_item, max_age, strategy)
                if result:
                    self._cache_stats["hits"] += 1
                    # Store in memory cache for faster access
                    self._memory_cache[key] = redis_item
                    await self._update_access_stats(redis_item)
                    return result

        # Try fallback strategies
        if strategy in [CacheStrategy.FALLBACK, CacheStrategy.PROGRESSIVE]:
            fallback_data = await self._get_fallback_data(key)
            if fallback_data:
                self._cache_stats["fallback_hits"] += 1
                return fallback_data

        self._cache_stats["misses"] += 1
        return None

    def calculate_validity_score(self, cached_data: Dict[str, Any]) -> float:
        """Calculate validity score for cached data.

        Args:
            cached_data: Cached data dictionary

        Returns:
            Validity score between 0 and 1
        """
        if "cached_at" not in cached_data:
            return 0.5  # Unknown age, assume medium validity

        try:
            cached_at = datetime.fromisoformat(cached_data["cached_at"])
            quality = cached_data.get("quality_score", 1.0)

            # Create temporary CachedItem for score calculation
            item = CachedItem(
                key="temp",
                data=cached_data.get("data", {}),
                cached_at=cached_at,
                quality_score=quality,
            )
            return item.validity_score
        except Exception as e:
            logger.error(f"Error calculating validity score: {e}")
            return 0.0

    async def set_with_quality(
        self,
        key: str,
        data: Dict[str, Any],
        quality_score: float = 1.0,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set cached data with quality score.

        Args:
            key: Cache key
            data: Data to cache
            quality_score: Quality score (0-1)
            ttl: Time to live in seconds
            metadata: Additional metadata
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl) if ttl else None

        item = CachedItem(
            key=key,
            data=data,
            cached_at=datetime.now(UTC),
            expires_at=expires_at,
            quality_score=quality_score,
            metadata=metadata or {},
        )

        # Store in memory cache
        self._memory_cache[key] = item

        # Store in Redis with extended TTL for fallback
        if self.redis_client:
            await self._store_in_redis(item, ttl * 3)  # Keep 3x longer for fallback

        # Store in fallback layer
        await self._store_fallback(item)

    async def warm_cache(self, keys: List[str], fetch_func: Optional[Any] = None) -> Dict[str, bool]:
        """Warm cache for specified keys.

        Args:
            keys: List of cache keys to warm
            fetch_func: Optional async function to fetch fresh data

        Returns:
            Dictionary of key: success status
        """
        results = {}

        for key in keys:
            try:
                # Check if already cached and fresh
                existing = await self.get_with_fallback(key, max_age=3600)
                if existing and self.calculate_validity_score(existing) > 0.8:
                    results[key] = True
                    continue

                # Fetch fresh data if function provided
                if fetch_func:
                    fresh_data = await fetch_func(key)
                    if fresh_data:
                        await self.set_with_quality(key, fresh_data)
                        results[key] = True
                    else:
                        results[key] = False
                else:
                    results[key] = False
            except Exception as e:
                logger.error(f"Error warming cache for {key}: {e}")
                results[key] = False

        return results

    async def clear_expired(self) -> int:
        """Clear expired items from cache.

        Returns:
            Number of items cleared
        """
        cleared = 0

        # Clear from memory cache
        expired_keys = [k for k, v in self._memory_cache.items() if v.is_expired]
        for key in expired_keys:
            del self._memory_cache[key]
            cleared += 1

        # Clear from Redis (Redis handles TTL automatically)
        # We just update statistics
        if self.redis_client:
            # Could implement manual cleanup if needed
            pass

        logger.info(f"Cleared {cleared} expired cache items")
        return cleared

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        total_requests = sum(self._cache_stats.values())
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self._cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
        }

    def _evaluate_cached_item(
        self, item: CachedItem, max_age: Optional[int], strategy: CacheStrategy
    ) -> Optional[dict]:
        """Evaluate if cached item should be returned.

        Args:
            item: Cached item
            max_age: Maximum acceptable age
            strategy: Retrieval strategy

        Returns:
            Data if acceptable, None otherwise
        """
        if strategy == CacheStrategy.STRICT:
            if max_age and item.age_seconds > max_age:
                return None
            if item.is_expired:
                return None
        elif strategy == CacheStrategy.FLEXIBLE:
            if max_age and item.age_seconds > max_age * 2:
                return None
            if item.validity_score < 0.3:
                return None
            if max_age and item.age_seconds > max_age:
                self._cache_stats["stale_hits"] += 1
                logger.warning(f"Returning stale cache for {item.key} (age: {item.age_seconds}s)")
        elif strategy == CacheStrategy.FALLBACK:
            if item.age_seconds > self.max_fallback_age:
                return None
        # PROGRESSIVE handled elsewhere

        return item.data

    async def _get_from_redis(self, key: str) -> Optional[CachedItem]:
        """Get cached item from Redis.

        Args:
            key: Cache key

        Returns:
            CachedItem or None
        """
        if not self.redis_client:
            return None

        try:
            # Try primary cache
            data = await self.redis_client.get(f"cache:primary:{key}")
            if data:
                item_dict = json.loads(data)
                return CachedItem.from_dict(item_dict)

            # Try fallback cache
            fallback_data = await self.redis_client.get(f"cache:fallback:{key}")
            if fallback_data:
                item_dict = json.loads(fallback_data)
                return CachedItem.from_dict(item_dict)
        except Exception as e:
            logger.error(f"Error getting from Redis: {e}")

        return None

    async def _store_in_redis(self, item: CachedItem, ttl: int) -> None:
        """Store item in Redis.

        Args:
            item: Item to store
            ttl: Time to live in seconds
        """
        if not self.redis_client:
            return

        try:
            key = f"cache:primary:{item.key}"
            await self.redis_client.set(key, json.dumps(item.to_dict()), ex=ttl)
        except Exception as e:
            logger.error(f"Error storing in Redis: {e}")

    async def _store_fallback(self, item: CachedItem) -> None:
        """Store item in fallback cache layer.

        Args:
            item: Item to store
        """
        if not self.redis_client:
            return

        try:
            # Store with longer TTL for fallback
            key = f"cache:fallback:{item.key}"
            fallback_ttl = self.max_fallback_age
            await self.redis_client.set(key, json.dumps(item.to_dict()), ex=fallback_ttl)

            # Track in fallback index
            await self.redis_client.zadd(
                "cache:fallback:index",
                {item.key: datetime.now(UTC).timestamp()},
            )
        except Exception as e:
            logger.error(f"Error storing fallback: {e}")

    async def _get_fallback_data(self, key: str) -> Optional[dict]:
        """Get data from fallback layers.

        Args:
            key: Cache key

        Returns:
            Fallback data or None
        """
        if not self.redis_client:
            return None

        try:
            # Try different fallback keys
            fallback_keys = [
                f"cache:fallback:{key}",
                f"cache:archive:{key}",
                f"cache:backup:{key}",
            ]

            for fallback_key in fallback_keys:
                data = await self.redis_client.get(fallback_key)
                if data:
                    item_dict = json.loads(data)
                    item = CachedItem.from_dict(item_dict)
                    if item.age_seconds < self.max_fallback_age:
                        logger.info(f"Using fallback cache for {key} (age: {item.age_seconds}s)")
                        return item.data
        except Exception as e:
            logger.error(f"Error getting fallback data: {e}")

        return None

    async def _update_access_stats(self, item: CachedItem) -> None:
        """Update access statistics for cached item.

        Args:
            item: Cached item that was accessed
        """
        item.access_count += 1
        item.last_accessed = datetime.now(UTC)

        # Update in Redis if available
        if self.redis_client:
            try:
                stats_key = f"cache:stats:{item.key}"
                # Use explicit typing to avoid mypy issues
                result1 = self.redis_client.hincrby(stats_key, "access_count", 1)
                result2 = self.redis_client.hset(stats_key, "last_accessed", item.last_accessed.isoformat())
                result3 = self.redis_client.expire(stats_key, 86400)  # 1 day TTL for stats

                # Await all results
                if hasattr(result1, "__await__"):
                    await result1
                if hasattr(result2, "__await__"):
                    await result2
                if hasattr(result3, "__await__"):
                    await result3
            except Exception as e:
                logger.error(f"Error updating access stats: {e}")
