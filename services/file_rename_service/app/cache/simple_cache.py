"""Simple in-memory cache for proposal results."""

import hashlib
import time
from typing import Any


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 3600) -> None:
        """Initialize cache with default TTL in seconds."""
        self._cache: dict[str, dict[str, Any]] = {}
        self.default_ttl = default_ttl

    def _generate_key(self, *args: Any) -> str:
        """Generate cache key from arguments."""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
        }

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        # Clean expired entries first
        current_time = time.time()
        expired_keys = [key for key, entry in self._cache.items() if current_time > entry["expires_at"]]
        for key in expired_keys:
            del self._cache[key]

        return len(self._cache)


# Global cache instance
proposal_cache = SimpleCache(default_ttl=1800)  # 30 minutes
