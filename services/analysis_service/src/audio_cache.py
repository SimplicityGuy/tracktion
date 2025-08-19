"""
Redis caching layer for audio analysis results.

This module provides caching functionality to avoid re-analyzing
audio files that have already been processed.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


class AudioCache:
    """
    Redis-based cache for audio analysis results.

    Provides caching with TTL, versioning, and cache invalidation
    for BPM detection and other audio analysis operations.
    """

    # Cache key prefixes
    BPM_PREFIX = "bpm"
    TEMPORAL_PREFIX = "temporal"

    # Default TTL values (in seconds)
    DEFAULT_TTL = 30 * 24 * 60 * 60  # 30 days
    FAILED_TTL = 60 * 60  # 1 hour
    LOW_CONFIDENCE_TTL = 7 * 24 * 60 * 60  # 7 days

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        algorithm_version: str = "1.0",
        use_xxh128: bool = True,
    ):
        """
        Initialize audio cache with Redis connection.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            algorithm_version: Version of analysis algorithms
            use_xxh128: Use xxHash128 (fast) instead of SHA256 (secure)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.algorithm_version = algorithm_version
        self.use_xxh128 = use_xxh128

        # Initialize Redis connection
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
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    def _generate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Generate hash of file contents for cache key.

        Args:
            file_path: Path to the audio file

        Returns:
            Hash string or None if hashing fails
        """
        try:
            if self.use_xxh128:
                # Use xxHash128 for faster hashing (requires xxhash library)
                try:
                    import xxhash

                    hasher = xxhash.xxh128()
                except ImportError:
                    logger.warning("xxhash not available, falling back to SHA256")
                    hasher = hashlib.sha256()
            else:
                hasher = hashlib.sha256()

            # Read file in chunks to handle large files
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

            return hasher.hexdigest()  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {str(e)}")
            return None

    def _build_cache_key(self, prefix: str, file_hash: str) -> str:
        """
        Build versioned cache key.

        Args:
            prefix: Cache key prefix (e.g., "bpm", "temporal")
            file_hash: Hash of the file contents

        Returns:
            Complete cache key
        """
        return f"{prefix}:{file_hash}:{self.algorithm_version}"

    def get_bpm_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached BPM analysis results.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached results or None if not found/expired
        """
        if not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.BPM_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for BPM analysis: {cache_key}")
                return json.loads(cached_data)  # type: ignore[no-any-return]
            else:
                logger.debug(f"Cache miss for BPM analysis: {cache_key}")
                return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached BPM results: {str(e)}")
            return None

    def set_bpm_results(
        self, file_path: str, results: Dict[str, Any], confidence: Optional[float] = None, failed: bool = False
    ) -> bool:
        """
        Cache BPM analysis results with appropriate TTL.

        Args:
            file_path: Path to the audio file
            results: BPM analysis results to cache
            confidence: Confidence score (affects TTL)
            failed: Whether analysis failed (short TTL)

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.BPM_PREFIX, file_hash)

        # Add metadata to cached results
        cache_data = {
            **results,
            "algorithm_version": self.algorithm_version,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
        }

        # Determine TTL based on confidence and success
        if failed:
            ttl = self.FAILED_TTL
        elif confidence is not None and confidence < 0.5:
            ttl = self.LOW_CONFIDENCE_TTL
        else:
            ttl = self.DEFAULT_TTL

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached BPM results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError) as e:
            logger.error(f"Failed to cache BPM results: {str(e)}")
            return False

    def get_temporal_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached temporal analysis results.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached results or None if not found/expired
        """
        if not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.TEMPORAL_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for temporal analysis: {cache_key}")
                return json.loads(cached_data)  # type: ignore[no-any-return]
            else:
                logger.debug(f"Cache miss for temporal analysis: {cache_key}")
                return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached temporal results: {str(e)}")
            return None

    def set_temporal_results(
        self, file_path: str, results: Dict[str, Any], stability_score: Optional[float] = None
    ) -> bool:
        """
        Cache temporal analysis results.

        Args:
            file_path: Path to the audio file
            results: Temporal analysis results to cache
            stability_score: Tempo stability (affects TTL)

        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.TEMPORAL_PREFIX, file_hash)

        # Add metadata
        cache_data = {
            **results,
            "algorithm_version": self.algorithm_version,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "file_path": file_path,
        }

        # Use shorter TTL for variable tempo tracks (may need re-analysis)
        if stability_score is not None and stability_score < 0.5:
            ttl = self.LOW_CONFIDENCE_TTL
        else:
            ttl = self.DEFAULT_TTL

        try:
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(cache_data, default=str),  # Handle datetime serialization
            )
            logger.debug(f"Cached temporal results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError) as e:
            logger.error(f"Failed to cache temporal results: {str(e)}")
            return False

    def invalidate_cache(self, file_path: str) -> bool:
        """
        Invalidate all cached results for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if invalidated successfully
        """
        if not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        # Delete all cache entries for this file
        keys_deleted = 0
        for prefix in [self.BPM_PREFIX, self.TEMPORAL_PREFIX]:
            cache_key = self._build_cache_key(prefix, file_hash)
            try:
                deleted = self.redis_client.delete(cache_key)
                keys_deleted += deleted
            except RedisError as e:
                logger.error(f"Failed to invalidate cache key {cache_key}: {str(e)}")

        logger.info(f"Invalidated {keys_deleted} cache entries for {file_path}")
        return keys_deleted > 0

    def flush_version_cache(self, version: Optional[str] = None) -> int:
        """
        Flush all cache entries for a specific algorithm version.

        Args:
            version: Algorithm version to flush (default: current version)

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        version = version or self.algorithm_version
        pattern = f"*:{version}"

        try:
            # Find all keys matching the version pattern
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Flushed {deleted} cache entries for version {version}")
                return int(deleted)
            return 0

        except RedisError as e:
            logger.error(f"Failed to flush version cache: {str(e)}")
            return 0

    def warm_cache(self, file_paths: list, analyzer_callback: Callable[[str], Dict[str, Any]]) -> int:
        """
        Pre-populate cache during off-peak hours.

        Args:
            file_paths: List of audio file paths to cache
            analyzer_callback: Function to analyze files not in cache

        Returns:
            Number of files cached
        """
        cached_count = 0

        for file_path in file_paths:
            # Check if already cached
            if self.get_bpm_results(file_path):
                continue

            try:
                # Analyze and cache
                results = analyzer_callback(file_path)
                if results:
                    self.set_bpm_results(file_path, results, confidence=results.get("confidence"))
                    cached_count += 1

            except Exception as e:
                logger.error(f"Failed to warm cache for {file_path}: {str(e)}")

        logger.info(f"Cache warming complete: {cached_count} files cached")
        return cached_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.redis_client:
            return {"connected": False}

        try:
            info = self.redis_client.info("memory")
            keyspace = self.redis_client.info("keyspace")

            # Count keys by type
            bpm_keys = len(self.redis_client.keys(f"{self.BPM_PREFIX}:*"))
            temporal_keys = len(self.redis_client.keys(f"{self.TEMPORAL_PREFIX}:*"))

            return {
                "connected": True,
                "memory_used": info.get("used_memory_human", "N/A"),
                "total_keys": keyspace.get(f"db{self.redis_db}", {}).get("keys", 0),
                "bpm_cached": bpm_keys,
                "temporal_cached": temporal_keys,
                "algorithm_version": self.algorithm_version,
            }

        except RedisError as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"connected": False, "error": str(e)}
