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
    """Redis-based cache for audio analysis results."""

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
        default_ttl: int = DEFAULT_TTL,
        algorithm_version: str = "1.0",
        use_xxh128: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Initialize audio cache with Redis connection.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            default_ttl: Default TTL for cache entries in seconds
            algorithm_version: Version of the BPM detection algorithm
            use_xxh128: Whether to use xxHash128 for file hashing (faster)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.algorithm_version = algorithm_version
        self.default_ttl = default_ttl
        self.use_xxh128 = use_xxh128
        self.redis_client: Optional[redis.Redis] = None

        if not self.enabled:
            logger.info("Audio cache disabled")
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
            logger.error(f"Failed to generate file hash: {str(e)}")
            return None

    def _build_cache_key(self, prefix: str, file_hash: str) -> str:
        """
        Build cache key with prefix and algorithm version.

        Args:
            prefix: Cache key prefix (BPM_PREFIX or TEMPORAL_PREFIX)
            file_hash: Hash of the file contents

        Returns:
            Complete cache key
        """
        return f"{prefix}:{file_hash}:{self.algorithm_version}"

    def get_bpm_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached BPM results for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached BPM results or None if not in cache
        """
        if not self.enabled or not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.BPM_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for BPM analysis: {cache_key}")
                # Redis returns str due to decode_responses=True
                return json.loads(str(cached_data))  # type: ignore[no-any-return]
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
        Cache BPM results for a file.

        Args:
            file_path: Path to the audio file
            results: BPM analysis results
            confidence: Confidence score (0-1)
            failed: Whether the analysis failed

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.BPM_PREFIX, file_hash)

        # Determine TTL based on confidence and failure status
        if failed:
            ttl = self.FAILED_TTL
        elif confidence is not None and confidence < 0.5:
            ttl = self.LOW_CONFIDENCE_TTL
        else:
            ttl = self.default_ttl

        # Add metadata
        cache_data = {
            **results,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached BPM results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache BPM results: {str(e)}")
            return False

    def get_temporal_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached temporal analysis results for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached temporal results or None if not in cache
        """
        if not self.enabled or not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.TEMPORAL_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for temporal analysis: {cache_key}")
                return json.loads(str(cached_data))  # type: ignore[no-any-return]
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
        Cache temporal analysis results for a file.

        Args:
            file_path: Path to the audio file
            results: Temporal analysis results
            stability_score: Tempo stability score (0-1)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.TEMPORAL_PREFIX, file_hash)

        # Use longer TTL for stable tempo tracks
        ttl = self.default_ttl if stability_score is None or stability_score > 0.8 else self.LOW_CONFIDENCE_TTL

        # Add metadata
        cache_data = {
            **results,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached temporal results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache temporal results: {str(e)}")
            return False

    def invalidate_cache(self, file_path: str) -> bool:
        """
        Invalidate cache entries for a specific file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if successfully invalidated, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        bpm_key = self._build_cache_key(self.BPM_PREFIX, file_hash)
        temporal_key = self._build_cache_key(self.TEMPORAL_PREFIX, file_hash)

        try:
            deleted_count = 0
            for key in [bpm_key, temporal_key]:
                if self.redis_client.delete(key):
                    deleted_count += 1
                    logger.debug(f"Invalidated cache key: {key}")

            return deleted_count > 0

        except RedisError as e:
            logger.error(f"Failed to invalidate cache: {str(e)}")
            return False

    def flush_version_cache(self) -> int:
        """
        Flush all cache entries for the current algorithm version.

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        pattern = f"*:{self.algorithm_version}"
        try:
            keys_result = self.redis_client.keys(pattern)
            # Handle both sync and async returns
            if hasattr(keys_result, "__await__"):
                keys = []  # Skip async results
            else:
                try:
                    keys = list(keys_result) if keys_result else []  # type: ignore[arg-type]
                except TypeError:
                    keys = []
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                # Handle both sync and async returns
                if hasattr(deleted_count, "__await__"):
                    count = 0  # Skip async results
                else:
                    try:
                        count = int(deleted_count) if deleted_count else 0  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        count = 0
                logger.info(f"Flushed {count} cache entries for version {self.algorithm_version}")
                return count
            return 0

        except RedisError as e:
            logger.error(f"Failed to flush version cache: {str(e)}")
            return 0

    def warm_cache(self, file_paths: list[str], processor: Callable[[str], Dict[str, Any]]) -> int:
        """
        Pre-populate cache with analysis results.

        Args:
            file_paths: List of audio file paths to process
            processor: Function to process each file and return results

        Returns:
            Number of files successfully cached
        """
        if not self.enabled or not self.redis_client:
            return 0

        cached_count = 0
        for file_path in file_paths:
            try:
                # Check if already cached
                if self.get_bpm_results(file_path):
                    logger.debug(f"File already cached: {file_path}")
                    continue

                # Process and cache
                results = processor(file_path)
                confidence = results.get("confidence", 1.0)
                if self.set_bpm_results(file_path, results, confidence=confidence):
                    cached_count += 1
                    logger.debug(f"Warmed cache for: {file_path}")

            except Exception as e:
                logger.error(f"Failed to warm cache for {file_path}: {str(e)}")

        logger.info(f"Cache warming complete: {cached_count}/{len(file_paths)} files cached")
        return cached_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False}

        try:
            # Get all keys matching our patterns
            bpm_keys_result = self.redis_client.keys(f"{self.BPM_PREFIX}:*")
            temporal_keys_result = self.redis_client.keys(f"{self.TEMPORAL_PREFIX}:*")
            # Handle both sync and async returns
            if hasattr(bpm_keys_result, "__await__"):
                bpm_keys = []  # Skip async results
            else:
                try:
                    bpm_keys = list(bpm_keys_result) if bpm_keys_result else []  # type: ignore[arg-type]
                except TypeError:
                    bpm_keys = []

            if hasattr(temporal_keys_result, "__await__"):
                temporal_keys = []  # Skip async results
            else:
                try:
                    temporal_keys = list(temporal_keys_result) if temporal_keys_result else []  # type: ignore[arg-type]
                except TypeError:
                    temporal_keys = []

            # Calculate sizes
            total_size = 0
            if bpm_keys or temporal_keys:
                all_keys = bpm_keys + temporal_keys
                for key in all_keys:
                    size = self.redis_client.memory_usage(key)
                    if size and not hasattr(size, "__await__"):
                        # Ensure size is an integer
                        try:
                            total_size += int(size) if size else 0  # type: ignore[arg-type]
                        except (TypeError, ValueError):
                            pass  # Skip invalid size values

            return {
                "enabled": True,
                "connected": True,
                "bpm_keys": len(bpm_keys),
                "temporal_keys": len(temporal_keys),
                "total_keys": len(bpm_keys) + len(temporal_keys),
                "total_size_bytes": total_size,
                "algorithm_version": self.algorithm_version,
            }

        except RedisError as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"enabled": True, "connected": False, "error": str(e)}
