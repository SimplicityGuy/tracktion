"""
Redis caching layer for audio analysis results.

This module provides caching functionality to avoid re-analyzing
audio files that have already been processed.
"""

import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import redis
import xxhash
from redis.exceptions import ConnectionError, RedisError

from services.analysis_service.src.exceptions import (
    AnalysisServiceError,
    InvalidAudioFileError,
    TransientStorageError,
)

logger = logging.getLogger(__name__)


class Hasher(Protocol):
    """Protocol for hash objects with update and hexdigest methods."""

    def update(self, data: bytes) -> None:
        """Update hasher with data."""
        ...

    def hexdigest(self) -> str:
        """Return hex digest."""
        ...


class AudioCache:
    """Redis-based cache for audio analysis results."""

    # Cache key prefixes
    BPM_PREFIX = "bpm"
    TEMPORAL_PREFIX = "temporal"
    KEY_PREFIX = "key"
    MOOD_PREFIX = "mood"

    # Default TTL values (in seconds)
    DEFAULT_TTL = 30 * 24 * 60 * 60  # 30 days
    FAILED_TTL = 60 * 60  # 1 hour
    LOW_CONFIDENCE_TTL = 7 * 24 * 60 * 60  # 7 days

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        default_ttl: int = DEFAULT_TTL,
        algorithm_version: str = "1.0",
        use_xxh128: bool = True,
        enabled: bool = True,
        connection_retry_attempts: int = 3,
        connection_retry_delay: float = 1.0,
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
            connection_retry_attempts: Number of retry attempts for Redis connection
            connection_retry_delay: Delay between retry attempts in seconds
        """
        self.enabled = enabled
        self.algorithm_version = algorithm_version
        self.default_ttl = default_ttl
        self.use_xxh128 = use_xxh128
        self.redis_client: redis.Redis | None = None
        self.connection_retry_attempts = connection_retry_attempts
        self.connection_retry_delay = connection_retry_delay

        if not self.enabled:
            logger.info("Audio cache disabled")
            return

        self._connect_with_retry()

    def _connect_with_retry(self) -> None:
        """Connect to Redis with retry logic."""

        for attempt in range(self.connection_retry_attempts):
            try:
                self.redis_client = redis.Redis(
                    host="localhost",  # Would normally get these from __init__ args
                    port=6379,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis successfully")
                return
            except (ConnectionError, RedisError) as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e!s}")
                if attempt < self.connection_retry_attempts - 1:
                    time.sleep(self.connection_retry_delay)
                else:
                    logger.error("All Redis connection attempts failed, disabling cache")
                    self.redis_client = None
                    self.enabled = False

    def _generate_file_hash(self, file_path: str) -> str | None:
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
                    hasher = cast("Hasher", xxhash.xxh128())
                except ImportError:
                    logger.warning("xxhash not available, falling back to SHA256")
                    hasher = cast("Hasher", hashlib.sha256())  # Both types implement same interface
            else:
                hasher = cast("Hasher", hashlib.sha256())  # Both types implement same interface

            # Read file in chunks to handle large files
            with Path(file_path).open("rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)

            return str(hasher.hexdigest())

        except FileNotFoundError as e:
            logger.error(f"File not found when generating hash: {file_path} - {e!s}")
            raise InvalidAudioFileError(f"Audio file not found: {file_path}") from e
        except PermissionError as e:
            logger.error(f"Permission denied reading file for hash: {file_path} - {e!s}")
            raise InvalidAudioFileError(f"Permission denied: {file_path}") from e
        except IsADirectoryError as e:
            logger.error(f"Attempted to hash directory instead of file: {file_path} - {e!s}")
            raise InvalidAudioFileError(f"Path is a directory: {file_path}") from e
        except OSError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "space" in error_msg or "quota" in error_msg:
                logger.error(f"Disk space/quota error reading file: {file_path} - {e!s}")
                raise TransientStorageError(f"Disk space error: {file_path}") from e
            if "network" in error_msg or "connection" in error_msg:
                logger.error(f"Network error reading file: {file_path} - {e!s}")
                raise TransientStorageError(f"Network error: {file_path}") from e
            logger.error(f"OS error reading file for hash: {file_path} - {e!s}")
            raise InvalidAudioFileError(f"File system error: {file_path}") from e
        except MemoryError as e:
            logger.error(f"Out of memory generating hash for large file: {file_path} - {e!s}")
            # Try with larger chunk size to reduce memory pressure
            try:
                logger.info(f"Retrying hash generation with larger chunks for {file_path}")
                hasher = cast("Hasher", hashlib.sha256())  # Use SHA256 as fallback for memory issues
                with Path(file_path).open("rb") as f:
                    while chunk := f.read(65536):  # 64KB chunks instead of 8KB
                        hasher.update(chunk)
                return str(hasher.hexdigest())
            except Exception as retry_error:
                logger.error(f"Hash generation retry failed: {retry_error!s}")
                raise AnalysisServiceError(f"Out of memory hashing file: {file_path}") from e
        except Exception as e:
            logger.error(f"Unexpected error generating file hash: {file_path} - {e!s}", exc_info=True)
            raise AnalysisServiceError(f"Hash generation failed: {file_path}: {e!s}") from e

    def _ensure_connection(self) -> bool:
        """Ensure Redis connection is healthy, reconnect if needed."""
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.ping()
            return True
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis connection lost: {e!s}. Attempting reconnection...")
            self._connect_with_retry()
            return self.redis_client is not None

    def _safe_redis_operation(self, operation: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Safely execute a Redis operation with connection checks and error handling."""
        if not self._ensure_connection():
            logger.debug(f"Skipping {operation} - Redis not available")
            return None

        try:
            return func(*args, **kwargs)
        except (ConnectionError, RedisError) as e:
            logger.error(f"Redis {operation} operation failed: {e!s}")
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

    def get_bpm_results(self, file_path: str) -> dict[str, Any] | None:
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

        def _get_from_redis():
            cached_data = self.redis_client.get(cache_key)  # type: ignore
            if cached_data:
                logger.debug(f"Cache hit for BPM analysis: {cache_key}")
                return json.loads(str(cached_data))
            return None

        result = self._safe_redis_operation("get", _get_from_redis)
        if result is None:
            logger.debug(f"Cache miss for BPM analysis: {cache_key}")
        return result  # type: ignore[no-any-return]

    def set_bpm_results(
        self,
        file_path: str,
        results: dict[str, Any],
        confidence: float | None = None,
        failed: bool = False,
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
            "cached_at": datetime.now(UTC).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        def _set_in_redis():
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))  # type: ignore
            logger.debug(f"Cached BPM results: {cache_key} (TTL: {ttl}s)")
            return True

        result = self._safe_redis_operation("set", _set_in_redis)
        return result is not None

    def get_temporal_results(self, file_path: str) -> dict[str, Any] | None:
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
            logger.debug(f"Cache miss for temporal analysis: {cache_key}")
            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached temporal results: {e!s}")
            return None

    def set_temporal_results(
        self,
        file_path: str,
        results: dict[str, Any],
        stability_score: float | None = None,
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
            "cached_at": datetime.now(UTC).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached temporal results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache temporal results: {e!s}")
            return False

    def get_key_results(self, file_path: str) -> dict[str, Any] | None:
        """
        Get cached key detection results for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached key results or None if not in cache
        """
        if not self.enabled or not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.KEY_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key analysis: {cache_key}")
                return json.loads(str(cached_data))  # type: ignore[no-any-return]
            logger.debug(f"Cache miss for key analysis: {cache_key}")
            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached key results: {e!s}")
            return None

    def set_key_results(
        self,
        file_path: str,
        results: dict[str, Any],
        confidence: float | None = None,
    ) -> bool:
        """
        Cache key detection results for a file.

        Args:
            file_path: Path to the audio file
            results: Key detection results
            confidence: Confidence score (0-1)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.KEY_PREFIX, file_hash)

        # Determine TTL based on confidence
        ttl = self.LOW_CONFIDENCE_TTL if confidence is not None and confidence < 0.6 else self.default_ttl

        # Add metadata
        cache_data = {
            **results,
            "cached_at": datetime.now(UTC).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached key results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache key results: {e!s}")
            return False

    def get_mood_results(self, file_path: str) -> dict[str, Any] | None:
        """
        Get cached mood analysis results for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Cached mood results or None if not in cache
        """
        if not self.enabled or not self.redis_client:
            return None

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return None

        cache_key = self._build_cache_key(self.MOOD_PREFIX, file_hash)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for mood analysis: {cache_key}")
                return json.loads(str(cached_data))  # type: ignore[no-any-return]
            logger.debug(f"Cache miss for mood analysis: {cache_key}")
            return None

        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve cached mood results: {e!s}")
            return None

    def set_mood_results(
        self,
        file_path: str,
        results: dict[str, Any],
        confidence: float | None = None,
    ) -> bool:
        """
        Cache mood analysis results for a file.

        Args:
            file_path: Path to the audio file
            results: Mood analysis results
            confidence: Overall confidence score (0-1)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        file_hash = self._generate_file_hash(file_path)
        if not file_hash:
            return False

        cache_key = self._build_cache_key(self.MOOD_PREFIX, file_hash)

        # Determine TTL based on confidence
        ttl = self.LOW_CONFIDENCE_TTL if confidence is not None and confidence < 0.6 else self.default_ttl

        # Add metadata
        cache_data = {
            **results,
            "cached_at": datetime.now(UTC).isoformat(),
            "algorithm_version": self.algorithm_version,
        }

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data))
            logger.debug(f"Cached mood results: {cache_key} (TTL: {ttl}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Failed to cache mood results: {e!s}")
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
        key_key = self._build_cache_key(self.KEY_PREFIX, file_hash)
        mood_key = self._build_cache_key(self.MOOD_PREFIX, file_hash)

        try:
            deleted_count = 0
            for key in [bpm_key, temporal_key, key_key, mood_key]:
                if self.redis_client.delete(key):
                    deleted_count += 1
                    logger.debug(f"Invalidated cache key: {key}")

            return deleted_count > 0

        except RedisError as e:
            logger.error(f"Failed to invalidate cache: {e!s}")
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
                    # Type cast for mypy - synchronous Redis client returns list[str]
                    keys = cast("list[str]", keys_result) if keys_result else []
                except TypeError:
                    keys = []
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                # Handle both sync and async returns
                if hasattr(deleted_count, "__await__"):
                    count = 0  # Skip async results
                else:
                    try:
                        # Type cast for mypy - synchronous Redis client returns int
                        count = deleted_count if deleted_count is not None else 0
                    except (TypeError, ValueError):
                        count = 0
                logger.info(f"Flushed {count} cache entries for version {self.algorithm_version}")
                return count
            return 0

        except RedisError as e:
            logger.error(f"Failed to flush version cache: {e!s}")
            return 0

    def warm_cache(self, file_paths: list[str], processor: Callable[[str], dict[str, Any]]) -> int:
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
                logger.error(f"Failed to warm cache for {file_path}: {e!s}")

        logger.info(f"Cache warming complete: {cached_count}/{len(file_paths)} files cached")
        return cached_count

    def get_cache_stats(self) -> dict[str, Any]:
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
            key_keys_result = self.redis_client.keys(f"{self.KEY_PREFIX}:*")
            mood_keys_result = self.redis_client.keys(f"{self.MOOD_PREFIX}:*")
            # Handle both sync and async returns
            if hasattr(bpm_keys_result, "__await__"):
                bpm_keys = []  # Skip async results
            else:
                try:
                    # Type cast for mypy - synchronous Redis client returns list[str]
                    bpm_keys = cast("list[str]", bpm_keys_result) if bpm_keys_result else []
                except TypeError:
                    bpm_keys = []

            if hasattr(temporal_keys_result, "__await__"):
                temporal_keys = []  # Skip async results
            else:
                try:
                    # Type cast for mypy - synchronous Redis client returns list[str]
                    temporal_keys = cast("list[str]", temporal_keys_result) if temporal_keys_result else []
                except TypeError:
                    temporal_keys = []

            if hasattr(key_keys_result, "__await__"):
                key_keys = []  # Skip async results
            else:
                try:
                    # Type cast for mypy - synchronous Redis client returns list[str]
                    key_keys = cast("list[str]", key_keys_result) if key_keys_result else []
                except TypeError:
                    key_keys = []

            if hasattr(mood_keys_result, "__await__"):
                mood_keys = []  # Skip async results
            else:
                try:
                    # Type cast for mypy - synchronous Redis client returns list[str]
                    mood_keys = cast("list[str]", mood_keys_result) if mood_keys_result else []
                except TypeError:
                    mood_keys = []

            # Calculate sizes
            total_size = 0
            if bpm_keys or temporal_keys or key_keys or mood_keys:
                all_keys = bpm_keys + temporal_keys + key_keys + mood_keys
                for key in all_keys:
                    size = self.redis_client.memory_usage(key)
                    if size and not hasattr(size, "__await__"):
                        # Ensure size is an integer
                        with contextlib.suppress(TypeError, ValueError):
                            # Type cast for mypy - synchronous Redis client returns int
                            total_size += cast("int", size) if size is not None else 0

            return {
                "enabled": True,
                "connected": True,
                "bpm_keys": len(bpm_keys),
                "temporal_keys": len(temporal_keys),
                "key_keys": len(key_keys),
                "mood_keys": len(mood_keys),
                "total_keys": len(bpm_keys) + len(temporal_keys) + len(key_keys) + len(mood_keys),
                "total_size_bytes": total_size,
                "algorithm_version": self.algorithm_version,
            }

        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e!s}")
            return {"enabled": True, "connected": False, "error": str(e)}
