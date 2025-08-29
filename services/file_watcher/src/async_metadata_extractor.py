"""Async metadata extraction for audio files."""

import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import structlog
from mutagen import File  # type: ignore

logger = structlog.get_logger()


class AsyncMetadataExtractor:
    """Async metadata extraction for audio files."""

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus", ".oga"}

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize async metadata extractor.

        Args:
            max_workers: Maximum number of threads for CPU-bound operations
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: dict[str, dict[str, Any]] = {}

    async def extract_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from audio file asynchronously.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary containing metadata
        """
        # Check cache first
        cache_key = await self._get_cache_key(file_path)
        if cache_key in self._cache:
            logger.debug("Metadata cache hit", file_path=file_path)
            return self._cache[cache_key]

        # Extract metadata in thread pool (CPU-bound operation)
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(self.executor, self._extract_metadata_sync, file_path)

        # Cache the result
        self._cache[cache_key] = metadata
        logger.info("Metadata extracted", file_path=file_path, metadata=metadata)

        return metadata

    def _extract_metadata_sync(self, file_path: str) -> dict[str, Any]:
        """Synchronously extract metadata (runs in thread pool).

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary containing metadata
        """
        try:
            # Get basic file info
            file_stat = os.stat(file_path)
            metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_size": file_stat.st_size,
                "modified_time": file_stat.st_mtime,
            }

            # Try to extract audio metadata using mutagen
            audio_file = File(file_path)
            if audio_file is not None:
                # Extract common metadata
                metadata.update(
                    {
                        "duration": getattr(audio_file.info, "length", None),
                        "bitrate": getattr(audio_file.info, "bitrate", None),
                        "sample_rate": getattr(audio_file.info, "sample_rate", None),
                        "channels": getattr(audio_file.info, "channels", None),
                        "format": audio_file.mime[0] if audio_file.mime else None,
                    }
                )

                # Extract tags
                if audio_file.tags:
                    tags = {}
                    for key in ["title", "artist", "album", "date", "genre", "track"]:
                        value = audio_file.tags.get(key)
                        if value:
                            # Handle list values (common in ID3 tags)
                            if isinstance(value, list) and value:
                                tags[key] = str(value[0])
                            else:
                                tags[key] = str(value)
                    metadata["tags"] = tags

            return metadata

        except Exception as e:
            logger.error("Failed to extract metadata", file_path=file_path, error=str(e))
            # Return basic metadata even on error
            return {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "error": str(e),
            }

    async def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file based on path and modification time.

        Args:
            file_path: Path to the file

        Returns:
            Cache key string
        """
        try:
            # Use file path and modification time for cache key
            stat = await asyncio.get_event_loop().run_in_executor(None, os.stat, file_path)
            key_string = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception:
            # If we can't get file stats, just use the path
            return hashlib.md5(file_path.encode()).hexdigest()

    async def extract_batch(self, file_paths: list[str]) -> dict[str, dict[str, Any]]:
        """Extract metadata from multiple files concurrently.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file paths to metadata
        """
        tasks = [self.extract_metadata(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        metadata_dict: dict[str, dict[str, Any]] = {}
        for path, result in zip(file_paths, results, strict=False):
            if isinstance(result, Exception):
                logger.error("Batch extraction error", file_path=path, error=str(result))
                metadata_dict[path] = {"error": str(result)}
            elif isinstance(result, dict):
                metadata_dict[path] = result
            else:
                # Shouldn't happen, but handle it gracefully
                metadata_dict[path] = {"error": "Unexpected result type"}

        return metadata_dict

    async def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()
        logger.info("Metadata cache cleared")

    async def get_cache_size(self) -> int:
        """Get the current cache size.

        Returns:
            Number of items in cache
        """
        return len(self._cache)

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("Metadata extractor shutdown")

    async def __aenter__(self) -> "AsyncMetadataExtractor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - ensures proper cleanup."""
        self.shutdown()


class AsyncMetadataProgressTracker:
    """Track progress of bulk metadata extraction operations."""

    def __init__(self) -> None:
        """Initialize progress tracker."""
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self._lock = asyncio.Lock()

    async def start_batch(self, total: int) -> None:
        """Start tracking a new batch.

        Args:
            total: Total number of files in batch
        """
        async with self._lock:
            self.total_files = total
            self.processed_files = 0
            self.failed_files = 0
            logger.info("Starting batch processing", total_files=total)

    async def update_progress(self, success: bool = True) -> None:
        """Update progress for a processed file.

        Args:
            success: Whether the file was processed successfully
        """
        async with self._lock:
            self.processed_files += 1
            if not success:
                self.failed_files += 1

            # Log progress every 10% or every 100 files
            if self.processed_files % max(self.total_files // 10, 100) == 0:
                progress_pct = (self.processed_files / self.total_files) * 100
                logger.info(
                    "Batch progress",
                    processed=self.processed_files,
                    total=self.total_files,
                    failed=self.failed_files,
                    progress_pct=f"{progress_pct:.1f}%",
                )

    async def get_progress(self) -> dict[str, Any]:
        """Get current progress statistics.

        Returns:
            Dictionary with progress information
        """
        async with self._lock:
            return {
                "total": self.total_files,
                "processed": self.processed_files,
                "failed": self.failed_files,
                "success_rate": (
                    ((self.processed_files - self.failed_files) / self.processed_files * 100)
                    if self.processed_files > 0
                    else 0
                ),
                "progress_pct": ((self.processed_files / self.total_files * 100) if self.total_files > 0 else 0),
            }
