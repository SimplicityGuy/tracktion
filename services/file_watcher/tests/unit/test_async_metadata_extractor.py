"""Test async metadata extraction functionality."""

import asyncio
import os
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest_asyncio

from src.async_metadata_extractor import AsyncMetadataExtractor, AsyncMetadataProgressTracker


@pytest_asyncio.fixture
async def metadata_extractor():
    """Create a metadata extractor for testing."""
    extractor = AsyncMetadataExtractor(max_workers=2)
    yield extractor
    extractor.shutdown()


@pytest_asyncio.fixture
async def test_audio_file() -> str:
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except Exception:
        pass


class TestAsyncMetadataExtractor:
    """Test async metadata extraction."""

    async def test_extract_metadata_basic(self, metadata_extractor: Any, test_audio_file: str) -> None:
        """Test basic metadata extraction."""
        metadata = await metadata_extractor.extract_metadata(test_audio_file)

        assert metadata is not None
        assert "file_path" in metadata
        assert "file_name" in metadata
        assert "file_size" in metadata
        assert metadata["file_path"] == test_audio_file
        assert metadata["file_name"].endswith(".mp3")
        assert metadata["file_size"] > 0

    async def test_metadata_caching(self, metadata_extractor: Any, test_audio_file: str) -> None:
        """Test that metadata is cached properly."""
        # First extraction
        metadata1 = await metadata_extractor.extract_metadata(test_audio_file)
        cache_size1 = await metadata_extractor.get_cache_size()

        # Second extraction (should hit cache)
        metadata2 = await metadata_extractor.extract_metadata(test_audio_file)
        cache_size2 = await metadata_extractor.get_cache_size()

        assert cache_size1 == 1
        assert cache_size2 == 1  # Cache size shouldn't increase
        assert metadata1 == metadata2

    async def test_batch_extraction(self, metadata_extractor: Any) -> None:
        """Test batch metadata extraction."""
        # Create multiple test files
        test_files = []
        for _i in range(5):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(b"test data " * 100)
                test_files.append(f.name)

        try:
            # Extract metadata for all files
            results = await metadata_extractor.extract_batch(test_files)

            assert len(results) == 5
            for path in test_files:
                assert path in results
                assert "file_name" in results[path]
                assert "file_size" in results[path]

        finally:
            # Cleanup
            for path in test_files:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    async def test_concurrent_extraction(self, metadata_extractor: Any) -> None:
        """Test concurrent metadata extraction."""
        # Create test files
        test_files = []
        for _i in range(10):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"audio " * 200)
                test_files.append(f.name)

        try:
            # Extract metadata concurrently
            start_time = time.time()
            tasks = [metadata_extractor.extract_metadata(path) for path in test_files]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            assert len(results) == 10
            for metadata in results:
                assert "file_path" in metadata
                assert "file_size" in metadata

            # Should be faster than sequential (thread pool helps)
            print(f"Concurrent extraction took {elapsed:.2f}s for 10 files")

        finally:
            # Cleanup
            for path in test_files:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    async def test_error_handling(self, metadata_extractor: Any) -> None:
        """Test error handling for non-existent files."""
        metadata = await metadata_extractor.extract_metadata("/nonexistent/file.mp3")

        assert metadata is not None
        assert "error" in metadata
        assert metadata["file_path"] == "/nonexistent/file.mp3"

    async def test_cache_operations(self, metadata_extractor: Any, test_audio_file: str) -> None:
        """Test cache clear and size operations."""
        # Add to cache
        await metadata_extractor.extract_metadata(test_audio_file)
        size_before = await metadata_extractor.get_cache_size()

        # Clear cache
        await metadata_extractor.clear_cache()
        size_after = await metadata_extractor.get_cache_size()

        assert size_before == 1
        assert size_after == 0

    @patch("src.async_metadata_extractor.File")
    async def test_mutagen_metadata_extraction(
        self, mock_file: Any, metadata_extractor: Any, test_audio_file: str
    ) -> None:
        """Test extraction with mutagen metadata."""
        # Mock mutagen File object
        mock_audio = MagicMock()
        mock_audio.info.length = 180.5
        mock_audio.info.bitrate = 320000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.mime = ["audio/mpeg"]
        mock_audio.tags = {
            "title": ["Test Song"],
            "artist": ["Test Artist"],
            "album": ["Test Album"],
            "date": ["2024"],
            "genre": ["Electronic"],
        }
        mock_file.return_value = mock_audio

        metadata = await metadata_extractor.extract_metadata(test_audio_file)

        assert metadata["duration"] == 180.5
        assert metadata["bitrate"] == 320000
        assert metadata["sample_rate"] == 44100
        assert metadata["channels"] == 2
        assert metadata["format"] == "audio/mpeg"
        assert "tags" in metadata
        assert metadata["tags"]["title"] == "Test Song"
        assert metadata["tags"]["artist"] == "Test Artist"


class TestAsyncMetadataProgressTracker:
    """Test progress tracking for bulk operations."""

    async def test_progress_tracking(self) -> None:
        """Test basic progress tracking."""
        tracker = AsyncMetadataProgressTracker()

        # Start batch
        await tracker.start_batch(100)

        # Update progress
        for _i in range(50):
            await tracker.update_progress(success=True)

        for _i in range(5):
            await tracker.update_progress(success=False)

        # Check progress
        progress = await tracker.get_progress()

        assert progress["total"] == 100
        assert progress["processed"] == 55
        assert progress["failed"] == 5
        assert progress["progress_pct"] == 55.0

    async def test_concurrent_progress_updates(self) -> None:
        """Test concurrent progress updates."""
        tracker = AsyncMetadataProgressTracker()
        await tracker.start_batch(200)

        # Simulate concurrent updates
        async def update_batch(count: int, success: bool):
            for _ in range(count):
                await tracker.update_progress(success=success)

        tasks = [
            update_batch(50, True),
            update_batch(50, True),
            update_batch(10, False),
        ]

        await asyncio.gather(*tasks)

        progress = await tracker.get_progress()
        assert progress["processed"] == 110
        assert progress["failed"] == 10
