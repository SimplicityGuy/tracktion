"""Test edge cases for async file watcher."""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.async_file_watcher import AsyncFileEventHandler, AsyncFileWatcherService
from src.async_message_publisher import AsyncMessagePublisher
from src.async_metadata_extractor import AsyncMetadataExtractor


class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_file_permissions_error(self) -> None:
        """Test handling of files with permission errors."""
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        mock_publisher.publish_file_event = AsyncMock(return_value=True)

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(10)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="test",
            loop=loop,
            semaphore=semaphore,
        )

        # Process file that doesn't exist (simulates permission error)
        await handler._process_file_async("/restricted/no_access.mp3", "created")

        # Should still publish event even if hash calculation fails
        mock_publisher.publish_file_event.assert_called_once()

    async def test_large_file_handling(self) -> None:
        """Test handling of very large files."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a 100MB file
            chunk = b"x" * (1024 * 1024)  # 1MB chunk
            for _ in range(100):
                f.write(chunk)
            large_file = f.name

        try:
            mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
            mock_publisher.publish_file_event = AsyncMock(return_value=True)

            loop = asyncio.get_event_loop()
            semaphore = asyncio.Semaphore(1)  # Limit concurrency for large files
            handler = AsyncFileEventHandler(
                publisher=mock_publisher,
                instance_id="test",
                loop=loop,
                semaphore=semaphore,
            )

            # Should handle large file without timeout
            await asyncio.wait_for(handler._process_file_async(large_file, "created"), timeout=30.0)

            mock_publisher.publish_file_event.assert_called_once()

        finally:
            os.unlink(large_file)

    async def test_concurrent_same_file_access(self) -> None:
        """Test multiple events for the same file."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"test data")
            test_file = f.name

        try:
            mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
            call_count = 0

            async def count_calls(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)
                return True

            mock_publisher.publish_file_event = count_calls

            loop = asyncio.get_event_loop()
            semaphore = asyncio.Semaphore(10)
            handler = AsyncFileEventHandler(
                publisher=mock_publisher,
                instance_id="test",
                loop=loop,
                semaphore=semaphore,
            )

            # Fire multiple events for same file
            tasks = []
            for event_type in ["created", "modified", "modified", "modified"]:
                task = asyncio.create_task(handler._process_file_async(test_file, event_type))
                tasks.append(task)

            await asyncio.gather(*tasks)

            # All events should be processed
            assert call_count == 4

        finally:
            os.unlink(test_file)

    async def test_rabbitmq_connection_failure(self) -> None:
        """Test handling of RabbitMQ connection failures."""
        publisher = AsyncMessagePublisher("amqp://invalid:5672/", "test")

        # Should handle connection failure gracefully
        with patch("src.async_message_publisher.aio_pika.connect_robust") as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            try:
                await publisher.connect()
            except Exception:
                pass  # Expected

            # Publish should return False when not connected
            result = await publisher.publish_file_event("created", "/test.mp3", "test")
            assert result is False

    async def test_metadata_extraction_timeout(self) -> None:
        """Test timeout handling in metadata extraction."""
        extractor = AsyncMetadataExtractor(max_workers=1)

        try:
            # Mock slow file operation
            with patch.object(extractor, "_extract_metadata_sync") as mock_extract:
                mock_extract.side_effect = lambda x: time.sleep(10)

                # Should not hang indefinitely
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(extractor.extract_metadata("/slow/file.mp3"), timeout=1.0)
        finally:
            extractor.shutdown()

    async def test_file_deleted_during_processing(self) -> None:
        """Test handling when file is deleted during processing."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"test")
            temp_file = f.name

        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        published_events = []

        async def capture_event(*args, **kwargs):
            published_events.append(kwargs)
            # Delete file after first event
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return True

        mock_publisher.publish_file_event = capture_event

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(1)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="test",
            loop=loop,
            semaphore=semaphore,
        )

        # Process file that gets deleted
        await handler._process_file_async(temp_file, "modified")

        # Should still publish event
        assert len(published_events) == 1
        assert published_events[0]["event_type"] == "modified"

    async def test_invalid_file_extensions(self) -> None:
        """Test that non-audio files are ignored."""
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        mock_publisher.publish_file_event = AsyncMock()

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(10)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="test",
            loop=loop,
            semaphore=semaphore,
        )

        # Test with non-audio file
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/document.pdf"

        handler.on_created(event)

        # Should not process non-audio files
        await asyncio.sleep(0.1)  # Give time for async processing
        mock_publisher.publish_file_event.assert_not_called()

    async def test_directory_with_many_files(self) -> None:
        """Test scanning directory with many nested files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Create nested directory structure
            for i in range(5):
                subdir = base_dir / f"artist_{i}"
                subdir.mkdir()
                for j in range(10):
                    album_dir = subdir / f"album_{j}"
                    album_dir.mkdir()
                    for k in range(5):
                        song_file = album_dir / f"song_{k}.mp3"
                        song_file.write_bytes(b"audio")

            # Total: 5 * 10 * 5 = 250 files

            mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
            event_count = 0

            async def count_events(*args, **kwargs):
                nonlocal event_count
                event_count += 1
                return True

            mock_publisher.publish_file_event = count_events
            mock_publisher.connect = AsyncMock()

            with patch.dict("os.environ", {"DATA_DIR": str(base_dir)}):
                with patch("src.async_file_watcher.AsyncMessagePublisher", return_value=mock_publisher):
                    service = AsyncFileWatcherService()
                    service.publisher = mock_publisher

                    await service.scan_existing_files()

            assert event_count == 250, f"Expected 250 events, got {event_count}"

    async def test_unicode_filenames(self) -> None:
        """Test handling of files with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with unicode names
            unicode_files = [
                Path(tmpdir) / "测试文件.mp3",
                Path(tmpdir) / "тест.wav",
                Path(tmpdir) / "🎵music🎵.flac",
                Path(tmpdir) / "café_société.mp3",
            ]

            for file_path in unicode_files:
                file_path.write_bytes(b"audio data")

            mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
            processed_files = []

            async def capture_files(*args, **kwargs):
                if "file_path" in kwargs:
                    processed_files.append(kwargs["file_path"])
                return True

            mock_publisher.publish_file_event = capture_files

            loop = asyncio.get_event_loop()
            semaphore = asyncio.Semaphore(10)
            handler = AsyncFileEventHandler(
                publisher=mock_publisher,
                instance_id="test",
                loop=loop,
                semaphore=semaphore,
            )

            # Process unicode files
            for file_path in unicode_files:
                await handler._process_file_async(str(file_path), "created")

            assert len(processed_files) == 4
            # Verify unicode preserved
            assert any("测试" in p for p in processed_files)
            assert any("тест" in p for p in processed_files)
