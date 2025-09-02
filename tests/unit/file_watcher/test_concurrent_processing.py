"""Test concurrent file processing capabilities."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles  # type: ignore
import pytest_asyncio

from services.file_watcher.src.async_file_watcher import AsyncFileEventHandler, AsyncFileWatcherService
from services.file_watcher.src.async_message_publisher import AsyncMessagePublisher


@pytest_asyncio.fixture
async def mock_publisher():
    """Create a mock async message publisher."""
    publisher = AsyncMock(spec=AsyncMessagePublisher)
    publisher.publish_file_event = AsyncMock(return_value=True)
    publisher.connect = AsyncMock()
    publisher.disconnect = AsyncMock()
    return publisher


@pytest_asyncio.fixture
async def async_event_handler(mock_publisher):
    """Create an async event handler for testing."""
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent operations for testing
    return AsyncFileEventHandler(
        publisher=mock_publisher,
        instance_id="test_instance",
        loop=loop,
        semaphore=semaphore,
    )


class TestConcurrentProcessing:
    """Test concurrent file processing capabilities."""

    async def test_semaphore_limits_concurrent_operations(self, async_event_handler, mock_publisher):
        """Test that semaphore properly limits concurrent operations."""
        # Track concurrent operations
        concurrent_count = 0
        max_concurrent = 0

        async def slow_publish(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate slow operation
            concurrent_count -= 1
            return True

        mock_publisher.publish_file_event = slow_publish

        # Create 30 file processing tasks
        tasks = []
        for i in range(30):
            task = asyncio.create_task(async_event_handler._process_file_async(f"/test/file_{i}.mp3", "created"))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Verify semaphore limited concurrent operations to 10
        assert max_concurrent <= 10, f"Max concurrent operations {max_concurrent} exceeded limit of 10"

    async def test_concurrent_file_processing_performance(self, async_event_handler, mock_publisher):
        """Test that concurrent processing improves performance."""
        # Track processing times
        start_times = {}
        end_times = {}

        async def track_timing(event_type, file_path, **kwargs):
            if file_path not in start_times:
                start_times[file_path] = time.time()
            await asyncio.sleep(0.05)  # Simulate processing time
            end_times[file_path] = time.time()
            return True

        mock_publisher.publish_file_event = track_timing

        # Process 20 files concurrently
        start = time.time()
        tasks = []
        for i in range(20):
            task = asyncio.create_task(async_event_handler._process_file_async(f"/test/file_{i}.mp3", "created"))
            tasks.append(task)

        await asyncio.gather(*tasks)
        total_time = time.time() - start

        # With concurrent processing, 20 files x 0.05s should take ~0.1-0.2s (not 1s)
        # (because semaphore allows 10 concurrent, so 2 batches)
        assert total_time < 0.5, f"Concurrent processing took too long: {total_time}s"
        assert len(end_times) == 20, "Not all files were processed"

    async def test_concurrent_hash_calculation(self, tmp_path):
        """Test concurrent hash calculation for multiple files."""
        # Create test files
        test_files = []
        for i in range(10):
            file_path = tmp_path / f"test_{i}.mp3"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(b"test data " * 1000)  # Write some data
            test_files.append(file_path)

        # Create event handler
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        mock_publisher.publish_file_event = AsyncMock(return_value=True)

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(5)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="test",
            loop=loop,
            semaphore=semaphore,
        )

        # Calculate hashes concurrently
        start = time.time()
        tasks = []
        for file_path in test_files:
            task = asyncio.create_task(handler._calculate_hashes_async(str(file_path)))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start

        # Verify all hashes were calculated
        assert len(results) == 10
        for sha256, xxh128 in results:
            assert len(sha256) == 64  # SHA256 hex is 64 chars
            assert len(xxh128) == 32  # XXH128 hex is 32 chars

        # Should be faster than sequential processing
        print(f"Concurrent hash calculation took {total_time:.2f}s for 10 files")

    async def test_batch_processing_in_scan(self, tmp_path, mock_publisher):
        """Test batch processing during initial scan."""
        # Create many test files
        for i in range(150):
            file_path = tmp_path / f"song_{i}.mp3"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(b"audio data")

        # Track batch processing
        call_times = []

        async def track_calls(*args, **kwargs):
            call_times.append(time.time())
            await asyncio.sleep(0.001)  # Small delay
            return True

        mock_publisher.publish_file_event = track_calls

        # Mock environment and create service
        with (
            patch.dict("os.environ", {"DATA_DIR": str(tmp_path)}),
            patch(
                "services.file_watcher.src.async_file_watcher.AsyncMessagePublisher",
                return_value=mock_publisher,
            ),
        ):
            service = AsyncFileWatcherService()
            service.publisher = mock_publisher
            await mock_publisher.connect()

            # Perform scan
            await service.scan_existing_files()

        # Verify all files were processed
        assert len(call_times) == 150, f"Expected 150 calls, got {len(call_times)}"

        # Verify batching occurred (should see clusters of calls)
        # Calculate time gaps between calls
        gaps = []
        for i in range(1, len(call_times)):
            gap = call_times[i] - call_times[i - 1]
            if gap > 0.01:  # Significant gap indicates batch boundary
                gaps.append(gap)

        # Should have at least one batch boundary (processing in batches of 100)
        assert len(gaps) >= 1, "No batch processing detected"

    async def test_concurrent_move_operations(self, async_event_handler, mock_publisher):
        """Test concurrent processing of move/rename operations."""
        # Track concurrent move operations
        move_count = 0

        async def track_moves(*args, **kwargs):
            nonlocal move_count
            move_count += 1
            await asyncio.sleep(0.02)
            return True

        mock_publisher.publish_file_event = track_moves

        # Process multiple move operations concurrently
        tasks = []
        for i in range(15):
            old_path = f"/old/file_{i}.mp3"
            new_path = f"/new/file_{i}.mp3"
            task = asyncio.create_task(async_event_handler._process_move_async(old_path, new_path))
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all moves were processed
        assert move_count == 15, f"Expected 15 moves, processed {move_count}"

    async def test_error_handling_in_concurrent_processing(self, async_event_handler, mock_publisher):
        """Test error handling during concurrent processing."""
        # Make some operations fail
        call_count = 0

        async def sometimes_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception("Simulated error")
            return True

        mock_publisher.publish_file_event = sometimes_fail

        # Process files, some will fail
        tasks = []
        for i in range(12):
            task = asyncio.create_task(async_event_handler._process_file_async(f"/test/file_{i}.mp3", "created"))
            tasks.append(task)

        # Should not raise exceptions (errors are caught and logged)
        await asyncio.gather(*tasks)

        # Verify attempts were made
        assert call_count == 12, f"Expected 12 calls, got {call_count}"

    async def test_future_handling_in_sync_context(self):
        """Test handling of Future objects from sync context."""
        # Create async components
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        mock_publisher.publish_file_event = AsyncMock(return_value=True)

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(5)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="test",
            loop=loop,
            semaphore=semaphore,
        )

        # Simulate sync event (like from watchdog)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/file.mp3"

        # Call sync method that schedules async work
        handler.on_created(event)

        # Verify Future was added to processing_tasks
        assert len(handler.processing_tasks) == 1
        assert isinstance(next(iter(handler.processing_tasks)), asyncio.Future)

        # Wait for the task to complete
        await handler.wait_for_tasks()

        # Verify the async operation was called
        mock_publisher.publish_file_event.assert_called_once()
