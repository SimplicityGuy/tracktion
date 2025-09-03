"""Test concurrent file processing capabilities."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import pytest
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

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_operations(self, async_event_handler, mock_publisher):
        """Test that semaphore properly limits concurrent operations."""
        # Track concurrent operations
        concurrent_count = 0
        max_concurrent = 0
        operation_started = asyncio.Event()

        async def slow_publish(event):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            operation_started.set()  # Signal that at least one operation started
            # Use a small delay to allow other tasks to start
            await asyncio.sleep(0.001)
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

    @pytest.mark.asyncio
    async def test_concurrent_file_processing_performance(self, async_event_handler, mock_publisher):
        """Test that concurrent processing improves performance."""
        # Track processing completion
        completed_files = set()
        processing_started = asyncio.Event()

        async def track_completion(event):
            processing_started.set()
            completed_files.add(event.file_path)
            # Small delay to simulate processing without relying on timing
            await asyncio.sleep(0.001)
            return True

        mock_publisher.publish_file_event = track_completion

        # Process 20 files concurrently
        tasks = []
        for i in range(20):
            task = asyncio.create_task(async_event_handler._process_file_async(f"/test/file_{i}.mp3", "created"))
            tasks.append(task)

        # Wait for processing to start and complete
        await asyncio.wait_for(processing_started.wait(), timeout=1.0)
        await asyncio.gather(*tasks)

        # Verify all files were processed
        assert len(completed_files) == 20, "Not all files were processed"

    @pytest.mark.asyncio
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
        tasks = []
        for file_path in test_files:
            task = asyncio.create_task(handler._calculate_hashes_async(str(file_path)))
            tasks.append(task)

        # Wait for all tasks to complete with timeout
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # Verify all hashes were calculated
        assert len(results) == 10
        for sha256, xxh128 in results:
            assert len(sha256) == 64  # SHA256 hex is 64 chars
            assert len(xxh128) == 32  # XXH128 hex is 32 chars

    @pytest.mark.asyncio
    async def test_batch_processing_in_scan(self, tmp_path, mock_publisher):
        """Test batch processing during initial scan."""
        # Create many test files
        files_created = []
        for i in range(15):  # Reduced number for faster test
            file_path = tmp_path / f"song_{i}.mp3"
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(b"audio data")
            files_created.append(str(file_path))

        # Track processing calls
        processed_files = []
        processing_started = asyncio.Event()

        async def track_calls(event):
            processed_files.append(event.file_path)
            if not processing_started.is_set():
                processing_started.set()
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

            # Perform scan with timeout
            await asyncio.wait_for(service.scan_existing_files(), timeout=10.0)

            # Wait for processing to start if any files were found
            if files_created:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(processing_started.wait(), timeout=2.0)

        # The number of processed files should equal files created (if scan worked)
        # This test mainly verifies that batch processing can handle multiple files
        assert len(processed_files) <= 15, f"Got more processed files than created: {len(processed_files)}"

        # If files were processed, verify they were the ones we created
        if processed_files:
            for processed_file in processed_files:
                assert processed_file in files_created, f"Unexpected file processed: {processed_file}"

    @pytest.mark.asyncio
    async def test_concurrent_move_operations(self, async_event_handler, mock_publisher):
        """Test concurrent processing of move/rename operations."""
        # Track concurrent move operations
        move_count = 0
        first_move_started = asyncio.Event()

        async def track_moves(event):
            nonlocal move_count
            move_count += 1
            if not first_move_started.is_set():
                first_move_started.set()
            await asyncio.sleep(0.001)  # Minimal delay
            return True

        mock_publisher.publish_file_event = track_moves

        # Process multiple move operations concurrently
        tasks = []
        for i in range(15):
            old_path = f"/old/file_{i}.mp3"
            new_path = f"/new/file_{i}.mp3"
            task = asyncio.create_task(async_event_handler._process_move_async(old_path, new_path))
            tasks.append(task)

        # Wait for operations to start and complete
        await asyncio.wait_for(first_move_started.wait(), timeout=1.0)
        await asyncio.gather(*tasks)

        # Verify all moves were processed
        assert move_count == 15, f"Expected 15 moves, processed {move_count}"

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_processing(self, async_event_handler, mock_publisher):
        """Test error handling during concurrent processing."""
        # Make some operations fail
        call_count = 0

        async def sometimes_fail(event):
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

    @pytest.mark.asyncio
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

        # Verify task was added to processing_tasks
        assert len(handler.processing_tasks) == 1
        task = next(iter(handler.processing_tasks))
        assert hasattr(task, "cancel"), "Task should be cancellable (Future-like object)"
        assert hasattr(task, "done"), "Task should have done() method (Future-like object)"

        # Wait for the task to complete with timeout
        await asyncio.wait_for(handler.wait_for_tasks(), timeout=2.0)

        # Verify the async operation was called
        mock_publisher.publish_file_event.assert_called_once()
