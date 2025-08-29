"""Performance benchmarks for async file watcher."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.async_file_watcher import AsyncFileEventHandler, AsyncFileWatcherService
from src.async_message_publisher import AsyncMessagePublisher
from src.async_metadata_extractor import AsyncMetadataExtractor


class TestPerformanceBenchmarks:
    """Performance benchmarks for async file operations."""

    async def test_1000_concurrent_file_events(self):
        """Test handling 1000+ concurrent file events."""
        # Create mock publisher that tracks timing
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        event_times = []

        async def track_publish(*args, **kwargs):
            event_times.append(time.time())
            await asyncio.sleep(0.001)  # Simulate network latency
            return True

        mock_publisher.publish_file_event = track_publish

        # Create event handler with reasonable concurrency limit
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(100)  # Max 100 concurrent
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="perf_test",
            loop=loop,
            semaphore=semaphore,
        )

        # Generate 1000 file events
        start_time = time.time()
        tasks = []

        for i in range(1000):
            task = asyncio.create_task(handler._process_file_async(f"/test/file_{i:04d}.mp3", "created"))
            tasks.append(task)

        # Wait for all events to process
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Verify performance
        assert len(event_times) == 1000, "All events should be processed"
        assert total_time < 30, f"Processing 1000 events took {total_time:.2f}s (should be < 30s)"

        # Calculate throughput
        throughput = 1000 / total_time
        print("\nPerformance Results:")
        print("  Total events: 1000")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} events/second")

        # Check 95th percentile response time
        if len(event_times) > 1:
            event_times.sort()
            response_times = []
            for i in range(1, len(event_times)):
                response_times.append(event_times[i] - event_times[i - 1])

            response_times.sort()
            p95_index = int(len(response_times) * 0.95)
            p95_response = response_times[p95_index] * 1000  # Convert to ms

            print(f"  95th percentile response: {p95_response:.1f}ms")
            assert p95_response < 100, f"95th percentile {p95_response:.1f}ms exceeds 100ms target"

    async def test_bulk_scan_performance(self):
        """Test performance of bulk file scanning."""
        # Create temporary directory with many files
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create 500 test files
            test_files = []
            for i in range(500):
                file_path = test_dir / f"song_{i:04d}.mp3"
                file_path.write_bytes(b"test audio data")
                test_files.append(file_path)

            # Mock publisher to track events
            mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
            publish_count = 0

            async def count_publishes(*args, **kwargs):
                nonlocal publish_count
                publish_count += 1
                return True

            mock_publisher.publish_file_event = count_publishes
            mock_publisher.connect = AsyncMock()

            # Create service and perform scan
            with patch.dict("os.environ", {"DATA_DIR": str(test_dir)}):
                with patch("src.async_file_watcher.AsyncMessagePublisher", return_value=mock_publisher):
                    service = AsyncFileWatcherService()
                    service.publisher = mock_publisher

                    start_time = time.time()
                    await service.scan_existing_files()
                    scan_time = time.time() - start_time

            # Verify performance
            assert publish_count == 500, f"Expected 500 events, got {publish_count}"
            assert scan_time < 10, f"Scanning 500 files took {scan_time:.2f}s (should be < 10s)"

            scan_rate = 500 / scan_time
            print("\nBulk Scan Performance:")
            print("  Files scanned: 500")
            print(f"  Scan time: {scan_time:.2f}s")
            print(f"  Scan rate: {scan_rate:.1f} files/second")

    async def test_metadata_extraction_performance(self):
        """Test performance of concurrent metadata extraction."""
        # Create test files
        with tempfile.TemporaryDirectory() as tmpdir:
            test_files = []
            for i in range(100):
                file_path = Path(tmpdir) / f"audio_{i:03d}.wav"
                file_path.write_bytes(b"RIFF" + b"test" * 1000)  # Fake WAV header
                test_files.append(str(file_path))

            # Create metadata extractor
            extractor = AsyncMetadataExtractor(max_workers=4)

            try:
                # Extract metadata for all files
                start_time = time.time()
                results = await extractor.extract_batch(test_files)
                extraction_time = time.time() - start_time

                # Verify performance
                assert len(results) == 100
                assert extraction_time < 5, f"Extracting 100 files took {extraction_time:.2f}s"

                extraction_rate = 100 / extraction_time
                print("\nMetadata Extraction Performance:")
                print("  Files processed: 100")
                print(f"  Extraction time: {extraction_time:.2f}s")
                print(f"  Extraction rate: {extraction_rate:.1f} files/second")

                # Test cache performance
                cache_start = time.time()
                await extractor.extract_batch(test_files[:50])
                cache_time = time.time() - cache_start

                assert cache_time < 0.1, f"Cached extraction took {cache_time:.2f}s"
                print(f"  Cache hit time (50 files): {cache_time * 1000:.1f}ms")

            finally:
                extractor.shutdown()

    async def test_resource_usage_optimization(self):
        """Test resource usage with concurrent operations."""
        # Track memory and CPU usage indirectly through timing
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        mock_publisher.publish_file_event = AsyncMock(return_value=True)

        # Test different concurrency limits
        concurrency_tests = [10, 50, 100, 200]
        results = []

        for max_concurrent in concurrency_tests:
            loop = asyncio.get_event_loop()
            semaphore = asyncio.Semaphore(max_concurrent)
            handler = AsyncFileEventHandler(
                publisher=mock_publisher,
                instance_id="resource_test",
                loop=loop,
                semaphore=semaphore,
            )

            # Process 200 events
            start_time = time.time()
            tasks = []
            for i in range(200):
                task = asyncio.create_task(handler._process_file_async(f"/test/file_{i}.mp3", "created"))
                tasks.append(task)

            await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            results.append({"concurrency": max_concurrent, "time": elapsed, "throughput": 200 / elapsed})

        print("\nConcurrency Optimization Results:")
        for result in results:
            print(
                f"  Concurrency {result['concurrency']:3d}: "
                f"{result['time']:.2f}s, "
                f"{result['throughput']:.1f} events/sec"
            )

        # Verify that higher concurrency improves performance (to a point)
        assert results[1]["throughput"] > results[0]["throughput"], "Higher concurrency should improve throughput"

    async def test_response_time_percentiles(self):
        """Test response time percentiles under load."""
        mock_publisher = AsyncMock(spec=AsyncMessagePublisher)
        response_times = []

        async def measure_response(*args, **kwargs):
            start = time.time()
            await asyncio.sleep(0.005)  # Simulate processing
            response_times.append((time.time() - start) * 1000)  # ms
            return True

        mock_publisher.publish_file_event = measure_response

        # Create handler
        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(50)
        handler = AsyncFileEventHandler(
            publisher=mock_publisher,
            instance_id="percentile_test",
            loop=loop,
            semaphore=semaphore,
        )

        # Generate load
        tasks = []
        for i in range(500):
            task = asyncio.create_task(handler._process_file_async(f"/test/file_{i}.mp3", "created"))
            tasks.append(task)

            # Stagger the requests slightly
            if i % 10 == 0:
                await asyncio.sleep(0.01)

        await asyncio.gather(*tasks)

        # Calculate percentiles
        response_times.sort()
        p50 = response_times[int(len(response_times) * 0.50)]
        p75 = response_times[int(len(response_times) * 0.75)]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]

        print("\nResponse Time Percentiles:")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P75: {p75:.1f}ms")
        print(f"  P95: {p95:.1f}ms (target < 100ms)")
        print(f"  P99: {p99:.1f}ms")

        # Verify P95 meets target
        assert p95 < 100, f"P95 response time {p95:.1f}ms exceeds 100ms target"
