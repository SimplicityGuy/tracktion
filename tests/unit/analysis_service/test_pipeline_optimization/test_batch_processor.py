"""Unit tests for batch processor functionality."""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from services.analysis_service.src.batch_processor import BatchConfig, BatchProcessor


class TestBatchConfig(unittest.TestCase):
    """Test BatchConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BatchConfig()
        self.assertEqual(config.min_workers, 1)
        self.assertEqual(config.max_workers, 10)
        self.assertEqual(config.default_workers, 4)
        self.assertEqual(config.batch_size, 10)
        self.assertEqual(config.max_batch_wait_seconds, 5.0)
        self.assertEqual(config.max_memory_per_worker_mb, 500.0)
        self.assertEqual(config.max_queue_size, 1000)

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        env_vars = {
            "BATCH_MIN_WORKERS": "2",
            "BATCH_MAX_WORKERS": "8",
            "BATCH_DEFAULT_WORKERS": "3",
            "BATCH_SIZE": "20",
            "BATCH_MAX_WAIT_SECONDS": "10.0",
            "BATCH_MAX_MEMORY_PER_WORKER_MB": "1024.0",
            "BATCH_MAX_QUEUE_SIZE": "500",
        }

        with patch.dict(os.environ, env_vars):
            config = BatchConfig.from_env()
            self.assertEqual(config.min_workers, 2)
            self.assertEqual(config.max_workers, 8)
            self.assertEqual(config.default_workers, 3)
            self.assertEqual(config.batch_size, 20)
            self.assertEqual(config.max_batch_wait_seconds, 10.0)
            self.assertEqual(config.max_memory_per_worker_mb, 1024.0)
            self.assertEqual(config.max_queue_size, 500)


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a mock processing function
        self.process_func = MagicMock(return_value={"status": "success"})
        self.config = BatchConfig(
            min_workers=1,
            max_workers=4,
            default_workers=2,
            batch_size=3,
        )
        self.processor = BatchProcessor(
            process_func=self.process_func,
            config=self.config,
        )

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.processor.shutdown(wait=True)

    def test_initialization(self) -> None:
        """Test batch processor initialization."""
        self.assertEqual(self.processor.worker_count, 2)
        self.assertEqual(self.processor.config.batch_size, 3)
        self.assertEqual(self.processor.processed_count, 0)
        self.assertEqual(self.processor.error_count, 0)

    def test_worker_count_validation(self) -> None:
        """Test that worker count is validated on initialization."""
        # Test with too many workers
        config = BatchConfig(min_workers=1, max_workers=2, default_workers=10)
        processor = BatchProcessor(self.process_func, config)
        self.assertEqual(processor.worker_count, 2)  # Clamped to max
        processor.shutdown(wait=False)

        # Test with too few workers
        config = BatchConfig(min_workers=5, max_workers=10, default_workers=1)
        processor = BatchProcessor(self.process_func, config)
        self.assertEqual(processor.worker_count, 5)  # Clamped to min
        processor.shutdown(wait=False)

    def test_add_to_batch(self) -> None:
        """Test adding items to batch."""
        # Add items below batch size
        self.processor.add_to_batch("file1.mp3", "rec1", "corr1")
        self.processor.add_to_batch("file2.mp3", "rec2", "corr2")

        # Batch should not be processed yet
        self.assertEqual(len(self.processor.current_batch), 2)
        self.process_func.assert_not_called()

        # Add one more to trigger batch processing
        self.processor.add_to_batch("file3.mp3", "rec3", "corr3")

        # Wait a bit for processing
        time.sleep(0.1)

        # Batch should be processed
        self.assertEqual(len(self.processor.current_batch), 0)
        self.assertEqual(self.process_func.call_count, 3)

    def test_process_single_file(self) -> None:
        """Test processing a single file."""
        result = self.processor._process_single_file("test.mp3", "rec123", "corr123")

        self.process_func.assert_called_once_with("test.mp3", "rec123")
        self.assertEqual(result, {"status": "success"})
        self.assertEqual(self.processor.processed_count, 1)
        self.assertEqual(self.processor.error_count, 0)

    def test_process_single_file_error(self) -> None:
        """Test error handling in single file processing."""
        self.process_func.side_effect = Exception("Processing error")

        result = self.processor._process_single_file("test.mp3", "rec123", "corr123")

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Processing error")
        self.assertEqual(self.processor.processed_count, 0)
        self.assertEqual(self.processor.error_count, 1)

    def test_process_batch(self) -> None:
        """Test processing a batch of files."""
        files = [
            ("file1.mp3", "rec1"),
            ("file2.mp3", "rec2"),
            ("file3.mp3", "rec3"),
        ]
        correlation_ids = ["corr1", "corr2", "corr3"]

        results = self.processor.process_batch(files, correlation_ids)

        self.assertEqual(len(results), 3)
        self.assertEqual(self.process_func.call_count, 3)
        for result in results:
            self.assertEqual(result, {"status": "success"})

    def test_process_batch_with_errors(self) -> None:
        """Test batch processing with some errors."""
        # Make second call fail
        self.process_func.side_effect = [
            {"status": "success"},
            Exception("Processing error"),
            {"status": "success"},
        ]

        files = [
            ("file1.mp3", "rec1"),
            ("file2.mp3", "rec2"),
            ("file3.mp3", "rec3"),
        ]

        results = self.processor.process_batch(files)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], {"status": "success"})
        self.assertIn("error", results[1])
        self.assertEqual(results[2], {"status": "success"})

    def test_flush_batch(self) -> None:
        """Test flushing partial batch."""
        # Add items below batch size
        self.processor.add_to_batch("file1.mp3", "rec1")
        self.processor.add_to_batch("file2.mp3", "rec2")

        # Flush the batch
        futures = self.processor.flush_batch()

        # Wait for futures to complete
        for future in futures:
            future.result()

        self.assertEqual(len(self.processor.current_batch), 0)
        self.assertEqual(self.process_func.call_count, 2)

    def test_adjust_worker_count(self) -> None:
        """Test adjusting worker count."""
        # Initial count
        self.assertEqual(self.processor.worker_count, 2)

        # Increase workers
        self.processor.adjust_worker_count(3)
        self.assertEqual(self.processor.worker_count, 3)

        # Try to exceed max
        self.processor.adjust_worker_count(10)
        self.assertEqual(self.processor.worker_count, 4)  # Clamped to max

        # Try to go below min
        self.processor.adjust_worker_count(0)
        self.assertEqual(self.processor.worker_count, 1)  # Clamped to min

    def test_get_statistics(self) -> None:
        """Test getting processing statistics."""
        # Process some files
        self.processor._process_single_file("file1.mp3", "rec1")
        self.processor._process_single_file("file2.mp3", "rec2")

        # Cause an error
        self.process_func.side_effect = Exception("Error")
        self.processor._process_single_file("file3.mp3", "rec3")

        stats = self.processor.get_statistics()

        self.assertEqual(stats["processed_count"], 2)
        self.assertEqual(stats["error_count"], 1)
        self.assertEqual(stats["total_count"], 3)
        self.assertAlmostEqual(stats["success_rate"], 66.67, places=1)
        self.assertEqual(stats["worker_count"], 2)

    def test_shutdown(self) -> None:
        """Test processor shutdown."""
        # Add some items to batch
        self.processor.add_to_batch("file1.mp3", "rec1")
        self.processor.add_to_batch("file2.mp3", "rec2")

        # Shutdown should process remaining items
        self.processor.shutdown(wait=True)

        # Batch should be processed
        self.assertEqual(self.process_func.call_count, 2)

    def test_context_manager(self) -> None:
        """Test using processor as context manager."""
        process_func = MagicMock(return_value={"status": "success"})

        with BatchProcessor(process_func, self.config) as processor:
            processor.add_to_batch("file1.mp3", "rec1")
            processor.add_to_batch("file2.mp3", "rec2")

        # Should be shut down after context exit
        # Remaining batch should be processed
        self.assertEqual(process_func.call_count, 2)


if __name__ == "__main__":
    unittest.main()
