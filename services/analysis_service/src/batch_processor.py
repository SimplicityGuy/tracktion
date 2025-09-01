"""Batch processing with configurable concurrency for the analysis service."""

import logging
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    # Worker pool configuration
    min_workers: int = 1
    max_workers: int = 10
    default_workers: int = 4

    # Batch sizes
    batch_size: int = 10
    max_batch_wait_seconds: float = 5.0

    # Resource limits
    max_memory_per_worker_mb: float = 500.0
    max_queue_size: int = 1000

    @classmethod
    def from_env(cls) -> "BatchConfig":
        """Create configuration from environment variables."""
        return cls(
            min_workers=int(os.getenv("BATCH_MIN_WORKERS", "1")),
            max_workers=int(os.getenv("BATCH_MAX_WORKERS", "10")),
            default_workers=int(os.getenv("BATCH_DEFAULT_WORKERS", "4")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_batch_wait_seconds=float(os.getenv("BATCH_MAX_WAIT_SECONDS", "5.0")),
            max_memory_per_worker_mb=float(os.getenv("BATCH_MAX_MEMORY_PER_WORKER_MB", "500.0")),
            max_queue_size=int(os.getenv("BATCH_MAX_QUEUE_SIZE", "1000")),
        )


class BatchProcessor:
    """Process audio files in batches with configurable concurrency."""

    def __init__(
        self,
        process_func: Callable[[str, str, str | None], dict[str, Any]],
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            process_func: Function to process individual files (file_path, recording_id, correlation_id) -> results
            config: Batch processing configuration
        """
        self.process_func = process_func
        self.config = config or BatchConfig.from_env()

        # Validate and adjust worker count
        worker_count = self.config.default_workers
        worker_count = max(self.config.min_workers, min(worker_count, self.config.max_workers))

        self.executor = ThreadPoolExecutor(max_workers=worker_count)
        self.worker_count = worker_count

        # Batch management
        self.current_batch: list[tuple[str, str, str | None]] = []
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()

        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.stats_lock = threading.Lock()

        # Processing queue for async batch accumulation
        self.processing_queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.stop_event = threading.Event()

        logger.info(f"Batch processor initialized with {worker_count} workers, batch size {self.config.batch_size}")

    def add_to_batch(
        self,
        file_path: str,
        recording_id: str,
        correlation_id: str | None = None,
    ) -> None:
        """Add a file to the current batch.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage
            correlation_id: Correlation ID for tracking
        """
        with self.batch_lock:
            self.current_batch.append((file_path, recording_id, correlation_id))

            # Process batch if it reaches the configured size
            if len(self.current_batch) >= self.config.batch_size:
                self._process_current_batch()

    def _process_current_batch(self) -> list[Future]:
        """Process the current batch of files.

        Returns:
            List of futures for the batch processing tasks
        """
        with self.batch_lock:
            if not self.current_batch:
                return []

            # Move current batch to processing
            batch_to_process = self.current_batch
            self.current_batch = []

        logger.info(f"Processing batch of {len(batch_to_process)} files")

        # Submit all files in the batch for parallel processing
        futures = []
        for file_path, recording_id, correlation_id in batch_to_process:
            future = self.executor.submit(
                self._process_single_file,
                file_path,
                recording_id,
                correlation_id,
            )
            futures.append(future)

        return futures

    def _process_single_file(
        self,
        file_path: str,
        recording_id: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a single file with error handling.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage
            correlation_id: Correlation ID for tracking

        Returns:
            Processing results or error information
        """
        try:
            logger.debug(
                f"Processing file: {file_path}",
                extra={"correlation_id": correlation_id},
            )

            # Call the actual processing function
            results = self.process_func(file_path, recording_id, correlation_id)

            # Update statistics
            with self.stats_lock:
                self.processed_count += 1

            logger.info(
                f"Successfully processed: {file_path}",
                extra={
                    "correlation_id": correlation_id,
                    "recording_id": recording_id,
                },
            )

            return results

        except Exception as e:
            logger.error(
                f"Error processing file {file_path}: {e}",
                extra={"correlation_id": correlation_id},
            )

            # Update error statistics
            with self.stats_lock:
                self.error_count += 1

            return {
                "error": str(e),
                "file_path": file_path,
                "recording_id": recording_id,
            }

    def process_batch(
        self,
        files: list[tuple[str, str]],
        correlation_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Process a batch of files and wait for completion.

        Args:
            files: List of (file_path, recording_id) tuples
            correlation_ids: Optional list of correlation IDs

        Returns:
            List of processing results
        """
        if correlation_ids is None:
            correlation_ids = [None] * len(files)  # type: ignore[list-item]

        # Submit all files for processing
        futures = []
        for (file_path, recording_id), correlation_id in zip(files, correlation_ids, strict=False):
            future = self.executor.submit(
                self._process_single_file,
                file_path,
                recording_id,
                correlation_id,
            )
            futures.append(future)

        # Wait for all processing to complete and collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 1 minute timeout per file
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append({"error": str(e)})

        return results

    def flush_batch(self) -> list[Future]:
        """Force processing of any pending items in the current batch.

        Returns:
            List of futures for the batch processing tasks
        """
        return self._process_current_batch()

    def adjust_worker_count(self, new_count: int, metrics_collector: Any | None = None) -> None:
        """Adjust the number of worker threads.

        Args:
            new_count: New number of worker threads
            metrics_collector: Optional metrics collector to update worker count
        """
        # Clamp to configured limits
        new_count = max(self.config.min_workers, min(new_count, self.config.max_workers))

        if new_count == self.worker_count:
            return

        logger.info(f"Adjusting worker count from {self.worker_count} to {new_count}")

        # Create new executor with adjusted worker count
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=new_count)
        self.worker_count = new_count

        # Update metrics if collector provided
        if metrics_collector:
            metrics_collector.update_worker_pool_size(new_count)

        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        with self.stats_lock:
            total = self.processed_count + self.error_count
            success_rate = (self.processed_count / total * 100) if total > 0 else 0

            return {
                "processed_count": self.processed_count,
                "error_count": self.error_count,
                "total_count": total,
                "success_rate": success_rate,
                "worker_count": self.worker_count,
                "pending_batch_size": len(self.current_batch),
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the batch processor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down batch processor")

        # Stop accepting new items
        self.stop_event.set()

        # Process any remaining items in the batch
        if self.current_batch:
            self._process_current_batch()

        # Shutdown the executor
        self.executor.shutdown(wait=wait)

        logger.info(f"Batch processor shutdown complete. Processed: {self.processed_count}, Errors: {self.error_count}")

    def __enter__(self) -> "BatchProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown(wait=True)
