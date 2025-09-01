"""
Async audio processing infrastructure for parallel analysis.

This module provides the core infrastructure for processing multiple audio files
simultaneously using asyncio and ThreadPoolExecutor for CPU-bound operations.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for audio analysis tasks."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AudioTaskConfig:
    """Configuration for audio processing tasks."""

    # Thread pool settings
    min_threads: int = 2
    max_threads_multiplier: float = 2.0  # Multiplier of CPU count
    max_threads_absolute: int = 32  # Absolute maximum threads

    # Resource limits
    max_concurrent_analyses: int | None = None  # None = CPU count * 2
    max_memory_per_file_mb: int = 100
    task_timeout_seconds: int = 30

    # Queue settings
    max_queue_size: int = 1000
    priority_queue_enabled: bool = True

    # Performance tuning
    cpu_affinity_enabled: bool = False
    dynamic_pool_sizing: bool = True
    cpu_utilization_target: float = 0.7  # Target 70% CPU utilization


class AsyncAudioProcessor:
    """
    Core async audio processing infrastructure.

    Manages thread pools, resource limits, and task scheduling for
    parallel audio analysis operations.
    """

    def __init__(self, config: AudioTaskConfig | None = None):
        """
        Initialize the async audio processor.

        Args:
            config: Configuration for the processor
        """
        self.config = config or AudioTaskConfig()

        # Determine CPU count and optimal thread pool size
        self.cpu_count = os.cpu_count() or 4
        self.optimal_thread_count = self._calculate_optimal_threads()

        # Initialize thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=self.optimal_thread_count,
            thread_name_prefix="audio_analysis",
        )

        # Initialize semaphore for concurrent analysis limit
        max_concurrent = self.config.max_concurrent_analyses or (self.cpu_count * 2)
        self.analysis_semaphore = asyncio.Semaphore(max_concurrent)

        # Task tracking
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.pending_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.completed_count = 0
        self.failed_count = 0

        # Resource monitoring
        self.process = psutil.Process()
        self._last_cpu_check = 0.0
        self._last_memory_check = 0.0

        logger.info(
            f"AsyncAudioProcessor initialized with {self.optimal_thread_count} threads, "
            f"max concurrent analyses: {max_concurrent}"
        )

    def _calculate_optimal_threads(self) -> int:
        """
        Calculate optimal number of threads based on CPU cores and config.

        Returns:
            Optimal thread count
        """
        # Base calculation: CPU count * multiplier
        optimal = int(self.cpu_count * self.config.max_threads_multiplier)

        # Apply bounds
        optimal = max(self.config.min_threads, optimal)
        optimal = min(self.config.max_threads_absolute, optimal)

        # For CPU-bound audio processing, slightly over-provision
        # to account for I/O waits during file loading
        if optimal < self.cpu_count * 1.5:
            optimal = min(int(self.cpu_count * 1.5), self.config.max_threads_absolute)

        return optimal

    async def process_audio_async(
        self,
        audio_file: str,
        processing_func: Callable,
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Process an audio file asynchronously with resource management.

        Args:
            audio_file: Path to the audio file
            processing_func: Synchronous processing function to run
            *args: Additional arguments for processing_func
            priority: Task priority level
            task_id: Optional unique identifier for the task
            **kwargs: Additional keyword arguments for processing_func

        Returns:
            Result from the processing function

        Raises:
            asyncio.TimeoutError: If processing exceeds timeout
            MemoryError: If memory limit is exceeded
        """
        task_id = task_id or f"audio_{audio_file}_{id(processing_func)}"

        # Check memory before starting
        if not await self._check_memory_available():
            raise MemoryError(f"Insufficient memory to process {audio_file}")

        # Acquire semaphore for rate limiting
        async with self.analysis_semaphore:
            logger.debug(f"Starting async processing for {audio_file} (priority: {priority.name})")

            try:
                # Run CPU-bound operation in thread pool with timeout
                result = await asyncio.wait_for(
                    self._run_in_executor(processing_func, audio_file, *args, **kwargs),
                    timeout=self.config.task_timeout_seconds,
                )

                self.completed_count += 1
                logger.debug(f"Completed processing for {audio_file}")
                return result

            except TimeoutError:
                self.failed_count += 1
                logger.error(f"Processing timeout for {audio_file} after {self.config.task_timeout_seconds}s")
                raise

            except Exception as e:
                self.failed_count += 1
                logger.error(f"Processing failed for {audio_file}: {e!s}")
                raise

            finally:
                # Clean up task tracking
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

    async def _run_in_executor(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Run a function in the thread pool executor.

        Args:
            func: Function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

    async def _check_memory_available(self) -> bool:
        """
        Check if sufficient memory is available for processing.

        Returns:
            True if memory is available, False otherwise
        """
        # Get available system memory
        available_mb = psutil.virtual_memory().available / (1024 * 1024)

        # Check if we have enough headroom
        if available_mb < self.config.max_memory_per_file_mb * 2:
            logger.warning(f"Low memory: {available_mb:.1f}MB available, need {self.config.max_memory_per_file_mb}MB")
            return False

        return True

    async def adjust_thread_pool_size(self) -> None:
        """
        Dynamically adjust thread pool size based on CPU utilization.

        This method monitors CPU usage and adjusts the thread pool size
        to maintain optimal performance.
        """
        if not self.config.dynamic_pool_sizing:
            return

        current_cpu = self.process.cpu_percent(interval=1.0) / 100.0

        if current_cpu < self.config.cpu_utilization_target - 0.1:
            # Under-utilized, can increase threads if not at max
            new_size = min(self.optimal_thread_count + 1, self.config.max_threads_absolute)
        elif current_cpu > self.config.cpu_utilization_target + 0.1:
            # Over-utilized, reduce threads if not at min
            new_size = max(self.optimal_thread_count - 1, self.config.min_threads)
        else:
            # Within target range
            return

        if new_size != self.optimal_thread_count:
            logger.info(f"Adjusting thread pool from {self.optimal_thread_count} to {new_size} threads")
            self.optimal_thread_count = new_size
            # Note: ThreadPoolExecutor doesn't support dynamic resizing,
            # so we'd need to create a new executor in a production system
            # For now, this is a placeholder for the logic

    async def batch_process_audio(
        self,
        audio_files: list[str],
        processing_func: Callable[..., Any],
        max_batch_size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process multiple audio files in parallel batches.

        Args:
            audio_files: List of audio file paths
            processing_func: Processing function to apply
            max_batch_size: Maximum files to process simultaneously
            **kwargs: Additional arguments for processing_func

        Returns:
            Dictionary mapping file paths to results or errors
        """
        max_batch = max_batch_size or self.cpu_count * 2
        results = {}

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(audio_files), max_batch):
            batch = audio_files[i : i + max_batch]

            # Create tasks for this batch
            tasks = []
            for audio_file in batch:
                task = asyncio.create_task(self.process_audio_async(audio_file, processing_func, **kwargs))
                tasks.append((audio_file, task))

            # Wait for batch to complete
            for audio_file, task in tasks:
                try:
                    result = await task
                    results[audio_file] = {"success": True, "result": result}
                except Exception as e:
                    results[audio_file] = {"success": False, "error": str(e)}

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Get current processor statistics.

        Returns:
            Dictionary with current stats
        """
        memory_info = self.process.memory_info()
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_count,
            "failed_tasks": self.failed_count,
            "thread_count": self.optimal_thread_count,
            "cpu_count": self.cpu_count,
            "memory_mb": memory_info.rss / (1024 * 1024),
            "cpu_percent": self.process.cpu_percent(),
        }

    async def shutdown(self) -> None:
        """Shutdown the processor and clean up resources."""
        logger.info("Shutting down AsyncAudioProcessor")

        # Cancel active tasks and wait for them to complete cancellation
        for task_id, task in list(self.active_tasks.items()):
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelling task {task_id}")
                with contextlib.suppress(asyncio.CancelledError):
                    # Expected when task is cancelled
                    await task
                logger.debug(f"Cancelled task {task_id}")

        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        logger.info("AsyncAudioProcessor shutdown complete")


class AudioAnalysisScheduler:
    """
    Task scheduler for audio analysis with priority queue support.
    """

    def __init__(self, processor: AsyncAudioProcessor):
        """
        Initialize the scheduler.

        Args:
            processor: AsyncAudioProcessor instance
        """
        self.processor = processor
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running = False
        self.scheduler_task: asyncio.Task | None = None

    async def schedule_analysis(
        self,
        audio_file: str,
        processing_func: Callable[..., Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Schedule an audio analysis task.

        Args:
            audio_file: Path to audio file
            processing_func: Function to process the audio
            priority: Task priority
            callback: Optional callback for completion
            **kwargs: Additional arguments for processing_func
        """
        # Priority queue expects (priority_value, item) tuples
        # Lower values have higher priority, so we negate the enum value
        priority_value = -priority.value
        task_item = (audio_file, processing_func, callback, kwargs)

        await self.priority_queue.put((priority_value, task_item))
        logger.debug(f"Scheduled {audio_file} with priority {priority.name}")

    async def start(self) -> None:
        """Start the scheduler."""
        if self.running:
            return

        self.running = True
        self.scheduler_task = asyncio.create_task(self._process_queue())
        logger.info("AudioAnalysisScheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.scheduler_task
        logger.info("AudioAnalysisScheduler stopped")

    async def _process_queue(self) -> None:
        """Process tasks from the priority queue."""
        while self.running:
            try:
                # Get next task from priority queue
                priority_value, task_item = await asyncio.wait_for(self.priority_queue.get(), timeout=1.0)

                audio_file, processing_func, callback, kwargs = task_item

                # Process the audio file
                try:
                    result = await self.processor.process_audio_async(audio_file, processing_func, **kwargs)

                    # Call callback if provided
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(audio_file, result, None)
                        else:
                            callback(audio_file, result, None)

                except Exception as e:
                    logger.error(f"Task failed for {audio_file}: {e!s}")
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(audio_file, None, e)
                        else:
                            callback(audio_file, None, e)

            except TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {e!s}")
