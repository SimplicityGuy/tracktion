"""
Resource management and limits for async audio processing.

This module provides resource management, including semaphores for concurrent
analysis limits, memory monitoring, task queuing, and prioritization.
"""

import asyncio
import heapq
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels (lower value = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class QueuedTask:
    """Represents a task in the priority queue."""

    priority: TaskPriority
    timestamp: float
    task_id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    future: Optional[asyncio.Future] = None

    def __lt__(self, other: "QueuedTask") -> bool:
        """Compare tasks for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, use FIFO based on timestamp
        return self.timestamp < other.timestamp


@dataclass
class ResourceLimits:
    """Resource limits configuration."""

    max_concurrent_analyses: Optional[int] = None  # None = CPU count * 2
    max_memory_mb: int = 4096  # Maximum total memory usage
    max_memory_per_task_mb: int = 100  # Maximum memory per task
    max_queue_size: int = 1000  # Maximum pending tasks
    cpu_threshold_percent: float = 80.0  # CPU usage threshold
    memory_threshold_percent: float = 75.0  # Memory usage threshold
    task_timeout_seconds: int = 30  # Default task timeout


class AsyncResourceManager:
    """
    Manages resources for async audio processing with limits and monitoring.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize the resource manager.

        Args:
            limits: Resource limits configuration
        """
        self.limits = limits or ResourceLimits()

        # Determine CPU count and set concurrent limit
        self.cpu_count = os.cpu_count() or 4
        max_concurrent = self.limits.max_concurrent_analyses or (self.cpu_count * 2)

        # Semaphore for concurrent analysis limit
        self.analysis_semaphore = asyncio.Semaphore(max_concurrent)

        # Priority queue for pending tasks
        self.task_queue: List[QueuedTask] = []
        self.queue_lock = asyncio.Lock()

        # Active task tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_start_times: Dict[str, float] = {}
        self.task_memory_usage: Dict[str, float] = {}

        # Resource monitoring
        self.process = psutil.Process()
        self.system_monitor_task: Optional[asyncio.Task] = None
        self.monitoring = False

        # Statistics
        self.total_tasks_processed = 0
        self.total_tasks_queued = 0
        self.total_tasks_rejected = 0
        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0

        logger.info(
            f"AsyncResourceManager initialized: max_concurrent={max_concurrent}, "
            f"max_memory={self.limits.max_memory_mb}MB"
        )

    async def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """
        Start resource monitoring.

        Args:
            interval_seconds: Monitoring interval
        """
        if self.monitoring:
            return

        self.monitoring = True
        self.system_monitor_task = asyncio.create_task(self._monitor_resources(interval_seconds))
        logger.info("Resource monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    async def acquire_resources(
        self,
        task_id: str,
        estimated_memory_mb: Optional[float] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire resources for a task.

        Args:
            task_id: Unique task identifier
            estimated_memory_mb: Estimated memory requirement
            priority: Task priority
            timeout: Acquisition timeout

        Returns:
            True if resources acquired, False otherwise
        """
        estimated_memory = estimated_memory_mb or self.limits.max_memory_per_task_mb

        # Check if we can accommodate the task
        if not await self._check_resource_availability(estimated_memory):
            logger.warning(f"Insufficient resources for task {task_id}")
            self.total_tasks_rejected += 1
            return False

        # Try to acquire semaphore with timeout
        try:
            if timeout:
                await asyncio.wait_for(self.analysis_semaphore.acquire(), timeout=timeout)
            else:
                await self.analysis_semaphore.acquire()

            # Track task
            self.task_start_times[task_id] = time.time()
            self.task_memory_usage[task_id] = estimated_memory
            self.total_tasks_processed += 1

            logger.debug(f"Resources acquired for task {task_id}")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Resource acquisition timeout for task {task_id}")
            return False

    async def release_resources(self, task_id: str) -> None:
        """
        Release resources for a task.

        Args:
            task_id: Task identifier
        """
        # Release semaphore
        self.analysis_semaphore.release()

        # Clean up tracking
        if task_id in self.task_start_times:
            del self.task_start_times[task_id]
        if task_id in self.task_memory_usage:
            del self.task_memory_usage[task_id]
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

        logger.debug(f"Resources released for task {task_id}")

        # Process queued tasks
        await self._process_queue()

    async def queue_task(
        self,
        task_id: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> asyncio.Future:
        """
        Queue a task for execution when resources are available.

        Args:
            task_id: Task identifier
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority

        Returns:
            Future that will be resolved when task completes

        Raises:
            ValueError: If queue is full
        """
        async with self.queue_lock:
            if len(self.task_queue) >= self.limits.max_queue_size:
                raise ValueError(f"Task queue full (max {self.limits.max_queue_size})")

            # Create future for result
            future = asyncio.get_event_loop().create_future()

            # Create queued task
            task = QueuedTask(
                priority=priority,
                timestamp=time.time(),
                task_id=task_id,
                func=func,
                args=args,
                kwargs=kwargs or {},
                future=future,
            )

            # Add to priority queue
            heapq.heappush(self.task_queue, task)
            self.total_tasks_queued += 1

            logger.debug(f"Task {task_id} queued with priority {priority.name}, queue size: {len(self.task_queue)}")

        # Try to process queue immediately
        asyncio.create_task(self._process_queue())

        return future

    async def _process_queue(self) -> None:
        """Process queued tasks when resources are available."""
        async with self.queue_lock:
            while self.task_queue:
                # Check if we have available resources
                if self.analysis_semaphore._value <= 0:
                    break

                # Get highest priority task
                task = heapq.heappop(self.task_queue)

                # Check if we can accommodate the task
                estimated_memory = self.limits.max_memory_per_task_mb
                if not await self._check_resource_availability(estimated_memory):
                    # Put task back in queue
                    heapq.heappush(self.task_queue, task)
                    break

                # Execute task
                asyncio.create_task(self._execute_queued_task(task))

    async def _execute_queued_task(self, task: QueuedTask) -> None:
        """
        Execute a queued task.

        Args:
            task: QueuedTask to execute
        """
        try:
            # Acquire resources
            acquired = await self.acquire_resources(
                task.task_id,
                estimated_memory_mb=self.limits.max_memory_per_task_mb,
                priority=task.priority,
            )

            if not acquired:
                task.future.set_exception(RuntimeError(f"Failed to acquire resources for {task.task_id}"))
                return

            # Execute function with timeout
            try:
                result = await asyncio.wait_for(
                    self._run_task_async(task.func, *task.args, **task.kwargs),
                    timeout=self.limits.task_timeout_seconds,
                )
                task.future.set_result(result)
            except asyncio.TimeoutError:
                task.future.set_exception(
                    asyncio.TimeoutError(f"Task {task.task_id} timed out after {self.limits.task_timeout_seconds}s")
                )
            except Exception as e:
                task.future.set_exception(e)

        finally:
            # Release resources
            await self.release_resources(task.task_id)

    async def _run_task_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a task asynchronously.

        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run in executor if not async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _check_resource_availability(self, required_memory_mb: float) -> bool:
        """
        Check if resources are available for a task.

        Args:
            required_memory_mb: Required memory in MB

        Returns:
            True if resources available
        """
        # Check memory
        memory_info = self.process.memory_info()
        current_memory_mb = memory_info.rss / (1024 * 1024)

        if current_memory_mb + required_memory_mb > self.limits.max_memory_mb:
            logger.warning(
                f"Memory limit would be exceeded: current={current_memory_mb:.1f}MB, "
                f"required={required_memory_mb:.1f}MB, limit={self.limits.max_memory_mb}MB"
            )
            return False

        # Check system memory
        system_memory = psutil.virtual_memory()
        if system_memory.percent > self.limits.memory_threshold_percent:
            logger.warning(f"System memory usage too high: {system_memory.percent:.1f}%")
            return False

        # Check CPU
        cpu_percent = self.process.cpu_percent()
        if cpu_percent > self.limits.cpu_threshold_percent:
            logger.warning(f"CPU usage too high: {cpu_percent:.1f}%")
            return False

        return True

    async def _monitor_resources(self, interval: float) -> None:
        """
        Monitor system resources periodically.

        Args:
            interval: Monitoring interval in seconds
        """
        while self.monitoring:
            try:
                # Get current resource usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = self.process.cpu_percent()

                # Update peaks
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)

                # Check for resource pressure
                system_memory = psutil.virtual_memory()
                if system_memory.percent > self.limits.memory_threshold_percent:
                    logger.warning(f"High system memory usage: {system_memory.percent:.1f}%")

                if cpu_percent > self.limits.cpu_threshold_percent:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

                # Log periodic stats
                logger.debug(
                    f"Resource usage - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, "
                    f"Active tasks: {len(self.active_tasks)}, Queued: {len(self.task_queue)}"
                )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                await asyncio.sleep(interval)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get resource manager statistics.

        Returns:
            Dictionary with current statistics
        """
        memory_info = self.process.memory_info()
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "total_processed": self.total_tasks_processed,
            "total_queued": self.total_tasks_queued,
            "total_rejected": self.total_tasks_rejected,
            "current_memory_mb": memory_info.rss / (1024 * 1024),
            "peak_memory_mb": self.peak_memory_mb,
            "current_cpu_percent": self.process.cpu_percent(),
            "peak_cpu_percent": self.peak_cpu_percent,
            "semaphore_available": self.analysis_semaphore._value,
        }

    async def wait_for_capacity(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for capacity to become available.

        Args:
            timeout: Maximum wait time

        Returns:
            True if capacity available, False if timeout
        """
        try:
            await asyncio.wait_for(self.analysis_semaphore.acquire(), timeout=timeout)
            self.analysis_semaphore.release()
            return True
        except asyncio.TimeoutError:
            return False

    async def shutdown(self) -> None:
        """Shutdown the resource manager."""
        logger.info("Shutting down AsyncResourceManager")

        # Stop monitoring
        await self.stop_monitoring()

        # Cancel queued tasks
        async with self.queue_lock:
            for task in self.task_queue:
                if task.future and not task.future.done():
                    task.future.cancel()
            self.task_queue.clear()

        # Cancel active tasks
        for task_id, active_task in self.active_tasks.items():
            if hasattr(active_task, "done") and not active_task.done():
                active_task.cancel()
                logger.debug(f"Cancelled active task {task_id}")

        logger.info("AsyncResourceManager shutdown complete")
