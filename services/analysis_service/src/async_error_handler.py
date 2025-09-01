"""
Error handling and recovery for async audio processing.

This module provides timeout management, retry logic, fallback strategies,
and resource cleanup for robust audio analysis.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Constants for error handling
MAX_ERROR_HISTORY = 1000


class ErrorType(Enum):
    """Types of errors in audio processing."""

    TIMEOUT = "timeout"
    MEMORY = "memory"
    CORRUPTED_FILE = "corrupted_file"
    ANALYSIS_FAILURE = "analysis_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context for error handling."""

    error_type: ErrorType
    error_message: str
    task_id: str
    audio_file: str | None = None
    retry_count: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    retry_on_timeout: bool = True
    retry_on_memory: bool = False
    retry_on_corruption: bool = False


class AsyncErrorHandler:
    """
    Handles errors and implements recovery strategies for async audio processing.
    """

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        enable_fallback: bool = True,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize error handler.

        Args:
            retry_policy: Retry policy configuration
            enable_fallback: Enable fallback strategies
            enable_circuit_breaker: Enable circuit breaker pattern
        """
        self.retry_policy = retry_policy or RetryPolicy()
        self.enable_fallback = enable_fallback
        self.enable_circuit_breaker = enable_circuit_breaker

        # Error tracking
        self.error_history: list[ErrorContext] = []
        self.error_counts: dict[ErrorType, int] = dict.fromkeys(ErrorType, 0)

        # Circuit breaker state
        self.circuit_open = False
        self.circuit_failures = 0
        self.circuit_threshold = 5
        self.circuit_timeout = 60.0  # seconds
        self.circuit_open_time = 0.0

        logger.info("AsyncErrorHandler initialized")

    async def handle_with_retry(
        self,
        func: Callable,
        *args: Any,
        task_id: str,
        audio_file: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Task identifier
            audio_file: Audio file being processed
            timeout: Execution timeout
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        if self._is_circuit_open():
            raise RuntimeError(f"Circuit breaker open for task {task_id}")

        last_error: Exception | None = None
        retry_count = 0

        while retry_count <= self.retry_policy.max_retries:
            try:
                result = await self._execute_with_timeout(func, timeout, *args, **kwargs)
                self._reset_circuit()
                return result

            except Exception as e:
                last_error = e
                error_type = self._handle_exception(e)

                # Record error
                await self._record_error(error_type, str(last_error), task_id, audio_file, retry_count)

                # Check if we should stop retrying
                if self._should_stop_retrying(e, retry_count):
                    break

                # Retry with delay
                await self._handle_retry_delay(retry_count, task_id)
                retry_count += 1

        # All retries failed
        self._trip_circuit()
        raise last_error or RuntimeError(f"Task {task_id} failed after {retry_count} retries")

    async def _execute_with_timeout(
        self,
        func: Callable,
        timeout: float | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function with optional timeout."""
        if timeout:
            return await asyncio.wait_for(self._execute_func(func, *args, **kwargs), timeout=timeout)
        return await self._execute_func(func, *args, **kwargs)

    def _handle_exception(self, exception: Exception) -> ErrorType:
        """Handle exception and return error type."""
        if isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT
        if isinstance(exception, MemoryError):
            return ErrorType.MEMORY
        if isinstance(exception, FileNotFoundError):
            return ErrorType.CORRUPTED_FILE
        return self._classify_error(exception)

    def _should_stop_retrying(self, exception: Exception, retry_count: int) -> bool:
        """Check if we should stop retrying based on exception type and policy."""
        if retry_count >= self.retry_policy.max_retries:
            return True

        if isinstance(exception, TimeoutError) and not self.retry_policy.retry_on_timeout:
            return True

        if isinstance(exception, MemoryError) and not self.retry_policy.retry_on_memory:
            return True

        return isinstance(exception, FileNotFoundError)  # No retry for file not found

    async def _handle_retry_delay(self, retry_count: int, task_id: str) -> None:
        """Handle retry delay logic."""
        delay = self._calculate_retry_delay(retry_count)
        logger.warning(
            f"Retry {retry_count + 1}/{self.retry_policy.max_retries} for task {task_id} after {delay:.1f}s delay"
        )
        await asyncio.sleep(delay)

    async def _execute_func(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute function, handling both sync and async.

        Args:
            func: Function to execute
            *args: Arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """
        Calculate delay before retry with exponential backoff.

        Args:
            retry_count: Current retry attempt

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.retry_policy.initial_delay_seconds * (self.retry_policy.exponential_base**retry_count),
            self.retry_policy.max_delay_seconds,
        )

        # Add jitter
        jitter = delay * self.retry_policy.jitter_factor * random.random()
        return delay + jitter

    def _classify_error(self, error: Exception) -> ErrorType:
        """
        Classify error type.

        Args:
            error: Exception to classify

        Returns:
            ErrorType classification
        """
        error_str = str(error).lower()

        if "timeout" in error_str:
            return ErrorType.TIMEOUT
        if "memory" in error_str:
            return ErrorType.MEMORY
        if "corrupt" in error_str or "invalid" in error_str:
            return ErrorType.CORRUPTED_FILE
        if "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK
        if "resource" in error_str:
            return ErrorType.RESOURCE_EXHAUSTED
        return ErrorType.UNKNOWN

    async def _record_error(
        self,
        error_type: ErrorType,
        message: str,
        task_id: str,
        audio_file: str | None,
        retry_count: int,
    ) -> None:
        """
        Record error for tracking.

        Args:
            error_type: Type of error
            message: Error message
            task_id: Task identifier
            audio_file: Audio file
            retry_count: Retry attempt number
        """
        context = ErrorContext(
            error_type=error_type,
            error_message=message,
            task_id=task_id,
            audio_file=audio_file,
            retry_count=retry_count,
            timestamp=time.time(),
        )

        self.error_history.append(context)
        self.error_counts[error_type] += 1

        # Limit history
        if len(self.error_history) > MAX_ERROR_HISTORY:
            self.error_history = self.error_history[-MAX_ERROR_HISTORY:]

        logger.error(
            f"Error recorded - Type: {error_type.value}, Task: {task_id}, File: {audio_file}, Message: {message}"
        )

    def _is_circuit_open(self) -> bool:
        """
        Check if circuit breaker is open.

        Returns:
            True if circuit is open
        """
        if not self.enable_circuit_breaker:
            return False

        if self.circuit_open:
            # Check if timeout has passed
            if time.time() - self.circuit_open_time > self.circuit_timeout:
                logger.info("Circuit breaker timeout expired, closing circuit")
                self.circuit_open = False
                self.circuit_failures = 0
                return False
            return True

        return False

    def _trip_circuit(self) -> None:
        """Trip the circuit breaker."""
        if not self.enable_circuit_breaker:
            return

        self.circuit_failures += 1
        if self.circuit_failures >= self.circuit_threshold:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.warning(f"Circuit breaker tripped after {self.circuit_failures} failures")

    def _reset_circuit(self) -> None:
        """Reset circuit breaker."""
        if self.circuit_failures > 0:
            self.circuit_failures = 0
            logger.debug("Circuit breaker reset")

    def get_error_stats(self) -> dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error stats
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": dict(self.error_counts),
            "circuit_open": self.circuit_open,
            "circuit_failures": self.circuit_failures,
            "recent_errors": len(self.error_history),
        }


class AudioFallbackHandler:
    """
    Provides fallback strategies for corrupted or problematic audio files.
    """

    def __init__(self) -> None:
        """Initialize fallback handler."""
        self.fallback_attempts = 0
        self.successful_fallbacks = 0

    async def process_with_fallback(
        self,
        audio_file: str,
        primary_func: Callable,
        fallback_funcs: list[Callable],
        **kwargs: Any,
    ) -> Any:
        """
        Process audio with fallback strategies.

        Args:
            audio_file: Audio file path
            primary_func: Primary processing function
            fallback_funcs: List of fallback functions
            **kwargs: Additional arguments

        Returns:
            Processing result

        Raises:
            Exception: If all strategies fail
        """
        self.fallback_attempts += 1

        # Try primary function
        try:
            result = await self._execute_strategy(primary_func, audio_file, **kwargs)
            self.successful_fallbacks += 1
            return result
        except Exception as e:
            logger.warning(f"Primary strategy failed for {audio_file}: {e!s}")

        # Try fallback strategies
        for i, fallback_func in enumerate(fallback_funcs):
            try:
                logger.info(f"Trying fallback strategy {i + 1} for {audio_file}")
                result = await self._execute_strategy(fallback_func, audio_file, **kwargs)
                self.successful_fallbacks += 1
                return result
            except Exception as e:
                logger.warning(f"Fallback strategy {i + 1} failed for {audio_file}: {e!s}")

        # All strategies failed
        raise RuntimeError(f"All processing strategies failed for {audio_file}")

    async def _execute_strategy(self, func: Callable, audio_file: str, **kwargs: Any) -> Any:
        """
        Execute a processing strategy.

        Args:
            func: Processing function
            audio_file: Audio file
            **kwargs: Additional arguments

        Returns:
            Processing result
        """
        if asyncio.iscoroutinefunction(func):
            return await func(audio_file, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, audio_file, **kwargs)

    async def repair_corrupted_audio(self, audio_file: str) -> str | None:
        """
        Attempt to repair corrupted audio file.

        Args:
            audio_file: Path to corrupted audio

        Returns:
            Path to repaired file or None
        """
        # This is a placeholder for actual repair logic
        # In production, you might use tools like ffmpeg to repair
        logger.warning(f"Attempting to repair corrupted audio: {audio_file}")

        # Simulated repair strategies:
        # 1. Try to re-encode with ffmpeg
        # 2. Extract audio from container
        # 3. Use error-tolerant decoder

        # For now, return None indicating repair not implemented
        return None


class ResourceCleanupManager:
    """
    Manages cleanup of resources after processing.
    """

    def __init__(self) -> None:
        """Initialize cleanup manager."""
        self.cleanup_tasks: list[asyncio.Task] = []
        self.resources_to_cleanup: dict[str, Any] = {}

    async def register_resource(
        self,
        resource_id: str,
        cleanup_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Register a resource for cleanup.

        Args:
            resource_id: Resource identifier
            cleanup_func: Cleanup function
            *args: Cleanup function arguments
            **kwargs: Cleanup function keyword arguments
        """
        self.resources_to_cleanup[resource_id] = (cleanup_func, args, kwargs)

    async def cleanup_resource(self, resource_id: str) -> None:
        """
        Clean up a specific resource.

        Args:
            resource_id: Resource identifier
        """
        if resource_id not in self.resources_to_cleanup:
            return

        cleanup_func, args, kwargs = self.resources_to_cleanup[resource_id]

        try:
            if asyncio.iscoroutinefunction(cleanup_func):
                await cleanup_func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cleanup_func, *args, **kwargs)

            del self.resources_to_cleanup[resource_id]
            logger.debug(f"Cleaned up resource: {resource_id}")

        except Exception as e:
            logger.error(f"Failed to clean up resource {resource_id}: {e!s}")

    async def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        tasks = []
        for resource_id in list(self.resources_to_cleanup.keys()):
            task = asyncio.create_task(self.cleanup_resource(resource_id))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Cleaned up {len(tasks)} resources")

    async def __aenter__(self) -> "ResourceCleanupManager":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        await self.cleanup_all()
