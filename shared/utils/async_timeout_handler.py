"""Async timeout and cancellation handling utilities."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, TypeVar

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger(__name__)

# Type variable for generic return types
T = TypeVar("T")

# Prometheus metrics
timeout_occurrences = Counter(
    "async_timeout_occurrences_total",
    "Total number of timeouts",
    ["service", "operation"],
)
cancellation_occurrences = Counter(
    "async_cancellation_occurrences_total",
    "Total number of cancellations",
    ["service", "operation"],
)
operation_duration = Histogram(
    "async_operation_duration_seconds",
    "Duration of async operations",
    ["service", "operation", "status"],
)


class TimeoutStrategy(Enum):
    """Timeout escalation strategies."""

    FIXED = "fixed"  # Use a fixed timeout
    LINEAR = "linear"  # Increase timeout linearly
    EXPONENTIAL = "exponential"  # Increase timeout exponentially
    ADAPTIVE = "adaptive"  # Adjust based on historical performance


class DeadlineExceeded(Exception):
    """Raised when a deadline is exceeded."""

    pass


class TimeoutConfig:
    """Configuration for timeout handling."""

    def __init__(
        self,
        default_timeout: float = 10.0,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        write_timeout: float = 10.0,
        total_timeout: float = 30.0,
        strategy: TimeoutStrategy = TimeoutStrategy.FIXED,
        escalation_factor: float = 1.5,
        max_timeout: float = 60.0,
    ) -> None:
        """Initialize timeout configuration.

        Args:
            default_timeout: Default timeout for operations
            connect_timeout: Timeout for connection establishment
            read_timeout: Timeout for reading data
            write_timeout: Timeout for writing data
            total_timeout: Total timeout for entire operation
            strategy: Timeout escalation strategy
            escalation_factor: Factor for timeout escalation
            max_timeout: Maximum allowed timeout
        """
        self.default_timeout = default_timeout
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self.total_timeout = total_timeout
        self.strategy = strategy
        self.escalation_factor = escalation_factor
        self.max_timeout = max_timeout


class TimeoutHandler:
    """Handles timeouts and cancellations for async operations."""

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        """Initialize timeout handler.

        Args:
            config: Optional timeout configuration
        """
        self.config = config or TimeoutConfig()
        self._operation_history: dict[str, list[float]] = {}
        self._timeout_counts: dict[str, int] = {}

    async def execute_with_timeout(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: float | None = None,
        service: str = "default",
        operation: str = "operation",
        **kwargs: Any,
    ) -> T:
        """Execute an async function with timeout.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            timeout: Optional timeout override
            service: Service name for metrics
            operation: Operation name for metrics
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            asyncio.TimeoutError: If operation times out
        """
        timeout_value = timeout or self._calculate_timeout(f"{service}.{operation}")
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_value,
            )

            # Record successful operation
            duration = time.time() - start_time
            self._record_operation(f"{service}.{operation}", duration, success=True)
            operation_duration.labels(
                service=service,
                operation=operation,
                status="success",
            ).observe(duration)

            logger.debug(
                "Operation completed",
                service=service,
                operation=operation,
                duration=duration,
                timeout=timeout_value,
            )

            return result

        except TimeoutError:
            duration = time.time() - start_time
            self._record_operation(f"{service}.{operation}", duration, success=False)
            timeout_occurrences.labels(service=service, operation=operation).inc()
            operation_duration.labels(
                service=service,
                operation=operation,
                status="timeout",
            ).observe(duration)

            logger.warning(
                "Operation timed out",
                service=service,
                operation=operation,
                timeout=timeout_value,
                duration=duration,
            )
            raise

        except asyncio.CancelledError:
            duration = time.time() - start_time
            cancellation_occurrences.labels(service=service, operation=operation).inc()
            operation_duration.labels(
                service=service,
                operation=operation,
                status="cancelled",
            ).observe(duration)

            logger.warning(
                "Operation cancelled",
                service=service,
                operation=operation,
                duration=duration,
            )
            raise

    def _calculate_timeout(self, operation_key: str) -> float:
        """Calculate timeout based on strategy and history.

        Args:
            operation_key: Key identifying the operation

        Returns:
            Calculated timeout value
        """
        base_timeout = self.config.default_timeout

        if self.config.strategy == TimeoutStrategy.FIXED:
            return base_timeout

        elif self.config.strategy == TimeoutStrategy.LINEAR:
            timeout_count = self._timeout_counts.get(operation_key, 0)
            timeout = base_timeout + (timeout_count * 2)
            return min(timeout, self.config.max_timeout)

        elif self.config.strategy == TimeoutStrategy.EXPONENTIAL:
            timeout_count = self._timeout_counts.get(operation_key, 0)
            timeout = base_timeout * (self.config.escalation_factor**timeout_count)
            return min(timeout, self.config.max_timeout)

        elif self.config.strategy == TimeoutStrategy.ADAPTIVE:
            history = self._operation_history.get(operation_key, [])
            if len(history) >= 5:
                # Use 95th percentile of recent operations
                sorted_history = sorted(history[-20:])  # Last 20 operations
                p95_index = int(len(sorted_history) * 0.95)
                timeout = sorted_history[p95_index] * 1.5  # Add 50% buffer
                return min(timeout, self.config.max_timeout)
            return base_timeout

        return base_timeout

    def _record_operation(self, operation_key: str, duration: float, success: bool) -> None:
        """Record operation duration and outcome.

        Args:
            operation_key: Key identifying the operation
            duration: Operation duration in seconds
            success: Whether operation succeeded
        """
        if success:
            # Record successful operation duration
            if operation_key not in self._operation_history:
                self._operation_history[operation_key] = []
            self._operation_history[operation_key].append(duration)

            # Keep only recent history (last 100 operations)
            if len(self._operation_history[operation_key]) > 100:
                self._operation_history[operation_key] = self._operation_history[operation_key][-100:]

            # Reset timeout count on success
            self._timeout_counts[operation_key] = 0
        else:
            # Increment timeout count
            self._timeout_counts[operation_key] = self._timeout_counts.get(operation_key, 0) + 1


class DeadlineManager:
    """Manages request deadlines and propagation."""

    def __init__(self) -> None:
        """Initialize deadline manager."""
        self._deadlines: dict[str, float] = {}

    def set_deadline(self, request_id: str, deadline: float) -> None:
        """Set a deadline for a request.

        Args:
            request_id: Unique request identifier
            deadline: Absolute deadline timestamp
        """
        self._deadlines[request_id] = deadline
        logger.debug(
            "Deadline set",
            request_id=request_id,
            deadline=deadline,
            remaining=deadline - time.time(),
        )

    def get_deadline(self, request_id: str) -> float | None:
        """Get deadline for a request.

        Args:
            request_id: Unique request identifier

        Returns:
            Deadline timestamp or None if not set
        """
        return self._deadlines.get(request_id)

    def check_deadline(self, request_id: str) -> None:
        """Check if deadline has been exceeded.

        Args:
            request_id: Unique request identifier

        Raises:
            DeadlineExceeded: If deadline has been exceeded
        """
        deadline = self._deadlines.get(request_id)
        if deadline and time.time() > deadline:
            raise DeadlineExceeded(f"Deadline exceeded for request {request_id}")

    def remaining_time(self, request_id: str) -> float | None:
        """Get remaining time until deadline.

        Args:
            request_id: Unique request identifier

        Returns:
            Remaining time in seconds or None if no deadline
        """
        deadline = self._deadlines.get(request_id)
        if deadline:
            remaining = deadline - time.time()
            return max(0, remaining)
        return None

    def clear_deadline(self, request_id: str) -> None:
        """Clear deadline for a request.

        Args:
            request_id: Unique request identifier
        """
        if request_id in self._deadlines:
            del self._deadlines[request_id]
            logger.debug("Deadline cleared", request_id=request_id)

    @asynccontextmanager
    async def deadline_context(
        self,
        request_id: str,
        timeout: float,
    ) -> AsyncIterator[None]:
        """Context manager for deadline-aware execution.

        Args:
            request_id: Unique request identifier
            timeout: Timeout in seconds

        Yields:
            None

        Raises:
            DeadlineExceeded: If deadline is exceeded
        """
        deadline = time.time() + timeout
        self.set_deadline(request_id, deadline)

        try:
            yield
        finally:
            self.clear_deadline(request_id)


class CancellationHandler:
    """Handles graceful cancellation of async operations."""

    def __init__(self) -> None:
        """Initialize cancellation handler."""
        self._cancellation_tokens: dict[str, asyncio.Event] = {}

    def create_token(self, operation_id: str) -> asyncio.Event:
        """Create a cancellation token for an operation.

        Args:
            operation_id: Unique operation identifier

        Returns:
            Cancellation token (asyncio.Event)
        """
        token = asyncio.Event()
        self._cancellation_tokens[operation_id] = token
        return token

    def request_cancellation(self, operation_id: str) -> bool:
        """Request cancellation of an operation.

        Args:
            operation_id: Unique operation identifier

        Returns:
            True if cancellation was requested, False if operation not found
        """
        token = self._cancellation_tokens.get(operation_id)
        if token:
            token.set()
            logger.info("Cancellation requested", operation_id=operation_id)
            return True
        return False

    def is_cancelled(self, operation_id: str) -> bool:
        """Check if cancellation has been requested.

        Args:
            operation_id: Unique operation identifier

        Returns:
            True if cancellation requested, False otherwise
        """
        token = self._cancellation_tokens.get(operation_id)
        return token.is_set() if token else False

    async def check_cancellation(self, operation_id: str) -> None:
        """Check for cancellation and raise if requested.

        Args:
            operation_id: Unique operation identifier

        Raises:
            asyncio.CancelledError: If cancellation was requested
        """
        if self.is_cancelled(operation_id):
            raise asyncio.CancelledError(f"Operation {operation_id} was cancelled")

    def clear_token(self, operation_id: str) -> None:
        """Clear cancellation token for an operation.

        Args:
            operation_id: Unique operation identifier
        """
        if operation_id in self._cancellation_tokens:
            del self._cancellation_tokens[operation_id]

    @asynccontextmanager
    async def cancellable_operation(
        self,
        operation_id: str,
    ) -> AsyncIterator[asyncio.Event]:
        """Context manager for cancellable operations.

        Args:
            operation_id: Unique operation identifier

        Yields:
            Cancellation token

        Example:
            async with cancellation_handler.cancellable_operation("op1") as token:
                while not token.is_set():
                    # Do work
                    await asyncio.sleep(0.1)
        """
        token = self.create_token(operation_id)
        try:
            yield token
        finally:
            self.clear_token(operation_id)


# Global instances
_timeout_handler: TimeoutHandler | None = None
_deadline_manager: DeadlineManager | None = None
_cancellation_handler: CancellationHandler | None = None


def get_timeout_handler(config: TimeoutConfig | None = None) -> TimeoutHandler:
    """Get or create global timeout handler.

    Args:
        config: Optional timeout configuration

    Returns:
        Global timeout handler instance
    """
    global _timeout_handler
    if _timeout_handler is None:
        _timeout_handler = TimeoutHandler(config)
    return _timeout_handler


def get_deadline_manager() -> DeadlineManager:
    """Get or create global deadline manager.

    Returns:
        Global deadline manager instance
    """
    global _deadline_manager
    if _deadline_manager is None:
        _deadline_manager = DeadlineManager()
    return _deadline_manager


def get_cancellation_handler() -> CancellationHandler:
    """Get or create global cancellation handler.

    Returns:
        Global cancellation handler instance
    """
    global _cancellation_handler
    if _cancellation_handler is None:
        _cancellation_handler = CancellationHandler()
    return _cancellation_handler
