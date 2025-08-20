"""Circuit breaker pattern implementation for resilient service operations."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Dict, Literal, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure threshold configuration
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open before closing
    timeout: float = 60.0  # Seconds to wait before trying half-open

    # Time window for counting failures
    failure_window: float = 60.0  # Time window for failure counting

    # What exceptions should trigger the circuit breaker
    expected_exceptions: tuple = (Exception,)

    # Optional fallback function
    fallback: Optional[Callable[[], Any]] = None

    # Monitoring hooks
    on_open: Optional[Callable[[str], None]] = None
    on_close: Optional[Callable[[str], None]] = None
    on_half_open: Optional[Callable[[str], None]] = None


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    fallback_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker for protecting service calls."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        metrics_collector: Optional[Any] = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker for logging
            config: Configuration for circuit breaker behavior
            metrics_collector: Optional metrics collector for monitoring
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = metrics_collector

        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # Failure tracking
        self._failure_times: list[float] = []
        self._last_open_time: Optional[float] = None
        self._half_open_successes = 0

        # Statistics
        self.stats = CircuitBreakerStats()

        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._last_open_time and (time.time() - self._last_open_time >= self.config.timeout):
                    self._transition_to_half_open()

            return self._state

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._last_open_time = time.time()
        self._half_open_successes = 0

        # Record state change
        self.stats.state_changes.append(
            {
                "from": old_state.value,
                "to": CircuitState.OPEN.value,
                "timestamp": self._last_open_time,
                "reason": f"Failure threshold ({self.config.failure_threshold}) exceeded",
            }
        )

        logger.warning(
            f"Circuit breaker '{self.name}' opened due to excessive failures",
            extra={
                "consecutive_failures": self.stats.consecutive_failures,
                "threshold": self.config.failure_threshold,
            },
        )

        # Call monitoring hook
        if self.config.on_open:
            try:
                self.config.on_open(self.name)
            except Exception as e:
                logger.error(f"Error in on_open hook: {e}")

        # Record metric if available
        if self.metrics and hasattr(self.metrics, "update_circuit_breaker_state"):
            self.metrics.update_circuit_breaker_state(self.name, "open")

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_times.clear()
        self._half_open_successes = 0
        self.stats.consecutive_failures = 0

        # Record state change
        self.stats.state_changes.append(
            {
                "from": old_state.value,
                "to": CircuitState.CLOSED.value,
                "timestamp": time.time(),
                "reason": "Service recovered",
            }
        )

        logger.info(
            f"Circuit breaker '{self.name}' closed - service recovered",
            extra={"consecutive_successes": self.stats.consecutive_successes},
        )

        # Call monitoring hook
        if self.config.on_close:
            try:
                self.config.on_close(self.name)
            except Exception as e:
                logger.error(f"Error in on_close hook: {e}")

        # Record metric if available
        if self.metrics and hasattr(self.metrics, "update_circuit_breaker_state"):
            self.metrics.update_circuit_breaker_state(self.name, "closed")

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._half_open_successes = 0

        # Record state change
        self.stats.state_changes.append(
            {
                "from": old_state.value,
                "to": CircuitState.HALF_OPEN.value,
                "timestamp": time.time(),
                "reason": f"Timeout ({self.config.timeout}s) expired",
            }
        )

        logger.info(f"Circuit breaker '{self.name}' half-open - testing recovery")

        # Call monitoring hook
        if self.config.on_half_open:
            try:
                self.config.on_half_open(self.name)
            except Exception as e:
                logger.error(f"Error in on_half_open hook: {e}")

        # Record metric if available
        if self.metrics and hasattr(self.metrics, "update_circuit_breaker_state"):
            self.metrics.update_circuit_breaker_state(self.name, "half-open")

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to_closed()

            # Clear old failures outside the window
            current_time = time.time()
            self._failure_times = [t for t in self._failure_times if current_time - t < self.config.failure_window]

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            current_time = time.time()
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure_time = current_time
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0

            # Add to failure times
            self._failure_times.append(current_time)

            # Remove old failures outside the window
            self._failure_times = [t for t in self._failure_times if current_time - t < self.config.failure_window]

            # Check state transitions
            if self._state == CircuitState.CLOSED:
                if len(self._failure_times) >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._transition_to_open()

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback is configured
            Exception: If function fails and is not in expected exceptions
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            self.stats.rejected_calls += 1

            # Use fallback if available
            if self.config.fallback:
                self.stats.fallback_calls += 1
                logger.debug(f"Circuit breaker '{self.name}' using fallback")
                return self.config.fallback()  # type: ignore[no-any-return]

            # Record metric if available
            if self.metrics and hasattr(self.metrics, "track_external_service_call"):
                self.metrics.track_external_service_call(self.name, False, 0.0)

            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN - calls are being rejected")

        # Try to execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.config.expected_exceptions:
            self._record_failure()

            # Record metric if available
            if self.metrics and hasattr(self.metrics, "track_external_service_call"):
                self.metrics.track_external_service_call(self.name, False, 0.0)

            raise

        except Exception as e:
            # Unexpected exceptions don't affect circuit state
            logger.error(f"Unexpected exception in circuit breaker '{self.name}': {e}", exc_info=True)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_times.clear()
            self._last_open_time = None
            self._half_open_successes = 0
            self.stats.consecutive_failures = 0

            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary containing circuit breaker statistics
        """
        with self._lock:
            success_rate = (
                (self.stats.successful_calls / self.stats.total_calls * 100) if self.stats.total_calls > 0 else 0
            )

            return {
                "name": self.name,
                "state": self.state.value,
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "fallback_calls": self.stats.fallback_calls,
                "success_rate": success_rate,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
                "recent_failures": len(self._failure_times),
                "state_changes": len(self.stats.state_changes),
            }

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Context manager exit.

        Records success or failure based on exception.
        """
        if exc_type is None:
            self._record_success()
        elif isinstance(exc_val, self.config.expected_exceptions):
            self._record_failure()

        return False  # Don't suppress exceptions


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


def circuit_breaker(
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable:
    """Decorator for applying circuit breaker to functions.

    Args:
        name: Name of the circuit breaker (defaults to function name)
        config: Circuit breaker configuration

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(breaker_name, config)

        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        # Add method to get breaker stats
        wrapper.get_circuit_stats = breaker.get_stats  # type: ignore[attr-defined]
        wrapper.reset_circuit = breaker.reset  # type: ignore[attr-defined]

        return wrapper

    return decorator
