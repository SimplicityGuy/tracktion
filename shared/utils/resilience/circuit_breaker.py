"""Circuit breaker pattern implementation for resilient service operations."""

import logging
import threading
import time
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Literal, TypeVar

from .config import (
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    ConfigurationManager,
    ServicePresets,
    ServiceType,
)
from .exceptions import CircuitOpenError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """Circuit breaker for protecting service calls."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        metrics_collector: Any | None = None,
        domain: str | None = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Name of the circuit breaker for logging
            config: Configuration for circuit breaker behavior
            metrics_collector: Optional metrics collector for monitoring
            domain: Domain this circuit breaker protects (extracted from URLs if not provided)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = metrics_collector
        self.domain = domain

        # State management
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # Failure tracking
        self._failure_times: list[float] = []
        self._last_open_time: float | None = None
        self._half_open_successes = 0

        # Statistics
        self.stats = CircuitBreakerStats()

        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._last_open_time
                and (time.time() - self._last_open_time >= self.config.timeout)
            ):
                # Check if we should transition to half-open
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

            raise CircuitOpenError(self.name)

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
            logger.error(
                f"Unexpected exception in circuit breaker '{self.name}': {e}",
                exc_info=True,
            )
            raise

    async def call_async(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
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

            raise CircuitOpenError(self.name)

        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
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
            logger.error(
                f"Unexpected exception in circuit breaker '{self.name}': {e}",
                exc_info=True,
            )
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

    def get_stats(self) -> dict[str, Any]:
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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Context manager exit.

        Records success or failure based on exception.
        """
        if exc_type is None:
            self._record_success()
        elif isinstance(exc_val, self.config.expected_exceptions):
            self._record_failure()

        return False  # Don't suppress exceptions


def circuit_breaker(
    name: str | None = None,
    config: CircuitBreakerConfig | None = None,
    domain: str | None = None,
    service_type: ServiceType | None = None,
) -> Callable:
    """Decorator for applying circuit breaker to functions.

    Args:
        name: Name of the circuit breaker (defaults to function name)
        config: Circuit breaker configuration
        domain: Domain for the circuit breaker
        service_type: Service type for preset configuration

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or func.__name__
        breaker = get_circuit_breaker(
            name=breaker_name,
            config=config,
            domain=domain,
            service_type=service_type,
        )

        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        # Add method to get breaker stats
        wrapper.get_circuit_stats = breaker.get_stats  # type: ignore[attr-defined]
        wrapper.reset_circuit = breaker.reset  # type: ignore[attr-defined]
        wrapper.circuit_breaker = breaker  # type: ignore[attr-defined]

        return wrapper

    return decorator


class CircuitBreakerManager:
    """Manages multiple circuit breakers with domain-based organization."""

    def __init__(self, config_manager: ConfigurationManager | None = None) -> None:
        """Initialize circuit breaker manager.

        Args:
            config_manager: Configuration manager for domain-specific settings
        """
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._config_manager = config_manager or ConfigurationManager()
        self._lock = threading.RLock()

    def get_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        domain: str | None = None,
        service_type: ServiceType | None = None,
        metrics_collector: Any | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Optional configuration (uses domain or default if not provided)
            domain: Domain for the circuit breaker
            service_type: Service type for preset configuration
            metrics_collector: Optional metrics collector

        Returns:
            Circuit breaker instance
        """
        with self._lock:
            key = f"{name}:{domain or 'default'}"

            if key not in self._circuit_breakers:
                # Determine configuration
                if config:
                    cb_config = config
                elif service_type:
                    cb_config = ServicePresets.get_preset(service_type)
                elif domain:
                    cb_config = self._config_manager.get_domain_config(domain)
                else:
                    cb_config = self._config_manager.get_default_config()

                # Set domain in config if provided
                if domain and not cb_config.domain:
                    cb_config.domain = domain

                self._circuit_breakers[key] = CircuitBreaker(
                    name=name,
                    config=cb_config,
                    metrics_collector=metrics_collector,
                    domain=domain,
                )

            return self._circuit_breakers[key]


# Global circuit breaker manager instance
_global_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    domain: str | None = None,
    service_type: ServiceType | None = None,
    metrics_collector: Any | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global manager.

    Args:
        name: Circuit breaker name
        config: Optional configuration
        domain: Domain for the circuit breaker
        service_type: Service type for preset configuration
        metrics_collector: Optional metrics collector

    Returns:
        Circuit breaker instance
    """
    return _global_manager.get_circuit_breaker(
        name=name,
        config=config,
        domain=domain,
        service_type=service_type,
        metrics_collector=metrics_collector,
    )
