"""
Resilience patterns for the tracklist service.

Provides circuit breaker, retry logic, and other resilience patterns.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from .exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time
            and (time.time() - self._last_failure_time > self.recovery_timeout)
        ):
            # Check if recovery timeout has passed
            logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

        return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ServiceUnavailableError: If circuit is open
            Exception: If function raises an exception
        """
        if self.state == CircuitState.OPEN:
            raise ServiceUnavailableError(
                f"Circuit breaker '{self.name}' is OPEN",
                service_name=self.name,
                retry_after=int(self.recovery_timeout),
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ServiceUnavailableError: If circuit is open
            Exception: If function raises an exception
        """
        if self.state == CircuitState.OPEN:
            raise ServiceUnavailableError(
                f"Circuit breaker '{self.name}' is OPEN",
                service_name=self.name,
                retry_after=int(self.recovery_timeout),
            )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            # Require multiple successes before closing
            if self._success_count >= 3:
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker '{self.name}' failure in HALF_OPEN state, reopening")
            self._state = CircuitState.OPEN

        elif self._failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker '{self.name}' threshold exceeded, opening circuit")
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count = 0
        logger.info(f"Circuit breaker '{self.name}' reset")


class ExponentialBackoff:
    """Exponential backoff retry strategy."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """Initialize exponential backoff.

        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Delay multiplier for each retry
            jitter: Add random jitter to delays
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt.

        Args:
            attempt: Attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (self.multiplier**attempt), self.max_delay)

        if self.jitter:
            # Add jitter of ±25%
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff: ExponentialBackoff | None = None,
    exceptions: tuple = (Exception,),
    **kwargs: Any,
) -> Any:
    """Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Function arguments
        max_attempts: Maximum number of attempts
        backoff: Backoff strategy (uses default if None)
        exceptions: Exception types to retry on
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Exception: Last exception if all retries fail
    """
    if backoff is None:
        backoff = ExponentialBackoff()

    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)

        except exceptions as e:
            last_exception = e

            if attempt < max_attempts - 1:
                delay = backoff.get_delay(attempt)
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed")

    if last_exception:
        raise last_exception

    raise RuntimeError("Retry failed without exception")


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        rate: float,
        capacity: int,
        name: str | None = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            rate: Tokens per second
            capacity: Maximum token capacity
            name: Optional name for logging
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name or "RateLimiter"

        self._tokens = float(capacity)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Seconds waited (0 if no wait)

        Raises:
            ValueError: If requesting more tokens than capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens, capacity is {self.capacity}")

        async with self._lock:
            wait_time = 0.0

            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now

            # Wait if not enough tokens
            if self._tokens < tokens:
                deficit = tokens - self._tokens
                wait_time = deficit / self.rate

                logger.debug(f"Rate limiter '{self.name}' waiting {wait_time:.2f}s for {tokens} tokens")

                await asyncio.sleep(wait_time)

                # Update tokens after waiting
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last_update = now

            # Consume tokens
            self._tokens -= tokens

            return wait_time

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        self._tokens = float(self.capacity)
        self._last_update = time.time()
        logger.debug(f"Rate limiter '{self.name}' reset")


class HealthCheck:
    """Health check for service dependencies."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        interval: float = 30.0,
    ) -> None:
        """Initialize health check.

        Args:
            name: Name of the service being checked
            check_func: Function that returns True if healthy
            interval: Seconds between health checks
        """
        self.name = name
        self.check_func = check_func
        self.interval = interval

        self._healthy = False
        self._last_check = 0.0
        self._consecutive_failures = 0

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if service is healthy
        """
        now = time.time()

        # Check if we need to run health check
        if now - self._last_check >= self.interval:
            try:
                self._healthy = self.check_func()

                if self._healthy:
                    if self._consecutive_failures > 0:
                        logger.info(f"Service '{self.name}' recovered")
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    logger.warning(f"Service '{self.name}' unhealthy (failures: {self._consecutive_failures})")

            except Exception as e:
                self._healthy = False
                self._consecutive_failures += 1
                logger.error(f"Health check for '{self.name}' failed: {e}")

            self._last_check = now

        return self._healthy
