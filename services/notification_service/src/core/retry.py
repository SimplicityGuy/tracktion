"""Retry logic and policies for notification delivery."""

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] | None = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt.

        Args:
            attempt: The attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = min(self.backoff_base ** (attempt - 1), self.backoff_max)
        if self.jitter:
            delay *= 0.5 + random.random()
        return delay


class RetryManager:
    """Manages retry logic for operations."""

    def __init__(self, policy: RetryPolicy | None = None):
        """Initialize retry manager.

        Args:
            policy: Retry policy to use (defaults to standard policy)
        """
        self.policy = policy or RetryPolicy()

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result from successful execution

        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Operation succeeded on attempt {attempt}")
                return result
            except Exception as e:
                last_exception = e

                # Check if we should retry this exception
                if self.policy.retry_on and not isinstance(e, self.policy.retry_on):
                    logger.error(f"Exception not in retry list: {e}")
                    raise

                if attempt < self.policy.max_attempts:
                    delay = self.policy.calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.policy.max_attempts} attempts failed. Last error: {e}")

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed without exception")


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call function through circuit breaker.

        Args:
            func: Async function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result from successful execution

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self.last_failure_time and (
                asyncio.get_event_loop().time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "half_open"
            else:
                raise RuntimeError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            if self.expected_exception and not isinstance(e, self.expected_exception):
                raise

            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
