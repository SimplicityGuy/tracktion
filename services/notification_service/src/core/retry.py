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
