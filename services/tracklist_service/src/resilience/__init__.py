"""
Resilience module for tracklist service.

This module provides error handling, circuit breaker patterns,
and retry logic for robust service operations.
"""

import asyncio
import random
import sys
from pathlib import Path

# Add the analysis service to Python path
analysis_service_path = Path(__file__).parent.parent.parent.parent / "analysis_service" / "src"
sys.path.insert(0, str(analysis_service_path))

from shared.utils.resilience import CircuitState, ServiceType, get_circuit_breaker  # noqa: E402

from .error_handler import (  # noqa: E402
    CircuitBreakerState,
    HealthCheck,
    ParseError,
    PartialExtractor,
    RateLimitError,
    ScrapingError,
    TracklistError,
    TracklistNotFoundError,
    async_retry,
    create_circuit_breaker,
    retry,
)

# For backward compatibility, re-export create_circuit_breaker as CircuitBreaker
CircuitBreaker = create_circuit_breaker

# Import from rate limiting module for backward compatibility
try:
    from services.tracklist_service.src.rate_limiting import RateLimiter
except ImportError:
    # Fallback if rate_limiting module doesn't have RateLimiter
    RateLimiter = None


# Create placeholder classes for missing functionality to maintain test compatibility
class ExponentialBackoff:
    """Exponential backoff strategy for retries."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.base_delay * (self.multiplier**attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.75 + random.random() * 0.5)  # ±25% jitter
        return delay


async def retry_with_backoff(
    func,
    max_attempts: int = 3,
    backoff: ExponentialBackoff | None = None,
    exceptions: tuple = (Exception,),
):
    """Retry function with exponential backoff."""
    if backoff is None:
        backoff = ExponentialBackoff()

    last_exception = None
    for attempt in range(max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = backoff.get_delay(attempt)
                await asyncio.sleep(delay)
            else:
                break

    if last_exception:
        raise last_exception
    return None


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitState",
    "ExponentialBackoff",
    "HealthCheck",
    "ParseError",
    "PartialExtractor",
    "RateLimitError",
    "RateLimiter",
    "ScrapingError",
    "ServiceType",
    "TracklistError",
    "TracklistNotFoundError",
    "async_retry",
    "create_circuit_breaker",
    "get_circuit_breaker",
    "retry",
    "retry_with_backoff",
]
