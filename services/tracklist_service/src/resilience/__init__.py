"""
Resilience module for tracklist service.

This module provides error handling, circuit breaker patterns,
and retry logic for robust service operations.
"""

from .error_handler import (
    CircuitBreaker,
    CircuitBreakerState,
    HealthCheck,
    ParseError,
    PartialExtractor,
    RateLimitError,
    ScrapingError,
    TracklistError,
    TracklistNotFoundError,
    async_retry,
    retry,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "HealthCheck",
    "ParseError",
    "PartialExtractor",
    "RateLimitError",
    "ScrapingError",
    "TracklistError",
    "TracklistNotFoundError",
    "async_retry",
    "retry",
]
