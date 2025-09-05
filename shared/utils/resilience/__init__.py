"""Resilience patterns and utilities for robust service operations.

This module provides circuit breaker patterns, retry logic, and other resilience
patterns to help services handle failures gracefully and recover from issues.

Main components:
- CircuitBreaker: Circuit breaker implementation with configurable thresholds
- CircuitBreakerConfig: Configuration for circuit breaker behavior
- Service presets: Predefined configurations for common service types
- Exception handling: Standardized exceptions for circuit breaker operations
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    circuit_breaker,
    get_circuit_breaker,
)
from .config import (
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    ConfigurationManager,
    DomainConfig,
    ServicePresets,
    ServiceType,
)
from .exceptions import (
    CircuitBreakerConfigurationError,
    CircuitBreakerError,
    CircuitBreakerTimeoutError,
    CircuitOpenError,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerConfigurationError",
    "CircuitBreakerError",
    "CircuitBreakerManager",
    "CircuitBreakerStats",
    "CircuitBreakerTimeoutError",
    "CircuitOpenError",
    "CircuitState",
    "ConfigurationManager",
    "DomainConfig",
    "ServicePresets",
    "ServiceType",
    "circuit_breaker",
    "get_circuit_breaker",
]
