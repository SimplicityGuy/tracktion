"""Circuit breaker exceptions for resilience patterns."""

from typing import Any


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize circuit breaker error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.details = details or {}


class CircuitOpenError(CircuitBreakerError):
    """Exception raised when circuit breaker is open and rejecting calls."""

    def __init__(
        self,
        name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize circuit open error.

        Args:
            name: Circuit breaker name
            details: Additional error details
        """
        message = f"Circuit breaker '{name}' is OPEN - calls are being rejected"
        error_details = details or {}
        error_details["circuit_name"] = name
        super().__init__(message, error_details)
        self.circuit_name = name


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Exception raised when circuit breaker operation times out."""

    def __init__(
        self,
        name: str,
        timeout: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize circuit breaker timeout error.

        Args:
            name: Circuit breaker name
            timeout: Timeout value
            details: Additional error details
        """
        message = f"Circuit breaker '{name}' operation timed out after {timeout}s"
        error_details = details or {}
        error_details.update({"circuit_name": name, "timeout": timeout})
        super().__init__(message, error_details)
        self.circuit_name = name
        self.timeout = timeout


class CircuitBreakerConfigurationError(CircuitBreakerError):
    """Exception raised for circuit breaker configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            details: Additional error details
        """
        error_details = details or {}
        if config_key is not None:
            error_details["config_key"] = config_key
        if config_value is not None:
            error_details["config_value"] = config_value
        super().__init__(message, error_details)
        self.config_key = config_key
        self.config_value = config_value
