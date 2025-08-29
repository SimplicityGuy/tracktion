"""Structured logging configuration for the analysis service."""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Performance tracking for log operations
LOG_PERFORMANCE_METRICS: Dict[str, Any] = {
    "total_logs": 0,
    "avg_format_time_ms": 0.0,
}


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        service_name: str = "analysis_service",
        include_hostname: bool = True,
        include_function: bool = True,
        include_thread: bool = True,
    ) -> None:
        """Initialize the structured formatter.

        Args:
            service_name: Name of the service for log identification
            include_hostname: Whether to include hostname in logs
            include_function: Whether to include function name in logs
            include_thread: Whether to include thread info in logs
        """
        super().__init__()
        self.service_name = service_name
        self.include_hostname = include_hostname
        self.include_function = include_function
        self.include_thread = include_thread

        # Get hostname once
        self.hostname = ""
        if include_hostname:
            import socket

            try:
                self.hostname = socket.gethostname()
            except Exception:
                self.hostname = "unknown"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        start_time = time.perf_counter()

        # Base log structure
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add optional fields
        if self.include_hostname and self.hostname:
            log_data["hostname"] = self.hostname

        if self.include_function:
            log_data["module"] = record.module
            log_data["function"] = record.funcName
            log_data["line"] = record.lineno

        if self.include_thread:
            log_data["thread"] = record.thread
            log_data["thread_name"] = record.threadName

        # Add extra fields (including correlation_id if present)
        if hasattr(record, "__dict__"):
            extras = {}
            for key, value in record.__dict__.items():
                # Skip standard LogRecord attributes
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "extra",
                ]:
                    # Handle special fields
                    if key == "correlation_id":
                        log_data["correlation_id"] = value
                    else:
                        extras[key] = value

            if extras:
                log_data["extra"] = extras

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_data["stack_info"] = self.formatStack(record.stack_info)

        # Track performance
        format_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_metrics(format_time)

        return json.dumps(log_data)

    def _update_performance_metrics(self, format_time_ms: float) -> None:
        """Update performance metrics for logging.

        Args:
            format_time_ms: Time taken to format log in milliseconds
        """
        global LOG_PERFORMANCE_METRICS
        total = LOG_PERFORMANCE_METRICS["total_logs"]
        avg_time = LOG_PERFORMANCE_METRICS["avg_format_time_ms"]

        # Update average
        new_total = total + 1
        new_avg = ((avg_time * total) + format_time_ms) / new_total

        LOG_PERFORMANCE_METRICS["total_logs"] = new_total
        LOG_PERFORMANCE_METRICS["avg_format_time_ms"] = new_avg


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def __init__(self, correlation_id: Optional[str] = None) -> None:
        """Initialize the filter.

        Args:
            correlation_id: Default correlation ID to use
        """
        super().__init__()
        self.correlation_id = correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record if not already present.

        Args:
            record: Log record to filter

        Returns:
            True to allow the record through
        """
        if not hasattr(record, "correlation_id") and self.correlation_id:
            record.correlation_id = self.correlation_id  # type: ignore[attr-defined]
        return True


class PerformanceLogger:
    """Context manager for logging operation performance."""

    def __init__(
        self,
        logger: logging.Logger,
        operation_name: str,
        correlation_id: Optional[str] = None,
        log_level: int = logging.INFO,
        threshold_ms: float = 1000.0,
    ) -> None:
        """Initialize performance logger.

        Args:
            logger: Logger instance to use
            operation_name: Name of the operation being timed
            correlation_id: Optional correlation ID
            log_level: Log level for timing messages
            threshold_ms: Threshold in ms for warning about slow operations
        """
        self.logger = logger
        self.operation_name = operation_name
        self.correlation_id = correlation_id
        self.log_level = log_level
        self.threshold_ms = threshold_ms
        self.start_time: float = 0

    def __enter__(self) -> "PerformanceLogger":
        """Enter context manager and start timing."""
        self.start_time = time.perf_counter()
        extra = {"correlation_id": self.correlation_id} if self.correlation_id else {}
        self.logger.log(
            self.log_level,
            f"Starting {self.operation_name}",
            extra=extra,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and log timing."""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        extra: Dict[str, Any] = {
            "duration_ms": elapsed_ms,
            "operation": self.operation_name,
        }
        if self.correlation_id:
            extra["correlation_id"] = self.correlation_id

        if exc_type:
            # Operation failed
            extra["error"] = str(exc_val)
            self.logger.error(
                f"{self.operation_name} failed after {elapsed_ms:.2f}ms",
                extra=extra,
            )
        elif elapsed_ms > self.threshold_ms:
            # Operation slow
            self.logger.warning(
                f"{self.operation_name} completed in {elapsed_ms:.2f}ms (threshold: {self.threshold_ms}ms)",
                extra=extra,
            )
        else:
            # Operation successful and within threshold
            self.logger.log(
                self.log_level,
                f"Completed {self.operation_name} in {elapsed_ms:.2f}ms",
                extra=extra,
            )


def configure_structured_logging(
    service_name: str = "analysis_service",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    include_console: bool = True,
    include_hostname: bool = True,
    include_function: bool = True,
    include_thread: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        service_name: Name of the service for log identification
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        include_console: Whether to include console output
        include_hostname: Whether to include hostname in logs
        include_function: Whether to include function info in logs
        include_thread: Whether to include thread info in logs
    """
    # Get log level from environment or parameter
    level_str = os.getenv("LOG_LEVEL", log_level).upper()
    level = getattr(logging, level_str, logging.INFO)

    # Create formatter
    formatter = StructuredFormatter(
        service_name=service_name,
        include_hostname=include_hostname,
        include_function=include_function,
        include_thread=include_thread,
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Add console handler if requested
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Set levels for specific loggers to reduce noise
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Log initial configuration
    root_logger.info(
        "Structured logging configured",
        extra={
            "service": service_name,
            "level": level_str,
            "handlers": {
                "console": include_console,
                "file": log_file is not None,
            },
        },
    )


def get_performance_metrics() -> Dict[str, Any]:
    """Get logging performance metrics.

    Returns:
        Dictionary with performance metrics
    """
    return LOG_PERFORMANCE_METRICS.copy()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
