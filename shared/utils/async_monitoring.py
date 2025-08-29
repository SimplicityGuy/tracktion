"""Async monitoring and observability utilities for external service calls."""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = structlog.get_logger(__name__)

# Prometheus metrics for external service monitoring
external_requests_total = Counter(
    "external_requests_total",
    "Total number of external service requests",
    ["service", "method", "endpoint", "status"],
)
external_request_duration = Histogram(
    "external_request_duration_seconds",
    "Duration of external service requests",
    ["service", "method", "endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
external_request_size = Summary(
    "external_request_size_bytes",
    "Size of external service request payloads",
    ["service", "direction"],  # direction: request/response
)
external_errors_total = Counter(
    "external_errors_total",
    "Total number of external service errors",
    ["service", "error_type", "endpoint"],
)
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Current state of circuit breaker (0=closed, 1=open, 2=half-open)",
    ["service"],
)
active_external_requests = Gauge(
    "active_external_requests",
    "Number of currently active external requests",
    ["service"],
)


class RequestStatus(Enum):
    """Status of an external request."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ExternalRequest:
    """Represents an external service request for monitoring."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    service_name: str = ""
    endpoint: str = ""
    method: str = "GET"
    status: RequestStatus = RequestStatus.PENDING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    request_size: int = 0
    response_size: int = 0
    status_code: int | None = None
    error_message: str | None = None
    retry_count: int = 0
    circuit_breaker_active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalServiceMonitor:
    """Monitor for external service calls with detailed tracking."""

    def __init__(self, service_name: str) -> None:
        """Initialize the external service monitor.

        Args:
            service_name: Name of the service being monitored
        """
        self.service_name = service_name
        self._active_requests: dict[str, ExternalRequest] = {}
        self._request_history: list[ExternalRequest] = []
        self._error_counts: dict[str, int] = {}
        self._success_rate_window: list[bool] = []
        self._latency_percentiles: dict[str, float] = {}

    def start_request(
        self,
        endpoint: str,
        method: str = "GET",
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExternalRequest:
        """Start tracking a new external request.

        Args:
            endpoint: API endpoint being called
            method: HTTP method
            correlation_id: Optional correlation ID for tracing
            metadata: Additional metadata to track

        Returns:
            ExternalRequest object for tracking
        """
        request = ExternalRequest(
            service_name=self.service_name,
            endpoint=endpoint,
            method=method,
            correlation_id=correlation_id or str(uuid.uuid4()),
            metadata=metadata or {},
        )

        self._active_requests[request.request_id] = request
        active_external_requests.labels(service=self.service_name).inc()

        logger.info(
            "External request started",
            service=self.service_name,
            endpoint=endpoint,
            method=method,
            request_id=request.request_id,
            correlation_id=request.correlation_id,
        )

        return request

    def complete_request(
        self,
        request: ExternalRequest,
        status_code: int | None = None,
        response_size: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Complete tracking of an external request.

        Args:
            request: The request object to complete
            status_code: HTTP status code if applicable
            response_size: Size of response in bytes
            error_message: Error message if request failed
        """
        request.end_time = time.time()
        request.duration_ms = (request.end_time - request.start_time) * 1000
        request.status_code = status_code
        request.response_size = response_size
        request.error_message = error_message

        # Determine status based on status code
        if status_code and 200 <= status_code < 300:
            request.status = RequestStatus.SUCCESS
            self._success_rate_window.append(True)
        elif status_code and status_code >= 500:
            request.status = RequestStatus.FAILURE
            self._success_rate_window.append(False)
            self._increment_error_count("server_error")
        elif status_code and status_code >= 400:
            request.status = RequestStatus.FAILURE
            self._success_rate_window.append(False)
            self._increment_error_count("client_error")
        elif error_message:
            if "timeout" in error_message.lower():
                request.status = RequestStatus.TIMEOUT
            elif "cancel" in error_message.lower():
                request.status = RequestStatus.CANCELLED
            else:
                request.status = RequestStatus.FAILURE
            self._success_rate_window.append(False)

        # Update metrics
        external_requests_total.labels(
            service=self.service_name,
            method=request.method,
            endpoint=request.endpoint,
            status=request.status.value,
        ).inc()

        if request.duration_ms:
            external_request_duration.labels(
                service=self.service_name,
                method=request.method,
                endpoint=request.endpoint,
            ).observe(request.duration_ms / 1000)

        if request.request_size:
            external_request_size.labels(
                service=self.service_name,
                direction="request",
            ).observe(request.request_size)

        if request.response_size:
            external_request_size.labels(
                service=self.service_name,
                direction="response",
            ).observe(request.response_size)

        # Remove from active requests
        if request.request_id in self._active_requests:
            del self._active_requests[request.request_id]
            active_external_requests.labels(service=self.service_name).dec()

        # Add to history (keep last 1000 requests)
        self._request_history.append(request)
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-1000:]

        # Keep success rate window to last 100 requests
        if len(self._success_rate_window) > 100:
            self._success_rate_window = self._success_rate_window[-100:]

        logger.info(
            "External request completed",
            service=self.service_name,
            endpoint=request.endpoint,
            method=request.method,
            status=request.status.value,
            duration_ms=request.duration_ms,
            status_code=status_code,
            request_id=request.request_id,
            correlation_id=request.correlation_id,
        )

    def record_error(
        self,
        error_type: str,
        endpoint: str,
        error_message: str,
        correlation_id: str | None = None,
    ) -> None:
        """Record an error for monitoring.

        Args:
            error_type: Type of error (e.g., timeout, connection, parse)
            endpoint: Endpoint where error occurred
            error_message: Detailed error message
            correlation_id: Optional correlation ID
        """
        external_errors_total.labels(
            service=self.service_name,
            error_type=error_type,
            endpoint=endpoint,
        ).inc()

        self._increment_error_count(error_type)

        logger.error(
            "External service error",
            service=self.service_name,
            error_type=error_type,
            endpoint=endpoint,
            error_message=error_message,
            correlation_id=correlation_id,
        )

    def update_circuit_breaker_state(self, state: str) -> None:
        """Update circuit breaker state for monitoring.

        Args:
            state: Circuit breaker state (closed, open, half-open)
        """
        state_value = {"closed": 0, "open": 1, "half-open": 2}.get(state.lower(), 0)
        circuit_breaker_state.labels(service=self.service_name).set(state_value)

        logger.info(
            "Circuit breaker state changed",
            service=self.service_name,
            state=state,
        )

    def get_success_rate(self) -> float:
        """Calculate current success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        if not self._success_rate_window:
            return 100.0
        success_count = sum(self._success_rate_window)
        return (success_count / len(self._success_rate_window)) * 100

    def get_error_rate(self) -> float:
        """Calculate current error rate.

        Returns:
            Error rate as percentage (0-100)
        """
        return 100.0 - self.get_success_rate()

    def get_latency_percentiles(self) -> dict[str, float]:
        """Calculate latency percentiles from recent requests.

        Returns:
            Dictionary of percentile values (p50, p95, p99)
        """
        durations = [r.duration_ms for r in self._request_history[-100:] if r.duration_ms is not None]

        if not durations:
            return {"p50": 0, "p95": 0, "p99": 0}

        durations.sort()
        n = len(durations)

        return {
            "p50": durations[min(int(n * 0.5), n - 1)],
            "p95": durations[min(int(n * 0.95), n - 1)] if n > 1 else durations[0],
            "p99": durations[min(int(n * 0.99), n - 1)] if n > 1 else durations[0],
        }

    def get_active_request_count(self) -> int:
        """Get count of currently active requests.

        Returns:
            Number of active requests
        """
        return len(self._active_requests)

    def _increment_error_count(self, error_type: str) -> None:
        """Increment error count for a specific error type.

        Args:
            error_type: Type of error
        """
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    def generate_alert(self, threshold: float = 10.0) -> dict[str, Any] | None:
        """Generate alert if error rate exceeds threshold.

        Args:
            threshold: Error rate threshold percentage

        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        error_rate = self.get_error_rate()
        if error_rate > threshold:
            return {
                "service": self.service_name,
                "error_rate": error_rate,
                "success_rate": self.get_success_rate(),
                "active_requests": self.get_active_request_count(),
                "error_counts": self._error_counts,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return None


class CorrelationIdPropagator:
    """Propagates correlation IDs through async call chains."""

    def __init__(self) -> None:
        """Initialize correlation ID propagator."""
        self._context_var: ContextVar[str | None] = ContextVar(
            "correlation_id",
            default=None,
        )

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID in current context.

        Args:
            correlation_id: Correlation ID to set
        """
        self._context_var.set(correlation_id)

    def get_correlation_id(self) -> str | None:
        """Get correlation ID from current context.

        Returns:
            Current correlation ID or None
        """
        return self._context_var.get()

    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID.

        Returns:
            New correlation ID
        """
        correlation_id = str(uuid.uuid4())
        self.set_correlation_id(correlation_id)
        return correlation_id


class ServiceHealthMonitor:
    """Monitors overall health of external services."""

    def __init__(self) -> None:
        """Initialize service health monitor."""
        self._service_monitors: dict[str, ExternalServiceMonitor] = {}
        self._health_status: dict[str, bool] = {}

    def get_or_create_monitor(self, service_name: str) -> ExternalServiceMonitor:
        """Get or create a monitor for a service.

        Args:
            service_name: Name of the service

        Returns:
            Service monitor instance
        """
        if service_name not in self._service_monitors:
            self._service_monitors[service_name] = ExternalServiceMonitor(service_name)
        return self._service_monitors[service_name]

    def check_health(self, service_name: str) -> bool:
        """Check health of a service based on recent metrics.

        Args:
            service_name: Name of the service

        Returns:
            True if service is healthy, False otherwise
        """
        monitor = self._service_monitors.get(service_name)
        if not monitor:
            return True  # Assume healthy if no data

        # Service is unhealthy if:
        # - Error rate > 50%
        # - P95 latency > 10 seconds
        # - More than 10 active requests (possible backup)
        error_rate = monitor.get_error_rate()
        latencies = monitor.get_latency_percentiles()
        active_count = monitor.get_active_request_count()

        is_healthy = (
            error_rate < 50
            and latencies.get("p95", 0) < 10000  # 10 seconds in ms
            and active_count < 10
        )

        self._health_status[service_name] = is_healthy
        return is_healthy

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get dashboard data for all monitored services.

        Returns:
            Dashboard data dictionary
        """
        dashboard = {
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {},
        }

        for service_name, monitor in self._service_monitors.items():
            dashboard["services"][service_name] = {  # type: ignore[index]
                "healthy": self.check_health(service_name),
                "success_rate": monitor.get_success_rate(),
                "error_rate": monitor.get_error_rate(),
                "latency_percentiles": monitor.get_latency_percentiles(),
                "active_requests": monitor.get_active_request_count(),
                "error_counts": monitor._error_counts,
            }

        return dashboard


# Global instances
_correlation_propagator: CorrelationIdPropagator | None = None
_health_monitor: ServiceHealthMonitor | None = None


def get_correlation_propagator() -> CorrelationIdPropagator:
    """Get or create global correlation ID propagator.

    Returns:
        Global correlation ID propagator instance
    """
    global _correlation_propagator
    if _correlation_propagator is None:
        _correlation_propagator = CorrelationIdPropagator()
    return _correlation_propagator


def get_health_monitor() -> ServiceHealthMonitor:
    """Get or create global service health monitor.

    Returns:
        Global service health monitor instance
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = ServiceHealthMonitor()
    return _health_monitor
