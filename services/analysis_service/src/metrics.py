"""
Metrics collection and exposure module for analysis service.

This module provides Prometheus metrics for monitoring the analysis pipeline,
including processing time, success/failure rates, queue depths, and throughput.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog
from flask import Flask, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:
    from collections.abc import Generator

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects and exposes metrics for the analysis pipeline."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """
        Initialize the metrics collector.

        Args:
            registry: Optional Prometheus registry. Creates new one if not provided.
        """
        self.registry = registry or CollectorRegistry()

        # Processing metrics
        self.processing_time = Histogram(
            "analysis_processing_seconds",
            "Time spent processing audio files",
            ["file_type", "status"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        self.files_processed = Counter(
            "analysis_files_processed_total",
            "Total number of files processed",
            ["file_type", "status"],
            registry=self.registry,
        )

        self.processing_errors = Counter(
            "analysis_processing_errors_total",
            "Total number of processing errors",
            ["error_type", "file_type"],
            registry=self.registry,
        )

        # Queue metrics
        self.queue_depth = Gauge(
            "analysis_queue_depth",
            "Number of items in the processing queue",
            ["priority"],
            registry=self.registry,
        )

        self.active_workers = Gauge(
            "analysis_active_workers",
            "Number of active worker threads",
            registry=self.registry,
        )

        # Batch processing metrics
        self.batch_size = Histogram(
            "analysis_batch_size",
            "Size of processed batches",
            buckets=(1, 5, 10, 25, 50, 100, 250, 500),
            registry=self.registry,
        )

        self.batch_processing_time = Histogram(
            "analysis_batch_processing_seconds",
            "Time spent processing batches",
            ["batch_size_category"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self.registry,
        )

        # Throughput metrics
        self.throughput_rate = Gauge(
            "analysis_throughput_files_per_minute",
            "Current throughput rate in files per minute",
            registry=self.registry,
        )

        # External service metrics
        self.external_service_calls = Counter(
            "analysis_external_service_calls_total",
            "Total number of external service calls",
            ["service", "status"],
            registry=self.registry,
        )

        self.external_service_latency = Histogram(
            "analysis_external_service_latency_seconds",
            "Latency of external service calls",
            ["service"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            "analysis_circuit_breaker_state",
            "Current state of circuit breaker (0=closed, 1=open, 2=half-open)",
            ["service"],
            registry=self.registry,
        )

        self.circuit_breaker_trips = Counter(
            "analysis_circuit_breaker_trips_total",
            "Total number of circuit breaker trips",
            ["service"],
            registry=self.registry,
        )

        # Resource utilization
        self.memory_usage = Gauge(
            "analysis_memory_usage_bytes",
            "Current memory usage in bytes",
            registry=self.registry,
        )

        # Track throughput
        self._throughput_window: list[tuple[float, int]] = []
        self._throughput_window_size = 60  # 1 minute window

        logger.info("Metrics collector initialized")

    @contextmanager
    def track_processing_time(self, file_type: str, status: str = "unknown") -> Generator[dict[str, Any]]:
        """
        Context manager to track file processing time.

        Args:
            file_type: Type of file being processed (e.g., 'flac', 'wav', 'mp3')
            status: Final status of processing

        Yields:
            Dictionary to store processing metadata
        """
        start_time = time.time()
        context: dict[str, Any] = {"file_type": file_type, "status": status}

        try:
            yield context
            status = context.get("status", "success")
        except Exception as e:
            status = "error"
            self.processing_errors.labels(error_type=type(e).__name__, file_type=file_type).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.processing_time.labels(file_type=file_type, status=status).observe(duration)
            self.files_processed.labels(file_type=file_type, status=status).inc()

            # Update throughput tracking
            self._update_throughput()

            logger.debug(
                "Tracked processing time",
                file_type=file_type,
                status=status,
                duration=duration,
            )

    @contextmanager
    def track_batch_processing(self, batch_size: int) -> Generator[None]:
        """
        Context manager to track batch processing metrics.

        Args:
            batch_size: Number of items in the batch
        """
        start_time = time.time()

        # Categorize batch size
        if batch_size <= 10:
            category = "small"
        elif batch_size <= 50:
            category = "medium"
        elif batch_size <= 100:
            category = "large"
        else:
            category = "xlarge"

        self.batch_size.observe(batch_size)

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.batch_processing_time.labels(batch_size_category=category).observe(duration)

            logger.debug(
                "Tracked batch processing",
                batch_size=batch_size,
                category=category,
                duration=duration,
            )

    def update_queue_depth(self, priority: str, depth: int) -> None:
        """
        Update the queue depth metric.

        Args:
            priority: Priority level (e.g., 'high', 'normal', 'low')
            depth: Current number of items in queue
        """
        self.queue_depth.labels(priority=priority).set(depth)
        logger.debug("Updated queue depth", priority=priority, depth=depth)

    def update_active_workers(self, count: int) -> None:
        """
        Update the number of active workers.

        Args:
            count: Number of active workers
        """
        self.active_workers.set(count)
        logger.debug("Updated active workers", count=count)

    def track_external_service_call(self, service: str, success: bool, latency: float) -> None:
        """
        Track an external service call.

        Args:
            service: Name of the external service
            success: Whether the call was successful
            latency: Call latency in seconds
        """
        status = "success" if success else "failure"
        self.external_service_calls.labels(service=service, status=status).inc()
        self.external_service_latency.labels(service=service).observe(latency)

        logger.debug(
            "Tracked external service call",
            service=service,
            status=status,
            latency=latency,
        )

    def update_circuit_breaker_state(self, service: str, state: str) -> None:
        """
        Update circuit breaker state metric.

        Args:
            service: Name of the service
            state: Circuit breaker state ('closed', 'open', 'half-open')
        """
        state_values = {"closed": 0, "open": 1, "half-open": 2}
        state_value = state_values.get(state, -1)
        self.circuit_breaker_state.labels(service=service).set(state_value)

        if state == "open":
            self.circuit_breaker_trips.labels(service=service).inc()

        logger.info("Updated circuit breaker state", service=service, state=state)

    def update_memory_usage(self, bytes_used: int) -> None:
        """
        Update memory usage metric.

        Args:
            bytes_used: Current memory usage in bytes
        """
        self.memory_usage.set(bytes_used)

    def _update_throughput(self) -> None:
        """Update throughput rate calculation."""
        current_time = time.time()

        # Add current file to window
        self._throughput_window.append((current_time, 1))

        # Remove old entries (older than window size)
        cutoff_time = current_time - self._throughput_window_size
        self._throughput_window = [(t, count) for t, count in self._throughput_window if t > cutoff_time]

        # Calculate throughput (files per minute)
        if self._throughput_window:
            total_files = sum(count for _, count in self._throughput_window)
            time_span = current_time - self._throughput_window[0][0]
            if time_span > 0:
                throughput = (total_files / time_span) * 60
                self.throughput_rate.set(throughput)

    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format.

        Returns:
            Metrics data in Prometheus text format
        """
        return bytes(generate_latest(self.registry))


class MetricsServer:
    """Flask server to expose metrics endpoint."""

    def __init__(self, collector: MetricsCollector, port: int = 8000) -> None:
        """
        Initialize the metrics server.

        Args:
            collector: MetricsCollector instance
            port: Port to serve metrics on
        """
        self.collector = collector
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up Flask routes for metrics endpoints."""

        @self.app.route("/metrics")
        def metrics() -> Response:
            """Expose metrics in Prometheus format."""
            return Response(self.collector.get_metrics(), mimetype=CONTENT_TYPE_LATEST)

        @self.app.route("/health")
        def health() -> dict[str, str]:
            """Basic health check endpoint."""
            return {"status": "healthy"}

    def run(self, host: str = "0.0.0.0") -> None:
        """
        Run the metrics server.

        Args:
            host: Host to bind to
        """
        logger.info(f"Starting metrics server on {host}:{self.port}")
        self.app.run(host=host, port=self.port)


# Global metrics instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        The global MetricsCollector instance
    """
    global _metrics_collector  # noqa: PLW0603 - Standard singleton pattern for global metrics collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (mainly for testing)."""
    global _metrics_collector  # noqa: PLW0603 - Standard singleton reset pattern for testing
    _metrics_collector = None
