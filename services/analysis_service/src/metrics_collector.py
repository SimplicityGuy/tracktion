"""Metrics collection and exposure for the analysis service using Prometheus."""

import logging
import time
from typing import Any, Callable, Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    push_to_gateway,
)

logger = logging.getLogger(__name__)

# Default buckets for processing time histogram (in seconds)
DEFAULT_TIME_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

# Default buckets for file size histogram (in MB)
DEFAULT_SIZE_BUCKETS = (0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)


class MetricsCollector:
    """Collects and exposes metrics for the analysis service."""

    def __init__(
        self,
        service_name: str = "analysis_service",
        namespace: str = "tracktion",
        registry: Optional[CollectorRegistry] = None,
        push_gateway_url: Optional[str] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            service_name: Name of the service for metric labeling
            namespace: Namespace for metric names
            registry: Prometheus collector registry (creates new if None)
            push_gateway_url: Optional URL for Prometheus push gateway
        """
        self.service_name = service_name
        self.namespace = namespace
        self.registry = registry or CollectorRegistry()
        self.push_gateway_url = push_gateway_url

        # Initialize metrics
        self._initialize_metrics()

        logger.info(f"Metrics collector initialized for {service_name}")

    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        # Processing metrics
        self.files_processed_total = Counter(
            name="files_processed_total",
            documentation="Total number of files processed",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["status", "file_format"],
            registry=self.registry,
        )

        self.processing_time_seconds = Histogram(
            name="processing_time_seconds",
            documentation="Time spent processing files in seconds",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["operation", "status"],
            buckets=DEFAULT_TIME_BUCKETS,
            registry=self.registry,
        )

        self.file_size_mb = Histogram(
            name="file_size_mb",
            documentation="Size of processed files in megabytes",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["file_format"],
            buckets=DEFAULT_SIZE_BUCKETS,
            registry=self.registry,
        )

        # Queue metrics
        self.queue_depth = Gauge(
            name="queue_depth",
            documentation="Current depth of the processing queue",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["queue_name"],
            registry=self.registry,
        )

        self.active_processing = Gauge(
            name="active_processing",
            documentation="Number of files currently being processed",
            namespace=self.namespace,
            subsystem=self.service_name,
            registry=self.registry,
        )

        # Analysis-specific metrics
        self.bpm_detection_accuracy = Histogram(
            name="bpm_detection_accuracy",
            documentation="BPM detection confidence scores",
            namespace=self.namespace,
            subsystem=self.service_name,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )

        self.key_detection_accuracy = Histogram(
            name="key_detection_accuracy",
            documentation="Key detection confidence scores",
            namespace=self.namespace,
            subsystem=self.service_name,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )

        self.mood_analysis_accuracy = Histogram(
            name="mood_analysis_accuracy",
            documentation="Mood analysis confidence scores",
            namespace=self.namespace,
            subsystem=self.service_name,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            name="errors_total",
            documentation="Total number of errors encountered",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["error_type", "operation"],
            registry=self.registry,
        )

        self.retries_total = Counter(
            name="retries_total",
            documentation="Total number of retry attempts",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["operation"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits_total = Counter(
            name="cache_hits_total",
            documentation="Total number of cache hits",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["cache_type"],
            registry=self.registry,
        )

        self.cache_misses_total = Counter(
            name="cache_misses_total",
            documentation="Total number of cache misses",
            namespace=self.namespace,
            subsystem=self.service_name,
            labelnames=["cache_type"],
            registry=self.registry,
        )

        # Resource metrics
        self.worker_pool_size = Gauge(
            name="worker_pool_size",
            documentation="Current size of the worker pool",
            namespace=self.namespace,
            subsystem=self.service_name,
            registry=self.registry,
        )

        self.memory_usage_mb = Gauge(
            name="memory_usage_mb",
            documentation="Current memory usage in megabytes",
            namespace=self.namespace,
            subsystem=self.service_name,
            registry=self.registry,
        )

    def record_file_processed(
        self,
        status: str,
        file_format: str,
        processing_time: float,
        file_size_mb: Optional[float] = None,
    ) -> None:
        """Record metrics for a processed file.

        Args:
            status: Processing status (success, failed, skipped)
            file_format: Format of the file (mp3, flac, wav, etc.)
            processing_time: Time taken to process in seconds
            file_size_mb: Size of the file in megabytes
        """
        # Increment counter
        self.files_processed_total.labels(status=status, file_format=file_format).inc()

        # Record processing time
        self.processing_time_seconds.labels(operation="file_processing", status=status).observe(processing_time)

        # Record file size if provided
        if file_size_mb is not None:
            self.file_size_mb.labels(file_format=file_format).observe(file_size_mb)

    def record_operation_time(self, operation: str, status: str, duration_seconds: float) -> None:
        """Record time for a specific operation.

        Args:
            operation: Name of the operation (bpm_detection, key_detection, etc.)
            status: Status of the operation (success, failed)
            duration_seconds: Duration in seconds
        """
        self.processing_time_seconds.labels(operation=operation, status=status).observe(duration_seconds)

    def update_queue_depth(self, queue_name: str, depth: int) -> None:
        """Update the queue depth metric.

        Args:
            queue_name: Name of the queue
            depth: Current queue depth
        """
        self.queue_depth.labels(queue_name=queue_name).set(depth)

    def update_active_processing(self, count: int) -> None:
        """Update the number of actively processing files.

        Args:
            count: Number of files being processed
        """
        self.active_processing.set(count)

    def record_analysis_confidence(self, analysis_type: str, confidence: float) -> None:
        """Record confidence score for an analysis operation.

        Args:
            analysis_type: Type of analysis (bpm, key, mood)
            confidence: Confidence score (0.0 to 1.0)
        """
        if analysis_type == "bpm":
            self.bpm_detection_accuracy.observe(confidence)
        elif analysis_type == "key":
            self.key_detection_accuracy.observe(confidence)
        elif analysis_type == "mood":
            self.mood_analysis_accuracy.observe(confidence)
        else:
            logger.warning(f"Unknown analysis type for confidence metric: {analysis_type}")

    def record_error(self, error_type: str, operation: str) -> None:
        """Record an error occurrence.

        Args:
            error_type: Type of error (timeout, connection, processing, etc.)
            operation: Operation during which error occurred
        """
        self.errors_total.labels(error_type=error_type, operation=operation).inc()

    def record_retry(self, operation: str) -> None:
        """Record a retry attempt.

        Args:
            operation: Operation being retried
        """
        self.retries_total.labels(operation=operation).inc()

    def record_cache_access(self, cache_type: str, hit: bool) -> None:
        """Record cache access.

        Args:
            cache_type: Type of cache (bpm, key, mood, temporal)
            hit: Whether it was a cache hit or miss
        """
        if hit:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses_total.labels(cache_type=cache_type).inc()

    def update_worker_pool_size(self, size: int) -> None:
        """Update the worker pool size metric.

        Args:
            size: Current size of the worker pool
        """
        self.worker_pool_size.set(size)

    def update_memory_usage(self, memory_mb: float) -> None:
        """Update memory usage metric.

        Args:
            memory_mb: Current memory usage in megabytes
        """
        self.memory_usage_mb.set(memory_mb)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return bytes(generate_latest(self.registry))

    def push_metrics(self, job: str = "analysis_service") -> None:
        """Push metrics to Prometheus push gateway.

        Args:
            job: Job name for push gateway
        """
        if not self.push_gateway_url:
            logger.warning("Push gateway URL not configured, skipping metrics push")
            return

        try:
            push_to_gateway(
                self.push_gateway_url,
                job=job,
                registry=self.registry,
            )
            logger.debug(f"Metrics pushed to gateway: {self.push_gateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")


class MetricsTimer:
    """Context manager for timing operations and recording metrics."""

    def __init__(
        self,
        metrics: MetricsCollector,
        operation: str,
        record_confidence: bool = False,
        analysis_type: Optional[str] = None,
    ) -> None:
        """Initialize the metrics timer.

        Args:
            metrics: MetricsCollector instance
            operation: Name of the operation being timed
            record_confidence: Whether to record confidence score
            analysis_type: Type of analysis for confidence recording
        """
        self.metrics = metrics
        self.operation = operation
        self.record_confidence = record_confidence
        self.analysis_type = analysis_type
        self.start_time: float = 0
        self.status = "success"
        self.confidence: Optional[float] = None

    def __enter__(self) -> "MetricsTimer":
        """Start timing the operation."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record metrics for the operation."""
        duration = time.perf_counter() - self.start_time

        # Determine status based on exception
        if exc_type:
            self.status = "failed"
            # Record error
            error_type = exc_type.__name__ if exc_type else "unknown"
            self.metrics.record_error(error_type, self.operation)

        # Record operation time
        self.metrics.record_operation_time(self.operation, self.status, duration)

        # Record confidence if applicable
        if self.record_confidence and self.confidence is not None and self.analysis_type:
            self.metrics.record_analysis_confidence(self.analysis_type, self.confidence)

    def set_confidence(self, confidence: float) -> None:
        """Set the confidence score for the operation.

        Args:
            confidence: Confidence score (0.0 to 1.0)
        """
        self.confidence = confidence


def create_metrics_endpoint(
    metrics_collector: MetricsCollector,
) -> Callable[[], tuple[bytes, int, Dict[str, str]]]:
    """Create a metrics endpoint handler.

    Args:
        metrics_collector: MetricsCollector instance

    Returns:
        Handler function that returns metrics response
    """

    def metrics_handler() -> tuple[bytes, int, Dict[str, str]]:
        """Handle metrics endpoint request.

        Returns:
            Tuple of (body, status_code, headers)
        """
        try:
            metrics_data = metrics_collector.get_metrics()
            return (
                metrics_data,
                200,
                {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            error_message = f"# Error generating metrics: {e}\n"
            return (
                error_message.encode("utf-8"),
                500,
                {"Content-Type": "text/plain; charset=utf-8"},
            )

    return metrics_handler
