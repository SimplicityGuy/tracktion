"""Unit tests for metrics collection and exposure."""

import time
import unittest
from unittest.mock import MagicMock, patch

from prometheus_client import CollectorRegistry

from services.analysis_service.src.metrics_collector import MetricsCollector, MetricsTimer, create_metrics_endpoint


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.registry = CollectorRegistry()
        self.collector = MetricsCollector(
            service_name="analysis_service",
            namespace="tracktion",
            registry=self.registry,
        )

    def test_initialization(self) -> None:
        """Test metrics collector initialization."""
        self.assertEqual(self.collector.service_name, "analysis_service")
        self.assertEqual(self.collector.namespace, "tracktion")
        self.assertIsNotNone(self.collector.files_processed_total)
        self.assertIsNotNone(self.collector.processing_time_seconds)
        self.assertIsNotNone(self.collector.queue_depth)

    def test_record_file_processed(self) -> None:
        """Test recording file processing metrics."""
        # Record successful processing
        self.collector.record_file_processed(
            status="success",
            file_format="mp3",
            processing_time=2.5,
            file_size_mb=10.5,
        )

        # Get metrics and verify
        metrics_output = self.collector.get_metrics().decode("utf-8")

        # Check that metrics were recorded
        self.assertIn("tracktion_analysis_service_files_processed_total", metrics_output)
        self.assertIn('status="success"', metrics_output)
        self.assertIn('file_format="mp3"', metrics_output)
        self.assertIn("tracktion_analysis_service_processing_time_seconds", metrics_output)
        self.assertIn("tracktion_analysis_service_file_size_mb", metrics_output)

    def test_record_operation_time(self) -> None:
        """Test recording operation time metrics."""
        self.collector.record_operation_time(
            operation="bpm_detection",
            status="success",
            duration_seconds=1.5,
        )

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_processing_time_seconds", metrics_output)
        self.assertIn('operation="bpm_detection"', metrics_output)

    def test_update_queue_depth(self) -> None:
        """Test updating queue depth metric."""
        self.collector.update_queue_depth("analysis_queue", 42)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_queue_depth", metrics_output)
        self.assertIn('queue_name="analysis_queue"', metrics_output)
        self.assertIn(" 42", metrics_output)  # The value should be 42

    def test_update_active_processing(self) -> None:
        """Test updating active processing count."""
        self.collector.update_active_processing(5)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_active_processing", metrics_output)
        self.assertIn(" 5", metrics_output)

    def test_record_analysis_confidence(self) -> None:
        """Test recording analysis confidence scores."""
        # Test BPM confidence
        self.collector.record_analysis_confidence("bpm", 0.95)

        # Test key confidence
        self.collector.record_analysis_confidence("key", 0.87)

        # Test mood confidence
        self.collector.record_analysis_confidence("mood", 0.72)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_bpm_detection_accuracy", metrics_output)
        self.assertIn("tracktion_analysis_service_key_detection_accuracy", metrics_output)
        self.assertIn("tracktion_analysis_service_mood_analysis_accuracy", metrics_output)

    def test_record_analysis_confidence_unknown_type(self) -> None:
        """Test recording confidence with unknown analysis type."""
        # Should log warning but not raise error
        self.collector.record_analysis_confidence("unknown", 0.5)

        # Should complete without exception
        metrics_output = self.collector.get_metrics()
        self.assertIsNotNone(metrics_output)

    def test_record_error(self) -> None:
        """Test recording error metrics."""
        self.collector.record_error("timeout", "file_processing")
        self.collector.record_error("connection", "rabbitmq")

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_errors_total", metrics_output)
        self.assertIn('error_type="timeout"', metrics_output)
        self.assertIn('operation="file_processing"', metrics_output)

    def test_record_retry(self) -> None:
        """Test recording retry metrics."""
        self.collector.record_retry("message_processing")

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_retries_total", metrics_output)
        self.assertIn('operation="message_processing"', metrics_output)

    def test_record_cache_access(self) -> None:
        """Test recording cache access metrics."""
        # Record cache hit
        self.collector.record_cache_access("bpm", hit=True)

        # Record cache miss
        self.collector.record_cache_access("key", hit=False)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_cache_hits_total", metrics_output)
        self.assertIn('cache_type="bpm"', metrics_output)
        self.assertIn("tracktion_analysis_service_cache_misses_total", metrics_output)
        self.assertIn('cache_type="key"', metrics_output)

    def test_update_worker_pool_size(self) -> None:
        """Test updating worker pool size metric."""
        self.collector.update_worker_pool_size(8)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_worker_pool_size", metrics_output)
        self.assertIn(" 8", metrics_output)

    def test_update_memory_usage(self) -> None:
        """Test updating memory usage metric."""
        self.collector.update_memory_usage(256.5)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_memory_usage_mb", metrics_output)
        self.assertIn(" 256.5", metrics_output)

    @patch("services.analysis_service.src.metrics_collector.push_to_gateway")
    def test_push_metrics_success(self, mock_push: MagicMock) -> None:
        """Test pushing metrics to gateway successfully."""
        collector = MetricsCollector(
            service_name="analysis_service",
            push_gateway_url="http://localhost:9091",
        )

        collector.push_metrics(job="test_job")

        mock_push.assert_called_once_with(
            "http://localhost:9091",
            job="test_job",
            registry=collector.registry,
        )

    def test_push_metrics_no_url(self) -> None:
        """Test pushing metrics without gateway URL configured."""
        # Should not raise error, just log warning
        self.collector.push_metrics(job="test_job")

        # No exception should be raised
        self.assertTrue(True)

    @patch("services.analysis_service.src.metrics_collector.push_to_gateway")
    def test_push_metrics_failure(self, mock_push: MagicMock) -> None:
        """Test handling push gateway failure."""
        mock_push.side_effect = Exception("Connection failed")

        collector = MetricsCollector(
            service_name="analysis_service",
            push_gateway_url="http://localhost:9091",
        )

        # Should not raise error, just log it
        collector.push_metrics(job="test_job")

        # Verify push was attempted
        mock_push.assert_called_once()


class TestMetricsTimer(unittest.TestCase):
    """Test MetricsTimer context manager."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.collector = MetricsCollector(
            service_name="analysis_service",
            namespace="tracktion",
        )

    def test_successful_operation(self) -> None:
        """Test timing a successful operation."""
        with MetricsTimer(self.collector, "test_operation") as timer:
            time.sleep(0.01)  # Simulate some work
            timer.status = "success"

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_processing_time_seconds", metrics_output)
        self.assertIn('operation="test_operation"', metrics_output)
        self.assertIn('status="success"', metrics_output)

    def test_failed_operation(self) -> None:
        """Test timing a failed operation."""
        try:
            with MetricsTimer(self.collector, "test_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        metrics_output = self.collector.get_metrics().decode("utf-8")

        # Check that error was recorded
        self.assertIn("tracktion_analysis_service_errors_total", metrics_output)
        self.assertIn('error_type="ValueError"', metrics_output)

        # Check that operation was recorded as failed
        self.assertIn('status="failed"', metrics_output)

    def test_operation_with_confidence(self) -> None:
        """Test recording operation with confidence score."""
        with MetricsTimer(
            self.collector,
            "bpm_detection",
            record_confidence=True,
            analysis_type="bpm",
        ) as timer:
            timer.set_confidence(0.92)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        self.assertIn("tracktion_analysis_service_bpm_detection_accuracy", metrics_output)

    def test_operation_without_confidence(self) -> None:
        """Test operation without confidence recording."""
        with MetricsTimer(self.collector, "test_operation") as timer:
            timer.set_confidence(0.5)  # Set but shouldn't be recorded

        metrics_output = self.collector.get_metrics().decode("utf-8")
        # Should have operation time but accuracy metrics should still be at 0 (not incremented)
        self.assertIn("tracktion_analysis_service_processing_time_seconds", metrics_output)
        # Check that accuracy metrics exist but weren't incremented (count = 0)
        self.assertIn(
            "tracktion_analysis_service_bpm_detection_accuracy_count 0.0",
            metrics_output,
        )


class TestMetricsEndpoint(unittest.TestCase):
    """Test metrics endpoint creation."""

    def test_create_metrics_endpoint_success(self) -> None:
        """Test creating and calling metrics endpoint successfully."""
        collector = MetricsCollector(service_name="analysis_service")

        # Record some metrics
        collector.record_file_processed("success", "mp3", 1.0)

        # Create endpoint handler
        handler = create_metrics_endpoint(collector)

        # Call handler
        body, status_code, headers = handler()

        self.assertEqual(status_code, 200)
        self.assertEqual(headers["Content-Type"], "text/plain; version=0.0.4; charset=utf-8")
        self.assertIn(b"tracktion_analysis_service_files_processed_total", body)

    @patch("services.analysis_service.src.metrics_collector.MetricsCollector.get_metrics")
    def test_create_metrics_endpoint_error(self, mock_get_metrics: MagicMock) -> None:
        """Test metrics endpoint error handling."""
        mock_get_metrics.side_effect = Exception("Metrics generation failed")

        collector = MetricsCollector(service_name="analysis_service")
        handler = create_metrics_endpoint(collector)

        # Call handler
        body, status_code, _headers = handler()

        self.assertEqual(status_code, 500)
        self.assertIn(b"Error generating metrics", body)


if __name__ == "__main__":
    unittest.main()
