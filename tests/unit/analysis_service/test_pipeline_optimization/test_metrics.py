"""
Unit tests for the metrics collection module.
"""

import time
from unittest.mock import Mock, patch

import pytest
from prometheus_client import CollectorRegistry

from services.analysis_service.src.metrics import (
    MetricsCollector,
    MetricsServer,
    get_metrics_collector,
    reset_metrics_collector,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Use a separate registry for each test to avoid conflicts
        self.registry = CollectorRegistry()
        self.collector = MetricsCollector(registry=self.registry)

    def test_initialization(self) -> None:
        """Test metrics collector initialization."""
        assert self.collector.registry == self.registry
        assert self.collector.processing_time is not None
        assert self.collector.files_processed is not None
        assert self.collector.processing_errors is not None
        assert self.collector.queue_depth is not None
        assert self.collector.active_workers is not None
        assert self.collector.batch_size is not None
        assert self.collector.batch_processing_time is not None
        assert self.collector.throughput_rate is not None
        assert self.collector.external_service_calls is not None
        assert self.collector.external_service_latency is not None
        assert self.collector.circuit_breaker_state is not None
        assert self.collector.circuit_breaker_trips is not None
        assert self.collector.memory_usage is not None

    def test_track_processing_time_success(self) -> None:
        """Test tracking successful file processing."""
        with self.collector.track_processing_time("flac", "success") as context:
            context["status"] = "success"
            time.sleep(0.01)  # Simulate processing

        # Check that metrics were recorded
        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_processing_seconds_bucket" in metrics_output
        assert "analysis_files_processed_total" in metrics_output
        assert 'file_type="flac"' in metrics_output
        assert 'status="success"' in metrics_output

    def test_track_processing_time_error(self) -> None:
        """Test tracking failed file processing."""
        with pytest.raises(ValueError):
            with self.collector.track_processing_time("wav", "processing"):
                raise ValueError("Processing failed")

        # Check that error metrics were recorded
        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_processing_errors_total" in metrics_output
        assert 'error_type="ValueError"' in metrics_output
        assert 'file_type="wav"' in metrics_output

    def test_track_batch_processing(self) -> None:
        """Test tracking batch processing metrics."""
        test_cases = [
            (5, "small"),
            (25, "medium"),
            (75, "large"),
            (150, "xlarge"),
        ]

        for batch_size, _expected_category in test_cases:
            with self.collector.track_batch_processing(batch_size):
                time.sleep(0.01)  # Simulate processing

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_batch_size_bucket" in metrics_output
        assert "analysis_batch_processing_seconds_bucket" in metrics_output
        for _, category in test_cases:
            assert f'batch_size_category="{category}"' in metrics_output

    def test_update_queue_depth(self) -> None:
        """Test updating queue depth metrics."""
        self.collector.update_queue_depth("high", 10)
        self.collector.update_queue_depth("normal", 50)
        self.collector.update_queue_depth("low", 100)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_queue_depth" in metrics_output
        assert 'priority="high"' in metrics_output
        assert 'priority="normal"' in metrics_output
        assert 'priority="low"' in metrics_output

    def test_update_active_workers(self) -> None:
        """Test updating active workers metric."""
        self.collector.update_active_workers(5)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_active_workers 5.0" in metrics_output

    def test_track_external_service_call(self) -> None:
        """Test tracking external service calls."""
        self.collector.track_external_service_call("neo4j", True, 0.150)
        self.collector.track_external_service_call("postgres", False, 1.500)

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_external_service_calls_total" in metrics_output
        assert 'service="neo4j"' in metrics_output
        assert 'service="postgres"' in metrics_output
        assert 'status="success"' in metrics_output
        assert 'status="failure"' in metrics_output
        assert "analysis_external_service_latency_seconds" in metrics_output

    def test_update_circuit_breaker_state(self) -> None:
        """Test updating circuit breaker state metrics."""
        self.collector.update_circuit_breaker_state("redis", "closed")
        self.collector.update_circuit_breaker_state("neo4j", "open")
        self.collector.update_circuit_breaker_state("postgres", "half-open")

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_circuit_breaker_state" in metrics_output
        assert 'service="redis"' in metrics_output
        assert 'service="neo4j"' in metrics_output
        assert 'service="postgres"' in metrics_output
        # Open state should increment trips counter
        assert "analysis_circuit_breaker_trips_total" in metrics_output

    def test_update_memory_usage(self) -> None:
        """Test updating memory usage metric."""
        self.collector.update_memory_usage(1024 * 1024 * 100)  # 100MB

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_memory_usage_bytes" in metrics_output
        assert "1.048576e+08" in metrics_output or "104857600" in metrics_output

    def test_throughput_calculation(self) -> None:
        """Test throughput rate calculation."""
        # Process multiple files to establish throughput
        for _ in range(10):
            with self.collector.track_processing_time("mp3", "success"):
                pass
            time.sleep(0.1)  # Simulate time between files

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_throughput_files_per_minute" in metrics_output

    def test_throughput_window_cleanup(self) -> None:
        """Test that old entries are removed from throughput window."""
        # Add some entries
        for _i in range(5):
            with self.collector.track_processing_time("flac", "success"):
                pass

        # Simulate time passing beyond window
        current_time = time.time()
        self.collector._throughput_window = [
            (current_time - 120, 1),  # Old entry
            (current_time - 30, 1),  # Recent entry
        ]

        # Process new file to trigger cleanup
        with self.collector.track_processing_time("flac", "success"):
            pass

        # Check that old entry was removed
        assert all(t > current_time - 65 for t, _ in self.collector._throughput_window)

    def test_context_manager_updates_status(self) -> None:
        """Test that context manager properly updates status from context."""
        with self.collector.track_processing_time("wav", "processing") as context:
            # Simulate changing status during processing
            context["status"] = "cached"

        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert 'status="cached"' in metrics_output

    def test_get_metrics_returns_bytes(self) -> None:
        """Test that get_metrics returns bytes in Prometheus format."""
        metrics_data = self.collector.get_metrics()
        assert isinstance(metrics_data, bytes)

        # Decode and check format
        metrics_text = metrics_data.decode("utf-8")
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text


class TestMetricsServer:
    """Tests for MetricsServer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.registry = CollectorRegistry()
        self.collector = MetricsCollector(registry=self.registry)
        self.server = MetricsServer(self.collector, port=8001)

    def test_initialization(self) -> None:
        """Test metrics server initialization."""
        assert self.server.collector == self.collector
        assert self.server.port == 8001
        assert self.server.app is not None

    def test_metrics_endpoint(self) -> None:
        """Test /metrics endpoint."""
        with self.server.app.test_client() as client:
            # Add some metrics
            self.collector.update_active_workers(3)

            # Request metrics
            response = client.get("/metrics")
            assert response.status_code == 200
            assert response.content_type.startswith("text/plain")

            # Check content
            data = response.data.decode("utf-8")
            assert "analysis_active_workers 3.0" in data

    def test_health_endpoint(self) -> None:
        """Test /health endpoint."""
        with self.server.app.test_client() as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json == {"status": "healthy"}

    @patch("services.analysis_service.src.metrics.Flask.run")
    def test_run_server(self, mock_run: Mock) -> None:
        """Test running the metrics server."""
        self.server.run(host="127.0.0.1")
        mock_run.assert_called_once_with(host="127.0.0.1", port=8001)


class TestGlobalMetricsCollector:
    """Tests for global metrics collector functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_metrics_collector()

    def test_get_metrics_collector_creates_instance(self) -> None:
        """Test that get_metrics_collector creates a singleton instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is not None
        assert collector1 is collector2  # Should be same instance

    def test_reset_metrics_collector(self) -> None:
        """Test resetting the global metrics collector."""
        collector1 = get_metrics_collector()
        reset_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is not collector2  # Should be different instances


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.registry = CollectorRegistry()
        self.collector = MetricsCollector(registry=self.registry)

    def test_complex_processing_scenario(self) -> None:
        """Test a complex processing scenario with multiple metrics."""
        # Update queue depths
        self.collector.update_queue_depth("high", 5)
        self.collector.update_queue_depth("normal", 20)

        # Set active workers
        self.collector.update_active_workers(4)

        # Process a batch
        with self.collector.track_batch_processing(25):
            # Process individual files
            for i in range(25):
                file_type = "flac" if i % 2 == 0 else "wav"
                try:
                    with self.collector.track_processing_time(file_type, "processing"):
                        # Simulate external service call
                        self.collector.track_external_service_call("postgres", True, 0.05)
                        if i == 10:
                            raise ValueError("Simulated error")
                except ValueError:
                    pass  # Error is tracked by context manager

        # Check circuit breaker
        self.collector.update_circuit_breaker_state("redis", "closed")

        # Update memory
        self.collector.update_memory_usage(500 * 1024 * 1024)  # 500MB

        # Get all metrics
        metrics_output = self.collector.get_metrics().decode("utf-8")

        # Verify various metrics are present
        assert "analysis_queue_depth" in metrics_output
        assert "analysis_active_workers 4.0" in metrics_output
        assert "analysis_batch_size" in metrics_output
        assert "analysis_files_processed_total" in metrics_output
        assert "analysis_processing_errors_total" in metrics_output
        assert "analysis_external_service_calls_total" in metrics_output
        assert "analysis_circuit_breaker_state" in metrics_output
        assert "analysis_memory_usage_bytes" in metrics_output

    def test_metric_labels_consistency(self) -> None:
        """Test that metric labels are consistent and valid."""
        # Process files with various labels
        file_types = ["flac", "wav", "mp3", "ogg"]
        statuses = ["success", "error", "cached", "skipped"]

        for file_type in file_types:
            for status in statuses[:2]:  # Only test success and error
                if status == "error":
                    with pytest.raises(RuntimeError):
                        with self.collector.track_processing_time(file_type, "processing"):
                            raise RuntimeError("Test error")
                else:
                    with self.collector.track_processing_time(file_type, status):
                        pass

        metrics_output = self.collector.get_metrics().decode("utf-8")

        # Check that all file types are present
        for file_type in file_types:
            assert f'file_type="{file_type}"' in metrics_output

    def test_concurrent_metric_updates(self) -> None:
        """Test that metrics can be updated concurrently (thread-safety)."""
        import threading

        def update_metrics() -> None:
            for _ in range(10):
                with self.collector.track_processing_time("mp3", "success"):
                    self.collector.update_active_workers(3)
                    self.collector.track_external_service_call("neo4j", True, 0.1)

        # Create multiple threads
        threads = [threading.Thread(target=update_metrics) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify metrics were collected
        metrics_output = self.collector.get_metrics().decode("utf-8")
        assert "analysis_files_processed_total" in metrics_output
        assert "analysis_external_service_calls_total" in metrics_output
