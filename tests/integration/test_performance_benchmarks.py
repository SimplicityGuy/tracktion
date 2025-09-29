"""
Comprehensive Performance Benchmarking Tests for Tracktion System.

This module provides extensive performance testing capabilities for the audio analysis
service, covering all aspects of system performance including response times, throughput,
resource utilization, database performance, message queue efficiency, and scalability.
"""

import asyncio
import contextlib
import gc
import hashlib
import json
import math
import random
import re
import statistics
import sys
import tempfile
import time
import tracemalloc
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Add service paths for imports
sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "services" / "analysis_service" / "src"),
)


from config import ServiceConfig


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Response time metrics (milliseconds)
    response_times: list[float] = field(default_factory=list)
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Throughput metrics
    requests_per_second: float = 0.0
    files_processed_per_second: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

    # Resource utilization
    cpu_usage_percent: list[float] = field(default_factory=list)
    memory_usage_mb: list[float] = field(default_factory=list)
    disk_io_read_mb: list[float] = field(default_factory=list)
    disk_io_write_mb: list[float] = field(default_factory=list)
    network_io_sent_mb: list[float] = field(default_factory=list)
    network_io_recv_mb: list[float] = field(default_factory=list)

    # Cache performance
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_response_times: list[float] = field(default_factory=list)

    # Database performance
    db_query_times: list[float] = field(default_factory=list)
    db_connection_times: list[float] = field(default_factory=list)
    db_connection_pool_usage: list[int] = field(default_factory=list)

    # Message queue performance
    queue_publish_times: list[float] = field(default_factory=list)
    queue_consume_times: list[float] = field(default_factory=list)
    queue_backlog_sizes: list[int] = field(default_factory=list)

    # System scalability
    concurrent_users: int = 0
    load_factor: float = 1.0
    saturation_point: int | None = None

    # Performance regression tracking
    baseline_comparison: dict[str, float] = field(default_factory=dict)
    regression_detected: bool = False
    performance_trend: str = "stable"  # stable, improving, degrading

    def calculate_statistics(self) -> None:
        """Calculate statistical metrics from raw data."""
        if self.response_times:
            self.min_response_time = min(self.response_times)
            self.max_response_time = max(self.response_times)
            self.avg_response_time = statistics.mean(self.response_times)
            self.p50_response_time = statistics.median(self.response_times)
            self.p95_response_time = self._percentile(self.response_times, 95)
            self.p99_response_time = self._percentile(self.response_times, 99)

        if self.total_requests > 0:
            self.error_rate = (self.failed_requests / self.total_requests) * 100

        if self.cache_hits + self.cache_misses > 0:
            self.cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100

    @staticmethod
    def _percentile(data: list[float], percentile: int) -> float:
        """Calculate the nth percentile of a dataset."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        lower_index = math.floor(index)
        upper_index = math.ceil(index)
        return sorted_data[lower_index] + (index - lower_index) * (sorted_data[upper_index] - sorted_data[lower_index])


@dataclass
class SLAThresholds:
    """Service Level Agreement thresholds for performance validation."""

    # Response time thresholds (milliseconds)
    max_avg_response_time: float = 2000.0  # 2 seconds
    max_p95_response_time: float = 5000.0  # 5 seconds
    max_p99_response_time: float = 10000.0  # 10 seconds

    # Throughput thresholds
    min_requests_per_second: float = 10.0
    min_files_per_second: float = 1.0
    max_error_rate: float = 1.0  # 1%

    # Resource utilization thresholds
    max_cpu_usage: float = 80.0  # 80%
    max_memory_usage: float = 1024.0  # 1GB

    # Cache performance thresholds
    min_cache_hit_rate: float = 85.0  # 85%
    max_cache_response_time: float = 50.0  # 50ms

    # Database performance thresholds
    max_db_query_time: float = 500.0  # 500ms
    max_db_connection_time: float = 100.0  # 100ms

    # Message queue thresholds
    max_queue_publish_time: float = 100.0  # 100ms
    max_queue_consume_time: float = 200.0  # 200ms
    max_queue_backlog: int = 1000


class ResourceMonitor:
    """Monitors system resource usage during performance tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = PerformanceMetrics()
        self._monitor_task = None
        self._start_time = None
        self._start_counters = None

    async def start_monitoring(self, interval: float = 0.1) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self._start_time = time.time()
        self._start_counters = {
            "disk_io": self.process.io_counters(),
            "net_io": psutil.net_io_counters(),
        }

        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self) -> PerformanceMetrics:
        """Stop resource monitoring and return collected metrics."""
        if not self.monitoring:
            return self.metrics

        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        return self.metrics

    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                if cpu_percent > 0:  # Skip first reading which is usually 0
                    self.metrics.cpu_usage_percent.append(cpu_percent)

                # Memory usage
                memory_info = self.process.memory_info()
                self.metrics.memory_usage_mb.append(memory_info.rss / 1024 / 1024)

                # I/O counters
                if self._start_counters:
                    try:
                        current_disk = self.process.io_counters()
                        current_net = psutil.net_io_counters()

                        # Calculate deltas since start
                        disk_read_mb = (
                            (current_disk.read_bytes - self._start_counters["disk_io"].read_bytes) / 1024 / 1024
                        )
                        disk_write_mb = (
                            (current_disk.write_bytes - self._start_counters["disk_io"].write_bytes) / 1024 / 1024
                        )
                        net_sent_mb = (current_net.bytes_sent - self._start_counters["net_io"].bytes_sent) / 1024 / 1024
                        net_recv_mb = (current_net.bytes_recv - self._start_counters["net_io"].bytes_recv) / 1024 / 1024

                        self.metrics.disk_io_read_mb.append(disk_read_mb)
                        self.metrics.disk_io_write_mb.append(disk_write_mb)
                        self.metrics.network_io_sent_mb.append(net_sent_mb)
                        self.metrics.network_io_recv_mb.append(net_recv_mb)
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass  # Skip if process terminated or access denied

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if individual metrics fail
                await asyncio.sleep(interval)


class PerformanceProfiler:
    """Profiler for detailed performance analysis."""

    def __init__(self):
        self.active_traces = {}

    @asynccontextmanager
    async def profile_memory(self, test_name: str):
        """Context manager for memory profiling."""
        tracemalloc.start()
        gc.collect()  # Clean up before measurement

        snapshot_before = tracemalloc.take_snapshot()

        try:
            yield
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Calculate memory difference
            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
            total_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

            print(f"\n{test_name} - Memory usage diff: {total_mb:.2f} MB")
            if len(top_stats) > 0:
                print("Top memory consumers:")
                for stat in top_stats[:5]:
                    print(f"  {stat.traceback.format()[-1]}: {stat.size_diff / 1024:.1f} KB")

    @asynccontextmanager
    async def profile_execution_time(self, test_name: str):
        """Context manager for execution time profiling."""
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"\n{test_name} - Execution time: {execution_time:.2f} ms")


class PerformanceBenchmarkSuite:
    """Main performance benchmarking suite."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.profiler = PerformanceProfiler()
        self.sla_thresholds = SLAThresholds()
        self.baseline_metrics = self._load_baseline_metrics()

    def _load_baseline_metrics(self) -> dict[str, float]:
        """Load baseline performance metrics for regression testing."""
        baseline_file = Path(__file__).parent / "performance_baseline.json"
        if baseline_file.exists():
            try:
                with baseline_file.open() as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _save_baseline_metrics(self, metrics: PerformanceMetrics, test_name: str) -> None:
        """Save current metrics as baseline for future comparisons."""
        baseline_file = Path(__file__).parent / "performance_baseline.json"

        baseline_data = self._load_baseline_metrics()
        baseline_data[test_name] = {
            "avg_response_time": metrics.avg_response_time,
            "p95_response_time": metrics.p95_response_time,
            "requests_per_second": metrics.requests_per_second,
            "error_rate": metrics.error_rate,
            "cache_hit_rate": metrics.cache_hit_rate,
            "timestamp": time.time(),
        }

        try:
            with baseline_file.open("w") as f:
                json.dump(baseline_data, f, indent=2)
        except OSError:
            pass  # Fail silently if we can't write baseline

    def _compare_with_baseline(self, metrics: PerformanceMetrics, test_name: str) -> None:
        """Compare current metrics with baseline and detect regressions."""
        if test_name not in self.baseline_metrics:
            return

        baseline = self.baseline_metrics[test_name]
        current = {
            "avg_response_time": metrics.avg_response_time,
            "p95_response_time": metrics.p95_response_time,
            "requests_per_second": metrics.requests_per_second,
            "error_rate": metrics.error_rate,
            "cache_hit_rate": metrics.cache_hit_rate,
        }

        regression_threshold = 0.2  # 20% worse than baseline
        improvement_threshold = 0.1  # 10% better than baseline

        regressions = []
        improvements = []

        for metric, current_value in current.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if baseline_value > 0:  # Avoid division by zero
                    change_ratio = (current_value - baseline_value) / baseline_value

                    # For response times and error rates, higher is worse
                    if metric in ["avg_response_time", "p95_response_time", "error_rate"]:
                        if change_ratio > regression_threshold:
                            regressions.append((metric, change_ratio))
                        elif change_ratio < -improvement_threshold:
                            improvements.append((metric, abs(change_ratio)))
                    # For throughput and cache hit rate, higher is better
                    elif change_ratio < -regression_threshold:
                        regressions.append((metric, abs(change_ratio)))
                    elif change_ratio > improvement_threshold:
                        improvements.append((metric, change_ratio))

        if regressions:
            metrics.regression_detected = True
            metrics.performance_trend = "degrading"
            print(f"\n⚠️  Performance regressions detected in {test_name}:")
            for metric, change in regressions:
                print(f"  - {metric}: {change * 100:.1f}% worse than baseline")
        elif improvements:
            metrics.performance_trend = "improving"
            print(f"\n✅ Performance improvements detected in {test_name}:")
            for metric, change in improvements:
                print(f"  - {metric}: {change * 100:.1f}% better than baseline")
        else:
            metrics.performance_trend = "stable"

        metrics.baseline_comparison = {f"{k}_baseline": v for k, v in baseline.items()}

    def _validate_sla(self, metrics: PerformanceMetrics) -> list[str]:
        """Validate performance metrics against SLA thresholds."""
        violations = []

        if metrics.avg_response_time > self.sla_thresholds.max_avg_response_time:
            violations.append(
                f"Average response time {metrics.avg_response_time:.1f}ms exceeds SLA "
                f"({self.sla_thresholds.max_avg_response_time:.1f}ms)"
            )

        if metrics.p95_response_time > self.sla_thresholds.max_p95_response_time:
            violations.append(
                f"P95 response time {metrics.p95_response_time:.1f}ms exceeds SLA "
                f"({self.sla_thresholds.max_p95_response_time:.1f}ms)"
            )

        if metrics.error_rate > self.sla_thresholds.max_error_rate:
            violations.append(
                f"Error rate {metrics.error_rate:.1f}% exceeds SLA ({self.sla_thresholds.max_error_rate:.1f}%)"
            )

        if metrics.requests_per_second < self.sla_thresholds.min_requests_per_second:
            violations.append(
                f"Requests per second {metrics.requests_per_second:.1f} below SLA "
                f"({self.sla_thresholds.min_requests_per_second:.1f})"
            )

        if metrics.cache_hit_rate < self.sla_thresholds.min_cache_hit_rate:
            violations.append(
                f"Cache hit rate {metrics.cache_hit_rate:.1f}% below SLA "
                f"({self.sla_thresholds.min_cache_hit_rate:.1f}%)"
            )

        return violations

    def _print_performance_report(self, metrics: PerformanceMetrics, test_name: str) -> None:
        """Print comprehensive performance report."""
        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE REPORT: {test_name}")
        print(f"{'=' * 60}")

        # Response time metrics
        print("\n📊 Response Time Metrics:")
        print(f"  Average: {metrics.avg_response_time:.1f}ms")
        print(f"  Min/Max: {metrics.min_response_time:.1f}ms / {metrics.max_response_time:.1f}ms")
        print(f"  P50 (Median): {metrics.p50_response_time:.1f}ms")
        print(f"  P95: {metrics.p95_response_time:.1f}ms")
        print(f"  P99: {metrics.p99_response_time:.1f}ms")

        # Throughput metrics
        print("\n🚀 Throughput Metrics:")
        print(f"  Requests per second: {metrics.requests_per_second:.1f}")
        print(f"  Files processed per second: {metrics.files_processed_per_second:.1f}")
        print(f"  Total requests: {metrics.total_requests}")
        print(f"  Successful requests: {metrics.successful_requests}")
        print(f"  Failed requests: {metrics.failed_requests}")
        print(f"  Error rate: {metrics.error_rate:.1f}%")

        # Resource utilization
        if metrics.cpu_usage_percent:
            print("\n💻 Resource Utilization:")
            print(f"  Average CPU usage: {statistics.mean(metrics.cpu_usage_percent):.1f}%")
            print(f"  Peak CPU usage: {max(metrics.cpu_usage_percent):.1f}%")

        if metrics.memory_usage_mb:
            print(f"  Average memory usage: {statistics.mean(metrics.memory_usage_mb):.1f} MB")
            print(f"  Peak memory usage: {max(metrics.memory_usage_mb):.1f} MB")

        # Cache performance
        if metrics.cache_hits + metrics.cache_misses > 0:
            print("\n🎯 Cache Performance:")
            print(f"  Hit rate: {metrics.cache_hit_rate:.1f}%")
            print(f"  Hits/Misses: {metrics.cache_hits}/{metrics.cache_misses}")
            if metrics.cache_response_times:
                print(f"  Average cache response time: {statistics.mean(metrics.cache_response_times):.1f}ms")

        # Database performance
        if metrics.db_query_times:
            print("\n🗄️  Database Performance:")
            print(f"  Average query time: {statistics.mean(metrics.db_query_times):.1f}ms")
            print(f"  Max query time: {max(metrics.db_query_times):.1f}ms")

        # Message queue performance
        if metrics.queue_publish_times:
            print("\n📨 Message Queue Performance:")
            print(f"  Average publish time: {statistics.mean(metrics.queue_publish_times):.1f}ms")
            print(f"  Average consume time: {statistics.mean(metrics.queue_consume_times):.1f}ms")

        # Scalability metrics
        if metrics.concurrent_users > 1:
            print("\n📈 Scalability Metrics:")
            print(f"  Concurrent users: {metrics.concurrent_users}")
            print(f"  Load factor: {metrics.load_factor:.1f}")
            if metrics.saturation_point:
                print(f"  Saturation point: {metrics.saturation_point} concurrent users")

        # Performance trend
        if metrics.performance_trend != "stable":
            trend_icon = "📈" if metrics.performance_trend == "improving" else "📉"
            print(f"\n{trend_icon} Performance Trend: {metrics.performance_trend.title()}")

        # SLA validation
        violations = self._validate_sla(metrics)
        if violations:
            print("\n❌ SLA Violations:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("\n✅ All SLA thresholds met!")

        print(f"{'=' * 60}\n")


@pytest.fixture
def performance_benchmark_suite():
    """Create performance benchmark suite with test configuration."""
    # Setup test configuration
    test_config = ServiceConfig()
    test_config.cache.enabled = True
    test_config.cache.redis_host = "localhost"
    test_config.performance.parallel_workers = 4
    test_config.performance.memory_limit_mb = 1024

    suite = PerformanceBenchmarkSuite(test_config)
    yield suite


@pytest.fixture
def mock_audio_files():
    """Create mock audio files for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "path": f"/mock/audio/file_{i:03d}.mp3",
            "size_mb": 5 + (i % 10),  # 5-15 MB files
            "duration_seconds": 180 + (i % 60),  # 3-4 minute songs
        }
        for i in range(100)
    ]


class TestServiceResponseTimeBenchmarks:
    """Test suite for service response time benchmarking."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_api_endpoint_response_times(self, performance_benchmark_suite, mock_audio_files):
        """Benchmark response times for API endpoints."""
        suite = performance_benchmark_suite

        # Mock API endpoints
        mock_endpoints = {
            "/health": self._mock_health_endpoint,
            "/analyze": self._mock_analyze_endpoint,
            "/status": self._mock_status_endpoint,
            "/metadata": self._mock_metadata_endpoint,
        }

        metrics = PerformanceMetrics()

        async with suite.profiler.profile_execution_time("API Response Times"):
            await suite.resource_monitor.start_monitoring()

            # Test each endpoint with different request patterns
            for endpoint, handler in mock_endpoints.items():
                endpoint_metrics = await self._benchmark_endpoint(
                    endpoint, handler, num_requests=100, concurrent_requests=10
                )

                # Aggregate metrics
                metrics.response_times.extend(endpoint_metrics.response_times)
                metrics.total_requests += endpoint_metrics.total_requests
                metrics.successful_requests += endpoint_metrics.successful_requests
                metrics.failed_requests += endpoint_metrics.failed_requests

            resource_metrics = await suite.resource_monitor.stop_monitoring()
            metrics.cpu_usage_percent = resource_metrics.cpu_usage_percent
            metrics.memory_usage_mb = resource_metrics.memory_usage_mb

        # Calculate final statistics
        metrics.calculate_statistics()
        elapsed_time = sum(metrics.response_times) / 1000  # Convert to seconds
        metrics.requests_per_second = metrics.total_requests / max(elapsed_time, 1)

        # Compare with baseline and validate SLA
        suite._compare_with_baseline(metrics, "api_endpoints")
        suite._print_performance_report(metrics, "API Endpoint Response Times")

        # Assertions
        assert metrics.avg_response_time < suite.sla_thresholds.max_avg_response_time
        assert metrics.p95_response_time < suite.sla_thresholds.max_p95_response_time
        assert metrics.error_rate <= suite.sla_thresholds.max_error_rate

        # Save as baseline if test passes
        suite._save_baseline_metrics(metrics, "api_endpoints")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_audio_processing_workflow_performance(self, performance_benchmark_suite, mock_audio_files):
        """Benchmark end-to-end audio processing workflow performance."""
        suite = performance_benchmark_suite

        # Mock components
        with (
            patch("metadata_extractor.MetadataExtractor") as mock_extractor,
            patch("model_manager.ModelManager") as mock_model_manager,
            patch("temporal_analyzer.TemporalAnalyzer") as mock_temporal,
            patch("storage_handler.StorageHandler") as mock_storage,
        ):
            # Configure mocks
            mock_extractor.return_value.extract_metadata = AsyncMock(
                return_value={"bpm": 120.0, "key": "C major", "duration": 180}
            )
            mock_model_manager.return_value.analyze_audio = AsyncMock(
                return_value={"mood": "happy", "genre": "electronic", "energy": 0.8}
            )
            mock_temporal.return_value.analyze = AsyncMock(
                return_value={"temporal_bpm": [119, 120, 121], "stability": 0.95}
            )
            mock_storage.return_value.store_results = AsyncMock(return_value=True)

            metrics = PerformanceMetrics()

            async with suite.profiler.profile_execution_time("Audio Processing Workflow"):
                await suite.resource_monitor.start_monitoring()

                # Process files in batches
                batch_size = 10
                total_processed = 0

                for i in range(0, len(mock_audio_files), batch_size):
                    batch = mock_audio_files[i : i + batch_size]
                    start_time = time.perf_counter()

                    # Simulate parallel processing of batch
                    tasks = []
                    for audio_file in batch:
                        task = asyncio.create_task(
                            self._process_audio_file(audio_file, mock_extractor, mock_temporal, mock_storage)
                        )
                        tasks.append(task)

                    # Wait for batch completion
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    end_time = time.perf_counter()
                    batch_time = (end_time - start_time) * 1000

                    # Record metrics
                    metrics.response_times.append(batch_time)
                    successful = sum(1 for r in results if not isinstance(r, Exception))
                    failed = len(results) - successful

                    metrics.successful_requests += successful
                    metrics.failed_requests += failed
                    total_processed += len(batch)

                metrics.total_requests = total_processed
                resource_metrics = await suite.resource_monitor.stop_monitoring()
                metrics.cpu_usage_percent = resource_metrics.cpu_usage_percent
                metrics.memory_usage_mb = resource_metrics.memory_usage_mb

            # Calculate final statistics
            metrics.calculate_statistics()
            total_time = sum(metrics.response_times) / 1000  # Convert to seconds
            metrics.files_processed_per_second = total_processed / max(total_time, 1)
            metrics.requests_per_second = metrics.files_processed_per_second

            # Performance analysis
            suite._compare_with_baseline(metrics, "audio_processing_workflow")
            suite._print_performance_report(metrics, "Audio Processing Workflow")

            # Assertions
            assert metrics.files_processed_per_second >= suite.sla_thresholds.min_files_per_second
            assert metrics.error_rate <= suite.sla_thresholds.max_error_rate
            assert max(metrics.memory_usage_mb) <= suite.sla_thresholds.max_memory_usage

            # Save baseline
            suite._save_baseline_metrics(metrics, "audio_processing_workflow")

    async def _benchmark_endpoint(
        self, endpoint: str, handler: Callable, num_requests: int = 100, concurrent_requests: int = 10
    ) -> PerformanceMetrics:
        """Benchmark a specific endpoint."""
        metrics = PerformanceMetrics()

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def make_request():
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    await handler()
                    success = True
                except Exception:
                    success = False

                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds

                return response_time, success

        # Execute requests
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        # Process results
        for response_time, success in results:
            metrics.response_times.append(response_time)
            metrics.total_requests += 1
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

        return metrics

    async def _process_audio_file(self, audio_file: dict, extractor, temporal, storage) -> dict:
        """Simulate processing a single audio file."""
        # Simulate processing delay based on file size
        processing_delay = audio_file["size_mb"] * 0.01  # 10ms per MB
        await asyncio.sleep(processing_delay)

        # Extract metadata
        metadata = await extractor.return_value.extract_metadata(audio_file["path"])

        # Temporal analysis
        temporal_data = await temporal.return_value.analyze(audio_file["path"])

        # Store results
        result = {**metadata, **temporal_data, "file_id": audio_file["id"]}
        await storage.return_value.store_results(result)

        return result

    async def _mock_health_endpoint(self):
        """Mock health check endpoint."""
        await asyncio.sleep(0.001)  # 1ms delay
        return {"status": "healthy", "timestamp": time.time()}

    async def _mock_analyze_endpoint(self):
        """Mock analysis endpoint."""
        await asyncio.sleep(0.05)  # 50ms delay
        return {"analysis_id": str(uuid.uuid4()), "status": "processing"}

    async def _mock_status_endpoint(self):
        """Mock status endpoint."""
        await asyncio.sleep(0.002)  # 2ms delay
        return {"queue_size": 10, "active_workers": 4}

    async def _mock_metadata_endpoint(self):
        """Mock metadata endpoint."""
        await asyncio.sleep(0.01)  # 10ms delay
        return {"metadata": {"bpm": 120, "key": "C major"}}


class TestThroughputBenchmarks:
    """Test suite for throughput benchmarking."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_requests_per_second_benchmark(self, performance_benchmark_suite):
        """Benchmark maximum requests per second handling capacity."""
        suite = performance_benchmark_suite

        # Test different load levels
        load_levels = [10, 50, 100, 200, 500]
        results = {}

        for rps_target in load_levels:
            print(f"\nTesting {rps_target} RPS target...")

            metrics = await self._run_throughput_test(target_rps=rps_target, duration_seconds=30, suite=suite)

            results[rps_target] = {
                "achieved_rps": metrics.requests_per_second,
                "error_rate": metrics.error_rate,
                "avg_response_time": metrics.avg_response_time,
                "p95_response_time": metrics.p95_response_time,
            }

            # Stop if error rate exceeds threshold
            if metrics.error_rate > 5.0:  # 5% error rate threshold
                print(f"Stopping throughput test at {rps_target} RPS due to high error rate")
                break

        # Analyze results to find optimal throughput
        optimal_rps = self._find_optimal_throughput(results)

        # Create summary metrics
        final_metrics = PerformanceMetrics()
        final_metrics.requests_per_second = optimal_rps

        suite._print_performance_report(final_metrics, "Throughput Benchmark")

        # Assertions
        assert optimal_rps >= suite.sla_thresholds.min_requests_per_second
        print(f"\n✅ Optimal throughput: {optimal_rps:.1f} RPS")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_file_processing_capacity(self, performance_benchmark_suite, mock_audio_files):
        """Benchmark file processing capacity."""
        suite = performance_benchmark_suite

        with patch("async_audio_processor.AsyncAudioProcessor") as mock_processor:
            # Configure processor mock
            mock_processor.return_value.process_file = AsyncMock(side_effect=self._mock_file_processing)

            processor = mock_processor.return_value

            metrics = PerformanceMetrics()

            async with suite.profiler.profile_execution_time("File Processing Capacity"):
                await suite.resource_monitor.start_monitoring()

                # Process files with different concurrency levels
                concurrency_levels = [1, 5, 10, 20]
                best_throughput = 0

                for concurrency in concurrency_levels:
                    print(f"\nTesting concurrency level: {concurrency}")

                    # Limit test files for faster execution
                    test_files = mock_audio_files[:50]

                    semaphore = asyncio.Semaphore(concurrency)

                    async def process_with_semaphore(file_data, sem=semaphore):
                        async with sem:
                            return await processor.process_file(file_data["path"])

                    # Process files
                    level_start = time.time()
                    tasks = [process_with_semaphore(f) for f in test_files]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    level_end = time.time()

                    # Calculate throughput for this level
                    level_duration = level_end - level_start
                    successful = sum(1 for r in results if not isinstance(r, Exception))
                    level_throughput = successful / level_duration

                    print(f"  Throughput: {level_throughput:.2f} files/sec")

                    # Track metrics
                    metrics.successful_requests += successful
                    metrics.failed_requests += len(results) - successful
                    metrics.total_requests += len(results)

                    best_throughput = max(best_throughput, level_throughput)

                resource_metrics = await suite.resource_monitor.stop_monitoring()
                metrics.cpu_usage_percent = resource_metrics.cpu_usage_percent
                metrics.memory_usage_mb = resource_metrics.memory_usage_mb

            # Calculate final metrics
            metrics.files_processed_per_second = best_throughput
            metrics.requests_per_second = best_throughput
            metrics.calculate_statistics()

            suite._compare_with_baseline(metrics, "file_processing_capacity")
            suite._print_performance_report(metrics, "File Processing Capacity")

            # Assertions
            assert best_throughput >= suite.sla_thresholds.min_files_per_second
            assert metrics.error_rate <= suite.sla_thresholds.max_error_rate

    async def _run_throughput_test(
        self, target_rps: int, duration_seconds: int, suite: PerformanceBenchmarkSuite
    ) -> PerformanceMetrics:
        """Run a throughput test at a specific RPS target."""
        metrics = PerformanceMetrics()
        request_interval = 1.0 / target_rps

        async def make_request():
            start_time = time.perf_counter()
            try:
                # Simulate request processing
                await asyncio.sleep(0.01 + (0.01 * (target_rps / 100)))  # Increasing delay with load
                success = True
            except Exception:
                success = False

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            return response_time, success

        # Generate requests at target rate
        start_time = time.time()
        tasks = []

        while time.time() - start_time < duration_seconds:
            task = asyncio.create_task(make_request())
            tasks.append(task)
            await asyncio.sleep(request_interval)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                metrics.failed_requests += 1
                metrics.response_times.append(10000)  # High penalty for exceptions
            else:
                response_time, success = result
                metrics.response_times.append(response_time)
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

        metrics.total_requests = len(results)
        metrics.calculate_statistics()

        # Calculate actual RPS
        actual_duration = time.time() - start_time
        metrics.requests_per_second = metrics.total_requests / actual_duration

        return metrics

    def _find_optimal_throughput(self, results: dict) -> float:
        """Find optimal throughput based on results."""
        optimal_rps = 0

        for rps_target, result in results.items():
            # Consider RPS optimal if:
            # 1. Error rate < 1%
            # 2. P95 response time < 5s
            # 3. Achieved RPS is close to target (within 80%)
            if (
                result["error_rate"] < 1.0
                and result["p95_response_time"] < 5000
                and result["achieved_rps"] >= rps_target * 0.8
            ):
                optimal_rps = result["achieved_rps"]

        return optimal_rps

    async def _mock_file_processing(self, file_path: str) -> dict:
        """Mock file processing with realistic delays."""
        # Simulate processing time based on file path (extract file number)
        match = re.search(r"file_(\d+)", file_path)
        file_num = int(match.group(1)) if match else 0

        # Variable processing time (50-200ms)
        processing_time = 0.05 + (file_num % 10) * 0.015
        await asyncio.sleep(processing_time)

        return {
            "file_path": file_path,
            "bpm": 120 + (file_num % 40),
            "key": ["C", "D", "E", "F", "G", "A", "B"][file_num % 7],
            "processing_time_ms": processing_time * 1000,
        }


class TestResourceUtilizationMonitoring:
    """Test suite for resource utilization monitoring."""

    @pytest.mark.integration
    async def test_memory_usage_monitoring(self, performance_benchmark_suite):
        """Test memory usage monitoring during various operations."""
        suite = performance_benchmark_suite

        async with suite.profiler.profile_memory("Memory Usage Test"):
            # Start resource monitoring
            await suite.resource_monitor.start_monitoring(interval=0.1)

            # Simulate memory-intensive operations
            await self._simulate_memory_intensive_workload()

            # Stop monitoring and get metrics
            metrics = await suite.resource_monitor.stop_monitoring()

        # Analyze memory usage
        if metrics.memory_usage_mb:
            avg_memory = statistics.mean(metrics.memory_usage_mb)
            peak_memory = max(metrics.memory_usage_mb)

            print("\nMemory Usage Analysis:")
            print(f"  Average: {avg_memory:.1f} MB")
            print(f"  Peak: {peak_memory:.1f} MB")

            # Assertions
            assert peak_memory <= suite.sla_thresholds.max_memory_usage
            assert avg_memory <= suite.sla_thresholds.max_memory_usage * 0.8  # 80% of max

    @pytest.mark.integration
    async def test_cpu_usage_monitoring(self, performance_benchmark_suite):
        """Test CPU usage monitoring during various operations."""
        suite = performance_benchmark_suite

        # Start resource monitoring
        await suite.resource_monitor.start_monitoring(interval=0.1)

        # Simulate CPU-intensive operations
        await self._simulate_cpu_intensive_workload()

        # Stop monitoring and get metrics
        metrics = await suite.resource_monitor.stop_monitoring()

        # Analyze CPU usage
        if metrics.cpu_usage_percent:
            avg_cpu = statistics.mean(metrics.cpu_usage_percent)
            peak_cpu = max(metrics.cpu_usage_percent)

            print("\nCPU Usage Analysis:")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Peak: {peak_cpu:.1f}%")

            # Assertions
            assert peak_cpu <= suite.sla_thresholds.max_cpu_usage
            assert avg_cpu <= suite.sla_thresholds.max_cpu_usage * 0.7  # 70% of max

    @pytest.mark.integration
    async def test_io_monitoring(self, performance_benchmark_suite):
        """Test I/O monitoring during file operations."""
        suite = performance_benchmark_suite

        # Start resource monitoring
        await suite.resource_monitor.start_monitoring(interval=0.1)

        # Simulate I/O intensive operations
        await self._simulate_io_intensive_workload()

        # Stop monitoring and get metrics
        metrics = await suite.resource_monitor.stop_monitoring()

        # Analyze I/O usage
        if metrics.disk_io_read_mb:
            total_read = sum(metrics.disk_io_read_mb)
            total_write = sum(metrics.disk_io_write_mb)

            print("\nI/O Usage Analysis:")
            print(f"  Total disk read: {total_read:.1f} MB")
            print(f"  Total disk write: {total_write:.1f} MB")

            # Basic assertions (I/O should be measurable but not excessive)
            assert total_read >= 0
            assert total_write >= 0

    async def _simulate_memory_intensive_workload(self):
        """Simulate memory-intensive operations."""
        # Create and manipulate large data structures
        large_lists = []

        for _ in range(10):
            # Create large list
            large_list = list(range(100000))
            large_lists.append(large_list)

            # Simulate processing
            await asyncio.sleep(0.1)

            # Transform data
            processed = [x * 2 for x in large_list[:50000]]
            large_lists.append(processed)

        # Cleanup
        large_lists.clear()
        gc.collect()

    async def _simulate_cpu_intensive_workload(self):
        """Simulate CPU-intensive operations."""
        # Mathematical computations
        for i in range(100):
            # Complex calculation
            _ = sum(j**2 for j in range(1000))

            # Allow other tasks to run
            if i % 10 == 0:
                await asyncio.sleep(0.001)

    async def _simulate_io_intensive_workload(self):
        """Simulate I/O intensive operations."""
        # Simulate file operations
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write data
            for i in range(100):
                data = f"Test data line {i}\n" * 1000
                temp_file.write(data.encode())

                if i % 10 == 0:
                    await asyncio.sleep(0.001)

        # Read data back
        with Path(temp_file.name).open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                await asyncio.sleep(0.001)

        # Cleanup
        Path(temp_file.name).unlink()


class TestDatabasePerformanceBenchmarks:
    """Test suite for database performance benchmarking."""

    @pytest.mark.integration
    async def test_database_query_performance(self, performance_benchmark_suite):
        """Benchmark database query performance."""
        suite = performance_benchmark_suite

        with patch("storage_handler.StorageHandler") as mock_storage:
            # Configure storage mock with realistic delays
            mock_storage.return_value.execute_query = AsyncMock(side_effect=self._mock_database_query)
            mock_storage.return_value.get_connection = AsyncMock(side_effect=self._mock_get_connection)

            storage = mock_storage.return_value
            metrics = PerformanceMetrics()

            # Test different query types
            query_types = [
                ("SELECT", "SELECT * FROM analysis_results WHERE id = ?"),
                ("INSERT", "INSERT INTO analysis_results (id, data) VALUES (?, ?)"),
                ("UPDATE", "UPDATE analysis_results SET data = ? WHERE id = ?"),
                (
                    "COMPLEX",
                    "SELECT a.*, b.* FROM analysis_results a JOIN metadata b "
                    "ON a.id = b.analysis_id WHERE a.created_at > ?",
                ),
            ]

            for query_type, query in query_types:
                print(f"\nBenchmarking {query_type} queries...")

                # Run multiple queries
                for i in range(50):
                    # Get connection
                    conn_start = time.perf_counter()
                    await storage.get_connection()
                    conn_end = time.perf_counter()

                    connection_time = (conn_end - conn_start) * 1000
                    metrics.db_connection_times.append(connection_time)

                    # Execute query
                    query_start = time.perf_counter()
                    await storage.execute_query(query, [str(uuid.uuid4()), "test_data"])
                    query_end = time.perf_counter()

                    query_time = (query_end - query_start) * 1000
                    metrics.db_query_times.append(query_time)

                    # Simulate connection pool usage
                    metrics.db_connection_pool_usage.append(min(10, i // 5 + 1))

            # Analyze results
            if metrics.db_query_times:
                avg_query_time = statistics.mean(metrics.db_query_times)
                max_query_time = max(metrics.db_query_times)
                avg_connection_time = statistics.mean(metrics.db_connection_times)

                print("\nDatabase Performance Analysis:")
                print(f"  Average query time: {avg_query_time:.1f}ms")
                print(f"  Max query time: {max_query_time:.1f}ms")
                print(f"  Average connection time: {avg_connection_time:.1f}ms")

                # Assertions
                assert avg_query_time <= suite.sla_thresholds.max_db_query_time
                assert avg_connection_time <= suite.sla_thresholds.max_db_connection_time

    @pytest.mark.integration
    async def test_connection_pool_performance(self, performance_benchmark_suite):
        """Test database connection pool performance under load."""
        with patch("storage_handler.StorageHandler"):
            # Configure connection pool mock
            connection_pool = asyncio.Queue(maxsize=10)
            for i in range(10):
                await connection_pool.put(f"connection_{i}")

            async def get_connection():
                start_time = time.perf_counter()
                conn = await connection_pool.get()
                end_time = time.perf_counter()

                # Simulate connection usage
                await asyncio.sleep(0.01)  # 10ms query time

                # Return connection to pool
                await connection_pool.put(conn)

                return (end_time - start_time) * 1000  # Return wait time in ms

            # Test concurrent connection requests
            concurrent_requests = 50
            tasks = [get_connection() for _ in range(concurrent_requests)]

            start_time = time.time()
            wait_times = await asyncio.gather(*tasks)
            end_time = time.time()

            # Analyze results
            avg_wait_time = statistics.mean(wait_times)
            max_wait_time = max(wait_times)
            total_throughput = concurrent_requests / (end_time - start_time)

            print("\nConnection Pool Performance:")
            print(f"  Average wait time: {avg_wait_time:.1f}ms")
            print(f"  Max wait time: {max_wait_time:.1f}ms")
            print(f"  Throughput: {total_throughput:.1f} requests/sec")

            # Assertions
            assert avg_wait_time <= 100.0  # Should not wait more than 100ms on average
            assert max_wait_time <= 500.0  # Max wait should be under 500ms

    async def _mock_database_query(self, query: str, params: list | None = None) -> dict:
        """Mock database query with realistic response times."""
        # Simulate different query complexities
        if "SELECT *" in query and "JOIN" in query:
            # Complex query
            await asyncio.sleep(0.05)  # 50ms
        elif "INSERT" in query or "UPDATE" in query:
            # Write query
            await asyncio.sleep(0.02)  # 20ms
        else:
            # Simple SELECT
            await asyncio.sleep(0.01)  # 10ms

        return {"rows": [{"id": str(uuid.uuid4()), "data": "mock_data"}]}

    async def _mock_get_connection(self):
        """Mock database connection acquisition."""
        # Simulate connection time
        await asyncio.sleep(0.005)  # 5ms connection time
        return MagicMock()


class TestMessageQueuePerformance:
    """Test suite for message queue performance benchmarking."""

    @pytest.mark.integration
    async def test_message_queue_throughput(self, performance_benchmark_suite):
        """Test message queue publishing and consuming throughput."""
        suite = performance_benchmark_suite

        with patch("async_message_consumer.AsyncMessageConsumer"):
            # Configure mock consumer
            published_messages = asyncio.Queue()
            consumed_messages = []

            async def mock_publish(message):
                start_time = time.perf_counter()
                await published_messages.put(message)
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000  # Return publish time in ms

            async def mock_consume():
                start_time = time.perf_counter()
                message = await published_messages.get()
                consumed_messages.append(message)
                end_time = time.perf_counter()
                return (end_time - start_time) * 1000  # Return consume time in ms

            metrics = PerformanceMetrics()

            # Test message publishing
            num_messages = 1000
            publish_tasks = []

            print(f"Publishing {num_messages} messages...")
            for i in range(num_messages):
                message = {
                    "id": str(uuid.uuid4()),
                    "type": "audio_analysis",
                    "data": {"file_path": f"/test/file_{i}.mp3"},
                    "timestamp": time.time(),
                }

                task = asyncio.create_task(mock_publish(message))
                publish_tasks.append(task)

            publish_times = await asyncio.gather(*publish_tasks)
            metrics.queue_publish_times.extend(publish_times)

            # Test message consuming
            print(f"Consuming {num_messages} messages...")
            consume_tasks = [mock_consume() for _ in range(num_messages)]
            consume_times = await asyncio.gather(*consume_tasks)
            metrics.queue_consume_times.extend(consume_times)

            # Analyze results
            avg_publish_time = statistics.mean(metrics.queue_publish_times)
            avg_consume_time = statistics.mean(metrics.queue_consume_times)

            # Calculate throughput
            total_time = max(*publish_times, *consume_times) / 1000  # Convert to seconds
            throughput = num_messages / total_time

            print("\nMessage Queue Performance:")
            print(f"  Average publish time: {avg_publish_time:.1f}ms")
            print(f"  Average consume time: {avg_consume_time:.1f}ms")
            print(f"  Throughput: {throughput:.1f} messages/sec")

            # Assertions
            assert avg_publish_time <= suite.sla_thresholds.max_queue_publish_time
            assert avg_consume_time <= suite.sla_thresholds.max_queue_consume_time
            assert len(consumed_messages) == num_messages  # All messages processed

    @pytest.mark.integration
    async def test_queue_backlog_handling(self, performance_benchmark_suite):
        """Test queue performance under backlog conditions."""
        suite = performance_benchmark_suite

        # Simulate queue with backlog
        queue_sizes = [0, 100, 500, 1000, 2000]
        metrics = PerformanceMetrics()

        for queue_size in queue_sizes:
            print(f"\nTesting queue with {queue_size} messages backlog...")

            # Create mock queue with backlog
            mock_queue = asyncio.Queue(maxsize=queue_size + 100)

            # Fill queue with backlog messages
            for _ in range(queue_size):
                await mock_queue.put({"id": str(uuid.uuid4()), "type": "backlog_message", "timestamp": time.time()})

            metrics.queue_backlog_sizes.append(queue_size)

            # Test processing time with backlog
            test_messages = 50
            start_time = time.time()

            # Add test messages
            for _ in range(test_messages):
                await mock_queue.put({"id": str(uuid.uuid4()), "type": "test_message", "timestamp": time.time()})

            # Process all messages (backlog + test)
            processed_count = 0
            while not mock_queue.empty():
                await mock_queue.get()  # Consume message
                processed_count += 1

                # Simulate processing time
                await asyncio.sleep(0.001)  # 1ms per message

            end_time = time.time()
            processing_time = end_time - start_time

            print(f"  Processed {processed_count} messages in {processing_time:.2f}s")
            print(f"  Rate: {processed_count / processing_time:.1f} messages/sec")

            # Check if backlog affects performance significantly
            if queue_size > 1000 and processing_time > 10:  # Performance degradation threshold
                print(f"  ⚠️  Performance degradation detected with backlog size {queue_size}")

        # Verify backlog doesn't exceed threshold
        max_backlog = max(metrics.queue_backlog_sizes)
        assert max_backlog <= suite.sla_thresholds.max_queue_backlog


class TestConcurrentUserSimulation:
    """Test suite for concurrent user simulation and load testing."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_user_scaling(self, performance_benchmark_suite):
        """Test system behavior with increasing concurrent users."""
        suite = performance_benchmark_suite

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 25, 50, 100]
        results = {}

        for concurrent_users in concurrency_levels:
            print(f"\nTesting with {concurrent_users} concurrent users...")

            metrics = await self._simulate_concurrent_users(concurrent_users, duration_seconds=30, suite=suite)

            results[concurrent_users] = {
                "avg_response_time": metrics.avg_response_time,
                "p95_response_time": metrics.p95_response_time,
                "requests_per_second": metrics.requests_per_second,
                "error_rate": metrics.error_rate,
                "successful_requests": metrics.successful_requests,
            }

            # Print results for this level
            print(f"  Response time (avg/p95): {metrics.avg_response_time:.1f}ms / {metrics.p95_response_time:.1f}ms")
            print(f"  RPS: {metrics.requests_per_second:.1f}")
            print(f"  Error rate: {metrics.error_rate:.1f}%")

            # Stop if system shows signs of saturation
            if metrics.error_rate > 10.0 or metrics.avg_response_time > 10000:  # 10s response time
                print(f"  💥 System saturation detected at {concurrent_users} users")
                break

        # Analyze scalability
        scalability_analysis = self._analyze_scalability(results)

        print("\n📈 Scalability Analysis:")
        print(f"  Optimal concurrent users: {scalability_analysis['optimal_users']}")
        print(f"  Saturation point: {scalability_analysis['saturation_point']}")
        print(f"  Linear scaling up to: {scalability_analysis['linear_scaling_limit']} users")

        # Create final metrics
        final_metrics = PerformanceMetrics()
        final_metrics.concurrent_users = scalability_analysis["optimal_users"]
        final_metrics.saturation_point = scalability_analysis["saturation_point"]

        suite._print_performance_report(final_metrics, "Concurrent User Scaling")

        # Assertions
        assert scalability_analysis["optimal_users"] >= 10  # Should handle at least 10 concurrent users
        assert scalability_analysis["saturation_point"] is None or scalability_analysis["saturation_point"] >= 25

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_burst_traffic_handling(self, performance_benchmark_suite):
        """Test system behavior during traffic bursts."""
        suite = performance_benchmark_suite

        metrics = PerformanceMetrics()

        async with suite.profiler.profile_execution_time("Burst Traffic Test"):
            await suite.resource_monitor.start_monitoring()

            # Normal load phase
            print("Phase 1: Normal load (10 RPS)")
            normal_metrics = await self._generate_load(rps=10, duration_seconds=30, label="normal")

            # Burst phase
            print("Phase 2: Traffic burst (100 RPS)")
            burst_metrics = await self._generate_load(rps=100, duration_seconds=10, label="burst")

            # Recovery phase
            print("Phase 3: Recovery (10 RPS)")
            recovery_metrics = await self._generate_load(rps=10, duration_seconds=20, label="recovery")

            resource_metrics = await suite.resource_monitor.stop_monitoring()

        # Aggregate metrics
        all_phases = [normal_metrics, burst_metrics, recovery_metrics]
        metrics.response_times = []
        metrics.total_requests = 0
        metrics.successful_requests = 0
        metrics.failed_requests = 0

        for phase_metrics in all_phases:
            metrics.response_times.extend(phase_metrics.response_times)
            metrics.total_requests += phase_metrics.total_requests
            metrics.successful_requests += phase_metrics.successful_requests
            metrics.failed_requests += phase_metrics.failed_requests

        metrics.calculate_statistics()
        metrics.cpu_usage_percent = resource_metrics.cpu_usage_percent
        metrics.memory_usage_mb = resource_metrics.memory_usage_mb

        # Analyze burst handling
        print("\nBurst Traffic Analysis:")
        print(f"  Normal phase error rate: {normal_metrics.error_rate:.1f}%")
        print(f"  Burst phase error rate: {burst_metrics.error_rate:.1f}%")
        print(f"  Recovery phase error rate: {recovery_metrics.error_rate:.1f}%")

        burst_impact = burst_metrics.error_rate - normal_metrics.error_rate
        recovery_effectiveness = recovery_metrics.error_rate <= normal_metrics.error_rate * 1.2

        suite._print_performance_report(metrics, "Burst Traffic Handling")

        # Assertions
        assert burst_impact <= 15.0  # Burst shouldn't increase error rate by more than 15%
        assert recovery_effectiveness  # System should recover after burst
        assert burst_metrics.avg_response_time <= 5000  # Response time shouldn't exceed 5s during burst

    async def _simulate_concurrent_users(
        self, concurrent_users: int, duration_seconds: int, suite: PerformanceBenchmarkSuite
    ) -> PerformanceMetrics:
        """Simulate concurrent users for scalability testing."""
        metrics = PerformanceMetrics()

        async def simulate_user():
            """Simulate a single user's behavior."""
            user_requests = 0
            user_start_time = time.time()

            while time.time() - user_start_time < duration_seconds:
                request_start = time.perf_counter()

                try:
                    # Simulate user actions
                    await self._simulate_user_request()
                    success = True
                except Exception:
                    success = False

                request_end = time.perf_counter()
                response_time = (request_end - request_start) * 1000

                # Thread-safe metrics update would be needed in real implementation
                metrics.response_times.append(response_time)
                user_requests += 1

                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

                # Random think time between requests (1-5 seconds)
                think_time = 1 + (user_requests % 5)
                await asyncio.sleep(think_time)

            return user_requests

        # Create user simulation tasks
        user_tasks = [simulate_user() for _ in range(concurrent_users)]

        start_time = time.time()
        user_request_counts = await asyncio.gather(*user_tasks, return_exceptions=True)
        end_time = time.time()

        # Calculate metrics
        total_duration = end_time - start_time
        metrics.total_requests = sum(count for count in user_request_counts if isinstance(count, int))
        metrics.requests_per_second = metrics.total_requests / total_duration
        metrics.concurrent_users = concurrent_users
        metrics.calculate_statistics()

        return metrics

    async def _generate_load(self, rps: int, duration_seconds: int, label: str) -> PerformanceMetrics:
        """Generate load at specified RPS for burst testing."""
        metrics = PerformanceMetrics()
        request_interval = 1.0 / rps

        async def make_request():
            start_time = time.perf_counter()
            try:
                await self._simulate_user_request()
                success = True
            except Exception:
                success = False

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            return response_time, success

        # Generate requests at target rate
        start_time = time.time()
        tasks = []

        while time.time() - start_time < duration_seconds:
            task = asyncio.create_task(make_request())
            tasks.append(task)
            await asyncio.sleep(request_interval)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                metrics.failed_requests += 1
                metrics.response_times.append(10000)  # High penalty for exceptions
            else:
                response_time, success = result
                metrics.response_times.append(response_time)
                if success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

        metrics.total_requests = len(results)
        metrics.calculate_statistics()

        return metrics

    async def _simulate_user_request(self):
        """Simulate a typical user request."""
        # Random request type
        request_types = ["health_check", "analyze_file", "get_status", "get_metadata"]
        request_type = request_types[int(time.time()) % len(request_types)]

        # Simulate different processing times
        if request_type == "health_check":
            await asyncio.sleep(0.001)  # 1ms
        elif request_type == "analyze_file":
            await asyncio.sleep(0.05)  # 50ms
        elif request_type == "get_status":
            await asyncio.sleep(0.002)  # 2ms
        else:  # get_metadata
            await asyncio.sleep(0.01)  # 10ms

    def _analyze_scalability(self, results: dict) -> dict:
        """Analyze scalability results."""
        optimal_users = 1
        saturation_point = None
        linear_scaling_limit = 1

        previous_rps = 0

        for users, metrics in results.items():
            current_rps = metrics["requests_per_second"]
            error_rate = metrics["error_rate"]
            response_time = metrics["avg_response_time"]

            # Check for optimal performance (good RPS, low error rate, reasonable response time)
            if error_rate <= 2.0 and response_time <= 2000:  # 2s response time threshold
                optimal_users = users

            # Check for saturation (high error rate or very slow response times)
            if (error_rate > 10.0 or response_time > 10000) and saturation_point is None:  # 10s response time
                saturation_point = users

            # Check for linear scaling (RPS should roughly scale with users)
            if previous_rps > 0:
                rps_scaling_factor = current_rps / previous_rps
                user_scaling_factor = users / (users - 1) if users > 1 else 1

                # If RPS scales linearly with users (within 80% efficiency)
                if rps_scaling_factor >= user_scaling_factor * 0.8:
                    linear_scaling_limit = users

            previous_rps = current_rps

        return {
            "optimal_users": optimal_users,
            "saturation_point": saturation_point,
            "linear_scaling_limit": linear_scaling_limit,
        }


class TestCachePerformanceOptimization:
    """Test suite for cache performance and optimization."""

    @pytest.mark.integration
    async def test_cache_hit_rate_optimization(self, performance_benchmark_suite):
        """Test cache hit rate under different access patterns."""
        suite = performance_benchmark_suite

        # Mock Redis cache
        cache_data = {}
        cache_stats = {"hits": 0, "misses": 0}

        async def mock_cache_get(key: str):
            start_time = time.perf_counter()

            if key in cache_data:
                cache_stats["hits"] += 1
                result = cache_data[key]
            else:
                cache_stats["misses"] += 1
                result = None

            # Simulate cache response time
            await asyncio.sleep(0.001)  # 1ms

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000

            return result, response_time

        async def mock_cache_set(key: str, value: any, ttl: int = 3600):
            cache_data[key] = value
            await asyncio.sleep(0.002)  # 2ms for write

        # Test different access patterns
        patterns = {
            "random": self._generate_random_access_pattern,
            "sequential": self._generate_sequential_access_pattern,
            "hotspot": self._generate_hotspot_access_pattern,
            "temporal": self._generate_temporal_access_pattern,
        }

        results = {}

        for pattern_name, pattern_generator in patterns.items():
            print(f"\nTesting {pattern_name} access pattern...")

            # Reset cache and stats
            cache_data.clear()
            cache_stats = {"hits": 0, "misses": 0}

            # Pre-populate cache with some data
            for i in range(100):
                await mock_cache_set(f"key_{i}", f"value_{i}")

            # Generate access pattern
            keys_to_access = pattern_generator(num_requests=1000)
            response_times = []

            # Execute cache operations
            for key in keys_to_access:
                result, response_time = await mock_cache_get(key)
                response_times.append(response_time)

                # Simulate cache miss - load from "database" and cache
                if result is None:
                    # Simulate database load time
                    await asyncio.sleep(0.05)  # 50ms database query

                    # Cache the result
                    await mock_cache_set(key, f"loaded_value_{key}")

            # Calculate metrics
            total_requests = len(keys_to_access)
            hit_rate = (cache_stats["hits"] / total_requests) * 100
            avg_response_time = statistics.mean(response_times)

            results[pattern_name] = {
                "hit_rate": hit_rate,
                "avg_response_time": avg_response_time,
                "total_requests": total_requests,
                "hits": cache_stats["hits"],
                "misses": cache_stats["misses"],
            }

            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Average response time: {avg_response_time:.1f}ms")
            print(f"  Hits/Misses: {cache_stats['hits']}/{cache_stats['misses']}")

        # Analyze results
        best_pattern = max(results.keys(), key=lambda x: results[x]["hit_rate"])
        worst_pattern = min(results.keys(), key=lambda x: results[x]["hit_rate"])

        print("\n🎯 Cache Performance Analysis:")
        print(f"  Best pattern: {best_pattern} ({results[best_pattern]['hit_rate']:.1f}% hit rate)")
        print(f"  Worst pattern: {worst_pattern} ({results[worst_pattern]['hit_rate']:.1f}% hit rate)")

        # Overall metrics
        total_hits = sum(r["hits"] for r in results.values())
        total_requests = sum(r["total_requests"] for r in results.values())
        overall_hit_rate = (total_hits / total_requests) * 100

        # Assertions
        assert overall_hit_rate >= suite.sla_thresholds.min_cache_hit_rate
        assert results[best_pattern]["hit_rate"] >= 90.0  # Best pattern should achieve >90%

        # Create final metrics
        metrics = PerformanceMetrics()
        metrics.cache_hits = total_hits
        metrics.cache_misses = total_requests - total_hits
        metrics.cache_hit_rate = overall_hit_rate
        metrics.cache_response_times = [r["avg_response_time"] for r in results.values()]

        suite._print_performance_report(metrics, "Cache Hit Rate Optimization")

    @pytest.mark.integration
    async def test_cache_eviction_performance(self, performance_benchmark_suite):
        """Test cache performance under memory pressure and eviction."""
        # Mock cache with limited capacity
        cache_capacity = 1000
        cache_data = {}
        access_times = {}  # For LRU tracking
        cache_stats = {"evictions": 0, "hits": 0, "misses": 0}

        async def mock_cache_with_eviction(key: str, value: str | None = None):
            """Mock cache operations with LRU eviction."""
            start_time = time.perf_counter()

            if value is not None:  # SET operation
                # Check if eviction is needed
                if len(cache_data) >= cache_capacity and key not in cache_data:
                    # Find LRU key
                    lru_key = min(access_times.keys(), key=access_times.get)
                    del cache_data[lru_key]
                    del access_times[lru_key]
                    cache_stats["evictions"] += 1

                cache_data[key] = value
                access_times[key] = time.time()

            elif key in cache_data:
                cache_stats["hits"] += 1
                access_times[key] = time.time()  # Update access time
                result = cache_data[key]
            else:
                cache_stats["misses"] += 1
                result = None

            # Simulate cache operation time
            await asyncio.sleep(0.002)  # 2ms

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000

            return result if value is None else True, response_time

        # Phase 1: Fill cache to capacity
        print("Phase 1: Filling cache to capacity...")
        for i in range(cache_capacity):
            await mock_cache_with_eviction(f"key_{i}", f"value_{i}")

        print(f"Cache filled: {len(cache_data)} items")

        # Phase 2: Test performance with evictions
        print("Phase 2: Testing performance with evictions...")
        eviction_response_times = []

        # Add more items than capacity to trigger evictions
        for i in range(cache_capacity, cache_capacity + 500):
            _result, response_time = await mock_cache_with_eviction(f"new_key_{i}", f"new_value_{i}")
            eviction_response_times.append(response_time)

        print(f"Evictions triggered: {cache_stats['evictions']}")

        # Phase 3: Test access patterns after evictions
        print("Phase 3: Testing access patterns after evictions...")
        access_response_times = []

        # Test mix of hot and cold data access
        for i in range(1000):
            key = f"new_key_{cache_capacity + (i % 100)}" if i % 3 == 0 else f"key_{i % cache_capacity}"

            _result, response_time = await mock_cache_with_eviction(key)
            access_response_times.append(response_time)

        # Analyze eviction performance
        avg_eviction_time = statistics.mean(eviction_response_times)
        avg_access_time = statistics.mean(access_response_times)
        hit_rate_after_eviction = (cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])) * 100

        print("\n🗑️ Cache Eviction Performance:")
        print(f"  Total evictions: {cache_stats['evictions']}")
        print(f"  Average eviction time: {avg_eviction_time:.1f}ms")
        print(f"  Average access time after evictions: {avg_access_time:.1f}ms")
        print(f"  Hit rate after evictions: {hit_rate_after_eviction:.1f}%")

        # Assertions
        assert avg_eviction_time <= 10.0  # Evictions should be fast (<10ms)
        assert hit_rate_after_eviction >= 60.0  # Should maintain reasonable hit rate
        assert cache_stats["evictions"] > 0  # Should have triggered evictions

    def _generate_random_access_pattern(self, num_requests: int) -> list[str]:
        """Generate random access pattern."""
        return [f"key_{random.randint(0, 199)}" for _ in range(num_requests)]

    def _generate_sequential_access_pattern(self, num_requests: int) -> list[str]:
        """Generate sequential access pattern."""
        return [f"key_{i % 200}" for i in range(num_requests)]

    def _generate_hotspot_access_pattern(self, num_requests: int) -> list[str]:
        """Generate hotspot access pattern (80/20 rule)."""
        keys = []
        for _ in range(num_requests):
            if random.random() < 0.8:  # 80% access to 20% of keys
                keys.append(f"key_{random.randint(0, 19)}")
            else:  # 20% access to remaining 80% of keys
                keys.append(f"key_{random.randint(20, 199)}")
        return keys

    def _generate_temporal_access_pattern(self, num_requests: int) -> list[str]:
        """Generate temporal access pattern (recent items more likely)."""
        keys = []
        for _ in range(num_requests):
            # Weight towards more recent keys
            weight = random.random() ** 2  # Square to bias towards 0
            key_index = int(weight * 200)
            keys.append(f"key_{key_index}")
        return keys


class TestEndToEndWorkflowPerformance:
    """Test suite for end-to-end workflow performance analysis."""

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_audio_analysis_workflow(self, performance_benchmark_suite, mock_audio_files):
        """Test complete end-to-end audio analysis workflow performance."""
        suite = performance_benchmark_suite

        # Mock all workflow components
        with (
            patch("wave_orchestrator.WaveOrchestrator") as mock_orchestrator,
            patch("async_message_consumer.AsyncMessageConsumer") as mock_consumer,
            patch("async_audio_processor.AsyncAudioProcessor") as mock_processor,
            patch("storage_handler.StorageHandler") as mock_storage,
            patch("model_manager.ModelManager") as mock_model_manager,
        ):
            # Configure mocks for realistic workflow
            mock_orchestrator.return_value.orchestrate = AsyncMock(side_effect=self._mock_workflow_orchestration)
            mock_consumer.return_value.process_message = AsyncMock(side_effect=self._mock_message_processing)
            mock_processor.return_value.process_audio = AsyncMock(side_effect=self._mock_audio_processing)
            mock_storage.return_value.store_analysis = AsyncMock(side_effect=self._mock_storage_operation)
            mock_model_manager.return_value.predict = AsyncMock(side_effect=self._mock_model_prediction)

            metrics = PerformanceMetrics()
            workflow_times = []

            async with (
                suite.profiler.profile_execution_time("Complete Workflow"),
                suite.profiler.profile_memory("Complete Workflow"),
            ):
                await suite.resource_monitor.start_monitoring()

                # Process files through complete workflow
                batch_size = 5
                processed_files = 0

                for i in range(0, min(50, len(mock_audio_files)), batch_size):
                    batch = mock_audio_files[i : i + batch_size]
                    batch_start = time.perf_counter()

                    # Process batch through workflow
                    tasks = []
                    for audio_file in batch:
                        task = asyncio.create_task(
                            self._run_complete_workflow(
                                audio_file,
                                mock_orchestrator.return_value,
                                mock_consumer.return_value,
                                mock_processor.return_value,
                                mock_storage.return_value,
                            )
                        )
                        tasks.append(task)

                    # Wait for batch completion
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    batch_end = time.perf_counter()
                    batch_time = (batch_end - batch_start) * 1000
                    workflow_times.append(batch_time)

                    # Count successful vs failed
                    successful = sum(1 for r in results if not isinstance(r, Exception))
                    failed = len(results) - successful

                    metrics.successful_requests += successful
                    metrics.failed_requests += failed
                    processed_files += len(batch)

                    print(f"Processed batch {i // batch_size + 1}: {successful}/{len(batch)} successful")

                resource_metrics = await suite.resource_monitor.stop_monitoring()

            # Calculate final metrics
            metrics.response_times = workflow_times
            metrics.total_requests = processed_files
            metrics.calculate_statistics()

            total_time = sum(workflow_times) / 1000  # Convert to seconds
            metrics.files_processed_per_second = processed_files / max(total_time, 1)
            metrics.requests_per_second = metrics.files_processed_per_second

            metrics.cpu_usage_percent = resource_metrics.cpu_usage_percent
            metrics.memory_usage_mb = resource_metrics.memory_usage_mb

            # Workflow-specific analysis
            print("\n🔄 End-to-End Workflow Analysis:")
            print(f"  Total files processed: {processed_files}")
            print(f"  Average workflow time: {metrics.avg_response_time:.1f}ms")
            print(f"  Workflow throughput: {metrics.files_processed_per_second:.1f} files/sec")
            print(f"  Success rate: {(metrics.successful_requests / metrics.total_requests) * 100:.1f}%")

            # Compare with baseline
            suite._compare_with_baseline(metrics, "complete_workflow")
            suite._print_performance_report(metrics, "Complete Audio Analysis Workflow")

            # Assertions
            assert metrics.error_rate <= 5.0  # Allow up to 5% error rate for complex workflow
            assert metrics.files_processed_per_second >= 0.5  # At least 0.5 files per second
            assert metrics.avg_response_time <= 10000  # Workflow should complete within 10s

            suite._save_baseline_metrics(metrics, "complete_workflow")

    @pytest.mark.integration
    async def test_workflow_bottleneck_identification(self, performance_benchmark_suite):
        """Identify bottlenecks in the audio analysis workflow."""
        # Mock workflow stages with different processing times
        stage_timings = {
            "message_intake": [],
            "file_validation": [],
            "metadata_extraction": [],
            "audio_analysis": [],
            "model_inference": [],
            "result_storage": [],
            "notification": [],
        }

        async def timed_stage(stage_name: str, processing_func):
            """Time a workflow stage."""
            start_time = time.perf_counter()
            result = await processing_func()
            end_time = time.perf_counter()

            stage_time = (end_time - start_time) * 1000
            stage_timings[stage_name].append(stage_time)

            return result

        # Process test files through timed workflow
        num_files = 20

        for i in range(num_files):
            # Run through all workflow stages
            try:
                # Stage 1: Message intake
                await timed_stage("message_intake", lambda: self._mock_stage_processing(0.005))  # 5ms

                # Stage 2: File validation
                await timed_stage("file_validation", lambda: self._mock_stage_processing(0.01))  # 10ms

                # Stage 3: Metadata extraction
                await timed_stage("metadata_extraction", lambda: self._mock_stage_processing(0.05))  # 50ms

                # Stage 4: Audio analysis (slowest stage)
                await timed_stage("audio_analysis", lambda: self._mock_stage_processing(0.2))  # 200ms

                # Stage 5: Model inference
                await timed_stage("model_inference", lambda: self._mock_stage_processing(0.1))  # 100ms

                # Stage 6: Result storage
                await timed_stage("result_storage", lambda: self._mock_stage_processing(0.02))  # 20ms

                # Stage 7: Notification
                await timed_stage("notification", lambda: self._mock_stage_processing(0.001))  # 1ms

            except Exception as e:
                print(f"Workflow failed for file {i}: {e}")

        # Analyze bottlenecks
        stage_analysis = {}
        total_workflow_time = 0

        for stage, timings in stage_timings.items():
            if timings:
                avg_time = statistics.mean(timings)
                max_time = max(timings)
                total_time = sum(timings)

                stage_analysis[stage] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "total_time": total_time,
                    "percentage": 0,  # Will calculate after total
                }
                total_workflow_time += total_time

        # Calculate percentages
        for analysis in stage_analysis.values():
            analysis["percentage"] = (analysis["total_time"] / total_workflow_time) * 100

        # Sort by total time to identify bottlenecks
        bottlenecks = sorted(stage_analysis.items(), key=lambda x: x[1]["total_time"], reverse=True)

        print("\n🔍 Workflow Bottleneck Analysis:")
        print(f"  Total workflow time: {total_workflow_time:.1f}ms")
        print("\n  Stage breakdown (by total time):")

        for stage, metrics in bottlenecks:
            print(f"    {stage}:")
            print(f"      Average: {metrics['avg_time']:.1f}ms")
            print(f"      Maximum: {metrics['max_time']:.1f}ms")
            print(f"      Percentage: {metrics['percentage']:.1f}%")

        # Identify top bottlenecks
        primary_bottleneck = bottlenecks[0]
        secondary_bottleneck = bottlenecks[1] if len(bottlenecks) > 1 else None

        print(f"\n  🚨 Primary bottleneck: {primary_bottleneck[0]} ({primary_bottleneck[1]['percentage']:.1f}%)")
        if secondary_bottleneck:
            print(
                f"  ⚠️  Secondary bottleneck: {secondary_bottleneck[0]} ({secondary_bottleneck[1]['percentage']:.1f}%)"
            )

        # Recommendations
        print("\n  💡 Optimization recommendations:")
        if primary_bottleneck[1]["percentage"] > 50:
            print(f"    - Focus on optimizing {primary_bottleneck[0]} (major bottleneck)")

        if secondary_bottleneck and secondary_bottleneck[1]["percentage"] > 20:
            print(f"    - Consider parallelizing {secondary_bottleneck[0]}")

        stages_under_10_percent = [s for s, m in bottlenecks if m["percentage"] < 10]
        if len(stages_under_10_percent) > 3:
            print(f"    - {len(stages_under_10_percent)} stages are well-optimized (<10% each)")

        # Assertions
        assert primary_bottleneck[1]["percentage"] < 80  # No single stage should dominate
        assert len([s for s, m in bottlenecks if m["avg_time"] > 1000]) == 0  # No stage >1s average

    async def _run_complete_workflow(self, audio_file: dict, orchestrator, consumer, processor, storage) -> dict:
        """Run complete workflow for a single audio file."""
        try:
            # Step 1: Message processing
            message = {"file_id": audio_file["id"], "file_path": audio_file["path"], "timestamp": time.time()}

            # Step 2: Orchestrate workflow
            orchestration_result = await orchestrator.orchestrate(message)

            # Step 3: Process message
            processing_result = await consumer.process_message(message)

            # Step 4: Audio processing
            audio_result = await processor.process_audio(audio_file["path"])

            # Step 5: Store results
            final_result = {**orchestration_result, **processing_result, **audio_result}

            storage_result = await storage.store_analysis(final_result)

            return {
                "status": "success",
                "file_id": audio_file["id"],
                "results": final_result,
                "storage_id": storage_result,
            }

        except Exception as e:
            return {"status": "error", "file_id": audio_file["id"], "error": str(e)}

    async def _mock_workflow_orchestration(self, message: dict) -> dict:
        """Mock workflow orchestration."""
        await asyncio.sleep(0.01)  # 10ms orchestration time
        return {
            "workflow_id": str(uuid.uuid4()),
            "stages": ["validate", "process", "analyze", "store"],
            "priority": "normal",
        }

    async def _mock_message_processing(self, message: dict) -> dict:
        """Mock message processing."""
        await asyncio.sleep(0.005)  # 5ms processing time
        return {"processed_at": time.time(), "queue_time": 0.001, "validation_status": "passed"}

    async def _mock_audio_processing(self, file_path: str) -> dict:
        """Mock audio processing."""
        # Simulate variable processing time based on file
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        processing_time = 0.05 + (int(file_hash[:2], 16) % 20) * 0.01  # 50-250ms

        await asyncio.sleep(processing_time)

        return {
            "bpm": 120 + (int(file_hash[:2], 16) % 40),
            "key": ["C", "D", "E", "F", "G", "A", "B"][int(file_hash[:1], 16) % 7],
            "energy": 0.5 + (int(file_hash[1:2], 16) / 32),
            "processing_time_ms": processing_time * 1000,
        }

    async def _mock_storage_operation(self, data: dict) -> str:
        """Mock storage operation."""
        await asyncio.sleep(0.02)  # 20ms storage time
        return str(uuid.uuid4())

    async def _mock_model_prediction(self, features: dict) -> dict:
        """Mock model prediction."""
        await asyncio.sleep(0.1)  # 100ms inference time
        return {"mood": "energetic", "genre": "electronic", "danceability": 0.8, "confidence": 0.95}

    async def _mock_stage_processing(self, delay: float) -> dict:
        """Mock a workflow stage with specified delay."""
        await asyncio.sleep(delay)
        return {"status": "completed", "timestamp": time.time()}


if __name__ == "__main__":
    # Run specific test suites for debugging
    pytest.main(
        [__file__ + "::TestServiceResponseTimeBenchmarks::test_api_endpoint_response_times", "-v", "--tb=short"]
    )
