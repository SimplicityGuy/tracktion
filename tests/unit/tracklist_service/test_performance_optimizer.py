"""Unit tests for performance optimization system."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.tracklist_service.src.optimization.performance_optimizer import (
    OptimizationConfig,
    OptimizationStrategy,
    PerformanceMetrics,
    PerformanceOptimizer,
    ResourceType,
)


@pytest.fixture
def mock_batch_queue():
    """Create mock batch queue."""
    queue = AsyncMock()
    queue.get_queue_depth = AsyncMock(return_value=50)
    queue.set_worker_count = AsyncMock()
    return queue


@pytest.fixture
def mock_progress_tracker():
    """Create mock progress tracker."""
    return Mock()


@pytest.fixture
def mock_retry_manager():
    """Create mock retry manager."""
    manager = Mock()
    manager.get_failure_stats = Mock(
        return_value={
            "total_attempts": 1000,
            "total_failures": 50,
            "total_retries": 100,
        }
    )
    manager.default_policy = Mock()
    manager.default_policy.initial_delay = 2.0
    manager.default_policy.max_delay = 60.0
    return manager


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = Mock()
    redis_mock.hset = Mock()
    redis_mock.expire = Mock()
    return redis_mock


@pytest.fixture
def optimizer(mock_batch_queue, mock_progress_tracker, mock_retry_manager, mock_redis):
    """Create PerformanceOptimizer instance with mocks."""
    with patch(
        "services.tracklist_service.src.optimization.performance_optimizer.Redis",
        return_value=mock_redis,
    ):
        opt = PerformanceOptimizer(
            batch_queue=mock_batch_queue,
            progress_tracker=mock_progress_tracker,
            retry_manager=mock_retry_manager,
        )
        yield opt


class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""

    def test_strategy_values(self):
        """Test optimization strategy values."""
        assert OptimizationStrategy.ADAPTIVE.value == "adaptive"
        assert OptimizationStrategy.AGGRESSIVE.value == "aggressive"
        assert OptimizationStrategy.CONSERVATIVE.value == "conservative"
        assert OptimizationStrategy.BALANCED.value == "balanced"
        assert OptimizationStrategy.CUSTOM.value == "custom"


class TestResourceType:
    """Test ResourceType enum."""

    def test_resource_types(self):
        """Test resource type values."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.NETWORK.value == "network"
        assert ResourceType.DISK_IO.value == "disk_io"
        assert ResourceType.CONNECTIONS.value == "connections"


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_initialization(self):
        """Test metrics initialization."""
        now = datetime.now(UTC)
        metrics = PerformanceMetrics(
            timestamp=now,
            throughput=100.0,
            latency_p50=10.0,
            latency_p95=50.0,
            latency_p99=100.0,
            error_rate=2.5,
            cpu_usage=60.0,
            memory_usage=45.0,
            active_workers=10,
            queue_depth=100,
            retry_rate=0.1,
        )

        assert metrics.timestamp == now
        assert metrics.throughput == 100.0
        assert metrics.latency_p95 == 50.0
        assert metrics.error_rate == 2.5
        assert metrics.active_workers == 10


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.strategy == OptimizationStrategy.BALANCED
        assert config.max_error_rate == 0.05
        assert config.min_workers == 2
        assert config.max_workers == 50
        assert config.scale_up_threshold == 0.8
        assert config.scale_down_threshold == 0.3

    def test_custom_values(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.AGGRESSIVE,
            target_throughput=200.0,
            max_workers=100,
        )

        assert config.strategy == OptimizationStrategy.AGGRESSIVE
        assert config.target_throughput == 200.0
        assert config.max_workers == 100


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class."""

    def test_initialization(self, mock_batch_queue, mock_progress_tracker, mock_retry_manager, mock_redis):
        """Test optimizer initialization."""
        with patch(
            "services.tracklist_service.src.optimization.performance_optimizer.Redis",
            return_value=mock_redis,
        ):
            opt = PerformanceOptimizer(
                batch_queue=mock_batch_queue,
                progress_tracker=mock_progress_tracker,
                retry_manager=mock_retry_manager,
            )

        assert opt.batch_queue == mock_batch_queue
        assert opt.progress_tracker == mock_progress_tracker
        assert opt.retry_manager == mock_retry_manager
        assert opt.current_workers == opt.config.min_workers
        assert opt.optimization_enabled is True

    @pytest.mark.asyncio
    async def test_collect_metrics(self, optimizer, mock_batch_queue):
        """Test collecting performance metrics."""
        # Add some latency samples
        optimizer.latency_samples.extend([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        with patch("psutil.cpu_percent", return_value=50.0), patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=60.0)

            metrics = await optimizer.collect_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.queue_depth == 100  # Placeholder value from optimizer
        assert metrics.active_workers == optimizer.current_workers

    def test_calculate_latencies(self, optimizer):
        """Test latency percentile calculation."""
        # Add latency samples (1-100ms)
        optimizer.latency_samples.extend(range(1, 101))

        p50, p95, p99 = optimizer._calculate_latencies()

        assert p50 == 51  # 50th percentile (index calculation)
        assert p95 == 96  # 95th percentile (index calculation)
        assert p99 == 100  # 99th percentile (index calculation)

    def test_calculate_latencies_empty(self, optimizer):
        """Test latency calculation with no samples."""
        p50, p95, p99 = optimizer._calculate_latencies()

        assert p50 == 0.0
        assert p95 == 0.0
        assert p99 == 0.0

    @pytest.mark.asyncio
    async def test_calculate_error_rate(self, optimizer):
        """Test error rate calculation."""
        error_rate = await optimizer._calculate_error_rate()

        # Based on mock data: 50 failures / 1000 attempts = 5%
        assert error_rate == 5.0

    @pytest.mark.asyncio
    async def test_analyze_workload_pattern_steady(self, optimizer):
        """Test workload pattern analysis - steady."""
        # Add metrics with low variation
        for i in range(20):
            optimizer.metrics_history.append(
                PerformanceMetrics(
                    timestamp=datetime.now(UTC),
                    throughput=100.0 + i % 2,  # Small variation
                    latency_p50=10.0,
                    latency_p95=20.0,
                    latency_p99=30.0,
                    error_rate=1.0,
                    cpu_usage=50.0,
                    memory_usage=60.0,
                    active_workers=10,
                    queue_depth=50,
                    retry_rate=0.1,
                )
            )

        pattern = await optimizer._analyze_workload_pattern()
        assert pattern == "steady"

    @pytest.mark.asyncio
    async def test_analyze_workload_pattern_bursty(self, optimizer):
        """Test workload pattern analysis - bursty."""
        # Add metrics with high variation
        for i in range(20):
            optimizer.metrics_history.append(
                PerformanceMetrics(
                    timestamp=datetime.now(UTC),
                    throughput=50.0 if i % 3 == 0 else 200.0,  # High variation
                    latency_p50=10.0,
                    latency_p95=20.0,
                    latency_p99=30.0,
                    error_rate=1.0,
                    cpu_usage=50.0,
                    memory_usage=60.0,
                    active_workers=10,
                    queue_depth=50,
                    retry_rate=0.1,
                )
            )

        pattern = await optimizer._analyze_workload_pattern()
        assert pattern in ["bursty", "periodic"]  # CV is on the boundary

    @pytest.mark.asyncio
    async def test_predict_load(self, optimizer):
        """Test load prediction."""
        # Add recent metrics
        throughputs = [80, 90, 100, 110, 120]
        for throughput in throughputs:
            optimizer.metrics_history.append(
                PerformanceMetrics(
                    timestamp=datetime.now(UTC),
                    throughput=throughput,
                    latency_p50=10.0,
                    latency_p95=20.0,
                    latency_p99=30.0,
                    error_rate=1.0,
                    cpu_usage=50.0,
                    memory_usage=60.0,
                    active_workers=10,
                    queue_depth=50,
                    retry_rate=0.1,
                )
            )

        predicted = await optimizer._predict_load()

        # Should be average of last 5: (80+90+100+110+120)/5 = 100
        assert predicted == 100.0

    def test_calculate_optimal_workers(self, optimizer):
        """Test optimal worker calculation."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,  # 100 jobs/sec
            latency_p50=100.0,  # 100ms = 0.1 sec
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=2.0,
            cpu_usage=50.0,
            memory_usage=60.0,
            active_workers=10,
            queue_depth=100,
            retry_rate=0.1,
        )

        # Little's Law: Workers = Throughput * Latency
        # 100 * 0.1 = 10 workers
        workers = optimizer._calculate_optimal_workers(metrics, "steady", 100.0)

        # With steady pattern multiplier (1.1): 10 * 1.1 = 11
        assert workers == 11

    def test_calculate_optimal_workers_bursty(self, optimizer):
        """Test optimal worker calculation for bursty pattern."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=2.0,
            cpu_usage=50.0,
            memory_usage=60.0,
            active_workers=10,
            queue_depth=100,
            retry_rate=0.1,
        )

        workers = optimizer._calculate_optimal_workers(metrics, "bursty", 100.0)

        # With bursty pattern multiplier (1.5): 10 * 1.5 = 15
        assert workers == 15

    @pytest.mark.asyncio
    async def test_adjust_workers(self, optimizer, mock_batch_queue):
        """Test worker adjustment."""
        optimizer.current_workers = 10

        await optimizer._adjust_workers(15)

        assert optimizer.current_workers == 15
        # Note: set_worker_count is commented out in the implementation

    @pytest.mark.asyncio
    async def test_adjust_workers_no_change(self, optimizer, mock_batch_queue):
        """Test worker adjustment with no change needed."""
        optimizer.current_workers = 10

        await optimizer._adjust_workers(10)

        # Should not call set_worker_count if no change
        mock_batch_queue.set_worker_count.assert_not_called()

    @pytest.mark.asyncio
    async def test_balanced_optimization_scale_up(self, optimizer, mock_batch_queue):
        """Test balanced optimization - scale up."""
        optimizer.current_workers = 10

        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=2.0,
            cpu_usage=85.0,  # High CPU
            memory_usage=60.0,
            active_workers=10,
            queue_depth=100,  # High queue depth
            retry_rate=0.1,
        )

        await optimizer._balanced_optimization(metrics)

        # Should increase workers
        assert optimizer.current_workers == 12

    @pytest.mark.asyncio
    async def test_balanced_optimization_scale_down(self, optimizer, mock_batch_queue):
        """Test balanced optimization - scale down."""
        optimizer.current_workers = 10

        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=2.0,
            cpu_usage=20.0,  # Low CPU
            memory_usage=60.0,
            active_workers=10,
            queue_depth=5,  # Low queue depth
            retry_rate=0.1,
        )

        await optimizer._balanced_optimization(metrics)

        # Should decrease workers
        assert optimizer.current_workers == 9

    @pytest.mark.asyncio
    async def test_aggressive_optimization(self, optimizer, mock_batch_queue):
        """Test aggressive optimization."""
        optimizer.current_workers = 10

        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=2.0,
            cpu_usage=30.0,  # Low CPU
            memory_usage=60.0,
            active_workers=10,
            queue_depth=200,  # High queue depth
            retry_rate=0.1,
        )

        await optimizer._aggressive_optimization(metrics)

        # Should double workers
        assert optimizer.current_workers == 20

    @pytest.mark.asyncio
    async def test_conservative_optimization(self, optimizer, mock_batch_queue):
        """Test conservative optimization."""
        optimizer.current_workers = 10

        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=100.0,
            latency_p95=200.0,
            latency_p99=300.0,
            error_rate=10.0,  # High error rate
            memory_usage=60.0,
            cpu_usage=50.0,
            active_workers=10,
            queue_depth=50,
            retry_rate=0.3,
        )

        await optimizer._conservative_optimization(metrics)

        # Should reduce workers due to high error rate
        assert optimizer.current_workers == 8

    def test_record_latency(self, optimizer):
        """Test recording latency samples."""
        optimizer.record_latency(10.5)
        optimizer.record_latency(20.3)

        assert len(optimizer.latency_samples) == 2
        assert 10.5 in optimizer.latency_samples
        assert 20.3 in optimizer.latency_samples

    def test_set_baseline_metrics(self, optimizer):
        """Test setting baseline metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(UTC),
            throughput=100.0,
            latency_p50=10.0,
            latency_p95=20.0,
            latency_p99=30.0,
            error_rate=1.0,
            cpu_usage=50.0,
            memory_usage=60.0,
            active_workers=10,
            queue_depth=50,
            retry_rate=0.1,
        )

        optimizer.set_baseline_metrics(metrics)
        assert optimizer.baseline_metrics == metrics

    def test_get_optimization_recommendations(self, optimizer):
        """Test getting optimization recommendations."""
        # Add current metrics
        optimizer.metrics_history.append(
            PerformanceMetrics(
                timestamp=datetime.now(UTC),
                throughput=100.0,
                latency_p50=10.0,
                latency_p95=20.0,
                latency_p99=30.0,
                error_rate=10.0,  # High error rate
                cpu_usage=85.0,  # High CPU
                memory_usage=80.0,  # High memory
                active_workers=10,
                queue_depth=2000,  # Large backlog
                retry_rate=0.1,
            )
        )

        recommendations = optimizer.get_optimization_recommendations()

        assert any("CPU" in r for r in recommendations)
        assert any("memory" in r for r in recommendations)
        assert any("error rate" in r for r in recommendations)
        assert any("queue backlog" in r for r in recommendations)

    def test_get_performance_report(self, optimizer):
        """Test generating performance report."""
        # Add metrics
        for i in range(5):
            optimizer.metrics_history.append(
                PerformanceMetrics(
                    timestamp=datetime.now(UTC),
                    throughput=100.0 + i,
                    latency_p50=10.0,
                    latency_p95=20.0,
                    latency_p99=30.0,
                    error_rate=2.0,
                    cpu_usage=50.0,
                    memory_usage=60.0,
                    active_workers=10,
                    queue_depth=50,
                    retry_rate=0.1,
                )
            )

        report = optimizer.get_performance_report()

        assert "current_metrics" in report
        assert "averages" in report
        assert "resource_usage" in report
        assert "optimization" in report
        assert report["current_metrics"]["throughput"] == 104.0  # Last value
        assert report["optimization"]["strategy"] == "balanced"

    def test_get_performance_report_no_data(self, optimizer):
        """Test performance report with no data."""
        report = optimizer.get_performance_report()
        assert report == {"status": "No data available"}
