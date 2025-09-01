"""Performance optimization for batch processing operations."""

import asyncio
import contextlib
import logging
import statistics
from asyncio import Task
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import psutil
from redis import Redis

from services.tracklist_service.src.progress.tracker import ProgressTracker
from services.tracklist_service.src.queue.batch_queue import BatchJobQueue
from services.tracklist_service.src.retry.retry_manager import RetryManager

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""

    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CUSTOM = "custom"


class ResourceType(Enum):
    """System resource types."""

    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK_IO = "disk_io"
    CONNECTIONS = "connections"


@dataclass
class PerformanceMetrics:
    """Performance metrics for batch processing."""

    timestamp: datetime
    throughput: float  # Jobs per second
    latency_p50: float  # Median latency
    latency_p95: float  # 95th percentile latency
    latency_p99: float  # 99th percentile latency
    error_rate: float  # Error percentage
    cpu_usage: float  # CPU percentage
    memory_usage: float  # Memory percentage
    active_workers: int
    queue_depth: int
    retry_rate: float


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""

    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    target_throughput: float | None = None  # Jobs per second
    max_latency_p95: float | None = None  # Maximum acceptable P95 latency
    max_error_rate: float = 0.05  # 5% error rate threshold
    min_workers: int = 2
    max_workers: int = 50
    scale_up_threshold: float = 0.8  # 80% resource usage
    scale_down_threshold: float = 0.3  # 30% resource usage
    monitoring_interval: int = 10  # Seconds
    optimization_interval: int = 60  # Seconds
    custom_rules: list[Callable[..., Any]] = field(default_factory=list)


@dataclass
class ResourceLimits:
    """Resource usage limits."""

    max_cpu_percent: float = 80.0
    max_memory_percent: float = 75.0
    max_connections: int = 1000
    max_disk_io_mbps: float = 100.0
    reserved_memory_mb: int = 512


class PerformanceOptimizer:
    """Optimizes batch processing performance dynamically."""

    def __init__(
        self,
        batch_queue: BatchJobQueue,
        progress_tracker: ProgressTracker,
        retry_manager: RetryManager,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """Initialize performance optimizer.

        Args:
            batch_queue: Batch queue instance
            progress_tracker: Progress tracker instance
            retry_manager: Retry manager instance
            redis_host: Redis host
            redis_port: Redis port
        """
        self.batch_queue = batch_queue
        self.progress_tracker = progress_tracker
        self.retry_manager = retry_manager
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Configuration
        self.config = OptimizationConfig()
        self.resource_limits = ResourceLimits()

        # Metrics storage
        self.metrics_history: deque[Any] = deque(maxlen=1000)
        self.latency_samples: deque[float] = deque(maxlen=10000)
        self.error_counts: defaultdict[str, int] = defaultdict(int)

        # Optimization state
        self.current_workers = self.config.min_workers
        self.last_optimization = datetime.now(UTC)
        self.optimization_enabled = True

        # Performance baselines
        self.baseline_metrics: PerformanceMetrics | None = None
        self.target_metrics: PerformanceMetrics | None = None

        # Monitoring tasks
        self.monitoring_task: Task[None] | None = None
        self.optimization_task: Task[None] | None = None

    async def start_monitoring(self) -> None:
        """Start performance monitoring and optimization."""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        if not self.optimization_task:
            self.optimization_task = asyncio.create_task(self._optimization_loop())

        logger.info("Performance monitoring and optimization started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring and optimization."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

        if self.optimization_task:
            self.optimization_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.optimization_task

        logger.info("Performance monitoring and optimization stopped")

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop."""
        while True:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)

                # Persist metrics
                await self._persist_metrics(metrics)

                # Check for anomalies
                await self._detect_anomalies(metrics)

                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.monitoring_interval)

    async def _optimization_loop(self) -> None:
        """Continuous optimization loop."""
        while True:
            try:
                if self.optimization_enabled:
                    await self.optimize_performance()

                await asyncio.sleep(self.config.optimization_interval)

            except Exception as e:
                logger.error(f"Optimization error: {e}")
                await asyncio.sleep(self.config.optimization_interval)

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics.

        Returns:
            Performance metrics
        """
        now = datetime.now(UTC)

        # Calculate throughput
        throughput = await self._calculate_throughput()

        # Calculate latencies
        latencies = self._calculate_latencies()

        # Calculate error rate
        error_rate = await self._calculate_error_rate()

        # Get resource usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent

        # Get queue status (mock for now, should be implemented in BatchJobQueue)
        queue_depth = 100  # Placeholder value

        # Get retry rate
        retry_stats = self.retry_manager.get_failure_stats()
        total_attempts = retry_stats.get("total_attempts", 1)
        retry_rate = retry_stats.get("total_retries", 0) / total_attempts if total_attempts > 0 else 0

        return PerformanceMetrics(
            timestamp=now,
            throughput=throughput,
            latency_p50=latencies[0],
            latency_p95=latencies[1],
            latency_p99=latencies[2],
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_workers=self.current_workers,
            queue_depth=queue_depth,
            retry_rate=retry_rate,
        )

    async def _calculate_throughput(self) -> float:
        """Calculate current throughput.

        Returns:
            Jobs per second
        """
        # Get completed jobs in last minute
        one_minute_ago = datetime.now(UTC) - timedelta(minutes=1)
        completed_count = 0

        # Check recent metrics
        for metrics in reversed(self.metrics_history):
            if metrics.timestamp < one_minute_ago:
                break
            completed_count += 1

        return completed_count / 60.0  # Jobs per second

    def _calculate_latencies(self) -> tuple[float, float, float]:
        """Calculate latency percentiles.

        Returns:
            Tuple of (P50, P95, P99) latencies
        """
        if not self.latency_samples:
            return (0.0, 0.0, 0.0)

        sorted_latencies = sorted(self.latency_samples)
        n = len(sorted_latencies)

        p50 = sorted_latencies[int(n * 0.5)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]

        return (p50, p95, p99)

    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate.

        Returns:
            Error rate as percentage
        """
        # Get error counts from retry manager
        stats = self.retry_manager.get_failure_stats()
        total = stats.get("total_attempts", 0)
        failures = stats.get("total_failures", 0)

        if total == 0:
            return 0.0

        return float((failures / total) * 100)

    async def optimize_performance(self) -> None:
        """Optimize performance based on current metrics."""
        if not self.metrics_history:
            return

        current_metrics = self.metrics_history[-1]

        # Apply optimization strategy
        if self.config.strategy == OptimizationStrategy.ADAPTIVE:
            await self._adaptive_optimization(current_metrics)
        elif self.config.strategy == OptimizationStrategy.AGGRESSIVE:
            await self._aggressive_optimization(current_metrics)
        elif self.config.strategy == OptimizationStrategy.CONSERVATIVE:
            await self._conservative_optimization(current_metrics)
        elif self.config.strategy == OptimizationStrategy.BALANCED:
            await self._balanced_optimization(current_metrics)
        elif self.config.strategy == OptimizationStrategy.CUSTOM:
            await self._custom_optimization(current_metrics)

        self.last_optimization = datetime.now(UTC)

    async def _adaptive_optimization(self, metrics: PerformanceMetrics) -> None:
        """Adaptive optimization based on workload patterns.

        Args:
            metrics: Current performance metrics
        """
        # Analyze workload pattern
        pattern = await self._analyze_workload_pattern()

        # Predict future load
        predicted_load = await self._predict_load()

        # Calculate optimal worker count
        optimal_workers = self._calculate_optimal_workers(metrics, pattern, predicted_load)

        # Apply changes
        await self._adjust_workers(optimal_workers)

        # Adjust rate limits
        await self._adjust_rate_limits(metrics, pattern)

        # Tune retry policies
        await self._tune_retry_policies(metrics)

    async def _aggressive_optimization(self, metrics: PerformanceMetrics) -> None:
        """Aggressive optimization for maximum throughput.

        Args:
            metrics: Current performance metrics
        """
        # Scale up aggressively if load is high
        if metrics.queue_depth > 100 or metrics.cpu_usage < 50:
            new_workers = min(self.current_workers * 2, self.config.max_workers)
            await self._adjust_workers(new_workers)

        # Increase rate limits
        await self._increase_rate_limits()

        # Reduce retry delays
        # Note: RetryPolicy uses base_delay, not initial_delay
        # self.retry_manager.default_policy.base_delay = 1.0
        # self.retry_manager.default_policy.max_delay = 30.0

    async def _conservative_optimization(self, metrics: PerformanceMetrics) -> None:
        """Conservative optimization for stability.

        Args:
            metrics: Current performance metrics
        """
        # Scale conservatively
        if metrics.error_rate > self.config.max_error_rate:
            # Reduce workers if errors are high
            new_workers = max(int(self.current_workers * 0.8), self.config.min_workers)
            await self._adjust_workers(new_workers)

        # Tighten rate limits
        await self._reduce_rate_limits()

        # Increase retry delays
        # Note: RetryPolicy uses base_delay, not initial_delay
        # self.retry_manager.default_policy.base_delay = 5.0
        # self.retry_manager.default_policy.max_delay = 300.0

    async def _balanced_optimization(self, metrics: PerformanceMetrics) -> None:
        """Balanced optimization for throughput and stability.

        Args:
            metrics: Current performance metrics
        """
        # Check resource usage
        if metrics.cpu_usage > self.config.scale_up_threshold * 100:
            # CPU is high, might need more workers
            if metrics.queue_depth > 50:
                new_workers = min(self.current_workers + 2, self.config.max_workers)
                await self._adjust_workers(new_workers)

        elif metrics.cpu_usage < self.config.scale_down_threshold * 100 and metrics.queue_depth < 10:
            # CPU is low and queue is small, might have too many workers
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            await self._adjust_workers(new_workers)

        # Adjust based on error rate
        if metrics.error_rate > self.config.max_error_rate:
            await self._reduce_rate_limits()
        elif metrics.error_rate < self.config.max_error_rate / 2:
            await self._increase_rate_limits()

    async def _custom_optimization(self, metrics: PerformanceMetrics) -> None:
        """Apply custom optimization rules.

        Args:
            metrics: Current performance metrics
        """
        for rule in self.config.custom_rules:
            try:
                await rule(self, metrics)
            except Exception as e:
                logger.error(f"Custom optimization rule failed: {e}")

    def _calculate_optimal_workers(self, metrics: PerformanceMetrics, pattern: str, predicted_load: float) -> int:
        """Calculate optimal number of workers.

        Args:
            metrics: Current metrics
            pattern: Workload pattern
            predicted_load: Predicted future load

        Returns:
            Optimal worker count
        """
        # Base calculation on Little's Law
        # Workers = Throughput * Latency

        target_throughput = self.config.target_throughput or metrics.throughput
        avg_latency = metrics.latency_p50 / 1000  # Convert to seconds

        base_workers = int(target_throughput * avg_latency)

        # Adjust for pattern
        if pattern == "bursty":
            base_workers = int(base_workers * 1.5)
        elif pattern == "steady":
            base_workers = int(base_workers * 1.1)

        # Adjust for predicted load
        if predicted_load > metrics.throughput * 1.5:
            base_workers = int(base_workers * 1.3)

        # Apply limits
        return max(self.config.min_workers, min(base_workers, self.config.max_workers))

    async def _analyze_workload_pattern(self) -> str:
        """Analyze workload pattern from metrics history.

        Returns:
            Pattern type (steady, bursty, periodic)
        """
        if len(self.metrics_history) < 10:
            return "unknown"

        # Calculate coefficient of variation
        throughputs = [m.throughput for m in self.metrics_history]
        if not throughputs or statistics.mean(throughputs) == 0:
            return "unknown"

        cv = statistics.stdev(throughputs) / statistics.mean(throughputs)

        if cv < 0.3:
            return "steady"
        if cv < 0.7:
            return "periodic"
        return "bursty"

    async def _predict_load(self) -> float:
        """Predict future load using simple moving average.

        Returns:
            Predicted throughput
        """
        if len(self.metrics_history) < 5:
            return 0.0

        # Simple moving average of last 5 measurements
        recent_throughputs = [m.throughput for m in list(self.metrics_history)[-5:]]

        return float(statistics.mean(recent_throughputs))

    async def _adjust_workers(self, target_workers: int) -> None:
        """Adjust number of workers.

        Args:
            target_workers: Target worker count
        """
        if target_workers == self.current_workers:
            return

        logger.info(f"Adjusting workers from {self.current_workers} to {target_workers}")

        # Update worker count in batch queue (mock for now, should be implemented)
        # await self.batch_queue.set_worker_count(target_workers)
        # Placeholder

        self.current_workers = target_workers

        # Persist configuration
        self.redis.hset("optimizer:config", "workers", str(target_workers))

    async def _adjust_rate_limits(self, metrics: PerformanceMetrics, pattern: str) -> None:
        """Adjust rate limits based on metrics and pattern.

        Args:
            metrics: Current metrics
            pattern: Workload pattern
        """
        # Implement rate limit adjustment logic
        # This would interact with the rate limiter component

    async def _increase_rate_limits(self) -> None:
        """Increase rate limits for higher throughput."""
        # Implement rate limit increase

    async def _reduce_rate_limits(self) -> None:
        """Reduce rate limits for stability."""
        # Implement rate limit reduction

    async def _tune_retry_policies(self, metrics: PerformanceMetrics) -> None:
        """Tune retry policies based on metrics.

        Args:
            metrics: Current metrics
        """
        if metrics.retry_rate > 0.2:  # High retry rate
            # Increase delays to reduce load
            # Note: RetryPolicy uses base_delay, not initial_delay
            # self.retry_manager.default_policy.base_delay *= 1.5
            pass
        elif metrics.retry_rate < 0.05:  # Low retry rate
            # Decrease delays for faster recovery
            # Note: RetryPolicy uses base_delay, not initial_delay
            # self.retry_manager.default_policy.base_delay *= 0.8
            pass

    async def _detect_anomalies(self, metrics: PerformanceMetrics) -> None:
        """Detect performance anomalies.

        Args:
            metrics: Current metrics
        """
        if not self.baseline_metrics:
            return

        # Check for significant deviations
        if metrics.throughput < self.baseline_metrics.throughput * 0.5:
            logger.warning(f"Throughput anomaly detected: {metrics.throughput}")

        if metrics.error_rate > self.baseline_metrics.error_rate * 2:
            logger.warning(f"Error rate anomaly detected: {metrics.error_rate}%")

        if metrics.latency_p95 > self.baseline_metrics.latency_p95 * 2:
            logger.warning(f"Latency anomaly detected: P95={metrics.latency_p95}ms")

    async def _persist_metrics(self, metrics: PerformanceMetrics) -> None:
        """Persist metrics to Redis.

        Args:
            metrics: Performance metrics
        """
        key = f"metrics:{metrics.timestamp.isoformat()}"
        self.redis.hset(
            key,
            mapping={
                "throughput": metrics.throughput,
                "latency_p50": metrics.latency_p50,
                "latency_p95": metrics.latency_p95,
                "latency_p99": metrics.latency_p99,
                "error_rate": metrics.error_rate,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "workers": metrics.active_workers,
                "queue_depth": metrics.queue_depth,
                "retry_rate": metrics.retry_rate,
            },
        )
        self.redis.expire(key, 86400)  # 24 hour TTL

    def record_latency(self, latency_ms: float) -> None:
        """Record latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_samples.append(latency_ms)

    def set_baseline_metrics(self, metrics: PerformanceMetrics | None = None) -> None:
        """Set baseline metrics for comparison.

        Args:
            metrics: Baseline metrics (uses current if None)
        """
        if metrics:
            self.baseline_metrics = metrics
        elif self.metrics_history:
            self.baseline_metrics = self.metrics_history[-1]

    def set_target_metrics(self, metrics: PerformanceMetrics) -> None:
        """Set target performance metrics.

        Args:
            metrics: Target metrics
        """
        self.target_metrics = metrics

    def get_optimization_recommendations(self) -> list[str]:
        """Get optimization recommendations based on current state.

        Returns:
            List of recommendations
        """
        recommendations = []

        if not self.metrics_history:
            return ["Insufficient data for recommendations"]

        current = self.metrics_history[-1]

        # CPU recommendations
        if current.cpu_usage > 80:
            recommendations.append("High CPU usage - consider scaling horizontally")
        elif current.cpu_usage < 20:
            recommendations.append("Low CPU usage - consider reducing workers")

        # Memory recommendations
        if current.memory_usage > 75:
            recommendations.append("High memory usage - optimize memory consumption")

        # Error rate recommendations
        if current.error_rate > 5:
            recommendations.append("High error rate - review retry policies and error handling")

        # Latency recommendations
        if self.config.max_latency_p95 and current.latency_p95 > self.config.max_latency_p95:
            recommendations.append("P95 latency exceeds target - optimize processing logic")

        # Queue depth recommendations
        if current.queue_depth > 1000:
            recommendations.append("Large queue backlog - increase processing capacity")

        return recommendations or ["System performing within normal parameters"]

    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report.

        Returns:
            Performance report dictionary
        """
        if not self.metrics_history:
            return {"status": "No data available"}

        recent_metrics = list(self.metrics_history)[-10:]

        return {
            "current_metrics": {
                "throughput": recent_metrics[-1].throughput,
                "latency_p50": recent_metrics[-1].latency_p50,
                "latency_p95": recent_metrics[-1].latency_p95,
                "error_rate": recent_metrics[-1].error_rate,
                "workers": self.current_workers,
            },
            "averages": {
                "avg_throughput": statistics.mean([m.throughput for m in recent_metrics]),
                "avg_latency": statistics.mean([m.latency_p50 for m in recent_metrics]),
                "avg_error_rate": statistics.mean([m.error_rate for m in recent_metrics]),
            },
            "resource_usage": {
                "cpu": recent_metrics[-1].cpu_usage,
                "memory": recent_metrics[-1].memory_usage,
                "queue_depth": recent_metrics[-1].queue_depth,
            },
            "optimization": {
                "strategy": self.config.strategy.value,
                "last_optimization": self.last_optimization.isoformat(),
                "recommendations": self.get_optimization_recommendations(),
            },
        }
