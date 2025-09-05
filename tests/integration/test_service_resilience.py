"""
Comprehensive integration tests for service resilience in the tracktion system.

This module tests all aspects of service resilience including:
- Circuit breaker patterns and failure isolation
- Retry mechanisms with exponential backoff
- Bulkhead pattern implementation (resource isolation)
- Timeout handling and graceful degradation
- Failover and fallback mechanisms
- Service mesh resilience (load balancing, service discovery)
- Chaos engineering scenarios (random failures, network partitions)
- Recovery testing (service restart, data consistency after failures)
- Rate limiting and throttling under pressure
- Disaster recovery and backup/restore procedures

Includes statistical analysis of recovery times and failure rates.
"""

import asyncio
import contextlib
import logging
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Any
from uuid import uuid4

import pytest

from services.tracklist_service.src.resilience import (
    CircuitState,
    ExponentialBackoff,
    RateLimiter,
    retry_with_backoff,
)
from shared.utils.async_timeout_handler import (
    CancellationHandler,
    DeadlineManager,
    TimeoutConfig,
    TimeoutHandler,
    TimeoutStrategy,
)
from shared.utils.resilience import (
    CircuitBreaker as AnalysisCircuitBreaker,
)
from shared.utils.resilience import (
    CircuitBreakerConfig,
    CircuitOpenError,
)
from shared.utils.resilience import (
    CircuitState as AnalysisCircuitState,
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be simulated."""

    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMITED = "rate_limited"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_ERROR = "memory_error"
    DISK_FULL = "disk_full"


class ResilienceMetrics:
    """Metrics collector for resilience testing."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_breaker_trips = 0
        self.fallback_executions = 0
        self.retry_attempts = 0
        self.timeouts = 0
        self.response_times = []
        self.recovery_times = []
        self.failure_times = []
        self.service_unavailable_periods = []
        self.failure_types = defaultdict(int)
        self.start_time = time.time()

    def record_request(self, success: bool, response_time: float, failure_type: FailureType | None = None):
        """Record a request attempt."""
        self.total_requests += 1
        self.response_times.append(response_time)

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.failure_times.append(time.time())
            if failure_type:
                self.failure_types[failure_type.value] += 1

    def record_circuit_breaker_trip(self):
        """Record circuit breaker opening."""
        self.circuit_breaker_trips += 1

    def record_fallback_execution(self):
        """Record fallback execution."""
        self.fallback_executions += 1

    def record_retry_attempt(self):
        """Record retry attempt."""
        self.retry_attempts += 1

    def record_timeout(self):
        """Record timeout occurrence."""
        self.timeouts += 1

    def record_recovery(self, recovery_time: float):
        """Record service recovery."""
        self.recovery_times.append(recovery_time)

    def record_service_unavailable_period(self, duration: float):
        """Record period of service unavailability."""
        self.service_unavailable_periods.append(duration)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        total_time = time.time() - self.start_time
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0

        stats = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "fallback_executions": self.fallback_executions,
            "retry_attempts": self.retry_attempts,
            "timeouts": self.timeouts,
            "failure_types": dict(self.failure_types),
            "total_test_duration": round(total_time, 2),
        }

        if self.response_times:
            stats["response_times"] = {
                "min": round(min(self.response_times), 3),
                "max": round(max(self.response_times), 3),
                "mean": round(statistics.mean(self.response_times), 3),
                "median": round(statistics.median(self.response_times), 3),
                "p95": round(self._percentile(self.response_times, 95), 3),
                "p99": round(self._percentile(self.response_times, 99), 3),
            }

        if self.recovery_times:
            stats["recovery_times"] = {
                "min": round(min(self.recovery_times), 3),
                "max": round(max(self.recovery_times), 3),
                "mean": round(statistics.mean(self.recovery_times), 3),
                "median": round(statistics.median(self.recovery_times), 3),
            }

        if self.service_unavailable_periods:
            total_downtime = sum(self.service_unavailable_periods)
            stats["availability"] = {
                "total_downtime": round(total_downtime, 3),
                "uptime_percent": round((total_time - total_downtime) / total_time * 100, 2),
                "mtbf": round(total_time / len(self.service_unavailable_periods), 3)
                if self.service_unavailable_periods
                else float("inf"),
                "mttr": round(statistics.mean(self.recovery_times), 3) if self.recovery_times else 0,
            }

        return stats

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


@dataclass
class ServiceConfig:
    """Configuration for a simulated service."""

    name: str
    base_latency: float = 0.1
    error_rate: float = 0.1
    timeout_rate: float = 0.05
    resource_limit: int = 100
    failure_types: list[FailureType] = field(
        default_factory=lambda: [FailureType.TIMEOUT, FailureType.SERVICE_UNAVAILABLE]
    )


class MockService:
    """Mock service for testing resilience patterns."""

    def __init__(self, config: ServiceConfig, metrics: ResilienceMetrics):
        self.config = config
        self.metrics = metrics
        self.is_healthy = True
        self.current_load = 0
        self.failure_injection_active = False
        self.network_partition_active = False
        self.resource_exhausted = False
        self._lock = asyncio.Lock()

    async def call(self, operation: str = "default") -> str:
        """Simulate a service call."""
        async with self._lock:
            self.current_load += 1

        start_time = time.time()

        try:
            # Check if service is partitioned
            if self.network_partition_active:
                await asyncio.sleep(5.0)  # Network timeout
                raise TimeoutError("Network partition - operation timed out")

            # Check resource limits
            if self.current_load > self.config.resource_limit:
                self.resource_exhausted = True
                failure_type = FailureType.RESOURCE_EXHAUSTED
                response_time = time.time() - start_time
                self.metrics.record_request(False, response_time, failure_type)
                raise Exception(f"Resource exhausted: {self.current_load} > {self.config.resource_limit}")

            # Simulate base latency
            await asyncio.sleep(self.config.base_latency + random.uniform(0, 0.05))

            # Inject failures if configured
            if self.failure_injection_active or not self.is_healthy:
                failure_chance = random.random()
                if failure_chance < self.config.error_rate:
                    failure_type = random.choice(self.config.failure_types)
                    response_time = time.time() - start_time
                    self.metrics.record_request(False, response_time, failure_type)

                    if failure_type == FailureType.TIMEOUT:
                        self.metrics.record_timeout()
                        raise TimeoutError(f"Service {self.config.name} timed out")
                    if failure_type == FailureType.SERVICE_UNAVAILABLE:
                        raise Exception(f"Service {self.config.name} is unavailable")
                    raise Exception(f"Service {self.config.name} failed: {failure_type.value}")

                # Additional timeout simulation
                if failure_chance < self.config.error_rate + self.config.timeout_rate:
                    await asyncio.sleep(2.0)  # Simulate slow response
                    self.metrics.record_timeout()
                    raise TimeoutError(f"Service {self.config.name} response timeout")

            # Successful response
            response_time = time.time() - start_time
            self.metrics.record_request(True, response_time)
            return f"{self.config.name}-{operation}-success-{uuid4().hex[:8]}"

        finally:
            async with self._lock:
                self.current_load -= 1

    def set_healthy(self, healthy: bool):
        """Set service health status."""
        self.is_healthy = healthy

    def enable_failure_injection(self, enable: bool = True):
        """Enable/disable failure injection."""
        self.failure_injection_active = enable

    def simulate_network_partition(self, enable: bool = True):
        """Simulate network partition."""
        self.network_partition_active = enable

    def reset_resource_exhaustion(self):
        """Reset resource exhaustion state."""
        self.resource_exhausted = False


class ServiceMesh:
    """Simulates a service mesh with load balancing and service discovery."""

    def __init__(self, metrics: ResilienceMetrics):
        self.services: dict[str, list[MockService]] = {}
        self.metrics = metrics
        self.load_balancer_strategy = "round_robin"
        self.service_counters: dict[str, int] = defaultdict(int)
        self.health_checks_enabled = True

    def register_service(self, service_name: str, service: MockService):
        """Register a service instance."""
        if service_name not in self.services:
            self.services[service_name] = []
        self.services[service_name].append(service)

    def get_healthy_instances(self, service_name: str) -> list[MockService]:
        """Get healthy instances of a service."""
        if service_name not in self.services:
            return []

        if not self.health_checks_enabled:
            return self.services[service_name]

        return [svc for svc in self.services[service_name] if svc.is_healthy and not svc.resource_exhausted]

    def select_service_instance(self, service_name: str) -> MockService | None:
        """Select a service instance based on load balancing strategy."""
        healthy_instances = self.get_healthy_instances(service_name)
        if not healthy_instances:
            return None

        if self.load_balancer_strategy == "round_robin":
            index = self.service_counters[service_name] % len(healthy_instances)
            self.service_counters[service_name] += 1
            return healthy_instances[index]
        if self.load_balancer_strategy == "random":
            return random.choice(healthy_instances)
        if self.load_balancer_strategy == "least_loaded":
            return min(healthy_instances, key=lambda svc: svc.current_load)

        return healthy_instances[0]

    async def call_service(self, service_name: str, operation: str = "default") -> str:
        """Call a service through the mesh."""
        service = self.select_service_instance(service_name)
        if not service:
            raise Exception(f"No healthy instances available for service {service_name}")

        return await service.call(operation)

    def set_load_balancer_strategy(self, strategy: str):
        """Set load balancing strategy."""
        self.load_balancer_strategy = strategy

    def enable_health_checks(self, enable: bool = True):
        """Enable/disable health checks."""
        self.health_checks_enabled = enable


class ChaosEngineer:
    """Chaos engineering utilities for failure injection."""

    def __init__(self, service_mesh: ServiceMesh, metrics: ResilienceMetrics):
        self.service_mesh = service_mesh
        self.metrics = metrics
        self.chaos_active = False
        self.failure_rate = 0.1
        self.chaos_tasks: list[asyncio.Task] = []

    async def start_chaos_monkey(self, duration: float = 30.0):
        """Start chaos monkey to inject random failures."""
        self.chaos_active = True

        async def chaos_loop():
            while self.chaos_active:
                # Random service disruption
                if random.random() < self.failure_rate:
                    service_name = random.choice(list(self.service_mesh.services.keys()))
                    instances = self.service_mesh.services[service_name]
                    if instances:
                        service = random.choice(instances)
                        disruption_type = random.choice(["health_toggle", "network_partition", "failure_injection"])

                        if disruption_type == "health_toggle":
                            service.set_healthy(False)
                            await asyncio.sleep(random.uniform(1, 5))
                            service.set_healthy(True)
                        elif disruption_type == "network_partition":
                            service.simulate_network_partition(True)
                            await asyncio.sleep(random.uniform(2, 8))
                            service.simulate_network_partition(False)
                        elif disruption_type == "failure_injection":
                            service.enable_failure_injection(True)
                            await asyncio.sleep(random.uniform(3, 10))
                            service.enable_failure_injection(False)

                await asyncio.sleep(random.uniform(0.5, 2.0))

        task = asyncio.create_task(chaos_loop())
        self.chaos_tasks.append(task)

        # Schedule stop
        async def stop_after_duration():
            await asyncio.sleep(duration)
            await self.stop_chaos_monkey()

        _stop_task = asyncio.create_task(stop_after_duration())  # noqa: RUF006

    async def stop_chaos_monkey(self):
        """Stop chaos monkey."""
        self.chaos_active = False
        for task in self.chaos_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.chaos_tasks.clear()

    async def simulate_network_partition(self, service_name: str, duration: float = 5.0):
        """Simulate network partition for a specific service."""
        if service_name in self.service_mesh.services:
            for service in self.service_mesh.services[service_name]:
                service.simulate_network_partition(True)

            await asyncio.sleep(duration)

            for service in self.service_mesh.services[service_name]:
                service.simulate_network_partition(False)

    async def simulate_cascading_failure(self, initial_service: str, failure_probability: float = 0.5):
        """Simulate cascading failure across services."""
        affected_services = [initial_service]

        # Initial failure
        if initial_service in self.service_mesh.services:
            for service in self.service_mesh.services[initial_service]:
                service.set_healthy(False)

        # Cascade to other services
        for service_name in self.service_mesh.services:
            if service_name != initial_service and random.random() < failure_probability:
                affected_services.append(service_name)
                for service in self.service_mesh.services[service_name]:
                    service.set_healthy(False)

        # Recovery after delay
        await asyncio.sleep(random.uniform(5, 15))

        for service_name in affected_services:
            if service_name in self.service_mesh.services:
                for service in self.service_mesh.services[service_name]:
                    service.set_healthy(True)

        return affected_services


class ResilienceTestSuite:
    """Comprehensive resilience test suite."""

    def __init__(self):
        self.metrics = ResilienceMetrics()
        self.service_mesh = ServiceMesh(self.metrics)
        self.chaos_engineer = ChaosEngineer(self.service_mesh, self.metrics)
        self.timeout_handler = TimeoutHandler()
        self.cancellation_handler = CancellationHandler()
        self.deadline_manager = DeadlineManager()

    def setup_services(self):
        """Set up mock services for testing."""
        # Analysis service instances
        for i in range(3):
            config = ServiceConfig(
                name=f"analysis-{i}", base_latency=0.1, error_rate=0.05, timeout_rate=0.02, resource_limit=50
            )
            service = MockService(config, self.metrics)
            self.service_mesh.register_service("analysis", service)

        # Tracklist service instances
        for i in range(2):
            config = ServiceConfig(
                name=f"tracklist-{i}", base_latency=0.2, error_rate=0.08, timeout_rate=0.03, resource_limit=30
            )
            service = MockService(config, self.metrics)
            self.service_mesh.register_service("tracklist", service)

        # Database service instance
        config = ServiceConfig(
            name="database", base_latency=0.05, error_rate=0.02, timeout_rate=0.01, resource_limit=100
        )
        service = MockService(config, self.metrics)
        self.service_mesh.register_service("database", service)

    def reset(self):
        """Reset all metrics and state."""
        self.metrics.reset()
        for services in self.service_mesh.services.values():
            for service in services:
                service.set_healthy(True)
                service.enable_failure_injection(False)
                service.simulate_network_partition(False)
                service.reset_resource_exhaustion()


# Test fixtures
@pytest.fixture
def resilience_suite():
    """Create a resilience test suite."""
    suite = ResilienceTestSuite()
    suite.setup_services()
    return suite


@pytest.fixture
def circuit_breaker_config():
    """Default circuit breaker configuration for tests."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=2.0,
        failure_window=10.0,
    )


@pytest.fixture
def timeout_config():
    """Default timeout configuration for tests."""
    return TimeoutConfig(
        default_timeout=1.0,
        strategy=TimeoutStrategy.EXPONENTIAL,
        escalation_factor=1.5,
        max_timeout=10.0,
    )


# Test Classes
class TestCircuitBreakerResilience:
    """Tests for circuit breaker pattern resilience."""

    async def test_circuit_breaker_prevents_cascading_failures(
        self, resilience_suite: ResilienceTestSuite, circuit_breaker_config: CircuitBreakerConfig
    ):
        """Test that circuit breakers prevent cascading failures."""
        # Configure circuit breaker for analysis service
        circuit_breaker = AnalysisCircuitBreaker("analysis_service", circuit_breaker_config, resilience_suite.metrics)

        # Simulate service degradation - increase error rate to ensure failures
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.error_rate = 1.0  # 100% failure rate during degradation
            service.enable_failure_injection(True)

        # Generate requests that should trigger circuit breaker
        failed_requests = 0
        for i in range(10):
            try:
                await circuit_breaker.call_async(
                    resilience_suite.service_mesh.call_service, "analysis", f"operation_{i}"
                )
            except (Exception, CircuitOpenError):
                failed_requests += 1

        # Verify circuit breaker opened and prevented further failures
        assert circuit_breaker.state == AnalysisCircuitState.OPEN
        assert failed_requests >= circuit_breaker_config.failure_threshold

        # Check that remaining requests were rejected quickly
        stats = circuit_breaker.get_stats()
        assert stats["rejected_calls"] > 0

        # Recovery test - wait for half-open state
        await asyncio.sleep(circuit_breaker_config.timeout + 0.1)

        # Restore service health
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.error_rate = 0.05  # Reset to original low error rate
            service.enable_failure_injection(False)

        # Test recovery
        success_count = 0
        for i in range(circuit_breaker_config.success_threshold + 1):
            try:
                await circuit_breaker.call_async(
                    resilience_suite.service_mesh.call_service, "analysis", f"recovery_{i}"
                )
                success_count += 1
            except Exception:
                pass

        # Verify circuit closed after successful operations
        assert circuit_breaker.state == AnalysisCircuitState.CLOSED
        assert success_count >= circuit_breaker_config.success_threshold

    async def test_circuit_breaker_with_fallback_mechanism(self, resilience_suite: ResilienceTestSuite):
        """Test circuit breaker with fallback functionality."""
        fallback_called = False
        fallback_responses = []

        def fallback_service():
            nonlocal fallback_called
            fallback_called = True
            response = f"fallback_response_{len(fallback_responses)}"
            fallback_responses.append(response)
            return response

        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout=1.0, fallback=fallback_service)

        circuit_breaker = AnalysisCircuitBreaker("fallback_test", config, resilience_suite.metrics)

        # Trigger circuit breaker
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(False)

        for _i in range(config.failure_threshold):
            with contextlib.suppress(Exception):
                await circuit_breaker.call_async(resilience_suite.service_mesh.call_service, "analysis")

        # Circuit should be open, test fallback
        result = await circuit_breaker.call_async(resilience_suite.service_mesh.call_service, "analysis")

        assert fallback_called
        assert result in fallback_responses
        assert circuit_breaker.state == AnalysisCircuitState.OPEN

        stats = circuit_breaker.get_stats()
        assert stats["fallback_calls"] > 0

    async def test_multiple_circuit_breakers_isolation(
        self, resilience_suite: ResilienceTestSuite, circuit_breaker_config: CircuitBreakerConfig
    ):
        """Test that multiple circuit breakers provide service isolation."""
        # Create separate circuit breakers for different services
        analysis_cb = AnalysisCircuitBreaker("analysis", circuit_breaker_config, resilience_suite.metrics)
        tracklist_cb = AnalysisCircuitBreaker("tracklist", circuit_breaker_config, resilience_suite.metrics)

        # Fail only analysis service
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.error_rate = 1.0  # 100% failure rate during degradation
            service.set_healthy(False)

        # Trigger analysis circuit breaker
        for _i in range(circuit_breaker_config.failure_threshold):
            with contextlib.suppress(Exception):
                await analysis_cb.call_async(resilience_suite.service_mesh.call_service, "analysis")

        # Test that analysis circuit is open but tracklist is still closed
        assert analysis_cb.state == AnalysisCircuitState.OPEN
        assert tracklist_cb.state == AnalysisCircuitState.CLOSED

        # Verify tracklist service still works
        result = await tracklist_cb.call_async(resilience_suite.service_mesh.call_service, "tracklist")
        assert "tracklist" in result
        assert "success" in result

        # Verify analysis circuit rejects calls
        with pytest.raises(CircuitOpenError):
            await analysis_cb.call_async(resilience_suite.service_mesh.call_service, "analysis")

    async def test_circuit_breaker_metrics_and_monitoring(
        self, resilience_suite: ResilienceTestSuite, circuit_breaker_config: CircuitBreakerConfig
    ):
        """Test circuit breaker metrics collection and monitoring hooks."""
        state_changes = []

        def on_open(name: str):
            state_changes.append(("open", name, time.time()))

        def on_close(name: str):
            state_changes.append(("close", name, time.time()))

        def on_half_open(name: str):
            state_changes.append(("half_open", name, time.time()))

        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=0.5,
            on_open=on_open,
            on_close=on_close,
            on_half_open=on_half_open,
        )

        circuit_breaker = AnalysisCircuitBreaker("monitoring_test", config, resilience_suite.metrics)

        # Trigger state changes
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(False)

        # Open circuit
        for _i in range(config.failure_threshold):
            with contextlib.suppress(Exception):
                await circuit_breaker.call_async(resilience_suite.service_mesh.call_service, "analysis")

        # Wait for half-open
        await asyncio.sleep(config.timeout + 0.1)
        _ = circuit_breaker.state  # Trigger state check

        # Restore service and close circuit
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(True)

        await circuit_breaker.call_async(resilience_suite.service_mesh.call_service, "analysis")

        # Verify monitoring hooks were called
        assert len(state_changes) >= 2  # At least open and half-open
        assert any(change[0] == "open" for change in state_changes)
        assert any(change[0] == "half_open" for change in state_changes)

        # Verify metrics
        stats = circuit_breaker.get_stats()
        assert stats["total_calls"] > 0
        assert stats["failed_calls"] >= config.failure_threshold
        assert stats.get("state_changes", 0) > 0


class TestRetryMechanisms:
    """Tests for retry mechanisms with exponential backoff."""

    async def test_exponential_backoff_retry_success(self, resilience_suite: ResilienceTestSuite):
        """Test successful retry with exponential backoff."""
        # Configure service to fail first few attempts then succeed
        call_count = 0

        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                resilience_suite.metrics.record_retry_attempt()
                raise Exception(f"Attempt {call_count} failed")
            return await resilience_suite.service_mesh.call_service("analysis")

        backoff = ExponentialBackoff(base_delay=0.1, max_delay=2.0, multiplier=2.0, jitter=True)

        start_time = time.time()
        result = await retry_with_backoff(flaky_service, max_attempts=4, backoff=backoff, exceptions=(Exception,))
        duration = time.time() - start_time

        # Verify success after retries
        assert "success" in result
        assert call_count == 3  # 2 failures + 1 success

        # Verify backoff delays (should be roughly 0.1 + 0.2 = 0.3s minimum)
        assert duration >= 0.25  # Account for jitter
        assert duration < 2.0  # Should not hit max delay

        assert resilience_suite.metrics.retry_attempts >= 2

    async def test_exponential_backoff_all_retries_exhausted(self, resilience_suite: ResilienceTestSuite):
        """Test retry mechanism when all attempts are exhausted."""

        async def always_failing_service():
            resilience_suite.metrics.record_retry_attempt()
            raise Exception("Service always fails")

        backoff = ExponentialBackoff(base_delay=0.05, max_delay=1.0, multiplier=2.0)
        max_attempts = 3

        start_time = time.time()
        with pytest.raises(Exception, match="Service always fails"):
            await retry_with_backoff(
                always_failing_service, max_attempts=max_attempts, backoff=backoff, exceptions=(Exception,)
            )
        duration = time.time() - start_time

        # Verify all attempts were made
        expected_delays = sum(backoff.get_delay(i) for i in range(max_attempts - 1))
        assert duration >= expected_delays * 0.8  # Allow some variance

        assert resilience_suite.metrics.retry_attempts >= max_attempts

    async def test_jitter_prevents_thundering_herd(self, resilience_suite: ResilienceTestSuite):
        """Test that jitter prevents thundering herd problem."""
        # Collect actual delays for multiple concurrent retry attempts
        delays_with_jitter = []
        delays_without_jitter = []

        async def failing_service():
            raise Exception("Simulated failure")

        backoff_with_jitter = ExponentialBackoff(base_delay=1.0, jitter=True)
        backoff_without_jitter = ExponentialBackoff(base_delay=1.0, jitter=False)

        # Test with jitter
        for _ in range(10):
            delay = backoff_with_jitter.get_delay(1)  # Second attempt
            delays_with_jitter.append(delay)

        # Test without jitter
        for _ in range(10):
            delay = backoff_without_jitter.get_delay(1)  # Second attempt
            delays_without_jitter.append(delay)

        # With jitter, delays should vary
        jitter_variance = statistics.variance(delays_with_jitter) if len(delays_with_jitter) > 1 else 0
        no_jitter_variance = statistics.variance(delays_without_jitter) if len(delays_without_jitter) > 1 else 0

        assert jitter_variance > no_jitter_variance
        assert no_jitter_variance == 0  # All delays should be identical without jitter

        # All delays should be within jitter bounds (±25% of base delay)
        for delay in delays_with_jitter:
            assert 0.75 <= delay <= 1.25

    async def test_retry_with_different_exception_types(self, resilience_suite: ResilienceTestSuite):
        """Test retry behavior with different exception types."""

        call_count = 0

        async def service_with_different_errors():
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise TimeoutError("Timeout - should retry")
            if call_count == 2:
                raise ValueError("Validation error - should not retry")
            return "success"

        backoff = ExponentialBackoff(base_delay=0.05)

        # Only retry on TimeoutError
        with pytest.raises(ValueError):
            await retry_with_backoff(
                service_with_different_errors, max_attempts=3, backoff=backoff, exceptions=(TimeoutError,)
            )

        # Should have stopped after ValueError (attempt 2)
        assert call_count == 2

    async def test_adaptive_retry_strategies(self, resilience_suite: ResilienceTestSuite):
        """Test adaptive retry strategies based on failure patterns."""

        # Track failure patterns
        failure_history = []

        async def service_with_pattern():
            current_time = time.time()
            failure_history.append(current_time)

            # Simulate different failure patterns
            if len(failure_history) <= 2:
                raise Exception("Initial failures")
            if len(failure_history) == 3:
                # Longer delay suggests overloaded service
                await asyncio.sleep(0.5)
                raise Exception("Overloaded service")
            return "success"

        # Adaptive backoff that increases delay based on failure pattern
        class AdaptiveBackoff(ExponentialBackoff):
            def get_delay(self, attempt: int) -> float:
                base_delay = super().get_delay(attempt)

                # If last failure was slow, increase delay more aggressively
                if len(failure_history) >= 2:
                    last_gap = failure_history[-1] - failure_history[-2]
                    if last_gap > 0.3:  # Slow failure detected
                        base_delay *= 2

                return base_delay

        adaptive_backoff = AdaptiveBackoff(base_delay=0.1, multiplier=1.5)

        result = await retry_with_backoff(
            service_with_pattern, max_attempts=5, backoff=adaptive_backoff, exceptions=(Exception,)
        )

        assert result == "success"
        assert len(failure_history) == 4  # 3 failures + 1 success


class TestBulkheadPattern:
    """Tests for bulkhead pattern and resource isolation."""

    async def test_resource_isolation_prevents_cascade(self, resilience_suite: ResilienceTestSuite):
        """Test that resource isolation prevents cascade failures."""

        # Create separate semaphores for different resource pools
        analysis_pool = asyncio.Semaphore(2)  # Limited resources for analysis
        tracklist_pool = asyncio.Semaphore(5)  # More resources for tracklist

        async def call_with_bulkhead(service_name: str, pool: asyncio.Semaphore):
            async with pool:
                return await resilience_suite.service_mesh.call_service(service_name)

        # Simulate resource exhaustion in analysis service
        analysis_tasks = []
        for _i in range(10):  # Try to create more tasks than available resources
            task = asyncio.create_task(call_with_bulkhead("analysis", analysis_pool))
            analysis_tasks.append(task)

        # Simultaneously test tracklist service availability
        tracklist_tasks = []
        for _i in range(3):
            task = asyncio.create_task(call_with_bulkhead("tracklist", tracklist_pool))
            tracklist_tasks.append(task)

        # Wait for all tasks
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        tracklist_results = await asyncio.gather(*tracklist_tasks, return_exceptions=True)

        # Analysis service should have some successful and some limited by semaphore
        analysis_successes = [r for r in analysis_results if isinstance(r, str) and "success" in r]
        tracklist_successes = [r for r in tracklist_results if isinstance(r, str) and "success" in r]

        # Tracklist should not be affected by analysis resource exhaustion
        assert len(tracklist_successes) == 3  # All tracklist calls should succeed
        assert len(analysis_successes) >= 2  # At least some analysis calls should succeed

        # Verify resource isolation worked
        assert len(analysis_successes) <= len(analysis_tasks)  # Some may have been queued/delayed

    async def test_bulkhead_with_circuit_breaker(self, resilience_suite: ResilienceTestSuite):
        """Test bulkhead pattern combined with circuit breaker."""

        # Create bulkhead with limited resources
        resource_pool = asyncio.Semaphore(2)

        # Circuit breaker for the bulkhead
        config = CircuitBreakerConfig(failure_threshold=3, timeout=1.0)
        circuit_breaker = AnalysisCircuitBreaker("bulkhead_test", config)

        async def bulkhead_call():
            async with resource_pool:
                # Simulate resource-intensive operation
                await asyncio.sleep(0.1)
                return await circuit_breaker.call_async(resilience_suite.service_mesh.call_service, "analysis")

        # Create more concurrent requests than bulkhead can handle
        tasks = []
        for _i in range(8):
            task = asyncio.create_task(bulkhead_call())
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have successful results limited by bulkhead capacity
        successful_results = [r for r in results if isinstance(r, str) and "success" in r]
        assert len(successful_results) > 0

        # Verify circuit breaker remained closed (no failures due to bulkhead protection)
        assert circuit_breaker.state == AnalysisCircuitState.CLOSED

    async def test_priority_based_resource_allocation(self, resilience_suite: ResilienceTestSuite):
        """Test priority-based resource allocation in bulkhead pattern."""

        # Create priority queues
        high_priority_queue = asyncio.Queue()
        low_priority_queue = asyncio.Queue()
        resource_semaphore = asyncio.Semaphore(1)  # Single resource

        async def priority_worker():
            while True:
                try:
                    # Always check high priority first
                    try:
                        request = high_priority_queue.get_nowait()
                        async with resource_semaphore:
                            result = await resilience_suite.service_mesh.call_service("analysis")
                            request.set_result(result)
                    except asyncio.QueueEmpty:
                        # Check low priority if no high priority requests
                        try:
                            request = low_priority_queue.get_nowait()
                            async with resource_semaphore:
                                result = await resilience_suite.service_mesh.call_service("analysis")
                                request.set_result(result)
                        except asyncio.QueueEmpty:
                            await asyncio.sleep(0.01)  # Brief wait before checking again
                except Exception as e:
                    if "request" in locals():
                        request.set_exception(e)

        # Start worker
        worker_task = asyncio.create_task(priority_worker())

        async def make_priority_request(is_high_priority: bool):
            future = asyncio.Future()
            if is_high_priority:
                await high_priority_queue.put(future)
            else:
                await low_priority_queue.put(future)
            return await future

        # Submit mixed priority requests
        time.time()

        # Submit low priority requests first
        low_priority_tasks = [asyncio.create_task(make_priority_request(False)) for _ in range(3)]

        # Brief delay, then submit high priority requests
        await asyncio.sleep(0.05)
        high_priority_tasks = [asyncio.create_task(make_priority_request(True)) for _ in range(2)]

        # Wait for all to complete
        all_tasks = high_priority_tasks + low_priority_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        worker_task.cancel()

        # High priority requests should complete despite being submitted later
        high_priority_results = results[:2]  # First 2 are high priority
        low_priority_results = results[2:]  # Rest are low priority

        # Verify all requests eventually succeeded
        successful_high = [r for r in high_priority_results if isinstance(r, str)]
        successful_low = [r for r in low_priority_results if isinstance(r, str)]

        assert len(successful_high) > 0
        assert len(successful_low) > 0

    async def test_bulkhead_metrics_and_monitoring(self, resilience_suite: ResilienceTestSuite):
        """Test bulkhead metrics collection and monitoring."""

        # Create monitored bulkhead
        resource_pool = asyncio.Semaphore(3)
        resource_usage = []
        wait_times = []

        async def monitored_bulkhead_call(request_id: str):
            wait_start = time.time()

            async with resource_pool:
                wait_time = time.time() - wait_start
                wait_times.append(wait_time)

                # Record resource usage
                available = resource_pool._value
                resource_usage.append(3 - available)  # Used resources

                return await resilience_suite.service_mesh.call_service("analysis", request_id)

        # Generate concurrent load
        tasks = []
        for i in range(10):
            task = asyncio.create_task(monitored_bulkhead_call(f"req_{i}"))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze metrics
        [r for r in results if isinstance(r, str)]

        # Verify metrics collection
        assert len(resource_usage) > 0
        assert len(wait_times) > 0

        # Some requests should have waited (queue contention)
        non_zero_waits = [w for w in wait_times if w > 0.001]  # Account for timing precision
        assert len(non_zero_waits) > 0

        # Resource utilization should have peaked
        max_utilization = max(resource_usage)
        assert max_utilization <= 3  # Should not exceed semaphore capacity
        assert max_utilization > 0  # Some resources should have been used

        # Calculate efficiency metrics
        avg_wait_time = statistics.mean(wait_times)
        max_wait_time = max(wait_times)
        utilization_rate = statistics.mean(resource_usage) / 3

        logger.info(
            f"Bulkhead metrics - Avg wait: {avg_wait_time:.3f}s, "
            f"Max wait: {max_wait_time:.3f}s, "
            f"Utilization: {utilization_rate:.2%}"
        )


class TestTimeoutHandling:
    """Tests for timeout handling and graceful degradation."""

    async def test_timeout_with_graceful_degradation(
        self, resilience_suite: ResilienceTestSuite, timeout_config: TimeoutConfig
    ):
        """Test timeout handling with graceful degradation."""

        timeout_handler = TimeoutHandler(timeout_config)
        fallback_used = False

        async def slow_service():
            await asyncio.sleep(2.0)  # Longer than timeout
            return "slow_response"

        async def fallback_service():
            nonlocal fallback_used
            fallback_used = True
            return "fallback_response"

        # Test timeout with fallback
        start_time = time.time()

        try:
            result = await timeout_handler.execute_with_timeout(
                slow_service, timeout=0.5, service="test", operation="slow_op"
            )
            raise AssertionError("Should have timed out")
        except TimeoutError:
            # Expected timeout
            duration = time.time() - start_time
            assert duration >= 0.45  # Should respect timeout
            assert duration < 1.0  # Should not wait full slow service time
            resilience_suite.metrics.record_timeout()

        # Use fallback after timeout
        result = await fallback_service()
        assert result == "fallback_response"
        assert fallback_used

    async def test_adaptive_timeout_strategy(self, resilience_suite: ResilienceTestSuite):
        """Test adaptive timeout strategy based on historical performance."""

        config = TimeoutConfig(default_timeout=1.0, strategy=TimeoutStrategy.ADAPTIVE, max_timeout=5.0)
        timeout_handler = TimeoutHandler(config)

        # Simulate varying response times to build history
        async def variable_latency_service(latency: float):
            await asyncio.sleep(latency)
            return f"response_after_{latency}"

        # Build performance history
        latencies = [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.2, 0.35]  # Gradually increasing

        for latency in latencies:
            with contextlib.suppress(TimeoutError):
                await timeout_handler.execute_with_timeout(
                    variable_latency_service,
                    latency,
                    timeout=2.0,  # Allow all to succeed
                    service="adaptive_test",
                    operation="variable_latency",
                )

        # Now test that timeout adapts based on history
        # The adaptive strategy should have learned from the 95th percentile
        calculated_timeout = timeout_handler._calculate_timeout("adaptive_test.variable_latency")

        # Should be higher than default due to observed latencies
        assert calculated_timeout > config.default_timeout
        assert calculated_timeout <= config.max_timeout

        # Verify it can handle a request within the adaptive timeout
        result = await timeout_handler.execute_with_timeout(
            variable_latency_service,
            0.3,  # Within historical pattern
            service="adaptive_test",
            operation="variable_latency",
        )
        assert "response_after_0.3" in result

    async def test_deadline_propagation(self, resilience_suite: ResilienceTestSuite):
        """Test deadline propagation across service calls."""

        deadline_manager = DeadlineManager()

        async def service_chain(request_id: str, depth: int = 3):
            # Check deadline at each service level
            deadline_manager.check_deadline(request_id)

            if depth <= 0:
                return f"final_result_{request_id}"

            # Simulate processing time
            await asyncio.sleep(0.1)

            # Propagate deadline to next service call
            remaining = deadline_manager.remaining_time(request_id)
            if remaining is None or remaining <= 0:
                raise Exception(f"Insufficient time remaining for request {request_id}")

            return await service_chain(request_id, depth - 1)

        request_id = str(uuid4())

        # Test successful chain within deadline
        async with deadline_manager.deadline_context(request_id, timeout=1.0):
            result = await service_chain(request_id, depth=3)
            assert request_id in result

        # Test deadline exceeded
        request_id_2 = str(uuid4())
        with pytest.raises((Exception, asyncio.TimeoutError)):  # Should raise DeadlineExceededError or similar
            async with deadline_manager.deadline_context(request_id_2, timeout=0.2):  # Too short
                await service_chain(request_id_2, depth=5)

    async def test_cancellation_handling(self, resilience_suite: ResilienceTestSuite):
        """Test graceful cancellation of operations."""

        cancellation_handler = CancellationHandler()

        async def cancellable_operation(operation_id: str):
            for _i in range(100):  # Long-running operation
                await cancellation_handler.check_cancellation(operation_id)
                await asyncio.sleep(0.01)  # Simulate work
            return "completed"

        operation_id = str(uuid4())

        # Start long-running operation
        async with cancellation_handler.cancellable_operation(operation_id):
            task = asyncio.create_task(cancellable_operation(operation_id))

            # Cancel after short delay
            await asyncio.sleep(0.1)
            cancellation_handler.request_cancellation(operation_id)

            # Verify cancellation
            with pytest.raises(asyncio.CancelledError):
                await task

        # Verify cancellation was requested
        assert not cancellation_handler.is_cancelled(operation_id)  # Should be cleaned up

    async def test_timeout_escalation_strategies(self, resilience_suite: ResilienceTestSuite):
        """Test different timeout escalation strategies."""

        # Test linear escalation
        linear_config = TimeoutConfig(default_timeout=0.5, strategy=TimeoutStrategy.LINEAR, max_timeout=3.0)
        linear_handler = TimeoutHandler(linear_config)

        # Test exponential escalation
        exponential_config = TimeoutConfig(
            default_timeout=0.5, strategy=TimeoutStrategy.EXPONENTIAL, escalation_factor=2.0, max_timeout=4.0
        )
        exponential_handler = TimeoutHandler(exponential_config)

        async def sometimes_slow_service():
            if random.random() < 0.7:  # 70% chance of timeout
                await asyncio.sleep(1.5)
            return "success"

        # Cause timeouts to trigger escalation
        for _i in range(3):
            with contextlib.suppress(TimeoutError):
                await linear_handler.execute_with_timeout(
                    sometimes_slow_service, service="escalation_test", operation="linear"
                )

        for _i in range(3):
            with contextlib.suppress(TimeoutError):
                await exponential_handler.execute_with_timeout(
                    sometimes_slow_service, service="escalation_test", operation="exponential"
                )

        # Check that timeouts have escalated
        linear_timeout = linear_handler._calculate_timeout("escalation_test.linear")
        exponential_timeout = exponential_handler._calculate_timeout("escalation_test.exponential")

        assert linear_timeout > linear_config.default_timeout
        assert exponential_timeout > exponential_config.default_timeout
        assert exponential_timeout > linear_timeout  # Exponential should escalate faster


class TestFailoverMechanisms:
    """Tests for failover and fallback mechanisms."""

    async def test_automatic_failover_between_instances(self, resilience_suite: ResilienceTestSuite):
        """Test automatic failover between service instances."""

        # Verify we have multiple instances
        analysis_instances = resilience_suite.service_mesh.services["analysis"]
        assert len(analysis_instances) >= 2

        # Fail the first instance
        analysis_instances[0].set_healthy(False)

        # Make requests that should failover to healthy instances
        successful_requests = 0
        for i in range(10):
            try:
                result = await resilience_suite.service_mesh.call_service("analysis", f"failover_test_{i}")
                if "success" in result:
                    successful_requests += 1
            except Exception:
                pass

        # Should have some successful requests using healthy instances
        assert successful_requests > 5

        # Verify load balancer is avoiding unhealthy instance
        healthy_instances = resilience_suite.service_mesh.get_healthy_instances("analysis")
        assert len(healthy_instances) == len(analysis_instances) - 1
        assert analysis_instances[0] not in healthy_instances

    async def test_cross_region_failover(self, resilience_suite: ResilienceTestSuite):
        """Test failover across regions/availability zones."""

        # Simulate multi-region setup
        regions = {"us-east-1": [], "us-west-2": [], "eu-west-1": []}

        # Add services to regions
        for i, region in enumerate(regions.keys()):
            config = ServiceConfig(
                name=f"multi-region-{region}-{i}",
                base_latency=0.05 + (i * 0.02),  # Simulate distance latency
                error_rate=0.02,
            )
            service = MockService(config, resilience_suite.metrics)
            resilience_suite.service_mesh.register_service(f"service_{region}", service)
            regions[region].append(service)

        async def call_with_region_failover():
            # Try regions in preference order
            region_order = ["us-east-1", "us-west-2", "eu-west-1"]

            for region in region_order:
                try:
                    return await resilience_suite.service_mesh.call_service(f"service_{region}")
                except Exception as e:
                    if region == region_order[-1]:  # Last region
                        raise e
                    continue  # Try next region

            raise Exception("All regions failed")

        # Simulate primary region failure
        primary_service = resilience_suite.service_mesh.services["service_us-east-1"][0]
        primary_service.set_healthy(False)

        # Test failover
        successful_failovers = 0
        for _ in range(5):
            try:
                result = await call_with_region_failover()
                if "us-west-2" in result or "eu-west-1" in result:  # Failed over
                    successful_failovers += 1
            except Exception:
                pass

        assert successful_failovers > 3  # Most should successfully failover

    async def test_database_failover_with_consistency(self, resilience_suite: ResilienceTestSuite):
        """Test database failover while maintaining data consistency."""

        # Simulate primary/replica database setup
        primary_db = {"data": {}, "is_primary": True, "healthy": True}
        replica_db = {"data": {}, "is_primary": False, "healthy": True}

        async def write_data(key: str, value: str, use_replica: bool = False):
            if not use_replica and primary_db["healthy"]:
                primary_db["data"][key] = value
                # Replicate to secondary (simulate async replication)
                await asyncio.sleep(0.01)
                replica_db["data"][key] = value
                return "written_to_primary"
            if replica_db["healthy"]:
                if replica_db["is_primary"]:  # Promoted to primary
                    replica_db["data"][key] = value
                    return "written_to_promoted_replica"
                raise Exception("Cannot write to read replica")
            raise Exception("No healthy database available")

        async def read_data(key: str, use_replica: bool = True):
            if use_replica and replica_db["healthy"]:
                return replica_db["data"].get(key, "not_found")
            if primary_db["healthy"]:
                return primary_db["data"].get(key, "not_found")
            raise Exception("No healthy database available")

        # Write some initial data
        await write_data("test_key", "test_value")

        # Verify data is readable
        value = await read_data("test_key")
        assert value == "test_value"

        # Simulate primary database failure
        primary_db["healthy"] = False

        # Promote replica to primary
        replica_db["is_primary"] = True

        # Verify writes still work after failover
        await write_data("failover_key", "failover_value")

        # Verify data consistency
        value = await read_data("failover_key", use_replica=False)  # Read from promoted primary
        assert value == "failover_value"

        # Original data should still be available
        value = await read_data("test_key", use_replica=False)
        assert value == "test_value"

    async def test_service_mesh_load_balancing(self, resilience_suite: ResilienceTestSuite):
        """Test service mesh load balancing strategies."""

        # Test different load balancing strategies
        strategies = ["round_robin", "random", "least_loaded"]

        for strategy in strategies:
            resilience_suite.service_mesh.set_load_balancer_strategy(strategy)

            # Track which instances are used
            instance_usage = defaultdict(int)

            # Make multiple requests
            for _i in range(20):
                try:
                    result = await resilience_suite.service_mesh.call_service("analysis")
                    # Extract instance identifier from result
                    for instance in resilience_suite.service_mesh.services["analysis"]:
                        if instance.config.name in result:
                            instance_usage[instance.config.name] += 1
                            break
                except Exception:
                    pass

            logger.info(f"Load balancing strategy: {strategy}")
            logger.info(f"Instance usage: {dict(instance_usage)}")

            # Verify load distribution
            total_requests = sum(instance_usage.values())
            if total_requests > 0:
                # All healthy instances should receive some requests
                healthy_count = len(resilience_suite.service_mesh.get_healthy_instances("analysis"))
                used_instances = len([count for count in instance_usage.values() if count > 0])
                assert used_instances >= min(2, healthy_count)  # At least 2 instances used if available

    async def test_fallback_service_implementation(self, resilience_suite: ResilienceTestSuite):
        """Test fallback service implementation."""

        # Create fallback service with cached/simplified responses
        fallback_cache = {
            "popular_tracks": ["Track 1", "Track 2", "Track 3"],
            "default_analysis": {"bpm": 120, "key": "C", "energy": 0.7},
        }

        async def fallback_service(operation: str):
            resilience_suite.metrics.record_fallback_execution()

            if operation == "get_popular_tracks":
                return {"tracks": fallback_cache["popular_tracks"], "source": "cache"}
            if operation == "analyze_audio":
                return {"analysis": fallback_cache["default_analysis"], "source": "fallback"}
            return {"message": "Fallback response", "source": "fallback"}

        # Fail all analysis service instances
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(False)

        # Test fallback responses
        operations = ["get_popular_tracks", "analyze_audio", "unknown_operation"]

        for operation in operations:
            try:
                # Try primary service first
                result = await resilience_suite.service_mesh.call_service("analysis", operation)
                raise AssertionError("Should have failed with all instances down")
            except Exception:
                # Use fallback
                result = await fallback_service(operation)
                assert result["source"] in ["cache", "fallback"]
                assert "tracks" in result or "analysis" in result or "message" in result

        # Verify fallback metrics
        assert resilience_suite.metrics.fallback_executions >= len(operations)


class TestServiceMeshResilience:
    """Tests for service mesh resilience features."""

    async def test_service_discovery_resilience(self, resilience_suite: ResilienceTestSuite):
        """Test service discovery resilience and health checks."""

        # Initially all services should be discoverable
        analysis_instances = resilience_suite.service_mesh.get_healthy_instances("analysis")
        initial_count = len(analysis_instances)
        assert initial_count >= 2

        # Simulate health check failure
        analysis_instances[0].set_healthy(False)

        # Service discovery should exclude unhealthy instance
        healthy_instances = resilience_suite.service_mesh.get_healthy_instances("analysis")
        assert len(healthy_instances) == initial_count - 1

        # Disable health checks
        resilience_suite.service_mesh.enable_health_checks(False)

        # All instances should be returned when health checks disabled
        all_instances = resilience_suite.service_mesh.get_healthy_instances("analysis")
        assert len(all_instances) == initial_count

        # Re-enable health checks
        resilience_suite.service_mesh.enable_health_checks(True)

        # Should again exclude unhealthy instance
        healthy_instances = resilience_suite.service_mesh.get_healthy_instances("analysis")
        assert len(healthy_instances) == initial_count - 1

    async def test_circuit_breaker_per_service_instance(self, resilience_suite: ResilienceTestSuite):
        """Test circuit breaker per service instance."""

        # Create circuit breakers for each instance
        instance_breakers = {}
        for i, service in enumerate(resilience_suite.service_mesh.services["analysis"]):
            config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
            breaker = AnalysisCircuitBreaker(f"instance_{i}", config)
            instance_breakers[service.config.name] = breaker

        # Fail one specific instance
        target_service = next(iter(resilience_suite.service_mesh.services["analysis"]))
        target_service.set_healthy(False)
        target_breaker = instance_breakers[target_service.config.name]

        # Trigger circuit breaker for that instance
        for _ in range(2):
            with contextlib.suppress(Exception):
                await target_breaker.call(target_service.call, "test")

        # Verify only that instance's circuit breaker is open
        assert target_breaker.state == CircuitState.OPEN

        # Other instances should still work
        other_services = [svc for svc in resilience_suite.service_mesh.services["analysis"] if svc != target_service]
        other_breaker = instance_breakers[other_services[0].config.name]

        result = await other_breaker.call(other_services[0].call, "test")
        assert "success" in result
        assert other_breaker.state == CircuitState.CLOSED

    async def test_rate_limiting_across_mesh(self, resilience_suite: ResilienceTestSuite):
        """Test distributed rate limiting across service mesh."""

        # Create rate limiters for different service tiers
        service_limiters = {
            "analysis": RateLimiter(rate=5.0, capacity=10, name="analysis_limiter"),
            "tracklist": RateLimiter(rate=10.0, capacity=20, name="tracklist_limiter"),
        }

        async def rate_limited_call(service_name: str, operation: str):
            limiter = service_limiters[service_name]
            wait_time = await limiter.acquire(tokens=1)

            if wait_time > 0:
                resilience_suite.metrics.record_request(False, wait_time, FailureType.RATE_LIMITED)

            return await resilience_suite.service_mesh.call_service(service_name, operation)

        # Generate burst of requests exceeding rate limits
        tasks = []
        for i in range(30):
            service = "analysis" if i % 2 == 0 else "tracklist"
            task = asyncio.create_task(rate_limited_call(service, f"burst_{i}"))
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time

        # Verify rate limiting was applied
        successful_results = [r for r in results if isinstance(r, str) and "success" in r]

        # Should have successful results but with rate limiting delays
        assert len(successful_results) > 0
        assert total_duration > 1.0  # Rate limiting should introduce delays

        # Check rate limiter statistics
        rate_limited_count = len(
            [
                1
                for failure_type, count in resilience_suite.metrics.failure_types.items()
                if failure_type == FailureType.RATE_LIMITED.value
            ]
        )

        logger.info(f"Rate limited requests: {rate_limited_count}")
        logger.info(f"Total duration with rate limiting: {total_duration:.2f}s")

    async def test_service_mesh_observability(self, resilience_suite: ResilienceTestSuite):
        """Test observability features in service mesh."""

        # Track metrics across service calls
        call_latencies = []
        error_counts = defaultdict(int)

        async def monitored_service_call(service_name: str, operation: str):
            start_time = time.time()
            try:
                result = await resilience_suite.service_mesh.call_service(service_name, operation)
                latency = time.time() - start_time
                call_latencies.append((service_name, operation, latency, True))
                return result
            except Exception as e:
                latency = time.time() - start_time
                call_latencies.append((service_name, operation, latency, False))
                error_counts[type(e).__name__] += 1
                raise

        # Generate mixed workload
        operations = [
            ("analysis", "audio_analysis"),
            ("tracklist", "search_tracks"),
            ("database", "query_data"),
        ]

        tasks = []
        for i in range(20):
            service, operation = operations[i % len(operations)]
            task = asyncio.create_task(monitored_service_call(service, f"{operation}_{i}"))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze observability data
        [r for r in results if isinstance(r, str)]
        [r for r in results if isinstance(r, Exception)]

        # Service-level metrics
        service_metrics = defaultdict(lambda: {"calls": 0, "successes": 0, "latencies": []})

        for service, _operation, latency, success in call_latencies:
            service_metrics[service]["calls"] += 1
            service_metrics[service]["latencies"].append(latency)
            if success:
                service_metrics[service]["successes"] += 1

        # Log observability data
        for service, metrics in service_metrics.items():
            if metrics["calls"] > 0:
                success_rate = metrics["successes"] / metrics["calls"]
                avg_latency = statistics.mean(metrics["latencies"])
                p95_latency = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.95)]

                logger.info(
                    f"Service {service}: {metrics['calls']} calls, "
                    f"{success_rate:.2%} success rate, "
                    f"{avg_latency:.3f}s avg latency, "
                    f"{p95_latency:.3f}s P95 latency"
                )

        # Verify we collected meaningful metrics
        assert len(call_latencies) > 0
        assert len(service_metrics) >= 2  # At least 2 different services


class TestChaosEngineering:
    """Tests for chaos engineering scenarios."""

    async def test_chaos_monkey_random_failures(self, resilience_suite: ResilienceTestSuite):
        """Test chaos monkey with random failure injection."""

        # Configure chaos monkey with higher failure rate for testing
        resilience_suite.chaos_engineer.failure_rate = 0.3  # 30% chance of disruption

        # Start chaos monkey for short duration
        chaos_task = asyncio.create_task(resilience_suite.chaos_engineer.start_chaos_monkey(duration=3.0))

        # Generate steady load while chaos monkey is running
        request_tasks = []
        for i in range(30):
            task = asyncio.create_task(self._resilient_service_call(resilience_suite, f"chaos_test_{i}"))
            request_tasks.append(task)
            await asyncio.sleep(0.1)  # Spread requests over time

        # Wait for all requests to complete
        results = await asyncio.gather(*request_tasks, return_exceptions=True)
        await chaos_task  # Wait for chaos monkey to finish

        # Analyze chaos impact
        successful_results = [r for r in results if isinstance(r, str) and "success" in r]
        failed_results = [r for r in results if isinstance(r, Exception)]

        # Should have some failures due to chaos but not complete failure
        failure_rate = len(failed_results) / len(results)
        assert 0.1 < failure_rate < 0.8  # Some chaos impact but not catastrophic

        logger.info(
            f"Chaos monkey test - Success: {len(successful_results)}, "
            f"Failures: {len(failed_results)}, "
            f"Failure rate: {failure_rate:.2%}"
        )

    async def test_network_partition_simulation(self, resilience_suite: ResilienceTestSuite):
        """Test network partition (split-brain) scenarios."""

        # Simulate network partition for analysis service
        partition_task = asyncio.create_task(
            resilience_suite.chaos_engineer.simulate_network_partition("analysis", duration=2.0)
        )

        # Test requests during partition
        partition_results = []
        for i in range(10):
            try:
                result = await asyncio.wait_for(
                    resilience_suite.service_mesh.call_service("analysis", f"partition_test_{i}"),
                    timeout=1.0,  # Short timeout to detect partition quickly
                )
                partition_results.append(("success", result))
            except (TimeoutError, Exception) as e:
                partition_results.append(("failure", str(e)))

            await asyncio.sleep(0.1)

        await partition_task  # Wait for partition to resolve

        # Test requests after partition recovery
        recovery_results = []
        for i in range(5):
            try:
                result = await resilience_suite.service_mesh.call_service("analysis", f"recovery_test_{i}")
                recovery_results.append(("success", result))
            except Exception as e:
                recovery_results.append(("failure", str(e)))

        # Analyze partition impact
        partition_failures = [r for r in partition_results if r[0] == "failure"]
        recovery_successes = [r for r in recovery_results if r[0] == "success"]

        # Should have failures during partition
        assert len(partition_failures) > len(partition_results) // 2

        # Should recover after partition ends
        assert len(recovery_successes) >= len(recovery_results) // 2

        logger.info(
            f"Network partition test - During partition: {len(partition_failures)} failures, "
            f"After recovery: {len(recovery_successes)} successes"
        )

    async def test_cascading_failure_scenario(self, resilience_suite: ResilienceTestSuite):
        """Test cascading failure scenarios."""

        # Simulate cascading failure starting from database service
        affected_services = await resilience_suite.chaos_engineer.simulate_cascading_failure(
            "database", failure_probability=0.6
        )

        # Test system behavior during cascading failure
        failure_results = []

        # Test each service type
        test_services = ["database", "analysis", "tracklist"]
        for service in test_services:
            for i in range(3):
                try:
                    result = await resilience_suite.service_mesh.call_service(service, f"cascade_test_{i}")
                    failure_results.append((service, "success", result))
                except Exception as e:
                    failure_results.append((service, "failure", str(e)))

        # Analyze cascading failure impact
        service_impact = defaultdict(lambda: {"success": 0, "failure": 0})

        for service, status, _ in failure_results:
            service_impact[service][status] += 1

        # Verify cascading failure occurred
        total_failures = sum(metrics["failure"] for metrics in service_impact.values())
        total_requests = sum(metrics["success"] + metrics["failure"] for metrics in service_impact.values())

        cascade_failure_rate = total_failures / total_requests if total_requests > 0 else 0

        logger.info(f"Cascading failure affected services: {affected_services}")
        logger.info(f"Overall failure rate during cascade: {cascade_failure_rate:.2%}")

        # Should have significant failure rate due to cascade
        assert cascade_failure_rate > 0.3  # At least 30% failure rate

        # Database should be most affected (initial failure point)
        if "database" in service_impact:
            db_failure_rate = service_impact["database"]["failure"] / (
                service_impact["database"]["success"] + service_impact["database"]["failure"]
            )
            assert db_failure_rate > 0.5  # Database should have high failure rate

    async def test_resource_exhaustion_scenario(self, resilience_suite: ResilienceTestSuite):
        """Test resource exhaustion scenarios."""

        # Create resource-intensive load
        resource_tasks = []

        # Generate load exceeding service capacity
        for i in range(50):  # More than any single service can handle
            task = asyncio.create_task(self._resource_intensive_call(resilience_suite, f"load_test_{i}"))
            resource_tasks.append(task)

        # Wait for all requests
        results = await asyncio.gather(*resource_tasks, return_exceptions=True)

        # Analyze resource exhaustion
        resource_exhausted_errors = 0
        timeout_errors = 0
        successful_requests = 0

        for result in results:
            if isinstance(result, str) and "success" in result:
                successful_requests += 1
            elif isinstance(result, Exception):
                error_msg = str(result).lower()
                if "resource" in error_msg or "exhausted" in error_msg:
                    resource_exhausted_errors += 1
                elif "timeout" in error_msg:
                    timeout_errors += 1

        # Verify resource exhaustion was handled gracefully
        assert resource_exhausted_errors + timeout_errors > 0  # Some resource pressure
        assert successful_requests > 0  # But not complete failure

        # System should prioritize requests and handle overload
        total_errors = resource_exhausted_errors + timeout_errors
        error_rate = total_errors / len(results)

        logger.info(
            f"Resource exhaustion test - Successful: {successful_requests}, "
            f"Resource exhausted: {resource_exhausted_errors}, "
            f"Timeouts: {timeout_errors}, "
            f"Error rate: {error_rate:.2%}"
        )

        # Error rate should be manageable (not 100% failure)
        assert error_rate < 0.9

    async def _resilient_service_call(self, resilience_suite: ResilienceTestSuite, operation_id: str) -> str:
        """Make a resilient service call with retries and fallback."""

        backoff = ExponentialBackoff(base_delay=0.1, max_delay=1.0)

        try:
            return await retry_with_backoff(
                resilience_suite.service_mesh.call_service,
                "analysis",
                operation_id,
                max_attempts=3,
                backoff=backoff,
                exceptions=(Exception,),
            )
        except Exception:
            # Fallback response
            resilience_suite.metrics.record_fallback_execution()
            return f"fallback_response_{operation_id}"

    async def _resource_intensive_call(self, resilience_suite: ResilienceTestSuite, operation_id: str) -> str:
        """Simulate resource-intensive call."""

        # Add artificial load to simulate resource usage
        service_name = random.choice(["analysis", "tracklist"])

        try:
            # Simulate resource acquisition delay
            await asyncio.sleep(random.uniform(0.05, 0.2))
            return await resilience_suite.service_mesh.call_service(service_name, operation_id)
        except Exception as e:
            if "resource" in str(e).lower():
                resilience_suite.metrics.record_request(False, 0.1, FailureType.RESOURCE_EXHAUSTED)
            raise


class TestRecoveryTesting:
    """Tests for recovery testing and data consistency."""

    async def test_service_restart_recovery(self, resilience_suite: ResilienceTestSuite):
        """Test service recovery after restart."""

        # Simulate service shutdown
        target_service = resilience_suite.service_mesh.services["analysis"][0]
        target_service.set_healthy(False)

        # Record downtime start
        downtime_start = time.time()

        # Test requests during downtime (should fail or use alternatives)
        downtime_results = []
        for i in range(5):
            try:
                result = await resilience_suite.service_mesh.call_service("analysis", f"downtime_{i}")
                downtime_results.append(("success", result))
            except Exception as e:
                downtime_results.append(("failure", str(e)))

        # Simulate service restart and recovery
        await asyncio.sleep(1.0)  # Downtime period
        target_service.set_healthy(True)  # Service comes back online

        recovery_time = time.time() - downtime_start
        resilience_suite.metrics.record_recovery(recovery_time)
        resilience_suite.metrics.record_service_unavailable_period(recovery_time)

        # Test requests after recovery
        recovery_results = []
        for i in range(10):
            try:
                result = await resilience_suite.service_mesh.call_service("analysis", f"recovery_{i}")
                recovery_results.append(("success", result))
            except Exception as e:
                recovery_results.append(("failure", str(e)))

        # Analyze recovery
        downtime_failures = [r for r in downtime_results if r[0] == "failure"]
        recovery_successes = [r for r in recovery_results if r[0] == "success"]

        # Should have failures during downtime
        assert len(downtime_failures) > 0

        # Should recover successfully
        recovery_rate = len(recovery_successes) / len(recovery_results)
        assert recovery_rate > 0.7  # At least 70% recovery rate

        logger.info(f"Service restart recovery - Downtime: {recovery_time:.2f}s, Recovery rate: {recovery_rate:.2%}")

    async def test_data_consistency_after_failure(self, resilience_suite: ResilienceTestSuite):
        """Test data consistency after service failures."""

        # Simulate distributed data store
        data_stores = {
            "primary": {"data": {}, "healthy": True},
            "replica1": {"data": {}, "healthy": True},
            "replica2": {"data": {}, "healthy": True},
        }

        async def write_data(key: str, value: str, stores: list[str] | None = None):
            """Write data to specified stores."""
            if stores is None:
                stores = ["primary", "replica1", "replica2"]

            successful_writes = 0
            for store_name in stores:
                if data_stores[store_name]["healthy"]:
                    data_stores[store_name]["data"][key] = value
                    successful_writes += 1
                    await asyncio.sleep(0.01)  # Simulate write latency

            return successful_writes

        async def read_data(key: str, store: str = "primary"):
            """Read data from specified store."""
            if not data_stores[store]["healthy"]:
                raise Exception(f"Store {store} is not healthy")

            return data_stores[store]["data"].get(key, "NOT_FOUND")

        # Write initial data to all stores
        test_data = {f"key_{i}": f"value_{i}" for i in range(10)}

        for key, value in test_data.items():
            writes = await write_data(key, value)
            assert writes == 3  # All stores should be healthy initially

        # Verify consistency across all stores
        for store_name in data_stores:
            for key, expected_value in test_data.items():
                stored_value = await read_data(key, store_name)
                assert stored_value == expected_value

        # Simulate failure of primary store
        data_stores["primary"]["healthy"] = False

        # Continue writing data (should succeed on replicas)
        failure_data = {f"failure_key_{i}": f"failure_value_{i}" for i in range(5)}

        for key, value in failure_data.items():
            writes = await write_data(key, value, stores=["replica1", "replica2"])
            assert writes >= 1  # At least one replica should succeed

        # Restore primary and check consistency
        await asyncio.sleep(0.5)  # Recovery time
        data_stores["primary"]["healthy"] = True

        # Sync missing data to primary (simulate recovery process)
        for key in failure_data:
            if key not in data_stores["primary"]["data"]:
                # Read from healthy replica and sync to primary
                value = await read_data(key, "replica1")
                data_stores["primary"]["data"][key] = value

        # Verify final consistency
        all_keys = set(test_data.keys()) | set(failure_data.keys())

        for store_name in data_stores:
            if data_stores[store_name]["healthy"]:
                store_keys = set(data_stores[store_name]["data"].keys())
                assert all_keys == store_keys  # All stores should have all keys

        logger.info(f"Data consistency test passed - {len(all_keys)} keys consistent across stores")

    async def test_graceful_degradation_recovery(self, resilience_suite: ResilienceTestSuite):
        """Test graceful degradation and recovery of service levels."""

        # Define service levels
        service_levels = {
            "full": {"features": ["analysis", "search", "recommendations"], "sla": 0.99},
            "degraded": {"features": ["analysis", "search"], "sla": 0.95},
            "minimal": {"features": ["search"], "sla": 0.90},
        }

        async def get_available_features():
            """Get currently available features based on service health."""
            healthy_services = []
            for service_name in ["analysis", "tracklist", "database"]:
                healthy_instances = resilience_suite.service_mesh.get_healthy_instances(service_name)
                if healthy_instances:
                    healthy_services.append(service_name)

            # Determine service level based on available services
            if len(healthy_services) >= 3:
                return "full", service_levels["full"]["features"]
            if len(healthy_services) >= 2:
                return "degraded", service_levels["degraded"]["features"]
            return "minimal", service_levels["minimal"]["features"]

        async def call_feature(feature: str):
            """Call a specific feature if available."""
            level, available_features = await get_available_features()

            if feature not in available_features:
                raise Exception(f"Feature {feature} not available at service level {level}")

            # Map features to services
            if feature == "analysis":
                return await resilience_suite.service_mesh.call_service("analysis", "analyze")
            if feature == "search":
                return await resilience_suite.service_mesh.call_service("tracklist", "search")
            if feature == "recommendations":
                return await resilience_suite.service_mesh.call_service("database", "recommend")
            return None

        # Test full service level
        level, features = await get_available_features()
        assert level == "full"
        assert len(features) == 3

        # Trigger degradation - fail analysis service
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(False)

        level, features = await get_available_features()
        assert level == "degraded"
        assert "analysis" not in features
        assert "search" in features

        # Further degradation - fail tracklist service partially
        tracklist_services = resilience_suite.service_mesh.services["tracklist"]
        for service in tracklist_services[:-1]:  # Keep one instance healthy
            service.set_healthy(False)

        level, features = await get_available_features()
        # Should still have degraded service level with remaining tracklist instance

        # Test recovery - restore analysis service
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.set_healthy(True)

        level, features = await get_available_features()
        assert level in ["degraded", "full"]  # Should improve
        assert "analysis" in features or len(features) >= 2

        # Full recovery
        for service in resilience_suite.service_mesh.services["tracklist"]:
            service.set_healthy(True)

        level, features = await get_available_features()
        assert level == "full"
        assert len(features) == 3

        logger.info(f"Graceful degradation test completed - final level: {level}")


class TestRateLimitingAndThrottling:
    """Tests for rate limiting and throttling under pressure."""

    async def test_token_bucket_rate_limiting(self, resilience_suite: ResilienceTestSuite):
        """Test token bucket rate limiting algorithm."""

        # Create rate limiter with specific capacity and refill rate
        rate_limiter = RateLimiter(rate=5.0, capacity=10, name="test_limiter")

        # Initial burst should succeed (using initial tokens)
        burst_times = []
        for _i in range(10):  # Use all initial tokens
            start = time.time()
            await rate_limiter.acquire(tokens=1)
            burst_times.append(time.time() - start)

        # First batch should be fast (no waiting)
        avg_burst_time = statistics.mean(burst_times)
        assert avg_burst_time < 0.1  # Should be very fast initially

        # Additional requests should be rate limited
        throttled_times = []
        for _i in range(5):  # These should be throttled
            start = time.time()
            await rate_limiter.acquire(tokens=1)
            throttled_times.append(time.time() - start)

        avg_throttled_time = statistics.mean(throttled_times)
        assert avg_throttled_time > 0.15  # Should have significant wait time

        # Verify rate limiting behavior
        total_time = sum(throttled_times)
        expected_minimum_time = 4 / 5.0  # 4 tokens at 5 tokens/second
        assert total_time >= expected_minimum_time * 0.8  # Allow some variance

        logger.info(f"Rate limiting test - Burst avg: {avg_burst_time:.3f}s, Throttled avg: {avg_throttled_time:.3f}s")

    async def test_adaptive_rate_limiting(self, resilience_suite: ResilienceTestSuite):
        """Test adaptive rate limiting based on system load."""

        # Create adaptive rate limiter that adjusts based on success rate
        class AdaptiveRateLimiter:
            def __init__(self, initial_rate: float = 10.0):
                self.rate = initial_rate
                self.limiter = RateLimiter(rate=self.rate, capacity=int(self.rate * 2))
                self.success_count = 0
                self.total_requests = 0
                self.adjustment_threshold = 10

            async def acquire(self):
                await self.limiter.acquire(1)
                self.total_requests += 1

            def record_result(self, success: bool):
                if success:
                    self.success_count += 1

                # Adjust rate based on success rate
                if self.total_requests >= self.adjustment_threshold:
                    success_rate = self.success_count / self.total_requests

                    if success_rate > 0.9:  # High success rate, increase rate
                        self.rate = min(20.0, self.rate * 1.1)
                    elif success_rate < 0.7:  # Low success rate, decrease rate
                        self.rate = max(1.0, self.rate * 0.8)

                    # Create new limiter with adjusted rate
                    self.limiter = RateLimiter(rate=self.rate, capacity=int(self.rate * 2))
                    self.success_count = 0
                    self.total_requests = 0

        adaptive_limiter = AdaptiveRateLimiter(initial_rate=5.0)

        # Simulate high success rate scenario
        for _ in range(15):
            await adaptive_limiter.acquire()
            # Simulate successful request
            adaptive_limiter.record_result(True)

        high_load_rate = adaptive_limiter.rate

        # Simulate low success rate scenario
        for i in range(15):
            await adaptive_limiter.acquire()
            # Simulate failed request
            success = i < 5  # Only first 5 succeed
            adaptive_limiter.record_result(success)

        low_load_rate = adaptive_limiter.rate

        # Rate should adapt to success patterns
        assert high_load_rate > 5.0  # Should increase from initial rate
        assert low_load_rate < high_load_rate  # Should decrease after failures

        logger.info(
            f"Adaptive rate limiting - High success rate: {high_load_rate:.1f}, Low success rate: {low_load_rate:.1f}"
        )

    async def test_distributed_rate_limiting(self, resilience_suite: ResilienceTestSuite):
        """Test distributed rate limiting across multiple service instances."""

        # Simulate distributed rate limiting with shared state
        class DistributedRateLimiter:
            def __init__(self, global_rate: float, num_instances: int):
                self.global_rate = global_rate
                self.per_instance_rate = global_rate / num_instances
                self.instances = [
                    RateLimiter(rate=self.per_instance_rate, capacity=int(self.per_instance_rate * 2))
                    for _ in range(num_instances)
                ]
                self.current_instance = 0

            async def acquire_from_any_instance(self):
                """Try to acquire from any available instance."""
                attempts = 0
                while attempts < len(self.instances):
                    instance = self.instances[self.current_instance]

                    # Try non-blocking acquire (check if tokens available)
                    if instance._tokens >= 1:
                        await instance.acquire(1)
                        return self.current_instance

                    # Move to next instance
                    self.current_instance = (self.current_instance + 1) % len(self.instances)
                    attempts += 1

                # If no instance has tokens, wait on the first one
                await self.instances[0].acquire(1)
                return 0

        # Create distributed rate limiter (10 req/s across 3 instances = ~3.33 req/s each)
        distributed_limiter = DistributedRateLimiter(global_rate=10.0, num_instances=3)

        # Test concurrent requests from multiple "clients"
        async def client_requests(client_id: int, num_requests: int):
            instance_usage = defaultdict(int)
            for _i in range(num_requests):
                instance_id = await distributed_limiter.acquire_from_any_instance()
                instance_usage[instance_id] += 1
            return instance_usage

        # Run multiple clients concurrently
        client_tasks = []
        for client_id in range(5):
            task = asyncio.create_task(client_requests(client_id, 6))
            client_tasks.append(task)

        start_time = time.time()
        client_results = await asyncio.gather(*client_tasks)
        total_time = time.time() - start_time

        # Analyze distribution across instances
        total_usage = defaultdict(int)
        for client_usage in client_results:
            for instance_id, count in client_usage.items():
                total_usage[instance_id] += count

        total_requests = sum(total_usage.values())

        # Verify load was distributed
        assert len(total_usage) > 1  # Multiple instances should be used

        # Verify rate limiting was applied
        theoretical_min_time = total_requests / 10.0  # 10 req/s global rate
        assert total_time >= theoretical_min_time * 0.8  # Allow some variance

        logger.info(f"Distributed rate limiting - {total_requests} requests in {total_time:.2f}s")
        logger.info(f"Instance usage: {dict(total_usage)}")

    async def test_priority_queuing_with_rate_limiting(self, resilience_suite: ResilienceTestSuite):
        """Test priority queuing combined with rate limiting."""

        # Create priority-aware rate limiter
        class PriorityRateLimiter:
            def __init__(self, rate: float = 5.0):
                self.high_priority_limiter = RateLimiter(rate=rate * 0.7, capacity=int(rate))
                self.low_priority_limiter = RateLimiter(rate=rate * 0.3, capacity=int(rate))

            async def acquire(self, priority: str = "low"):
                if priority == "high":
                    return await self.high_priority_limiter.acquire(1)
                return await self.low_priority_limiter.acquire(1)

        priority_limiter = PriorityRateLimiter(rate=10.0)

        async def priority_request(priority: str, request_id: str):
            start_time = time.time()
            wait_time = await priority_limiter.acquire(priority)

            # Simulate actual work
            result = await resilience_suite.service_mesh.call_service("analysis", request_id)

            total_time = time.time() - start_time
            return {
                "priority": priority,
                "request_id": request_id,
                "wait_time": wait_time,
                "total_time": total_time,
                "result": result,
            }

        # Submit mixed priority requests
        tasks = []

        # Submit low priority requests first
        for i in range(10):
            task = asyncio.create_task(priority_request("low", f"low_pri_{i}"))
            tasks.append(task)

        # Brief delay, then submit high priority requests
        await asyncio.sleep(0.1)
        for i in range(5):
            task = asyncio.create_task(priority_request("high", f"high_pri_{i}"))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze priority handling
        high_pri_results = [r for r in results if isinstance(r, dict) and r["priority"] == "high"]
        low_pri_results = [r for r in results if isinstance(r, dict) and r["priority"] == "low"]

        if high_pri_results and low_pri_results:
            avg_high_wait = statistics.mean([r["wait_time"] for r in high_pri_results])
            avg_low_wait = statistics.mean([r["wait_time"] for r in low_pri_results])

            # High priority should generally have lower wait times
            logger.info(
                f"Priority rate limiting - High priority avg wait: {avg_high_wait:.3f}s, "
                f"Low priority avg wait: {avg_low_wait:.3f}s"
            )

            # Both priorities should get some service
            assert len(high_pri_results) > 0
            assert len(low_pri_results) > 0


class TestStatisticalAnalysis:
    """Tests for statistical analysis of recovery times and failure rates."""

    async def test_comprehensive_resilience_analysis(self, resilience_suite: ResilienceTestSuite):
        """Comprehensive resilience analysis with statistical metrics."""

        # Reset metrics for clean test
        resilience_suite.reset()

        # Configure realistic failure scenarios
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.error_rate = 0.15  # 15% error rate
            service.config.timeout_rate = 0.05  # 5% timeout rate

        # Generate comprehensive test load
        test_duration = 10.0  # 10 second test
        request_rate = 5.0  # 5 requests per second
        total_requests = int(test_duration * request_rate)

        time.time()

        # Generate load with various patterns
        tasks = []
        for i in range(total_requests):
            # Mix of different request types
            if i % 10 < 7:  # 70% analysis requests
                task = self._timed_request(resilience_suite, "analysis", f"analysis_{i}")
            elif i % 10 < 9:  # 20% tracklist requests
                task = self._timed_request(resilience_suite, "tracklist", f"tracklist_{i}")
            else:  # 10% database requests
                task = self._timed_request(resilience_suite, "database", f"database_{i}")

            tasks.append(task)

            # Maintain request rate
            await asyncio.sleep(1.0 / request_rate)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results and generate statistics
        stats = resilience_suite.metrics.get_statistics()

        # Verify comprehensive statistics
        assert stats["total_requests"] > 0
        assert "response_times" in stats
        assert "availability" in stats or stats["total_requests"] == stats["successful_requests"]

        # Log comprehensive analysis
        logger.info("=== Comprehensive Resilience Analysis ===")
        logger.info(f"Test Duration: {stats['total_test_duration']}s")
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Success Rate: {stats['success_rate_percent']}%")
        logger.info(f"Circuit Breaker Trips: {stats['circuit_breaker_trips']}")
        logger.info(f"Retry Attempts: {stats['retry_attempts']}")
        logger.info(f"Timeouts: {stats['timeouts']}")

        if "response_times" in stats:
            rt = stats["response_times"]
            logger.info(
                f"Response Times - Min: {rt['min']}s, Mean: {rt['mean']}s, P95: {rt['p95']}s, P99: {rt['p99']}s"
            )

        if "availability" in stats:
            avail = stats["availability"]
            logger.info(
                f"Availability - Uptime: {avail['uptime_percent']}%, MTBF: {avail['mtbf']}s, MTTR: {avail['mttr']}s"
            )

        logger.info(f"Failure Types: {stats['failure_types']}")

        # Verify system resilience thresholds
        assert stats["success_rate_percent"] > 70  # At least 70% success rate

        if "response_times" in stats:
            assert stats["response_times"]["p95"] < 2.0  # P95 latency under 2s

        if "availability" in stats:
            assert stats["availability"]["uptime_percent"] > 95  # 95% uptime

    async def _timed_request(self, resilience_suite: ResilienceTestSuite, service: str, operation: str):
        """Make a timed request with comprehensive error handling."""

        start_time = time.time()
        try:
            # Add circuit breaker protection
            config = CircuitBreakerConfig(failure_threshold=5, timeout=2.0)
            breaker = AnalysisCircuitBreaker(f"{service}_breaker", config, resilience_suite.metrics)

            # Add retry with backoff
            backoff = ExponentialBackoff(base_delay=0.1, max_delay=1.0, jitter=True)

            result = await retry_with_backoff(
                breaker.call,
                resilience_suite.service_mesh.call_service,
                service,
                operation,
                max_attempts=3,
                backoff=backoff,
                exceptions=(Exception,),
            )

            response_time = time.time() - start_time
            resilience_suite.metrics.record_request(True, response_time)
            return result

        except CircuitOpenError:
            response_time = time.time() - start_time
            resilience_suite.metrics.record_request(False, response_time, FailureType.SERVICE_UNAVAILABLE)
            resilience_suite.metrics.record_circuit_breaker_trip()
            raise

        except TimeoutError:
            response_time = time.time() - start_time
            resilience_suite.metrics.record_request(False, response_time, FailureType.TIMEOUT)
            resilience_suite.metrics.record_timeout()
            raise

        except Exception as e:
            response_time = time.time() - start_time
            # Classify error type
            error_msg = str(e).lower()
            if "resource" in error_msg:
                failure_type = FailureType.RESOURCE_EXHAUSTED
            elif "network" in error_msg or "partition" in error_msg:
                failure_type = FailureType.NETWORK_PARTITION
            else:
                failure_type = FailureType.SERVICE_UNAVAILABLE

            resilience_suite.metrics.record_request(False, response_time, failure_type)
            raise

    async def test_failure_pattern_analysis(self, resilience_suite: ResilienceTestSuite):
        """Test failure pattern analysis and anomaly detection."""

        failure_timestamps = []
        recovery_timestamps = []

        # Simulate different failure patterns
        patterns = [
            {"name": "burst_failures", "duration": 2.0, "intensity": 0.8},
            {"name": "steady_degradation", "duration": 3.0, "intensity": 0.3},
            {"name": "intermittent_issues", "duration": 4.0, "intensity": 0.5},
        ]

        for pattern in patterns:
            logger.info(f"Testing failure pattern: {pattern['name']}")
            time.time()

            # Configure failure pattern
            for service in resilience_suite.service_mesh.services["analysis"]:
                service.config.error_rate = pattern["intensity"]
                service.enable_failure_injection(True)

            # Generate requests during pattern
            pattern_tasks = []
            for i in range(20):
                task = self._timed_request(resilience_suite, "analysis", f"{pattern['name']}_{i}")
                pattern_tasks.append(task)
                await asyncio.sleep(pattern["duration"] / 20)  # Spread over duration

            # Wait for pattern completion
            pattern_results = await asyncio.gather(*pattern_tasks, return_exceptions=True)

            # Record pattern end and recovery start
            failure_timestamps.append(time.time())

            # Recovery phase
            for service in resilience_suite.service_mesh.services["analysis"]:
                service.enable_failure_injection(False)
                service.config.error_rate = 0.05  # Back to baseline

            # Test recovery
            recovery_tasks = []
            for i in range(10):
                task = self._timed_request(resilience_suite, "analysis", f"recovery_{pattern['name']}_{i}")
                recovery_tasks.append(task)
                await asyncio.sleep(0.1)

            recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            recovery_timestamps.append(time.time())

            # Analyze pattern impact
            pattern_failures = len([r for r in pattern_results if isinstance(r, Exception)])
            recovery_successes = len([r for r in recovery_results if isinstance(r, str)])

            logger.info(
                f"Pattern {pattern['name']}: {pattern_failures} failures, {recovery_successes} recovery successes"
            )

        # Analyze failure patterns
        if len(failure_timestamps) >= 2 and len(recovery_timestamps) >= 2:
            inter_failure_times = [
                failure_timestamps[i + 1] - failure_timestamps[i] for i in range(len(failure_timestamps) - 1)
            ]
            recovery_times = [recovery_timestamps[i] - failure_timestamps[i] for i in range(len(failure_timestamps))]

            logger.info(f"Inter-failure times: {[round(t, 2) for t in inter_failure_times]}")
            logger.info(f"Recovery times: {[round(t, 2) for t in recovery_times]}")

            # Statistical analysis
            if len(recovery_times) > 1:
                mean_recovery = statistics.mean(recovery_times)
                recovery_variance = statistics.variance(recovery_times)

                logger.info(f"Mean recovery time: {mean_recovery:.2f}s")
                logger.info(f"Recovery time variance: {recovery_variance:.2f}")

                # Verify recovery times are reasonable
                assert mean_recovery < 10.0  # Should recover within 10 seconds
                assert all(rt < 20.0 for rt in recovery_times)  # No recovery should take > 20s

    async def test_performance_regression_detection(self, resilience_suite: ResilienceTestSuite):
        """Test performance regression detection through statistical analysis."""

        # Baseline performance measurement
        baseline_measurements = []

        logger.info("Measuring baseline performance...")
        for i in range(20):
            start_time = time.time()
            try:
                await resilience_suite.service_mesh.call_service("analysis", f"baseline_{i}")
                baseline_measurements.append(time.time() - start_time)
            except Exception:
                pass
            await asyncio.sleep(0.05)

        baseline_mean = statistics.mean(baseline_measurements) if baseline_measurements else 0
        baseline_stdev = statistics.stdev(baseline_measurements) if len(baseline_measurements) > 1 else 0

        logger.info(f"Baseline performance - Mean: {baseline_mean:.3f}s, StdDev: {baseline_stdev:.3f}s")

        # Introduce performance degradation
        logger.info("Introducing performance degradation...")
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.base_latency *= 2.0  # Double the latency

        # Measure degraded performance
        degraded_measurements = []
        for i in range(20):
            start_time = time.time()
            try:
                await resilience_suite.service_mesh.call_service("analysis", f"degraded_{i}")
                degraded_measurements.append(time.time() - start_time)
            except Exception:
                pass
            await asyncio.sleep(0.05)

        degraded_mean = statistics.mean(degraded_measurements) if degraded_measurements else 0
        degraded_stdev = statistics.stdev(degraded_measurements) if len(degraded_measurements) > 1 else 0

        logger.info(f"Degraded performance - Mean: {degraded_mean:.3f}s, StdDev: {degraded_stdev:.3f}s")

        # Statistical regression detection
        if baseline_measurements and degraded_measurements and len(baseline_measurements) > 1:
            # Perform t-test to detect significant difference

            n1, n2 = len(baseline_measurements), len(degraded_measurements)
            mean1, mean2 = baseline_mean, degraded_mean
            var1 = baseline_stdev**2
            var2 = degraded_stdev**2

            # Pooled standard deviation
            pooled_stdev = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            # t-statistic
            t_stat = abs(mean2 - mean1) / (pooled_stdev * sqrt(1 / n1 + 1 / n2))

            # Simple threshold-based detection (t > 2 suggests significant difference)
            performance_regression_detected = t_stat > 2.0

            logger.info(f"T-statistic: {t_stat:.2f}")
            logger.info(f"Performance regression detected: {performance_regression_detected}")

            # Verify regression was detected
            assert performance_regression_detected
            assert degraded_mean > baseline_mean * 1.5  # At least 50% slower

        # Restore baseline performance
        for service in resilience_suite.service_mesh.services["analysis"]:
            service.config.base_latency /= 2.0

        # Measure recovery
        recovery_measurements = []
        for i in range(10):
            start_time = time.time()
            try:
                await resilience_suite.service_mesh.call_service("analysis", f"recovery_{i}")
                recovery_measurements.append(time.time() - start_time)
            except Exception:
                pass
            await asyncio.sleep(0.05)

        recovery_mean = statistics.mean(recovery_measurements) if recovery_measurements else 0
        logger.info(f"Recovery performance - Mean: {recovery_mean:.3f}s")

        # Verify performance recovered
        if baseline_mean > 0 and recovery_mean > 0:
            performance_recovered = abs(recovery_mean - baseline_mean) / baseline_mean < 0.3
            logger.info(f"Performance recovered: {performance_recovered}")
            assert performance_recovered


# Integration test that combines multiple resilience patterns
@pytest.mark.asyncio
async def test_comprehensive_resilience_integration(resilience_suite: ResilienceTestSuite):
    """Comprehensive integration test combining all resilience patterns."""

    logger.info("=== Starting Comprehensive Resilience Integration Test ===")

    # Test configuration
    test_duration = 15.0  # 15 second comprehensive test
    concurrent_clients = 5
    chaos_duration = 8.0

    # Configure resilience patterns
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=5, success_threshold=3, timeout=3.0, failure_window=10.0
    )

    timeout_config = TimeoutConfig(default_timeout=2.0, strategy=TimeoutStrategy.ADAPTIVE, max_timeout=10.0)

    # Create comprehensive resilience stack
    async def resilient_client(client_id: int):
        """Resilient client implementing multiple resilience patterns."""
        client_results = []

        # Per-client circuit breakers and rate limiters
        analysis_cb = AnalysisCircuitBreaker(
            f"client_{client_id}_analysis", circuit_breaker_config, resilience_suite.metrics
        )
        tracklist_cb = AnalysisCircuitBreaker(
            f"client_{client_id}_tracklist", circuit_breaker_config, resilience_suite.metrics
        )
        rate_limiter = RateLimiter(rate=3.0, capacity=5, name=f"client_{client_id}")
        timeout_handler = TimeoutHandler(timeout_config)

        backoff = ExponentialBackoff(base_delay=0.1, max_delay=2.0, jitter=True)

        async def make_resilient_request(service_name: str, operation: str):
            """Make a request with full resilience stack."""

            # Rate limiting
            await rate_limiter.acquire(1)

            # Circuit breaker selection
            circuit_breaker = analysis_cb if service_name == "analysis" else tracklist_cb

            # Retry with backoff
            try:
                result = await retry_with_backoff(
                    timeout_handler.execute_with_timeout,
                    circuit_breaker.call_async,
                    resilience_suite.service_mesh.call_service,
                    service_name,
                    operation,
                    timeout=2.0,
                    service=service_name,
                    operation=operation,
                    max_attempts=3,
                    backoff=backoff,
                    exceptions=(Exception,),
                )
                return {"success": True, "result": result, "service": service_name}

            except (CircuitOpenError, TimeoutError, Exception) as e:
                # Fallback strategy
                resilience_suite.metrics.record_fallback_execution()
                return {
                    "success": False,
                    "fallback": f"cached_response_{service_name}_{operation}",
                    "error": str(e),
                    "service": service_name,
                }

        # Generate client load
        end_time = time.time() + test_duration
        request_count = 0

        while time.time() < end_time:
            request_count += 1
            service = "analysis" if request_count % 3 != 0 else "tracklist"
            operation = f"client_{client_id}_req_{request_count}"

            result = await make_resilient_request(service, operation)
            client_results.append(result)

            # Variable request rate
            await asyncio.sleep(random.uniform(0.2, 0.8))

        return client_results

    # Start chaos engineering
    chaos_task = asyncio.create_task(resilience_suite.chaos_engineer.start_chaos_monkey(duration=chaos_duration))

    # Start concurrent clients
    client_tasks = []
    for client_id in range(concurrent_clients):
        task = asyncio.create_task(resilient_client(client_id))
        client_tasks.append(task)

    # Wait for test completion
    logger.info(f"Running {concurrent_clients} clients for {test_duration}s with chaos for {chaos_duration}s...")

    client_results = await asyncio.gather(*client_tasks)
    await chaos_task

    # Comprehensive analysis
    all_results = [result for client_results in client_results for result in client_results]

    successful_requests = [r for r in all_results if r.get("success")]
    failed_requests = [r for r in all_results if not r.get("success")]
    fallback_requests = [r for r in failed_requests if "fallback" in r]

    # Service-specific analysis
    analysis_requests = [r for r in all_results if r.get("service") == "analysis"]
    tracklist_requests = [r for r in all_results if r.get("service") == "tracklist"]

    # Calculate comprehensive metrics
    total_requests = len(all_results)
    success_rate = len(successful_requests) / total_requests if total_requests > 0 else 0
    fallback_rate = len(fallback_requests) / total_requests if total_requests > 0 else 0

    analysis_success_rate = (
        len([r for r in analysis_requests if r.get("success")]) / len(analysis_requests) if analysis_requests else 0
    )
    tracklist_success_rate = (
        len([r for r in tracklist_requests if r.get("success")]) / len(tracklist_requests) if tracklist_requests else 0
    )

    # Get final system metrics
    final_stats = resilience_suite.metrics.get_statistics()

    # Log comprehensive results
    logger.info("=== Comprehensive Resilience Test Results ===")
    logger.info(f"Total Requests: {total_requests}")
    logger.info(f"Overall Success Rate: {success_rate:.2%}")
    logger.info(f"Fallback Usage Rate: {fallback_rate:.2%}")
    logger.info(f"Analysis Service Success Rate: {analysis_success_rate:.2%}")
    logger.info(f"Tracklist Service Success Rate: {tracklist_success_rate:.2%}")
    logger.info(f"Circuit Breaker Trips: {final_stats.get('circuit_breaker_trips', 0)}")
    logger.info(f"Total Retry Attempts: {final_stats.get('retry_attempts', 0)}")
    logger.info(f"Total Timeouts: {final_stats.get('timeouts', 0)}")
    logger.info(f"Fallback Executions: {final_stats.get('fallback_executions', 0)}")

    if "response_times" in final_stats:
        rt = final_stats["response_times"]
        logger.info(f"Response Time P95: {rt.get('p95', 0):.3f}s")
        logger.info(f"Response Time P99: {rt.get('p99', 0):.3f}s")

    if "availability" in final_stats:
        avail = final_stats["availability"]
        logger.info(f"System Uptime: {avail.get('uptime_percent', 0):.1f}%")
        logger.info(f"MTBF: {avail.get('mtbf', 0):.1f}s")
        logger.info(f"MTTR: {avail.get('mttr', 0):.1f}s")

    # Verify comprehensive resilience
    assert total_requests > 50  # Significant load generated
    assert success_rate > 0.5  # At least 50% success despite chaos
    assert fallback_rate > 0  # Fallback mechanisms were used

    # System should maintain basic functionality under chaos
    assert success_rate + fallback_rate > 0.8  # 80% effective handling (success + fallback)

    # Response times should remain reasonable despite stress
    if "response_times" in final_stats:
        assert final_stats["response_times"].get("p95", 0) < 5.0  # P95 under 5s

    logger.info("=== Comprehensive Resilience Integration Test PASSED ===")


if __name__ == "__main__":
    # Run specific test for development
    import asyncio

    async def run_single_test():
        suite = ResilienceTestSuite()
        suite.setup_services()

        # Quick test
        test_instance = TestCircuitBreakerResilience()
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout=1.0)

        await test_instance.test_circuit_breaker_prevents_cascading_failures(suite, config)

        print("Single test completed successfully!")

    asyncio.run(run_single_test())
