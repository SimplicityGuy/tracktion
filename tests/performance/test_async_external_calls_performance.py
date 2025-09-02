"""Performance tests for async external service calls."""

import asyncio
import contextlib
import random
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.utils.async_http_client import AsyncHTTPClient, AsyncHTTPClientFactory, HTTPClientConfig
from shared.utils.connection_pool_manager import ConnectionPoolManager


class MockExternalService:
    """Mock external service for performance testing."""

    def __init__(self, latency: float = 0.1, error_rate: float = 0.0):
        """Initialize mock service.

        Args:
            latency: Response latency in seconds
            error_rate: Percentage of requests that should fail (0-1)
        """
        self.latency = latency
        self.error_rate = error_rate
        self.request_count = 0
        self.error_count = 0

    async def handle_request(self, method: str, url: str) -> httpx.Response:
        """Simulate handling a request."""
        self.request_count += 1

        # Simulate network latency
        await asyncio.sleep(self.latency)

        # Simulate errors based on error rate

        if random.random() < self.error_rate:
            self.error_count += 1
            raise httpx.NetworkError(f"Simulated error for {url}")

        # Return mock response
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.text = f"Response for {url}"
        response.headers = {"content-type": "text/plain"}
        response.raise_for_status = MagicMock()
        return response


@pytest.fixture
def mock_service():
    """Create mock external service."""
    return MockExternalService(latency=0.01, error_rate=0.05)


@pytest.fixture
async def http_factory():
    """Create HTTP factory for testing."""
    config = HTTPClientConfig(
        timeout=5.0,
        max_connections=50,
        retry_attempts=2,
        retry_delay=0.1,
        circuit_breaker_fail_max=10,
    )
    factory = AsyncHTTPClientFactory(config)
    yield factory
    await factory.close()


@pytest.fixture
async def pool_manager():
    """Create connection pool manager for testing."""
    manager = ConnectionPoolManager(
        min_connections=10,
        max_connections=50,
        health_check_interval=60.0,  # Longer interval for tests
    )
    yield manager
    await manager.close_all_pools()


class TestAsyncHTTPClientPerformance:
    """Performance tests for async HTTP client."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, http_factory, mock_service):
        """Test performance of concurrent requests."""
        client = AsyncHTTPClient(http_factory)
        num_requests = 100
        max_concurrent = 20

        async def make_request(i: int) -> float:
            """Make a single request and return duration."""
            start = time.time()
            with patch.object(http_factory, "get_httpx_client") as mock_get:
                mock_client = AsyncMock()
                mock_client.request = mock_service.handle_request

                mock_context = AsyncMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_client)
                mock_context.__aexit__ = AsyncMock()
                mock_get.return_value = mock_context

                with contextlib.suppress(Exception):
                    await client.request_with_circuit_breaker(
                        service_name="test_service",
                        method="GET",
                        url=f"https://example.com/api/{i}",
                    )  # Some requests may fail due to error rate

            return time.time() - start

        # Run concurrent requests
        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(i: int) -> float:
            async with semaphore:
                return await make_request(i)

        tasks = [limited_request(i) for i in range(num_requests)]
        durations = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Calculate metrics
        successful_durations = [d for d in durations if d is not None]
        avg_duration = sum(successful_durations) / len(successful_durations)
        p95_duration = sorted(successful_durations)[int(len(successful_durations) * 0.95)]
        p99_duration = sorted(successful_durations)[int(len(successful_durations) * 0.99)]

        # Performance assertions
        assert total_time < 1.0  # Should complete 100 requests in under 1 second
        assert avg_duration < 0.1  # Average request should be under 100ms
        assert p95_duration < 0.2  # 95% of requests under 200ms
        assert p99_duration < 0.5  # 99% of requests under 500ms

        print("\nConcurrent Requests Performance:")
        print(f"  Total requests: {num_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Requests/sec: {num_requests / total_time:.1f}")
        print(f"  Avg duration: {avg_duration * 1000:.1f}ms")
        print(f"  P95 duration: {p95_duration * 1000:.1f}ms")
        print(f"  P99 duration: {p99_duration * 1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_batch_requests_performance(self, http_factory, mock_service):
        """Test performance of batch request processing."""
        client = AsyncHTTPClient(http_factory)
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            requests = [
                {
                    "service_name": "test_service",
                    "method": "GET",
                    "url": f"https://example.com/api/{i}",
                }
                for i in range(batch_size)
            ]

            with patch.object(client, "request_with_circuit_breaker") as mock_request:
                mock_request.side_effect = [mock_service.handle_request("GET", req["url"]) for req in requests]

                start_time = time.time()
                responses = await client.batch_requests(requests, max_concurrent=20)
                duration = time.time() - start_time

                assert len(responses) == batch_size
                assert duration < batch_size * 0.02  # Should be much faster than sequential

                print(f"\nBatch size {batch_size}:")
                print(f"  Duration: {duration:.3f}s")
                print(f"  Requests/sec: {batch_size / duration:.1f}")


class TestConnectionPoolPerformance:
    """Performance tests for connection pool manager."""

    @pytest.mark.asyncio
    async def test_pool_warmup_performance(self, pool_manager):
        """Test performance of pool warmup."""
        client = pool_manager.create_pool("test_service", "https://example.com")

        with patch.object(client, "head") as mock_head:

            async def mock_head_response():
                await asyncio.sleep(0.01)  # Simulate network latency
                return MagicMock(status_code=200)

            mock_head.return_value = mock_head_response()

            start_time = time.time()
            await pool_manager.warmup_pool("test_service", num_connections=10)
            warmup_time = time.time() - start_time

            # Warmup should be fast due to parallel connections
            assert warmup_time < 0.1  # Should complete in under 100ms
            print(f"\nPool warmup time for 10 connections: {warmup_time:.3f}s")

    @pytest.mark.asyncio
    async def test_pool_request_tracking_overhead(self, pool_manager):
        """Test overhead of request tracking."""
        client = pool_manager.create_pool("test_service")
        num_requests = 1000

        with patch.object(client, "request") as mock_request:

            async def mock_response():
                return MagicMock()

            mock_request.return_value = mock_response()

            # Measure with tracking
            start_time = time.time()
            for _ in range(num_requests):
                await pool_manager.request_with_tracking(
                    "test_service",
                    "GET",
                    "https://example.com/api",
                )
            tracked_time = time.time() - start_time

            # Measure without tracking (direct calls)
            start_time = time.time()
            for _ in range(num_requests):
                await client.request("GET", "https://example.com/api")
            direct_time = time.time() - start_time

            overhead_percentage = ((tracked_time - direct_time) / direct_time) * 100

            # Tracking overhead should be minimal
            assert overhead_percentage < 10  # Less than 10% overhead
            print("\nRequest tracking overhead:")
            print(f"  Direct calls: {direct_time:.3f}s")
            print(f"  Tracked calls: {tracked_time:.3f}s")
            print(f"  Overhead: {overhead_percentage:.1f}%")

    @pytest.mark.asyncio
    async def test_health_check_performance(self, pool_manager):
        """Test performance of health checks."""
        services = [f"service_{i}" for i in range(10)]

        for service in services:
            client = pool_manager.create_pool(service, f"https://{service}.com")

            with patch.object(client, "head") as mock_head:
                mock_head.return_value = asyncio.create_task(asyncio.coroutine(lambda: MagicMock(status_code=200))())

        # Run health checks for all services
        start_time = time.time()
        tasks = [pool_manager._perform_health_check(service) for service in services]
        results = await asyncio.gather(*tasks)
        health_check_time = time.time() - start_time

        assert all(results)  # All should be healthy
        assert health_check_time < 1.0  # Should complete in under 1 second
        print(f"\nHealth check time for {len(services)} services: {health_check_time:.3f}s")


class TestCircuitBreakerPerformance:
    """Performance tests for circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self, http_factory):
        """Test overhead of circuit breaker."""
        client = AsyncHTTPClient(http_factory)
        num_requests = 100

        async def successful_request():
            """Simulate a successful request."""
            await asyncio.sleep(0.001)
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            return response

        # Test with circuit breaker
        with patch.object(http_factory, "get_httpx_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.request = successful_request

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client)
            mock_context.__aexit__ = AsyncMock()
            mock_get.return_value = mock_context

            start_time = time.time()
            for _ in range(num_requests):
                await client.request_with_circuit_breaker(
                    service_name="test_service",
                    method="GET",
                    url="https://example.com",
                )
            cb_time = time.time() - start_time

        # Circuit breaker overhead should be minimal
        assert cb_time < num_requests * 0.002  # Less than 2ms per request
        print(f"\nCircuit breaker overhead for {num_requests} requests:")
        print(f"  Total time: {cb_time:.3f}s")
        print(f"  Per request: {(cb_time / num_requests) * 1000:.2f}ms")

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_performance(self, http_factory):
        """Test performance of circuit breaker recovery."""
        client = AsyncHTTPClient(http_factory)
        breaker = http_factory.get_circuit_breaker("recovery_test")
        breaker.fail_max = 5
        breaker.reset_timeout = 0.1  # Fast reset for testing

        failure_count = 0

        async def flaky_request(method: str, url: str):
            """Simulate a flaky service that recovers."""
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:
                raise httpx.NetworkError("Service unavailable")
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()
            return response

        with patch.object(http_factory, "get_httpx_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.request = flaky_request

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client)
            mock_context.__aexit__ = AsyncMock()
            mock_get.return_value = mock_context

            # Configure no retries for clearer testing
            client.retry_handler.config.retry_attempts = 1

            start_time = time.time()

            # Cause circuit to open
            for _ in range(5):
                with contextlib.suppress(Exception):
                    await client.request_with_circuit_breaker(
                        service_name="recovery_test",
                        method="GET",
                        url="https://example.com",
                    )

            assert breaker.state == "open"

            # Wait for reset
            await asyncio.sleep(0.15)

            # Circuit should recover quickly
            response = await client.request_with_circuit_breaker(
                service_name="recovery_test",
                method="GET",
                url="https://example.com",
            )

            recovery_time = time.time() - start_time

            assert response is not None
            assert recovery_time < 0.3  # Should recover quickly
            print(f"\nCircuit breaker recovery time: {recovery_time:.3f}s")


class TestRetryMechanismPerformance:
    """Performance tests for retry mechanism."""

    @pytest.mark.asyncio
    async def test_retry_with_jitter_distribution(self, http_factory):
        """Test that retry jitter provides good distribution."""
        client = AsyncHTTPClient(http_factory)
        client.retry_handler.config.retry_delay = 0.01
        client.retry_handler.config.retry_attempts = 2

        retry_delays = []

        async def fail_once():
            """Fail on first attempt, track retry delay."""
            # random imported at top level

            if not hasattr(fail_once, "attempt"):
                fail_once.attempt = 0
            fail_once.attempt += 1

            if fail_once.attempt == 1:
                fail_once.start_time = time.time()
                raise httpx.NetworkError("First attempt fails")
            retry_delays.append(time.time() - fail_once.start_time)
            return MagicMock()

        # Run multiple retries to check jitter distribution
        num_tests = 20
        for _ in range(num_tests):
            fail_once.attempt = 0
            await client.retry_handler.execute_with_retry(fail_once)

        # Check that jitter provides variation
        min_delay = min(retry_delays)
        max_delay = max(retry_delays)
        avg_delay = sum(retry_delays) / len(retry_delays)

        assert min_delay >= 0.01  # At least base delay
        assert max_delay <= 1.1  # Not more than base + max jitter
        assert 0.01 <= avg_delay <= 0.6  # Reasonable average

        print(f"\nRetry jitter distribution ({num_tests} samples):")
        print(f"  Min delay: {min_delay * 1000:.1f}ms")
        print(f"  Max delay: {max_delay * 1000:.1f}ms")
        print(f"  Avg delay: {avg_delay * 1000:.1f}ms")
