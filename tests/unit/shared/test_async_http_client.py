"""Unit tests for async HTTP client utilities."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.utils.async_http_client import (
    AsyncHTTPClient,
    AsyncHTTPClientFactory,
    HTTPClientConfig,
    RetryHandler,
    cleanup_global_factory,
    get_global_http_factory,
)


@pytest.fixture
async def http_config():
    """Create test HTTP client configuration."""
    return HTTPClientConfig(
        timeout=1.0,
        max_keepalive_connections=5,
        max_connections=10,
        retry_attempts=2,
        retry_delay=0.1,
        retry_max_delay=1.0,
        circuit_breaker_fail_max=2,
        circuit_breaker_reset_timeout=1,
    )


@pytest.fixture
async def factory(http_config):
    """Create test HTTP client factory."""
    factory = AsyncHTTPClientFactory(http_config)
    yield factory
    await factory.close()


@pytest.fixture
async def retry_handler(http_config):
    """Create test retry handler."""
    return RetryHandler(http_config)


class TestHTTPClientConfig:
    """Test HTTP client configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HTTPClientConfig()
        assert config.timeout == 10.0
        assert config.max_keepalive_connections == 20
        assert config.max_connections == 50
        assert config.user_agent == "tracktion/1.0"
        assert config.retry_attempts == 3
        assert config.circuit_breaker_fail_max == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HTTPClientConfig(
            timeout=5.0,
            max_connections=100,
            retry_attempts=5,
            user_agent="test/1.0",
        )
        assert config.timeout == 5.0
        assert config.max_connections == 100
        assert config.retry_attempts == 5
        assert config.user_agent == "test/1.0"


class TestAsyncHTTPClientFactory:
    """Test async HTTP client factory."""

    @pytest.mark.asyncio
    async def test_get_httpx_client(self, factory):
        """Test getting httpx client from factory."""
        async with factory.get_httpx_client() as client:
            assert isinstance(client, httpx.AsyncClient)
            assert client.timeout.connect == factory.config.timeout
            # Verify same client is returned on subsequent calls
            async with factory.get_httpx_client() as client2:
                assert client is client2

    @pytest.mark.asyncio
    async def test_get_aiohttp_session(self, factory):
        """Test getting aiohttp session from factory."""
        async with factory.get_aiohttp_session() as session:
            assert session is not None
            # Verify same session is returned on subsequent calls
            async with factory.get_aiohttp_session() as session2:
                assert session is session2

    def test_get_circuit_breaker(self, factory):
        """Test getting circuit breaker for service."""
        breaker1 = factory.get_circuit_breaker("service1")
        breaker2 = factory.get_circuit_breaker("service1")
        breaker3 = factory.get_circuit_breaker("service2")

        # Same breaker for same service
        assert breaker1 is breaker2
        # Different breaker for different service
        assert breaker1 is not breaker3
        # Check configuration
        assert breaker1.fail_max == factory.config.circuit_breaker_fail_max
        assert breaker1.reset_timeout == factory.config.circuit_breaker_reset_timeout

    @pytest.mark.asyncio
    async def test_close(self, factory):
        """Test closing factory resources."""
        async with factory.get_httpx_client():
            pass
        async with factory.get_aiohttp_session():
            pass

        await factory.close()
        # After closing, clients should be None
        assert factory._httpx_client is None
        assert factory._aiohttp_session is None


class TestRetryHandler:
    """Test retry handler functionality."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, retry_handler):
        """Test successful execution without retry."""

        async def success_func():
            return "success"

        result = await retry_handler.execute_with_retry(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, retry_handler):
        """Test retry on network errors."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.NetworkError("Network error")
            return "success"

        result = await retry_handler.execute_with_retry(flaky_func)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, retry_handler):
        """Test retry on timeout errors."""
        call_count = 0

        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("Timeout")
            return "success"

        result = await retry_handler.execute_with_retry(timeout_func)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retry_attempts(self, retry_handler):
        """Test maximum retry attempts are respected."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise httpx.NetworkError("Always fails")

        with pytest.raises(httpx.NetworkError):
            await retry_handler.execute_with_retry(always_fail)

        assert call_count == retry_handler.config.retry_attempts

    @pytest.mark.asyncio
    async def test_retry_with_jitter(self, retry_handler):
        """Test that retry includes jitter to prevent thundering herd."""
        retry_handler.config.retry_delay = 0.1
        retry_handler.config.retry_max_delay = 1.0

        async def fail_once():
            if not hasattr(fail_once, "called"):
                fail_once.called = True
                raise httpx.NetworkError("Network error")
            return "success"

        start_time = asyncio.get_event_loop().time()
        result = await retry_handler.execute_with_retry(fail_once)
        end_time = asyncio.get_event_loop().time()

        assert result == "success"
        # Should have some delay due to retry + jitter
        assert end_time - start_time >= 0.1


class TestAsyncHTTPClient:
    """Test high-level async HTTP client."""

    @pytest.mark.asyncio
    async def test_request_with_circuit_breaker_success(self, factory):
        """Test successful request with circuit breaker."""
        client = AsyncHTTPClient(factory)

        with patch.object(factory, "get_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_response.raise_for_status = MagicMock()

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client)
            mock_context.__aexit__ = AsyncMock()
            mock_get_client.return_value = mock_context

            response = await client.request_with_circuit_breaker(
                service_name="test_service",
                method="GET",
                url="https://example.com",
            )

            assert response == mock_response
            mock_client.request.assert_called_once_with("GET", "https://example.com")

    @pytest.mark.asyncio
    async def test_request_with_circuit_breaker_failure(self, factory):
        """Test request failure with circuit breaker."""
        client = AsyncHTTPClient(factory)

        with patch.object(factory, "get_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.NetworkError("Network error"))

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client)
            mock_context.__aexit__ = AsyncMock()
            mock_get_client.return_value = mock_context

            # Configure retry handler to not retry (for test speed)
            client.retry_handler.config.retry_attempts = 1

            with pytest.raises(httpx.NetworkError):  # Specific exception type
                await client.request_with_circuit_breaker(
                    service_name="test_service",
                    method="GET",
                    url="https://example.com",
                )

    @pytest.mark.asyncio
    async def test_batch_requests(self, factory):
        """Test batch request execution."""
        client = AsyncHTTPClient(factory)

        with patch.object(client, "request_with_circuit_breaker") as mock_request:
            mock_response1 = MagicMock()
            mock_response2 = MagicMock()
            mock_request.side_effect = [mock_response1, mock_response2]

            requests = [
                {
                    "service_name": "service1",
                    "method": "GET",
                    "url": "https://example1.com",
                },
                {
                    "service_name": "service2",
                    "method": "POST",
                    "url": "https://example2.com",
                },
            ]

            responses = await client.batch_requests(requests, max_concurrent=2)

            assert len(responses) == 2
            assert responses[0] == mock_response1
            assert responses[1] == mock_response2
            assert mock_request.call_count == 2


class TestGlobalFactory:
    """Test global factory management."""

    @pytest.mark.asyncio
    async def test_get_global_factory(self):
        """Test getting global factory instance."""
        factory1 = get_global_http_factory()
        factory2 = get_global_http_factory()
        assert factory1 is factory2

        await cleanup_global_factory()

    @pytest.mark.asyncio
    async def test_cleanup_global_factory(self):
        """Test cleaning up global factory."""
        factory = get_global_http_factory()
        assert factory is not None

        await cleanup_global_factory()

        # After cleanup, should create new instance
        factory2 = get_global_http_factory()
        assert factory2 is not factory

        await cleanup_global_factory()


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, factory):
        """Test that circuit breaker opens after consecutive failures."""
        client = AsyncHTTPClient(factory)
        breaker = factory.get_circuit_breaker("failing_service")

        # Configure for faster testing
        breaker.fail_max = 2
        breaker.reset_timeout = 0.5

        with patch.object(factory, "get_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.NetworkError("Network error"))

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_client)
            mock_context.__aexit__ = AsyncMock()
            mock_get_client.return_value = mock_context

            # Configure no retries for faster testing
            client.retry_handler.config.retry_attempts = 1

            # First failure
            with pytest.raises(httpx.NetworkError):
                await client.request_with_circuit_breaker(
                    service_name="failing_service",
                    method="GET",
                    url="https://example.com",
                )

            # Second failure should open circuit
            with pytest.raises(httpx.NetworkError):
                await client.request_with_circuit_breaker(
                    service_name="failing_service",
                    method="GET",
                    url="https://example.com",
                )

            # Circuit should now be open
            assert breaker.state == "open"

            # Wait for reset timeout
            await asyncio.sleep(0.6)

            # Circuit should be half-open now
            assert breaker.state == "half-open"
