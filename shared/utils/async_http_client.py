"""Async HTTP client utilities with connection pooling, retry logic, and circuit breaker."""

import asyncio
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from pybreaker import CircuitBreaker  # type: ignore[import-not-found]

logger = structlog.get_logger(__name__)


class HTTPClientConfig:
    """Configuration for async HTTP clients."""

    def __init__(
        self,
        timeout: float = 10.0,
        max_keepalive_connections: int = 20,
        max_connections: int = 50,
        keepalive_expiry: float = 30.0,
        user_agent: str = "tracktion/1.0",
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        retry_factor: float = 2.0,
        circuit_breaker_fail_max: int = 5,
        circuit_breaker_reset_timeout: int = 60,
    ) -> None:
        """Initialize HTTP client configuration.

        Args:
            timeout: Request timeout in seconds
            max_keepalive_connections: Maximum number of keepalive connections
            max_connections: Maximum total connections
            keepalive_expiry: Keepalive timeout in seconds
            user_agent: User agent string for requests
            retry_attempts: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
            retry_max_delay: Maximum retry delay in seconds
            retry_factor: Exponential backoff factor
            circuit_breaker_fail_max: Failures before circuit opens
            circuit_breaker_reset_timeout: Seconds before attempting reset
        """
        self.timeout = timeout
        self.max_keepalive_connections = max_keepalive_connections
        self.max_connections = max_connections
        self.keepalive_expiry = keepalive_expiry
        self.user_agent = user_agent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.retry_factor = retry_factor
        self.circuit_breaker_fail_max = circuit_breaker_fail_max
        self.circuit_breaker_reset_timeout = circuit_breaker_reset_timeout


class AsyncHTTPClientFactory:
    """Factory for creating configured async HTTP clients."""

    def __init__(self, config: HTTPClientConfig | None = None) -> None:
        """Initialize the HTTP client factory.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or HTTPClientConfig()
        self._httpx_client: httpx.AsyncClient | None = None
        self._aiohttp_session: ClientSession | None = None
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    @asynccontextmanager
    async def get_httpx_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get or create an httpx async client.

        Yields:
            Configured httpx async client
        """
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    max_connections=self.config.max_connections,
                    keepalive_expiry=self.config.keepalive_expiry,
                ),
                headers={"User-Agent": self.config.user_agent},
            )
        try:
            yield self._httpx_client
        except Exception as e:
            logger.error("Error with httpx client", error=str(e))
            raise

    @asynccontextmanager
    async def get_aiohttp_session(self) -> AsyncIterator[ClientSession]:
        """Get or create an aiohttp session.

        Yields:
            Configured aiohttp session
        """
        if self._aiohttp_session is None:
            connector = TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_keepalive_connections,
                keepalive_timeout=self.config.keepalive_expiry,
            )
            timeout = ClientTimeout(total=self.config.timeout)
            self._aiohttp_session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": self.config.user_agent},
            )
        try:
            yield self._aiohttp_session
        except Exception as e:
            logger.error("Error with aiohttp session", error=str(e))
            raise

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service.

        Args:
            service_name: Name of the service to protect

        Returns:
            Circuit breaker instance for the service
        """
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker(
                fail_max=self.config.circuit_breaker_fail_max,
                reset_timeout=self.config.circuit_breaker_reset_timeout,
                name=service_name,
            )
        return self._circuit_breakers[service_name]

    async def close(self) -> None:
        """Close all HTTP clients and sessions."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None
        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter."""

    def __init__(self, config: HTTPClientConfig | None = None) -> None:
        """Initialize retry handler.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or HTTPClientConfig()

    async def execute_with_retry(
        self,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        delay = self.config.retry_delay

        for attempt in range(self.config.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError, ConnectionError) as e:
                last_exception = e
                if attempt < self.config.retry_attempts - 1:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 1)
                    sleep_time = min(delay + jitter, self.config.retry_max_delay)

                    logger.warning(
                        "Request failed, retrying",
                        attempt=attempt + 1,
                        max_attempts=self.config.retry_attempts,
                        sleep_time=sleep_time,
                        error=str(e),
                    )

                    await asyncio.sleep(sleep_time)
                    delay *= self.config.retry_factor
                else:
                    logger.error(
                        "All retry attempts failed",
                        attempts=self.config.retry_attempts,
                        error=str(e),
                    )

        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed with unknown error")


class AsyncHTTPClient:
    """High-level async HTTP client with retry and circuit breaker support."""

    def __init__(
        self,
        factory: AsyncHTTPClientFactory,
        retry_handler: RetryHandler | None = None,
    ) -> None:
        """Initialize the async HTTP client.

        Args:
            factory: HTTP client factory
            retry_handler: Optional retry handler, creates one if not provided
        """
        self.factory = factory
        self.retry_handler = retry_handler or RetryHandler(factory.config)

    async def request_with_circuit_breaker(
        self,
        service_name: str,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with circuit breaker protection.

        Args:
            service_name: Name of the service for circuit breaker
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            HTTP response

        Raises:
            Exception: If circuit is open or request fails
        """
        circuit_breaker = self.factory.get_circuit_breaker(service_name)

        async def _make_request() -> httpx.Response:
            async with self.factory.get_httpx_client() as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

        # Circuit breaker integration with proper pybreaker usage
        # The pybreaker library handles state management internally
        try:
            # Create a wrapper for the circuit breaker to handle async calls
            @circuit_breaker  # type: ignore[misc]
            async def _protected_request() -> httpx.Response:
                return await self.retry_handler.execute_with_retry(_make_request)  # type: ignore[no-any-return]

            # Execute the protected request
            result = await _protected_request()
            return result  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(
                "Request failed with circuit breaker",
                service=service_name,
                url=url,
                error=str(e),
                circuit_state=str(circuit_breaker.state),
            )
            raise

    async def batch_requests(
        self,
        requests: list[dict[str, Any]],
        max_concurrent: int = 10,
    ) -> list[httpx.Response]:
        """Execute multiple requests concurrently with rate limiting.

        Args:
            requests: List of request parameters
            max_concurrent: Maximum concurrent requests

        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited_request(req: dict[str, Any]) -> httpx.Response:
            async with semaphore:
                return await self.request_with_circuit_breaker(**req)

        tasks = [_limited_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


# Global client factory instance
_global_factory: AsyncHTTPClientFactory | None = None


def get_global_http_factory(
    config: HTTPClientConfig | None = None,
) -> AsyncHTTPClientFactory:
    """Get or create the global HTTP client factory.

    Args:
        config: Optional configuration for the factory

    Returns:
        Global HTTP client factory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = AsyncHTTPClientFactory(config)
    return _global_factory


async def cleanup_global_factory() -> None:
    """Cleanup the global HTTP client factory."""
    global _global_factory
    if _global_factory:
        await _global_factory.close()
        _global_factory = None
