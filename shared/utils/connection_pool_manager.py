"""Connection pool manager with health checks and monitoring."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# Prometheus metrics for pool monitoring
pool_connections_active = Gauge(
    "http_pool_connections_active",
    "Number of active HTTP connections",
    ["service"],
)
pool_connections_idle = Gauge(
    "http_pool_connections_idle",
    "Number of idle HTTP connections",
    ["service"],
)
pool_connections_total = Gauge(
    "http_pool_connections_total",
    "Total number of HTTP connections",
    ["service"],
)
pool_wait_time = Histogram(
    "http_pool_wait_time_seconds",
    "Time waiting for connection from pool",
    ["service"],
)
pool_health_checks = Counter(
    "http_pool_health_checks_total",
    "Total number of pool health checks",
    ["service", "status"],
)


@dataclass
class PoolStatistics:
    """Statistics for connection pool monitoring."""

    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    requests_served: int = 0
    average_wait_time: float = 0.0
    health_check_failures: int = 0
    last_health_check: float | None = None
    pool_efficiency: float = 0.0  # Percentage of time connections are active


class ConnectionPoolManager:
    """Manages connection pools with health checks and monitoring."""

    def __init__(
        self,
        min_connections: int = 10,
        max_connections: int = 50,
        keepalive_timeout: float = 30.0,
        connection_ttl: float = 300.0,  # 5 minutes
        health_check_interval: float = 60.0,
        health_check_timeout: float = 5.0,
    ) -> None:
        """Initialize the connection pool manager.

        Args:
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
            keepalive_timeout: Keepalive timeout in seconds
            connection_ttl: Connection time-to-live in seconds
            health_check_interval: Interval between health checks in seconds
            health_check_timeout: Timeout for health check requests
        """
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.keepalive_timeout = keepalive_timeout
        self.connection_ttl = connection_ttl
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout

        self._pools: dict[str, httpx.AsyncClient] = {}
        self._pool_stats: dict[str, PoolStatistics] = {}
        self._health_check_tasks: dict[str, asyncio.Task] = {}
        self._connection_timestamps: dict[str, dict[int, float]] = {}
        self._request_queue_times: dict[str, list[float]] = {}

    def create_pool(self, service_name: str, base_url: str | None = None) -> httpx.AsyncClient:
        """Create a new connection pool for a service.

        Args:
            service_name: Name of the service
            base_url: Optional base URL for the service

        Returns:
            Configured httpx client with connection pool
        """
        if service_name in self._pools:
            return self._pools[service_name]

        # Configure the pool
        limits = httpx.Limits(
            max_keepalive_connections=self.min_connections,
            max_connections=self.max_connections,
            keepalive_expiry=self.keepalive_timeout,
        )

        timeout = httpx.Timeout(
            connect=5.0,
            read=10.0,
            write=10.0,
            pool=30.0,  # Max time to wait for connection from pool
        )

        if base_url is not None:
            client = httpx.AsyncClient(
                base_url=base_url,
                limits=limits,
                timeout=timeout,
                http2=True,  # Enable HTTP/2 for better multiplexing
            )
        else:
            client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True,  # Enable HTTP/2 for better multiplexing
            )

        self._pools[service_name] = client
        self._pool_stats[service_name] = PoolStatistics()
        self._connection_timestamps[service_name] = {}
        self._request_queue_times[service_name] = []

        # Start health check task
        self._start_health_check(service_name)

        logger.info(
            "Connection pool created",
            service=service_name,
            min_connections=self.min_connections,
            max_connections=self.max_connections,
        )

        return client

    def _start_health_check(self, service_name: str) -> None:
        """Start health check task for a service pool."""
        if service_name in self._health_check_tasks:
            return

        async def health_check_loop() -> None:
            """Periodic health check loop."""
            while service_name in self._pools:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._perform_health_check(service_name)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "Health check error",
                        service=service_name,
                        error=str(e),
                    )

        task = asyncio.create_task(health_check_loop())
        self._health_check_tasks[service_name] = task

    async def _perform_health_check(self, service_name: str) -> bool:
        """Perform health check on a connection pool.

        Args:
            service_name: Name of the service

        Returns:
            True if health check passed, False otherwise
        """
        client = self._pools.get(service_name)
        if not client:
            return False

        stats = self._pool_stats[service_name]
        stats.last_health_check = time.time()

        try:
            # Perform a simple HEAD request to check connectivity
            if client.base_url:
                response = await client.head("/", timeout=self.health_check_timeout)
                success = response.status_code < 500
            else:
                # If no base URL, consider pool healthy
                success = True

            if success:
                pool_health_checks.labels(service=service_name, status="success").inc()
                logger.debug("Health check passed", service=service_name)
            else:
                pool_health_checks.labels(service=service_name, status="failure").inc()
                stats.health_check_failures += 1
                logger.warning("Health check failed", service=service_name)

            # Clean up old connections
            await self._cleanup_stale_connections(service_name)

            # Update metrics
            self._update_metrics(service_name)

            return success  # type: ignore[no-any-return] # Boolean value from health check comparison, known to be bool

        except Exception as e:
            pool_health_checks.labels(service=service_name, status="error").inc()
            stats.health_check_failures += 1
            logger.error(
                "Health check error",
                service=service_name,
                error=str(e),
            )
            return False

    async def _cleanup_stale_connections(self, service_name: str) -> None:
        """Clean up stale connections that exceeded TTL.

        Args:
            service_name: Name of the service
        """
        timestamps = self._connection_timestamps.get(service_name, {})
        current_time = time.time()
        stale_connections = []

        for conn_id, created_at in timestamps.items():
            if current_time - created_at > self.connection_ttl:
                stale_connections.append(conn_id)

        for conn_id in stale_connections:
            del timestamps[conn_id]
            logger.debug(
                "Cleaned up stale connection",
                service=service_name,
                connection_id=conn_id,
            )

    def _update_metrics(self, service_name: str) -> None:
        """Update Prometheus metrics for a service pool.

        Args:
            service_name: Name of the service
        """
        stats = self.get_statistics(service_name)

        pool_connections_active.labels(service=service_name).set(stats.active_connections)
        pool_connections_idle.labels(service=service_name).set(stats.idle_connections)
        pool_connections_total.labels(service=service_name).set(stats.total_connections)

    def get_statistics(self, service_name: str) -> PoolStatistics:
        """Get statistics for a service pool.

        Args:
            service_name: Name of the service

        Returns:
            Pool statistics
        """
        stats = self._pool_stats.get(service_name, PoolStatistics())

        # Calculate average wait time
        queue_times = self._request_queue_times.get(service_name, [])
        if queue_times:
            # Keep only recent samples (last 100)
            recent_times = queue_times[-100:]
            stats.average_wait_time = sum(recent_times) / len(recent_times)
            self._request_queue_times[service_name] = recent_times

        # Estimate active/idle connections (simplified)
        total_conns = len(self._connection_timestamps.get(service_name, {}))
        stats.total_connections = min(total_conns, self.max_connections)
        stats.idle_connections = max(0, self.min_connections - stats.active_connections)

        # Calculate pool efficiency
        if stats.total_connections > 0:
            stats.pool_efficiency = (stats.active_connections / stats.total_connections) * 100

        return stats

    async def warmup_pool(self, service_name: str, num_connections: int = 5) -> None:
        """Pre-warm the connection pool with initial connections.

        Args:
            service_name: Name of the service
            num_connections: Number of connections to pre-establish
        """
        client = self._pools.get(service_name)
        if not client:
            logger.warning("No pool found for warmup", service=service_name)
            return

        logger.info(
            "Warming up connection pool",
            service=service_name,
            connections=num_connections,
        )

        # Make parallel HEAD requests to establish connections
        tasks = [
            client.head("/", timeout=5.0) for _ in range(min(num_connections, self.min_connections)) if client.base_url
        ]

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(
                    "Pool warmup completed",
                    service=service_name,
                    connections=len(tasks),
                )
            except Exception as e:
                logger.warning(
                    "Pool warmup partially failed",
                    service=service_name,
                    error=str(e),
                )

    async def request_with_tracking(
        self,
        service_name: str,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a request with connection tracking.

        Args:
            service_name: Name of the service
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            HTTP response
        """
        client = self._pools.get(service_name)
        if not client:
            raise ValueError(f"No pool found for service: {service_name}")

        stats = self._pool_stats[service_name]

        # Track queue time
        queue_start = time.time()
        stats.active_connections += 1

        try:
            response = await client.request(method, url, **kwargs)

            # Record queue time
            queue_time = time.time() - queue_start
            self._request_queue_times.setdefault(service_name, []).append(queue_time)
            pool_wait_time.labels(service=service_name).observe(queue_time)

            stats.requests_served += 1
            return response

        finally:
            stats.active_connections -= 1

    async def close_pool(self, service_name: str) -> None:
        """Close a connection pool and cleanup resources.

        Args:
            service_name: Name of the service
        """
        # Cancel health check task
        if service_name in self._health_check_tasks:
            self._health_check_tasks[service_name].cancel()
            del self._health_check_tasks[service_name]

        # Close the client
        if service_name in self._pools:
            await self._pools[service_name].aclose()
            del self._pools[service_name]

        # Cleanup tracking data
        if service_name in self._pool_stats:
            del self._pool_stats[service_name]
        if service_name in self._connection_timestamps:
            del self._connection_timestamps[service_name]
        if service_name in self._request_queue_times:
            del self._request_queue_times[service_name]

        logger.info("Connection pool closed", service=service_name)

    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        services = list(self._pools.keys())
        for service in services:
            await self.close_pool(service)

        logger.info("All connection pools closed")

    @asynccontextmanager
    async def pool_context(
        self,
        service_name: str,
        base_url: str | None = None,
    ) -> AsyncIterator[httpx.AsyncClient]:
        """Context manager for automatic pool management.

        Args:
            service_name: Name of the service
            base_url: Optional base URL for the service

        Yields:
            Configured httpx client with connection pool

        Example:
            async with manager.pool_context("api_service") as client:
                response = await client.get("/endpoint")
        """
        client = self.create_pool(service_name, base_url)
        try:
            yield client
        finally:
            await self.close_pool(service_name)


class ConnectionPoolManagerSingleton:
    """Singleton wrapper for ConnectionPoolManager."""

    _instance: ConnectionPoolManager | None = None

    def __new__(cls) -> "ConnectionPoolManagerSingleton":
        """Get the singleton ConnectionPoolManager instance."""
        if cls._instance is None:
            cls._instance = ConnectionPoolManager()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup the singleton instance."""
        if cls._instance:
            await cls._instance.close_all_pools()
            cls._instance = None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


def get_global_pool_manager() -> "ConnectionPoolManagerSingleton":
    """Get or create the singleton connection pool manager.

    Returns:
        Singleton connection pool manager instance
    """
    return ConnectionPoolManagerSingleton()


async def cleanup_global_pool_manager() -> None:
    """Cleanup the singleton connection pool manager."""
    await ConnectionPoolManagerSingleton.cleanup()
