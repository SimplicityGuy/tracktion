"""
Graceful shutdown handler for the analysis service.

This module provides functionality to handle graceful shutdown of the service,
ensuring in-flight requests are completed and resources are properly cleaned up.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown of the analysis service."""

    def __init__(
        self,
        timeout: float = 30.0,
        drain_timeout: float = 10.0,
        force_kill_timeout: float = 5.0,
    ) -> None:
        """
        Initialize the graceful shutdown handler.

        Args:
            timeout: Maximum time to wait for in-flight requests to complete (seconds)
            drain_timeout: Maximum time to wait for queue draining (seconds)
            force_kill_timeout: Time to wait before force killing after timeout (seconds)
        """
        self.timeout = timeout
        self.drain_timeout = drain_timeout
        self.force_kill_timeout = force_kill_timeout

        # Shutdown state
        self._shutdown_event = threading.Event()
        self._shutdown_initiated = False
        self._shutdown_complete = False
        self._lock = threading.RLock()

        # Track in-flight requests
        self._in_flight_requests: set[str] = set()
        self._request_lock = threading.RLock()

        # Shutdown hooks
        self._shutdown_hooks: list[Callable[[], None]] = []

        # Resources to clean up
        self._resources: dict[str, Any] = {}

        # Statistics
        self._shutdown_start_time: float | None = None
        self._requests_completed = 0
        self._requests_aborted = 0

        logger.info(
            "Graceful shutdown handler initialized",
            timeout=timeout,
            drain_timeout=drain_timeout,
            force_kill_timeout=force_kill_timeout,
        )

    def register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Windows compatibility
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

        logger.info("Signal handlers registered for graceful shutdown")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received shutdown signal: {signal_name}")

        # Initiate shutdown
        threading.Thread(target=self.shutdown, name="shutdown-thread").start()

    def register_shutdown_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a hook to be called during shutdown.

        Args:
            hook: Function to call during shutdown
        """
        with self._lock:
            self._shutdown_hooks.append(hook)
            hook_name = getattr(hook, "__name__", str(hook))
            logger.debug(f"Registered shutdown hook: {hook_name}")

    def register_resource(self, name: str, resource: Any) -> None:
        """
        Register a resource to be cleaned up during shutdown.

        Args:
            name: Resource name
            resource: Resource object (should have close() or cleanup() method)
        """
        with self._lock:
            self._resources[name] = resource
            logger.debug(f"Registered resource for cleanup: {name}")

    @contextmanager
    def track_request(self, request_id: str) -> Any:
        """
        Context manager to track in-flight requests.

        Args:
            request_id: Unique identifier for the request

        Yields:
            None

        Raises:
            RuntimeError: If shutdown is in progress and new requests are not accepted
        """
        # Check if we're shutting down
        if self._shutdown_initiated:
            raise RuntimeError("Service is shutting down, not accepting new requests")

        # Register request
        with self._request_lock:
            self._in_flight_requests.add(request_id)
            logger.debug(f"Request started: {request_id}")

        try:
            yield
        finally:
            # Unregister request
            with self._request_lock:
                self._in_flight_requests.discard(request_id)
                if self._shutdown_initiated:
                    self._requests_completed += 1
                logger.debug(f"Request completed: {request_id}")

    def is_shutting_down(self) -> bool:
        """
        Check if shutdown is in progress.

        Returns:
            True if shutdown has been initiated
        """
        return self._shutdown_initiated

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """
        Wait for shutdown to complete.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if shutdown completed within timeout
        """
        return self._shutdown_event.wait(timeout)

    def shutdown(self) -> None:
        """Perform graceful shutdown."""
        with self._lock:
            if self._shutdown_initiated:
                logger.warning("Shutdown already in progress")
                return

            self._shutdown_initiated = True
            self._shutdown_start_time = time.time()

        logger.info("Starting graceful shutdown")

        try:
            # Step 1: Stop accepting new requests (already done by setting _shutdown_initiated)
            logger.info("Step 1: Stopped accepting new requests")

            # Step 2: Call shutdown hooks
            self._call_shutdown_hooks()

            # Step 3: Wait for in-flight requests to complete
            self._wait_for_requests()

            # Step 4: Drain queues
            self._drain_queues()

            # Step 5: Clean up resources
            self._cleanup_resources()

            # Step 6: Final cleanup
            self._final_cleanup()

            # Mark shutdown as complete
            self._shutdown_complete = True
            self._shutdown_event.set()

            elapsed = time.time() - self._shutdown_start_time
            logger.info(
                "Graceful shutdown completed",
                elapsed_seconds=elapsed,
                requests_completed=self._requests_completed,
                requests_aborted=self._requests_aborted,
            )

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}", exc_info=True)
            self._force_shutdown()

    def _call_shutdown_hooks(self) -> None:
        """Call all registered shutdown hooks."""
        logger.info(f"Step 2: Calling {len(self._shutdown_hooks)} shutdown hooks")

        for hook in self._shutdown_hooks:
            try:
                hook_name = getattr(hook, "__name__", str(hook))
                logger.debug(f"Calling shutdown hook: {hook_name}")
                hook()
            except Exception as e:
                logger.error(f"Error in shutdown hook: {e}", exc_info=True)

    def _wait_for_requests(self) -> None:
        """Wait for in-flight requests to complete."""
        logger.info(
            f"Step 3: Waiting for {len(self._in_flight_requests)} in-flight requests",
            timeout=self.timeout,
        )

        start_time = time.time()
        check_interval = 0.5

        while self._in_flight_requests and (time.time() - start_time) < self.timeout:
            remaining = len(self._in_flight_requests)
            elapsed = time.time() - start_time

            logger.debug(
                "Waiting for requests to complete",
                remaining=remaining,
                elapsed=elapsed,
            )

            time.sleep(check_interval)

        # Check if any requests are still in flight
        if self._in_flight_requests:
            self._requests_aborted = len(self._in_flight_requests)
            logger.warning(
                f"Timeout reached, aborting {self._requests_aborted} requests",
                request_ids=list(self._in_flight_requests),
            )
        else:
            logger.info("All in-flight requests completed")

    def _drain_queues(self) -> None:
        """Drain message queues before shutdown."""
        logger.info("Step 4: Draining queues", timeout=self.drain_timeout)

        # This would be implemented based on the specific queue system
        # For now, it's a placeholder that can be extended
        start_time = time.time()

        # Simulate queue draining
        while (time.time() - start_time) < self.drain_timeout:
            # Check if there are messages to process
            # This would interact with RabbitMQ or other queue systems
            queue_size = self._get_queue_size()
            if queue_size == 0:
                logger.info("All queues drained")
                break

            logger.debug(f"Queue size: {queue_size}")
            time.sleep(0.5)
        else:
            logger.warning("Queue drain timeout reached")

    def _get_queue_size(self) -> int:
        """
        Get the current queue size.

        Returns:
            Number of messages in queue (placeholder implementation)
        """
        # Placeholder - would connect to actual queue system
        return 0

    def _cleanup_resources(self) -> None:
        """Clean up registered resources."""
        logger.info(f"Step 5: Cleaning up {len(self._resources)} resources")

        for name, resource in self._resources.items():
            try:
                logger.debug(f"Cleaning up resource: {name}")

                # Try different cleanup methods
                if hasattr(resource, "close"):
                    resource.close()
                elif hasattr(resource, "cleanup"):
                    resource.cleanup()
                elif hasattr(resource, "shutdown"):
                    resource.shutdown()
                elif hasattr(resource, "dispose"):
                    resource.dispose()
                else:
                    logger.warning(f"No cleanup method found for resource: {name}")

            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}", exc_info=True)

    def _final_cleanup(self) -> None:
        """Perform final cleanup tasks."""
        logger.info("Step 6: Performing final cleanup")

        # Ensure all database transactions are committed/rolled back
        # This would interact with actual database connections

        # Close any remaining connections
        # This would close Redis, Neo4j, PostgreSQL connections

        # Final logging
        logger.info("Final cleanup completed")

    def _force_shutdown(self) -> None:
        """Force shutdown after timeout."""
        logger.error(f"Force shutdown initiated after {self.force_kill_timeout} seconds")

        time.sleep(self.force_kill_timeout)

        # Force exit
        os._exit(1)

    def get_stats(self) -> dict[str, Any]:
        """
        Get shutdown statistics.

        Returns:
            Dictionary containing shutdown statistics
        """
        stats: dict[str, Any] = {
            "shutdown_initiated": self._shutdown_initiated,
            "shutdown_complete": self._shutdown_complete,
            "in_flight_requests": len(self._in_flight_requests),
            "requests_completed": self._requests_completed,
            "requests_aborted": self._requests_aborted,
            "registered_hooks": len(self._shutdown_hooks),
            "registered_resources": len(self._resources),
        }

        if self._shutdown_start_time is not None:
            stats["shutdown_duration"] = time.time() - self._shutdown_start_time

        return stats


class AsyncGracefulShutdownHandler(GracefulShutdownHandler):
    """Async version of the graceful shutdown handler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the async graceful shutdown handler."""
        super().__init__(*args, **kwargs)
        self._async_shutdown_hooks: list[Callable[[], Any]] = []

    def register_async_shutdown_hook(self, hook: Callable[[], Any]) -> None:
        """
        Register an async hook to be called during shutdown.

        Args:
            hook: Async function to call during shutdown
        """
        with self._lock:
            self._async_shutdown_hooks.append(hook)
            hook_name = getattr(hook, "__name__", str(hook))
            logger.debug(f"Registered async shutdown hook: {hook_name}")

    async def async_shutdown(self) -> None:
        """Perform async graceful shutdown."""
        with self._lock:
            if self._shutdown_initiated:
                logger.warning("Shutdown already in progress")
                return

            self._shutdown_initiated = True
            self._shutdown_start_time = time.time()

        logger.info("Starting async graceful shutdown")

        try:
            # Step 1: Stop accepting new requests
            logger.info("Step 1: Stopped accepting new requests")

            # Step 2: Call shutdown hooks (both sync and async)
            await self._call_async_shutdown_hooks()

            # Step 3: Wait for in-flight requests
            await self._async_wait_for_requests()

            # Step 4: Drain queues
            await self._async_drain_queues()

            # Step 5: Clean up resources
            await self._async_cleanup_resources()

            # Step 6: Final cleanup
            self._final_cleanup()

            # Mark shutdown as complete
            self._shutdown_complete = True
            self._shutdown_event.set()

            elapsed = time.time() - self._shutdown_start_time
            logger.info(
                "Async graceful shutdown completed",
                elapsed_seconds=elapsed,
                requests_completed=self._requests_completed,
                requests_aborted=self._requests_aborted,
            )

        except Exception as e:
            logger.error(f"Error during async graceful shutdown: {e}", exc_info=True)
            self._force_shutdown()

    async def _call_async_shutdown_hooks(self) -> None:
        """Call all registered async shutdown hooks."""
        # Call sync hooks first
        self._call_shutdown_hooks()

        # Then call async hooks
        logger.info(f"Calling {len(self._async_shutdown_hooks)} async shutdown hooks")

        tasks = []
        for hook in self._async_shutdown_hooks:
            hook_name = getattr(hook, "__name__", str(hook))
            logger.debug(f"Calling async shutdown hook: {hook_name}")
            tasks.append(asyncio.create_task(hook()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _async_wait_for_requests(self) -> None:
        """Async wait for in-flight requests."""
        logger.info(f"Step 3: Async waiting for {len(self._in_flight_requests)} requests")

        start_time = time.time()

        while self._in_flight_requests and (time.time() - start_time) < self.timeout:
            await asyncio.sleep(0.5)

        if self._in_flight_requests:
            self._requests_aborted = len(self._in_flight_requests)
            logger.warning(f"Timeout, aborting {self._requests_aborted} requests")

    async def _async_drain_queues(self) -> None:
        """Async drain message queues."""
        logger.info("Step 4: Async draining queues")

        start_time = time.time()

        while (time.time() - start_time) < self.drain_timeout:
            queue_size = await self._async_get_queue_size()
            if queue_size == 0:
                logger.info("All queues drained")
                break

            await asyncio.sleep(0.5)

    async def _async_get_queue_size(self) -> int:
        """Async get queue size."""
        # Placeholder for async queue interaction
        return 0

    async def _async_cleanup_resources(self) -> None:
        """Async cleanup resources."""
        logger.info(f"Step 5: Async cleaning up {len(self._resources)} resources")

        tasks = []
        for name, resource in self._resources.items():
            if asyncio.iscoroutinefunction(getattr(resource, "close", None)):
                tasks.append(resource.close())
            else:
                # Handle sync cleanup in thread
                self._cleanup_single_resource(name, resource)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _cleanup_single_resource(self, name: str, resource: Any) -> None:
        """Clean up a single resource."""
        try:
            if hasattr(resource, "close"):
                resource.close()
            elif hasattr(resource, "cleanup"):
                resource.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up {name}: {e}")


# Global instance
_shutdown_handler: GracefulShutdownHandler | None = None


def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get the global shutdown handler instance."""
    global _shutdown_handler  # noqa: PLW0603 - Standard singleton pattern for global shutdown handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
    return _shutdown_handler


def reset_shutdown_handler() -> None:
    """Reset the global shutdown handler (mainly for testing)."""
    global _shutdown_handler  # noqa: PLW0603 - Standard singleton reset pattern for testing
    _shutdown_handler = None
