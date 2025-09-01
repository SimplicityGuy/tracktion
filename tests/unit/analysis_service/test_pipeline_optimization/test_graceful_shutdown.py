"""
Unit tests for graceful shutdown handler.
"""

import asyncio
import signal
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.graceful_shutdown import (
    AsyncGracefulShutdownHandler,
    GracefulShutdownHandler,
    get_shutdown_handler,
    reset_shutdown_handler,
)


class TestGracefulShutdownHandler:
    """Tests for GracefulShutdownHandler class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.handler = GracefulShutdownHandler(
            timeout=2.0,
            drain_timeout=1.0,
            force_kill_timeout=0.5,
        )

    def test_initialization(self) -> None:
        """Test handler initialization."""
        assert self.handler.timeout == 2.0
        assert self.handler.drain_timeout == 1.0
        assert self.handler.force_kill_timeout == 0.5
        assert not self.handler.is_shutting_down()
        assert len(self.handler._shutdown_hooks) == 0
        assert len(self.handler._resources) == 0

    @patch("signal.signal")
    def test_register_signal_handlers(self, mock_signal: Mock) -> None:
        """Test signal handler registration."""
        self.handler.register_signal_handlers()

        # Should register SIGTERM and SIGINT
        calls = mock_signal.call_args_list
        registered_signals = [call[0][0] for call in calls]

        assert signal.SIGTERM in registered_signals
        assert signal.SIGINT in registered_signals

    def test_register_shutdown_hook(self) -> None:
        """Test registering shutdown hooks."""
        hook1 = Mock()
        hook2 = Mock()

        self.handler.register_shutdown_hook(hook1)
        self.handler.register_shutdown_hook(hook2)

        assert len(self.handler._shutdown_hooks) == 2
        assert hook1 in self.handler._shutdown_hooks
        assert hook2 in self.handler._shutdown_hooks

    def test_register_resource(self) -> None:
        """Test registering resources for cleanup."""
        resource1 = Mock(spec=["close"])
        resource2 = Mock(spec=["cleanup"])

        self.handler.register_resource("db_connection", resource1)
        self.handler.register_resource("cache_client", resource2)

        assert len(self.handler._resources) == 2
        assert self.handler._resources["db_connection"] == resource1
        assert self.handler._resources["cache_client"] == resource2

    def test_track_request_normal(self) -> None:
        """Test tracking requests during normal operation."""
        request_id = "req-123"

        with self.handler.track_request(request_id):
            # Request should be tracked
            assert request_id in self.handler._in_flight_requests

        # Request should be removed after completion
        assert request_id not in self.handler._in_flight_requests

    def test_track_request_during_shutdown(self) -> None:
        """Test that new requests are rejected during shutdown."""
        self.handler._shutdown_initiated = True

        with pytest.raises(RuntimeError) as exc_info, self.handler.track_request("req-456"):
            pass

        assert "shutting down" in str(exc_info.value)

    def test_track_request_exception_cleanup(self) -> None:
        """Test that requests are cleaned up even if exception occurs."""
        request_id = "req-789"

        with pytest.raises(ValueError), self.handler.track_request(request_id):
            assert request_id in self.handler._in_flight_requests
            raise ValueError("Test error")

        # Request should still be removed
        assert request_id not in self.handler._in_flight_requests

    def test_shutdown_hooks_called(self) -> None:
        """Test that shutdown hooks are called during shutdown."""
        hook1 = Mock()
        hook2 = Mock()

        self.handler.register_shutdown_hook(hook1)
        self.handler.register_shutdown_hook(hook2)

        self.handler.shutdown()

        hook1.assert_called_once()
        hook2.assert_called_once()
        assert self.handler.is_shutting_down()

    def test_shutdown_hook_error_handling(self) -> None:
        """Test that errors in shutdown hooks don't stop shutdown."""
        good_hook = Mock()
        bad_hook = Mock(side_effect=RuntimeError("Hook error"))

        self.handler.register_shutdown_hook(bad_hook)
        self.handler.register_shutdown_hook(good_hook)

        # Should not raise
        self.handler.shutdown()

        # Good hook should still be called
        good_hook.assert_called_once()
        assert self.handler.is_shutting_down()

    def test_resource_cleanup(self) -> None:
        """Test that resources are cleaned up during shutdown."""
        resource1 = Mock(spec=["close"])
        resource2 = Mock(spec=["cleanup"])
        resource3 = Mock(spec=["shutdown"])
        resource4 = Mock(spec=["dispose"])

        self.handler.register_resource("r1", resource1)
        self.handler.register_resource("r2", resource2)
        self.handler.register_resource("r3", resource3)
        self.handler.register_resource("r4", resource4)

        self.handler.shutdown()

        resource1.close.assert_called_once()
        resource2.cleanup.assert_called_once()
        resource3.shutdown.assert_called_once()
        resource4.dispose.assert_called_once()

    def test_resource_cleanup_error_handling(self) -> None:
        """Test that errors in resource cleanup don't stop shutdown."""
        good_resource = Mock(spec=["close"])
        bad_resource = Mock(spec=["close"])
        bad_resource.close.side_effect = RuntimeError("Cleanup error")

        self.handler.register_resource("bad", bad_resource)
        self.handler.register_resource("good", good_resource)

        # Should not raise
        self.handler.shutdown()

        # Good resource should still be cleaned up
        good_resource.close.assert_called_once()

    def test_wait_for_in_flight_requests(self) -> None:
        """Test waiting for in-flight requests to complete."""
        # Start some requests
        self.handler._in_flight_requests.add("req-1")
        self.handler._in_flight_requests.add("req-2")

        # Create a thread to complete requests after delay
        def complete_requests() -> None:
            time.sleep(0.5)
            self.handler._in_flight_requests.clear()
            self.handler._requests_completed = 2

        thread = threading.Thread(target=complete_requests)
        thread.start()

        # Wait should complete when requests are done
        self.handler._wait_for_requests()
        thread.join()

        assert len(self.handler._in_flight_requests) == 0
        assert self.handler._requests_completed == 2
        assert self.handler._requests_aborted == 0

    def test_wait_for_requests_timeout(self) -> None:
        """Test timeout when waiting for in-flight requests."""
        # Set a very short timeout
        self.handler.timeout = 0.1

        # Add requests that won't complete
        self.handler._in_flight_requests.add("req-1")
        self.handler._in_flight_requests.add("req-2")

        # Wait should timeout
        self.handler._wait_for_requests()

        # Requests should be marked as aborted
        assert self.handler._requests_aborted == 2

    def test_wait_for_shutdown(self) -> None:
        """Test waiting for shutdown to complete."""

        # Start shutdown in a thread
        def do_shutdown() -> None:
            time.sleep(0.5)
            self.handler._shutdown_event.set()

        thread = threading.Thread(target=do_shutdown)
        thread.start()

        # Wait should return True when shutdown completes
        result = self.handler.wait_for_shutdown(timeout=2.0)
        thread.join()

        assert result is True

    def test_wait_for_shutdown_timeout(self) -> None:
        """Test timeout when waiting for shutdown."""
        # Don't actually trigger shutdown
        result = self.handler.wait_for_shutdown(timeout=0.1)
        assert result is False

    def test_shutdown_idempotent(self) -> None:
        """Test that shutdown can only be initiated once."""
        hook = Mock()
        self.handler.register_shutdown_hook(hook)

        # First shutdown
        self.handler.shutdown()
        assert hook.call_count == 1

        # Second shutdown should be ignored
        self.handler.shutdown()
        assert hook.call_count == 1  # Not called again

    def test_get_stats(self) -> None:
        """Test getting shutdown statistics."""
        # Set up some state
        self.handler._in_flight_requests.add("req-1")
        self.handler.register_shutdown_hook(Mock())
        self.handler.register_resource("db", Mock())

        stats = self.handler.get_stats()

        assert stats["shutdown_initiated"] is False
        assert stats["shutdown_complete"] is False
        assert stats["in_flight_requests"] == 1
        assert stats["registered_hooks"] == 1
        assert stats["registered_resources"] == 1
        assert "shutdown_duration" not in stats

    def test_get_stats_after_shutdown(self) -> None:
        """Test statistics after shutdown."""
        self.handler._requests_completed = 5
        self.handler._requests_aborted = 2

        self.handler.shutdown()
        stats = self.handler.get_stats()

        assert stats["shutdown_initiated"] is True
        assert stats["shutdown_complete"] is True
        assert stats["requests_completed"] == 5
        assert stats["requests_aborted"] == 2
        assert "shutdown_duration" in stats

    @patch("os._exit")
    def test_force_shutdown(self, mock_exit: Mock) -> None:
        """Test force shutdown after timeout."""
        self.handler.force_kill_timeout = 0.1
        self.handler._force_shutdown()

        # Should call os._exit
        mock_exit.assert_called_once_with(1)

    def test_signal_handler_triggers_shutdown(self) -> None:
        """Test that signal handler triggers shutdown."""
        with patch.object(self.handler, "shutdown") as mock_shutdown:
            # Simulate signal
            self.handler._signal_handler(signal.SIGTERM, None)

            # Give thread time to start
            time.sleep(0.1)

            # Shutdown should be called
            assert mock_shutdown.called

    def test_complete_shutdown_flow(self) -> None:
        """Test complete shutdown flow with all components."""
        # Set up hooks and resources
        hook = Mock()
        resource = Mock(spec=["close"])

        self.handler.register_shutdown_hook(hook)
        self.handler.register_resource("test_resource", resource)

        # Add an in-flight request
        self.handler._in_flight_requests.add("req-1")

        # Clear request after short delay
        def clear_request() -> None:
            time.sleep(0.2)
            self.handler._in_flight_requests.clear()

        thread = threading.Thread(target=clear_request)
        thread.start()

        # Perform shutdown
        self.handler.shutdown()
        thread.join()

        # Verify everything was called
        hook.assert_called_once()
        resource.close.assert_called_once()
        assert self.handler.is_shutting_down()
        assert self.handler._shutdown_complete


class TestAsyncGracefulShutdownHandler:
    """Tests for AsyncGracefulShutdownHandler class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.handler = AsyncGracefulShutdownHandler(
            timeout=2.0,
            drain_timeout=1.0,
            force_kill_timeout=0.5,
        )

    def test_initialization(self) -> None:
        """Test async handler initialization."""
        assert isinstance(self.handler, GracefulShutdownHandler)
        assert hasattr(self.handler, "_async_shutdown_hooks")
        assert len(self.handler._async_shutdown_hooks) == 0

    def test_register_async_shutdown_hook(self) -> None:
        """Test registering async shutdown hooks."""

        async def async_hook() -> None:
            pass

        self.handler.register_async_shutdown_hook(async_hook)
        assert len(self.handler._async_shutdown_hooks) == 1
        assert async_hook in self.handler._async_shutdown_hooks

    @pytest.mark.asyncio
    async def test_async_shutdown_hooks_called(self) -> None:
        """Test that async shutdown hooks are called."""
        sync_hook = Mock()
        async_hook = Mock(return_value=asyncio.sleep(0))

        self.handler.register_shutdown_hook(sync_hook)
        self.handler.register_async_shutdown_hook(async_hook)

        await self.handler.async_shutdown()

        sync_hook.assert_called_once()
        async_hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_wait_for_requests(self) -> None:
        """Test async waiting for in-flight requests."""
        self.handler._in_flight_requests.add("req-1")

        # Clear request after delay
        async def clear_request() -> None:
            await asyncio.sleep(0.2)
            self.handler._in_flight_requests.clear()

        task = asyncio.create_task(clear_request())
        await self.handler._async_wait_for_requests()
        await task

        assert len(self.handler._in_flight_requests) == 0

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self) -> None:
        """Test async resource cleanup."""
        # Create async resource
        async_resource = MagicMock()
        async_resource.close = Mock(return_value=asyncio.sleep(0))

        # Create sync resource
        sync_resource = Mock(spec=["close"])

        self.handler.register_resource("async_res", async_resource)
        self.handler.register_resource("sync_res", sync_resource)

        await self.handler._async_cleanup_resources()

        async_resource.close.assert_called_once()
        sync_resource.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_async_shutdown(self) -> None:
        """Test complete async shutdown flow."""
        sync_hook = Mock()
        async_hook = Mock(return_value=asyncio.sleep(0))
        resource = Mock(spec=["close"])

        self.handler.register_shutdown_hook(sync_hook)
        self.handler.register_async_shutdown_hook(async_hook)
        self.handler.register_resource("resource", resource)

        await self.handler.async_shutdown()

        sync_hook.assert_called_once()
        async_hook.assert_called_once()
        resource.close.assert_called_once()
        assert self.handler.is_shutting_down()
        assert self.handler._shutdown_complete


class TestGlobalShutdownHandler:
    """Tests for global shutdown handler functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_shutdown_handler()

    def test_get_shutdown_handler_creates_instance(self) -> None:
        """Test that get_shutdown_handler creates a singleton."""
        handler1 = get_shutdown_handler()
        handler2 = get_shutdown_handler()

        assert handler1 is not None
        assert handler1 is handler2

    def test_reset_shutdown_handler(self) -> None:
        """Test resetting the global shutdown handler."""
        handler1 = get_shutdown_handler()
        reset_shutdown_handler()
        handler2 = get_shutdown_handler()

        assert handler1 is not handler2
