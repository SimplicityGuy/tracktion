"""
Unit tests for the circuit breaker pattern implementation.
"""

import contextlib
import threading
import time
from unittest.mock import Mock

import pytest

from shared.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 60.0
        assert config.failure_window == 60.0
        assert config.expected_exceptions == (Exception,)
        assert config.fallback is None
        assert config.on_open is None
        assert config.on_close is None
        assert config.on_half_open is None

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        fallback_func = Mock(return_value="fallback")
        on_open_func = Mock()

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout=30.0,
            failure_window=120.0,
            expected_exceptions=(ValueError, RuntimeError),
            fallback=fallback_func,
            on_open=on_open_func,
        )

        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout == 30.0
        assert config.failure_window == 120.0
        assert config.expected_exceptions == (ValueError, RuntimeError)
        assert config.fallback == fallback_func
        assert config.on_open == on_open_func


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,  # Short timeout for testing
            failure_window=10.0,
        )
        self.breaker = CircuitBreaker("test_breaker", self.config)

    def test_initialization(self) -> None:
        """Test circuit breaker initialization."""
        assert self.breaker.name == "test_breaker"
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.config == self.config
        assert isinstance(self.breaker.stats, CircuitBreakerStats)

    def test_successful_calls(self) -> None:
        """Test that successful calls don't trigger the circuit."""

        def success_func() -> str:
            return "success"

        # Multiple successful calls
        for _ in range(10):
            result = self.breaker.call(success_func)
            assert result == "success"

        # Circuit should remain closed
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.stats.successful_calls == 10
        assert self.breaker.stats.failed_calls == 0
        assert self.breaker.stats.consecutive_successes == 10

    def test_circuit_opens_on_failures(self) -> None:
        """Test that circuit opens after failure threshold."""

        def failing_func() -> None:
            raise ValueError("Test failure")

        # Cause failures up to threshold
        for i in range(self.config.failure_threshold):
            with pytest.raises(ValueError):
                self.breaker.call(failing_func)

            # Check state after each failure
            if i < self.config.failure_threshold - 1:
                assert self.breaker.state == CircuitState.CLOSED
            else:
                assert self.breaker.state == CircuitState.OPEN

        # Circuit should be open
        assert self.breaker.stats.failed_calls == self.config.failure_threshold
        assert self.breaker.stats.consecutive_failures == self.config.failure_threshold

    def test_circuit_rejects_calls_when_open(self) -> None:
        """Test that calls are rejected when circuit is open."""

        def test_func() -> str:
            return "test"

        # Open the circuit
        self._open_circuit()

        # Calls should be rejected
        with pytest.raises(CircuitOpenError) as exc_info:
            self.breaker.call(test_func)

        assert "test_breaker" in str(exc_info.value)
        assert "OPEN" in str(exc_info.value)
        assert self.breaker.stats.rejected_calls == 1

    def test_circuit_transitions_to_half_open(self) -> None:
        """Test transition from open to half-open after timeout."""
        # Open the circuit
        self._open_circuit()
        assert self.breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(self.config.timeout + 0.1)

        # Should transition to half-open on next state check
        assert self.breaker.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self) -> None:
        """Test transition from half-open to closed on successes."""

        def success_func() -> str:
            return "success"

        # Open the circuit and wait for half-open
        self._open_circuit()
        time.sleep(self.config.timeout + 0.1)
        assert self.breaker.state == CircuitState.HALF_OPEN

        # Successful calls in half-open state
        for i in range(self.config.success_threshold):
            result = self.breaker.call(success_func)
            assert result == "success"

            # Check state after each success
            if i < self.config.success_threshold - 1:
                assert self.breaker.state == CircuitState.HALF_OPEN
            else:
                assert self.breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self) -> None:
        """Test transition from half-open back to open on failure."""

        def failing_func() -> None:
            raise ValueError("Test failure")

        # Open the circuit and wait for half-open
        self._open_circuit()
        time.sleep(self.config.timeout + 0.1)
        assert self.breaker.state == CircuitState.HALF_OPEN

        # Single failure in half-open state should reopen
        with pytest.raises(ValueError):
            self.breaker.call(failing_func)

        assert self.breaker.state == CircuitState.OPEN

    def test_fallback_function(self) -> None:
        """Test fallback function when circuit is open."""
        fallback_mock = Mock(return_value="fallback_result")
        self.breaker.config.fallback = fallback_mock

        # Open the circuit
        self._open_circuit()

        # Call should return fallback result
        result = self.breaker.call(lambda: "normal")
        assert result == "fallback_result"
        fallback_mock.assert_called_once()
        assert self.breaker.stats.fallback_calls == 1

    def test_state_change_hooks(self) -> None:
        """Test that state change hooks are called."""
        on_open_mock = Mock()
        on_close_mock = Mock()
        on_half_open_mock = Mock()

        self.breaker.config.on_open = on_open_mock
        self.breaker.config.on_close = on_close_mock
        self.breaker.config.on_half_open = on_half_open_mock

        def failing_func() -> None:
            raise ValueError("Test")

        def success_func() -> str:
            return "success"

        # Trigger open state
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ValueError):
                self.breaker.call(failing_func)

        on_open_mock.assert_called_once_with("test_breaker")

        # Trigger half-open state
        time.sleep(self.config.timeout + 0.1)
        _ = self.breaker.state  # Check state to trigger transition

        on_half_open_mock.assert_called_once_with("test_breaker")

        # Trigger closed state
        for _ in range(self.config.success_threshold):
            self.breaker.call(success_func)

        on_close_mock.assert_called_once_with("test_breaker")

    def test_failure_window_cleanup(self) -> None:
        """Test that old failures are cleaned up outside the window."""

        def failing_func() -> None:
            raise ValueError("Test")

        # Record some failures
        for _ in range(2):
            with pytest.raises(ValueError):
                self.breaker.call(failing_func)

        # Wait for failures to become old (but not timeout)
        time.sleep(0.5)

        # Manually set failure times to be old
        with self.breaker._lock:
            current_time = time.time()
            self.breaker._failure_times = [
                current_time - self.config.failure_window - 1,  # Old
                current_time - 1,  # Recent
            ]

        # New failure should trigger cleanup
        with pytest.raises(ValueError):
            self.breaker.call(failing_func)

        # Only recent failures should remain
        assert len(self.breaker._failure_times) == 2  # The recent one + new one

    def test_unexpected_exceptions_dont_affect_state(self) -> None:
        """Test that unexpected exceptions don't affect circuit state."""

        def unexpected_error() -> None:
            raise TypeError("Unexpected")

        # Configure to expect only ValueError
        self.breaker.config.expected_exceptions = (ValueError,)

        # Unexpected exception shouldn't affect state
        with pytest.raises(TypeError):
            self.breaker.call(unexpected_error)

        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.stats.failed_calls == 0

    def test_reset_circuit(self) -> None:
        """Test manual circuit reset."""
        # Open the circuit
        self._open_circuit()
        assert self.breaker.state == CircuitState.OPEN
        assert self.breaker.stats.consecutive_failures > 0

        # Reset the circuit
        self.breaker.reset()

        # Should be closed and cleared
        assert self.breaker.state == CircuitState.CLOSED
        assert self.breaker.stats.consecutive_failures == 0
        assert len(self.breaker._failure_times) == 0

    def test_get_stats(self) -> None:
        """Test getting circuit breaker statistics."""

        def test_func() -> str:
            return "test"

        def failing_func() -> None:
            raise ValueError("Test")

        # Generate some statistics
        self.breaker.call(test_func)
        with pytest.raises(ValueError):
            self.breaker.call(failing_func)

        stats = self.breaker.get_stats()

        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 50.0
        assert "consecutive_failures" in stats
        assert "consecutive_successes" in stats
        assert "recent_failures" in stats

    def test_context_manager_success(self) -> None:
        """Test circuit breaker as context manager with success."""
        with self.breaker:
            # Successful operation
            pass

        assert self.breaker.stats.successful_calls == 1
        assert self.breaker.stats.failed_calls == 0

    def test_context_manager_failure(self) -> None:
        """Test circuit breaker as context manager with failure."""
        with pytest.raises(ValueError), self.breaker:
            raise ValueError("Test failure")

        assert self.breaker.stats.successful_calls == 0
        assert self.breaker.stats.failed_calls == 1

    def test_context_manager_unexpected_exception(self) -> None:
        """Test context manager with unexpected exception."""
        self.breaker.config.expected_exceptions = (ValueError,)

        with pytest.raises(TypeError), self.breaker:
            raise TypeError("Unexpected")

        # Unexpected exception shouldn't count as failure
        assert self.breaker.stats.failed_calls == 0

    def test_metrics_integration(self) -> None:
        """Test integration with metrics collector."""
        metrics_mock = Mock()
        metrics_mock.update_circuit_breaker_state = Mock()
        metrics_mock.track_external_service_call = Mock()
        breaker = CircuitBreaker("metrics_test", self.config, metrics_mock)

        def failing_func() -> None:
            raise ValueError("Test")

        # Trigger failures to open circuit
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        # Should update circuit breaker state to open
        metrics_mock.update_circuit_breaker_state.assert_called_with("metrics_test", "open")

        # Should record failure
        metrics_mock.track_external_service_call.assert_called_with("metrics_test", False, 0.0)

        # Circuit should now be open, try another call
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "test")

        # Should record rejection as failed call
        assert metrics_mock.track_external_service_call.call_count > 1

    def test_state_changes_recorded(self) -> None:
        """Test that state changes are recorded in stats."""

        def failing_func() -> None:
            raise ValueError("Test")

        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with pytest.raises(ValueError):
                self.breaker.call(failing_func)

        # Check state changes
        assert len(self.breaker.stats.state_changes) > 0
        last_change = self.breaker.stats.state_changes[-1]
        assert last_change["to"] == "open"
        assert "reason" in last_change
        assert "timestamp" in last_change

    def _open_circuit(self) -> None:
        """Helper to open the circuit breaker."""
        self._open_circuit_for(self.breaker)

    def _open_circuit_for(self, breaker: CircuitBreaker) -> None:
        """Helper to open a specific circuit breaker."""

        def failing_func() -> None:
            raise ValueError("Test failure")

        for _ in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(failing_func)


class TestCircuitBreakerDecorator:
    """Tests for circuit breaker decorator."""

    def test_decorator_basic(self) -> None:
        """Test basic decorator functionality."""

        @circuit_breaker(name="test_decorated")
        def test_function(value: str) -> str:
            return f"result: {value}"

        result = test_function("test")
        assert result == "result: test"

    def test_decorator_with_failures(self) -> None:
        """Test decorator with function failures."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker(name="failing_func", config=config)
        def failing_function() -> None:
            raise ValueError("Test failure")

        # Trigger failures
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                failing_function()

        # Circuit should be open
        with pytest.raises(CircuitOpenError):
            failing_function()

    def test_decorator_stats_access(self) -> None:
        """Test accessing stats through decorated function."""

        @circuit_breaker(name="stats_func")
        def test_function() -> str:
            return "success"

        # Call function
        test_function()

        # Access stats
        stats = test_function.get_circuit_stats()  # type: ignore[attr-defined]
        assert stats["name"] == "stats_func"
        assert stats["successful_calls"] == 1

    def test_decorator_reset_access(self) -> None:
        """Test resetting circuit through decorated function."""
        config = CircuitBreakerConfig(failure_threshold=1)

        @circuit_breaker(config=config)
        def test_function(fail: bool = False) -> str:
            if fail:
                raise ValueError("Test")
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            test_function(fail=True)

        # Should be open
        with pytest.raises(CircuitOpenError):
            test_function()

        # Reset circuit
        test_function.reset_circuit()  # type: ignore[attr-defined]

        # Should work now
        result = test_function()
        assert result == "success"

    def test_decorator_uses_function_name(self) -> None:
        """Test that decorator uses function name if no name provided."""

        @circuit_breaker()
        def my_special_function() -> str:
            return "test"

        result = my_special_function()
        assert result == "test"

        stats = my_special_function.get_circuit_stats()  # type: ignore[attr-defined]
        assert stats["name"] == "my_special_function"


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_complex_scenario(self) -> None:
        """Test a complex usage scenario."""
        fallback_called = False

        def fallback() -> str:
            nonlocal fallback_called
            fallback_called = True
            return "fallback"

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=0.5,
            fallback=fallback,
        )

        breaker = CircuitBreaker("complex_test", config)

        def unreliable_service(should_fail: bool = False) -> str:
            if should_fail:
                raise ValueError("Service error")
            return "success"

        # Initial successful calls
        for _ in range(5):
            result = breaker.call(unreliable_service, should_fail=False)
            assert result == "success"

        # Trigger failures to open circuit
        for _ in range(config.failure_threshold):
            with pytest.raises(ValueError):
                breaker.call(unreliable_service, should_fail=True)

        # Circuit is open, should use fallback
        result = breaker.call(unreliable_service)
        assert result == "fallback"
        assert fallback_called

        # Wait for half-open
        time.sleep(config.timeout + 0.1)

        # Service recovered, close circuit
        for _ in range(config.success_threshold):
            result = breaker.call(unreliable_service, should_fail=False)
            assert result == "success"

        # Circuit should be closed
        assert breaker.state == CircuitState.CLOSED

        # Verify final stats
        stats = breaker.get_stats()
        assert stats["total_calls"] > 0
        assert stats["fallback_calls"] == 1
        assert stats["state"] == "closed"

    def test_concurrent_access(self) -> None:
        """Test circuit breaker with concurrent access."""

        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
        )
        breaker = CircuitBreaker("concurrent_test", config)

        def worker(fail: bool) -> None:
            def test_func() -> str:
                if fail:
                    raise ValueError("Test")
                return "success"

            with contextlib.suppress(ValueError, CircuitOpenError):
                breaker.call(test_func)

        # Create threads for concurrent access
        threads = []
        for i in range(20):
            fail = i < 10  # First 10 fail, next 10 succeed
            thread = threading.Thread(target=worker, args=(fail,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check stats
        stats = breaker.get_stats()
        assert stats["total_calls"] >= 10  # At least some calls should succeed
        assert stats["failed_calls"] >= 10  # At least the failing calls
