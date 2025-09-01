"""Tests for resilience patterns."""

import asyncio
import time

import pytest

from services.tracklist_service.src.exceptions import ServiceUnavailableError
from services.tracklist_service.src.resilience import (
    CircuitBreaker,
    CircuitState,
    ExponentialBackoff,
    HealthCheck,
    RateLimiter,
    retry_with_backoff,
)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=10)

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0

    def test_successful_call(self):
        """Test successful function call."""
        cb = CircuitBreaker("test", failure_threshold=3)

        def test_func(x):
            return x * 2

        result = cb.call(test_func, 5)

        assert result == 10
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_failed_call(self):
        """Test failed function call."""
        cb = CircuitBreaker("test", failure_threshold=3, expected_exception=ValueError)

        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            cb.call(test_func)

        assert cb._failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test", failure_threshold=3, expected_exception=ValueError)

        def test_func():
            raise ValueError("Test error")

        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(test_func)

        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 3

    def test_open_circuit_rejects_calls(self):
        """Test open circuit rejects calls."""
        cb = CircuitBreaker("test", failure_threshold=1, expected_exception=ValueError)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected
        def good_func():
            return "success"

        with pytest.raises(ServiceUnavailableError) as exc_info:
            cb.call(good_func)

        assert "Circuit breaker 'test' is OPEN" in str(exc_info.value)

    def test_circuit_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms timeout
            expected_exception=ValueError,
        )

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # State should transition to half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Test half-open circuit closes on success."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        def failing_func():
            raise ValueError("Test error")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Wait for recovery timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Successful calls in half-open state
        for _ in range(3):  # Requires 3 successes
            result = cb.call(success_func)
            assert result == "success"

        # Circuit should be closed
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Test half-open circuit reopens on failure."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Wait for recovery timeout
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open state
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Circuit should be open again
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async function call with circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=3)

        async def async_func(x):
            return x * 2

        result = await cb.async_call(async_func, 5)

        assert result == 10
        assert cb.state == CircuitState.CLOSED

    def test_reset(self):
        """Test resetting circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=1, expected_exception=ValueError)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0


class TestExponentialBackoff:
    """Test ExponentialBackoff functionality."""

    def test_get_delay_without_jitter(self):
        """Test delay calculation without jitter."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, multiplier=2.0, jitter=False)

        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0
        assert backoff.get_delay(4) == 10.0  # Capped at max_delay
        assert backoff.get_delay(5) == 10.0  # Still capped

    def test_get_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0, multiplier=2.0, jitter=True)

        # With jitter, delay should be within ±25% of base value
        delay = backoff.get_delay(1)  # Base would be 2.0
        assert 1.5 <= delay <= 2.5

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test successful retry with backoff."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await retry_with_backoff(
            func,
            max_attempts=5,
            backoff=ExponentialBackoff(base_delay=0.01),
            exceptions=(ValueError,),
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_all_fail(self):
        """Test retry with backoff when all attempts fail."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Error {call_count}")

        with pytest.raises(ValueError) as exc_info:
            await retry_with_backoff(
                func,
                max_attempts=3,
                backoff=ExponentialBackoff(base_delay=0.01),
                exceptions=(ValueError,),
            )

        assert "Error 3" in str(exc_info.value)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_unexpected_exception(self):
        """Test retry doesn't catch unexpected exceptions."""

        async def func():
            raise TypeError("Unexpected error")

        with pytest.raises(TypeError):
            await retry_with_backoff(
                func,
                max_attempts=3,
                backoff=ExponentialBackoff(base_delay=0.01),
                exceptions=(ValueError,),  # Only retrying ValueError
            )


class TestRateLimiter:
    """Test RateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_acquire_tokens_available(self):
        """Test acquiring tokens when available."""
        limiter = RateLimiter(rate=10.0, capacity=10, name="test")

        wait_time = await limiter.acquire(5)

        assert wait_time == 0.0
        assert limiter._tokens == 5.0

    @pytest.mark.asyncio
    async def test_acquire_tokens_wait_required(self):
        """Test acquiring tokens requires waiting."""
        limiter = RateLimiter(rate=10.0, capacity=10, name="test")

        # Deplete all tokens
        await limiter.acquire(10)

        # Try to acquire more tokens
        start_time = time.time()
        wait_time = await limiter.acquire(1)
        elapsed = time.time() - start_time

        assert wait_time > 0
        assert elapsed >= 0.09  # Should wait at least 0.1 seconds (with some tolerance)

    @pytest.mark.asyncio
    async def test_acquire_too_many_tokens(self):
        """Test acquiring more tokens than capacity."""
        limiter = RateLimiter(rate=10.0, capacity=10, name="test")

        with pytest.raises(ValueError) as exc_info:
            await limiter.acquire(15)

        assert "Cannot acquire 15 tokens" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refilling over time."""
        limiter = RateLimiter(rate=100.0, capacity=10, name="test")

        # Deplete some tokens
        await limiter.acquire(5)
        assert limiter._tokens == 5.0

        # Wait for refill
        await asyncio.sleep(0.05)  # 50ms should refill 5 tokens at 100/s

        # Acquire should not wait
        wait_time = await limiter.acquire(5)
        assert wait_time == 0.0

    def test_reset(self):
        """Test resetting rate limiter."""
        limiter = RateLimiter(rate=10.0, capacity=10, name="test")

        # Deplete tokens
        asyncio.run(limiter.acquire(8))
        assert limiter._tokens == 2.0

        # Reset
        limiter.reset()

        assert limiter._tokens == 10.0


class TestHealthCheck:
    """Test HealthCheck functionality."""

    def test_healthy_check(self):
        """Test healthy service check."""

        def check_func():
            return True

        health_check = HealthCheck("test_service", check_func, interval=0.1)

        assert health_check.is_healthy() is True
        assert health_check._consecutive_failures == 0

    def test_unhealthy_check(self):
        """Test unhealthy service check."""

        def check_func():
            return False

        health_check = HealthCheck("test_service", check_func, interval=0.1)

        assert health_check.is_healthy() is False
        assert health_check._consecutive_failures == 1

    def test_check_exception(self):
        """Test health check with exception."""

        def check_func():
            raise Exception("Check failed")

        health_check = HealthCheck("test_service", check_func, interval=0.1)

        assert health_check.is_healthy() is False
        assert health_check._consecutive_failures == 1

    def test_check_interval(self):
        """Test health check respects interval."""
        call_count = 0

        def check_func():
            nonlocal call_count
            call_count += 1
            return True

        health_check = HealthCheck("test_service", check_func, interval=1.0)

        # First check
        assert health_check.is_healthy() is True
        assert call_count == 1

        # Immediate second check should use cached result
        assert health_check.is_healthy() is True
        assert call_count == 1  # Not called again

        # Wait for interval
        time.sleep(1.1)

        # Should check again
        assert health_check.is_healthy() is True
        assert call_count == 2

    def test_recovery_detection(self):
        """Test service recovery detection."""
        healthy = False

        def check_func():
            return healthy

        health_check = HealthCheck("test_service", check_func, interval=0.1)

        # Initially unhealthy
        assert health_check.is_healthy() is False
        assert health_check._consecutive_failures == 1

        # Wait for interval
        time.sleep(0.15)

        # Now healthy
        healthy = True
        assert health_check.is_healthy() is True
        assert health_check._consecutive_failures == 0
