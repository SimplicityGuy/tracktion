"""Tests for retry logic and circuit breaker."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from services.notification_service.src.core.retry import (
    RetryManager,
    RetryPolicy,
)
from shared.utils.resilience import CircuitOpenError, ServiceType, get_circuit_breaker
from shared.utils.resilience.config import CircuitBreakerConfig


class TestRetryPolicy:
    """Test RetryPolicy configuration."""

    def test_default_policy(self) -> None:
        """Test default retry policy configuration."""
        policy = RetryPolicy()

        assert policy.max_attempts == 3
        assert policy.backoff_base == 2.0
        assert policy.backoff_max == 60.0
        assert policy.jitter is True
        assert policy.retry_on is None

    def test_custom_policy(self) -> None:
        """Test custom retry policy configuration."""
        policy = RetryPolicy(
            max_attempts=5,
            backoff_base=3.0,
            backoff_max=120.0,
            jitter=False,
            retry_on=(ValueError, TypeError),
        )

        assert policy.max_attempts == 5
        assert policy.backoff_base == 3.0
        assert policy.backoff_max == 120.0
        assert policy.jitter is False
        assert policy.retry_on == (ValueError, TypeError)

    def test_calculate_delay_without_jitter(self) -> None:
        """Test delay calculation without jitter."""
        policy = RetryPolicy(backoff_base=2.0, jitter=False)

        assert policy.calculate_delay(1) == 1.0  # 2^0
        assert policy.calculate_delay(2) == 2.0  # 2^1
        assert policy.calculate_delay(3) == 4.0  # 2^2
        assert policy.calculate_delay(4) == 8.0  # 2^3

    def test_calculate_delay_with_max(self) -> None:
        """Test delay calculation respects maximum."""
        policy = RetryPolicy(backoff_base=2.0, backoff_max=5.0, jitter=False)

        assert policy.calculate_delay(1) == 1.0
        assert policy.calculate_delay(2) == 2.0
        assert policy.calculate_delay(3) == 4.0
        assert policy.calculate_delay(4) == 5.0  # Capped at max
        assert policy.calculate_delay(5) == 5.0  # Still capped

    def test_calculate_delay_with_jitter(self) -> None:
        """Test delay calculation with jitter."""
        policy = RetryPolicy(backoff_base=2.0, jitter=True)

        # With jitter, delay should be between 50% and 150% of base
        delay = policy.calculate_delay(2)  # Base would be 2.0
        assert 1.0 <= delay <= 3.0


class TestRetryManager:
    """Test RetryManager functionality."""

    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        """Test successful execution on first attempt."""
        manager = RetryManager()
        mock_func = AsyncMock(return_value="success")

        result = await manager.execute(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry on failure."""
        policy = RetryPolicy(max_attempts=3, backoff_base=0.01, jitter=False)
        manager = RetryManager(policy)

        # Fail twice, succeed on third
        mock_func = AsyncMock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])

        result = await manager.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self) -> None:
        """Test when all retry attempts fail."""
        policy = RetryPolicy(max_attempts=3, backoff_base=0.01, jitter=False)
        manager = RetryManager(policy)

        mock_func = AsyncMock(side_effect=ValueError("persistent failure"))

        with pytest.raises(ValueError, match="persistent failure"):
            await manager.execute(mock_func)

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_only_specified_exceptions(self) -> None:
        """Test that only specified exceptions are retried."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_on=(ValueError,),
            backoff_base=0.01,
            jitter=False,
        )
        manager = RetryManager(policy)

        # TypeError should not be retried
        mock_func = AsyncMock(side_effect=TypeError("wrong type"))

        with pytest.raises(TypeError, match="wrong type"):
            await manager.execute(mock_func)

        # Should fail immediately without retries
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_delay(self) -> None:
        """Test that backoff delay is applied between retries."""
        policy = RetryPolicy(max_attempts=3, backoff_base=0.1, jitter=False)
        manager = RetryManager(policy)

        mock_func = AsyncMock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])

        start_time = asyncio.get_event_loop().time()
        result = await manager.execute(mock_func)
        end_time = asyncio.get_event_loop().time()

        assert result == "success"
        # Should have delays of 0.1 and 0.2 seconds
        assert end_time - start_time >= 0.3


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    @pytest.mark.asyncio
    async def test_successful_calls(self) -> None:
        """Test circuit breaker with successful calls."""
        breaker = get_circuit_breaker(name="test_successful_calls", service_type=ServiceType.EXTERNAL_SERVICE)
        mock_func = AsyncMock(return_value="success")

        # Multiple successful calls
        for _ in range(5):
            result = await breaker.call_async(mock_func)
            assert result == "success"

        # Shared circuit breaker uses CircuitState enum
        assert breaker.state.name == "CLOSED"
        assert breaker.stats.failed_calls == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self) -> None:
        """Test circuit opens after threshold failures."""
        breaker = get_circuit_breaker(name="test_circuit_opens", service_type=ServiceType.EXTERNAL_SERVICE)
        mock_func = AsyncMock(side_effect=ValueError("fail"))

        # First few failures (EXTERNAL_SERVICE default threshold is 10)
        for _ in range(9):
            with pytest.raises(ValueError):
                await breaker.call_async(mock_func)

        assert breaker.state.name == "CLOSED"
        assert breaker.stats.failed_calls == 9

        # Tenth failure opens circuit
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        assert breaker.state.name == "OPEN"
        assert breaker.stats.failed_calls == 10

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self) -> None:
        """Test that open circuit rejects calls."""
        # Create a circuit breaker with low threshold for quick opening
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = get_circuit_breaker(
            name="test_open_rejects", service_type=ServiceType.EXTERNAL_SERVICE, config=config
        )
        mock_func = AsyncMock(side_effect=[ValueError("fail"), "success"])

        # First call fails and opens circuit
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        assert breaker.state.name == "OPEN"

        # Subsequent call should be rejected immediately
        with pytest.raises(CircuitOpenError):
            await breaker.call_async(mock_func)

        # Function should only be called once
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_recovery(self) -> None:
        """Test circuit recovery after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = get_circuit_breaker(
            name="test_circuit_recovery", service_type=ServiceType.EXTERNAL_SERVICE, config=config
        )
        mock_func = AsyncMock(side_effect=[ValueError("fail")] + ["success"] * 10)

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        assert breaker.state.name == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should go to half-open and allow call
        result = await breaker.call_async(mock_func)
        assert result == "success"
        assert breaker.state.name == "HALF_OPEN"

        # Need success_threshold (5 for EXTERNAL_SERVICE) successful calls to close
        for _ in range(4):  # Already had 1 successful call
            result = await breaker.call_async(mock_func)
            assert result == "success"

        assert breaker.state.name == "CLOSED"
        assert breaker.stats.consecutive_failures == 0  # This should be 0, not failed_calls

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self) -> None:
        """Test that failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = get_circuit_breaker(
            name="test_half_open_failure", service_type=ServiceType.EXTERNAL_SERVICE, config=config
        )
        mock_func = AsyncMock(side_effect=ValueError("persistent fail"))

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        assert breaker.state.name == "OPEN"
        assert breaker.stats.failed_calls == 2

    @pytest.mark.asyncio
    async def test_expected_exception_filtering(self) -> None:
        """Test that all exceptions trigger circuit in shared implementation."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = get_circuit_breaker(
            name="test_exception_filtering", service_type=ServiceType.EXTERNAL_SERVICE, config=config
        )
        mock_func = AsyncMock()

        # In shared implementation, all exceptions trigger circuit breaker
        mock_func.side_effect = TypeError("wrong type")
        with pytest.raises(TypeError):
            await breaker.call_async(mock_func)

        assert breaker.state.name == "CLOSED"
        assert breaker.stats.failed_calls == 1

        # ValueError should also affect circuit
        mock_func.side_effect = ValueError("value error")
        with pytest.raises(ValueError):
            await breaker.call_async(mock_func)

        assert breaker.stats.failed_calls == 2

    def test_reset(self) -> None:
        """Test circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = get_circuit_breaker(name="test_reset", service_type=ServiceType.EXTERNAL_SERVICE, config=config)

        # Manually set failure count for testing
        breaker.stats.failed_calls = 5

        breaker.reset()

        assert breaker.state.name == "CLOSED"
        assert breaker.stats.consecutive_failures == 0  # Reset resets consecutive, not total failed_calls
