"""Unit tests for retry management and error recovery."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from services.tracklist_service.src.queue.batch_queue import Job, JobPriority
from services.tracklist_service.src.retry.retry_manager import (
    CircuitBreaker,
    FailedJob,
    FailureType,
    RetryManager,
    RetryPolicy,
    RetryStrategy,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = Mock()
    redis_mock.hset = Mock()
    redis_mock.expire = Mock()
    redis_mock.hget = Mock(return_value=None)
    return redis_mock


@pytest.fixture
def mock_rabbitmq():
    """Create mock RabbitMQ channel."""
    channel_mock = Mock()
    channel_mock.queue_declare = Mock()
    channel_mock.basic_publish = Mock()

    connection_mock = Mock()
    connection_mock.channel.return_value = channel_mock

    return connection_mock, channel_mock


@pytest.fixture
def retry_manager(mock_redis, mock_rabbitmq):
    """Create RetryManager instance with mocks."""
    connection_mock, channel_mock = mock_rabbitmq

    with (
        patch(
            "services.tracklist_service.src.retry.retry_manager.Redis",
            return_value=mock_redis,
        ),
        patch(
            "services.tracklist_service.src.retry.retry_manager.pika.BlockingConnection",
            return_value=connection_mock,
        ),
    ):
        return RetryManager()


@pytest.fixture
def sample_job():
    """Create sample job."""
    return Job(
        id="job-123",
        batch_id="batch-456",
        url="http://example.com/test",
        priority=JobPriority.NORMAL,
        user_id="user123",
        created_at=datetime.now(UTC),
    )


class TestRetryStrategy:
    """Test RetryStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.FIBONACCI.value == "fibonacci"
        assert RetryStrategy.FIXED.value == "fixed"
        assert RetryStrategy.ADAPTIVE.value == "adaptive"


class TestFailureType:
    """Test FailureType enum."""

    def test_failure_type_values(self):
        """Test failure type enum values."""
        assert FailureType.NETWORK.value == "network"
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.RATE_LIMIT.value == "rate_limit"
        assert FailureType.AUTH.value == "auth"
        assert FailureType.SERVER.value == "server"
        assert FailureType.CLIENT.value == "client"
        assert FailureType.PARSE.value == "parse"
        assert FailureType.UNKNOWN.value == "unknown"


class TestRetryPolicy:
    """Test RetryPolicy class."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=600.0,
            strategy=RetryStrategy.EXPONENTIAL,
        )

        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 600.0
        assert policy.strategy == RetryStrategy.EXPONENTIAL
        assert policy.jitter is True

    def test_exponential_delay(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, strategy=RetryStrategy.EXPONENTIAL, jitter=False)

        assert policy.get_delay(0) == 1.0  # 1 * 2^0
        assert policy.get_delay(1) == 2.0  # 1 * 2^1
        assert policy.get_delay(2) == 4.0  # 1 * 2^2
        assert policy.get_delay(3) == 8.0  # 1 * 2^3

    def test_linear_delay(self):
        """Test linear backoff calculation."""
        policy = RetryPolicy(base_delay=2.0, strategy=RetryStrategy.LINEAR, jitter=False)

        assert policy.get_delay(1) == 2.0  # 2 * 1
        assert policy.get_delay(2) == 4.0  # 2 * 2
        assert policy.get_delay(3) == 6.0  # 2 * 3

    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        policy = RetryPolicy(base_delay=5.0, strategy=RetryStrategy.FIXED, jitter=False)

        assert policy.get_delay(1) == 5.0
        assert policy.get_delay(2) == 5.0
        assert policy.get_delay(10) == 5.0

    def test_fibonacci_delay(self):
        """Test Fibonacci backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, strategy=RetryStrategy.FIBONACCI, jitter=False)

        assert policy.get_delay(0) == 1.0  # Base
        assert policy.get_delay(1) == 1.0  # 1 * 1
        assert policy.get_delay(2) == 2.0  # 1 * 2
        assert policy.get_delay(3) == 3.0  # 1 * 3
        assert policy.get_delay(4) == 5.0  # 1 * 5

    def test_max_delay_cap(self):
        """Test maximum delay cap."""
        policy = RetryPolicy(
            base_delay=100.0,
            max_delay=200.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
        )

        # Should be capped at max_delay
        assert policy.get_delay(10) == 200.0

    def test_jitter(self):
        """Test jitter application."""
        policy = RetryPolicy(base_delay=10.0, strategy=RetryStrategy.FIXED, jitter=True)

        delay = policy.get_delay(1)
        # With jitter (0.5 + random()), delay should be between 5 and 15
        assert 5.0 <= delay <= 15.0

    def test_failure_specific_policy(self):
        """Test failure-specific policy overrides."""
        policy = RetryPolicy(
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False,
            failure_policies={
                FailureType.RATE_LIMIT: {
                    "base_delay": 10.0,
                    "strategy": RetryStrategy.FIXED,
                }
            },
        )

        # Normal failure uses default
        assert policy.get_delay(1, FailureType.NETWORK) == 2.0

        # Rate limit uses override
        assert policy.get_delay(1, FailureType.RATE_LIMIT) == 10.0
        assert policy.get_delay(2, FailureType.RATE_LIMIT) == 10.0  # Fixed


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30, half_open_max_calls=2)

        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.half_open_max_calls == 2
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_success_in_closed_state(self):
        """Test successful calls in closed state."""
        cb = CircuitBreaker(failure_threshold=3)

        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_failure_opens_circuit(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)

        def fail_func():
            raise Exception("Test failure")

        # First two failures don't open circuit
        for _ in range(2):
            with pytest.raises(Exception, match="Test failure"):
                cb.call(fail_func)
            assert cb.state == "closed"

        # Third failure opens circuit
        with pytest.raises(Exception, match="Test failure"):
            cb.call(fail_func)
        assert cb.state == "open"

    def test_open_circuit_blocks_calls(self):
        """Test open circuit blocks calls."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.state = "open"
        cb.state_changed_at = datetime.now(UTC)

        def test_func():
            return "test"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            cb.call(test_func)

    def test_half_open_transition(self):
        """Test transition to half-open after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        # Open the circuit
        cb.state = "open"
        cb.state_changed_at = datetime.now(UTC) - timedelta(seconds=2)

        def test_func():
            return "test"

        # Should transition to half-open and allow call
        result = cb.call(test_func)
        assert result == "test"
        assert cb.state == "half_open"

    def test_half_open_to_closed(self):
        """Test successful calls in half-open close circuit."""
        cb = CircuitBreaker(half_open_max_calls=2)
        cb.state = "half_open"

        def success_func():
            return "success"

        # First successful call
        cb.call(success_func)
        assert cb.state == "half_open"

        # Second successful call closes circuit
        cb.call(success_func)
        assert cb.state == "closed"

    def test_half_open_to_open(self):
        """Test failure in half-open reopens circuit."""
        cb = CircuitBreaker()
        cb.state = "half_open"

        def fail_func():
            raise Exception("Test failure")

        with pytest.raises(Exception, match="Test failure"):
            cb.call(fail_func)

        assert cb.state == "open"

    def test_get_state(self):
        """Test getting circuit breaker state."""
        cb = CircuitBreaker()
        cb.failure_count = 2
        cb.last_failure = datetime.now(UTC)

        state = cb.get_state()

        assert state["state"] == "closed"
        assert state["failure_count"] == 2
        assert state["last_failure"] is not None


class TestRetryManager:
    """Test RetryManager class."""

    def test_initialization(self, mock_redis, mock_rabbitmq):
        """Test retry manager initialization."""
        connection_mock, channel_mock = mock_rabbitmq

        with (
            patch(
                "services.tracklist_service.src.retry.retry_manager.Redis",
                return_value=mock_redis,
            ),
            patch(
                "services.tracklist_service.src.retry.retry_manager.pika.BlockingConnection",
                return_value=connection_mock,
            ),
        ):
            manager = RetryManager()

        assert manager.redis == mock_redis
        assert manager.default_policy is not None
        assert manager.domain_policies == {}
        assert len(manager.circuit_breakers) == 0

        # Check queue declaration
        channel_mock.queue_declare.assert_called()

    def test_classify_failure_network(self, retry_manager):
        """Test network failure classification."""
        errors = [
            "Connection refused",
            "Network unreachable",
            "DNS resolution failed",
            "SSL certificate error",
        ]

        for error in errors:
            assert retry_manager.classify_failure(error) == FailureType.NETWORK

    def test_classify_failure_timeout(self, retry_manager):
        """Test timeout failure classification."""
        errors = ["Request timeout", "Operation timed out", "Timeout exceeded"]

        for error in errors:
            assert retry_manager.classify_failure(error) == FailureType.TIMEOUT

    def test_classify_failure_rate_limit(self, retry_manager):
        """Test rate limit failure classification."""
        errors = [
            "Rate limit exceeded",
            "Too many requests",
            "Error 429: Too Many Requests",
        ]

        for error in errors:
            assert retry_manager.classify_failure(error) == FailureType.RATE_LIMIT

    def test_classify_failure_auth(self, retry_manager):
        """Test auth failure classification."""
        errors = ["Unauthorized access", "403 Forbidden", "401 Unauthorized"]

        for error in errors:
            assert retry_manager.classify_failure(error) == FailureType.AUTH

    def test_classify_failure_server(self, retry_manager):
        """Test server failure classification."""
        errors = [
            "500 Internal Server Error",
            "502 Bad Gateway",
            "503 Service Unavailable",
        ]

        for error in errors:
            assert retry_manager.classify_failure(error) == FailureType.SERVER

    def test_classify_failure_unknown(self, retry_manager):
        """Test unknown failure classification."""
        error = "Some random error message"
        assert retry_manager.classify_failure(error) == FailureType.UNKNOWN

    @pytest.mark.asyncio
    async def test_handle_failure_first_attempt(self, retry_manager, sample_job):
        """Test handling first failure."""
        error = "Network connection failed"

        result = await retry_manager.handle_failure(sample_job, error)

        assert result is True  # Should retry
        assert sample_job.id in retry_manager.failed_jobs

        failed_job = retry_manager.failed_jobs[sample_job.id]
        assert failed_job.error == error
        assert failed_job.failure_type == FailureType.NETWORK
        assert failed_job.retry_count == 0

    @pytest.mark.asyncio
    async def test_handle_failure_max_retries(self, retry_manager, sample_job):
        """Test handling failure after max retries."""
        retry_manager.default_policy.max_retries = 2

        # Simulate previous failures
        retry_manager.failed_jobs[sample_job.id] = FailedJob(
            job=sample_job,
            error="Previous error",
            failure_type=FailureType.NETWORK,
            failed_at=datetime.now(UTC),
            retry_count=2,  # Already at max
        )

        result = await retry_manager.handle_failure(sample_job, "Network error")

        assert result is False  # Should not retry
        assert sample_job.id not in retry_manager.failed_jobs  # Moved to DLQ

    @pytest.mark.asyncio
    async def test_handle_failure_circuit_open(self, retry_manager, sample_job):
        """Test handling failure with open circuit."""
        domain = "example.com"
        retry_manager.circuit_breakers[domain].state = "open"
        retry_manager.circuit_breakers[domain].state_changed_at = datetime.now(UTC)

        result = await retry_manager.handle_failure(sample_job, "Network error")

        assert result is False  # Should not retry due to open circuit

    @pytest.mark.asyncio
    async def test_schedule_retry(self, retry_manager, sample_job, mock_rabbitmq):
        """Test scheduling job retry."""
        _, channel_mock = mock_rabbitmq
        delay = 10.0

        await retry_manager.schedule_retry(sample_job, delay)

        # Check RabbitMQ publish
        channel_mock.basic_publish.assert_called_once()

        # Check Redis storage
        retry_manager.redis.hset.assert_called()
        retry_manager.redis.expire.assert_called()

    @pytest.mark.asyncio
    async def test_process_retry_queue(self, retry_manager, sample_job):
        """Test processing retry queue."""
        # Add failed job ready for retry
        retry_manager.failed_jobs[sample_job.id] = FailedJob(
            job=sample_job,
            error="Test error",
            failure_type=FailureType.NETWORK,
            failed_at=datetime.now(UTC) - timedelta(seconds=60),
            next_retry=datetime.now(UTC) - timedelta(seconds=1),  # Past due
        )

        ready_jobs = await retry_manager.process_retry_queue()

        assert len(ready_jobs) == 1
        assert ready_jobs[0].id == sample_job.id
        assert sample_job.id not in retry_manager.failed_jobs  # Removed

    @pytest.mark.asyncio
    async def test_process_retry_queue_not_ready(self, retry_manager, sample_job):
        """Test processing retry queue with jobs not ready."""
        # Add failed job not ready for retry
        retry_manager.failed_jobs[sample_job.id] = FailedJob(
            job=sample_job,
            error="Test error",
            failure_type=FailureType.NETWORK,
            failed_at=datetime.now(UTC),
            next_retry=datetime.now(UTC) + timedelta(seconds=60),  # Future
        )

        ready_jobs = await retry_manager.process_retry_queue()

        assert len(ready_jobs) == 0
        assert sample_job.id in retry_manager.failed_jobs  # Still there

    def test_set_domain_policy(self, retry_manager):
        """Test setting domain-specific policy."""
        policy = RetryPolicy(max_retries=10, base_delay=5.0, strategy=RetryStrategy.LINEAR)

        retry_manager.set_domain_policy("example.com", policy)

        assert "example.com" in retry_manager.domain_policies
        assert retry_manager.domain_policies["example.com"] == policy

    def test_get_failure_stats(self, retry_manager):
        """Test getting failure statistics."""
        # Add some failure stats
        retry_manager.failure_stats["example.com"]["network"] = 5
        retry_manager.failure_stats["example.com"]["timeout"] = 3
        retry_manager.failure_stats["other.com"]["server"] = 2

        # Get stats for specific domain
        stats = retry_manager.get_failure_stats("example.com")
        assert stats["domain"] == "example.com"
        assert stats["failures"]["network"] == 5
        assert stats["failures"]["timeout"] == 3

        # Get all stats
        all_stats = retry_manager.get_failure_stats()
        assert "example.com" in all_stats
        assert "other.com" in all_stats

    def test_reset_circuit_breaker(self, retry_manager):
        """Test manually resetting circuit breaker."""
        domain = "example.com"
        retry_manager.circuit_breakers[domain].state = "open"

        retry_manager.reset_circuit_breaker(domain)

        assert retry_manager.circuit_breakers[domain].state == "closed"

    @pytest.mark.asyncio
    async def test_move_to_dlq(self, retry_manager, sample_job, mock_rabbitmq):
        """Test moving job to dead letter queue."""
        _, channel_mock = mock_rabbitmq
        error = "Final error"

        # Add failed job info
        retry_manager.failed_jobs[sample_job.id] = FailedJob(
            job=sample_job,
            error=error,
            failure_type=FailureType.UNKNOWN,
            failed_at=datetime.now(UTC),
            retry_count=3,
        )

        await retry_manager._move_to_dlq(sample_job, error)

        # Check DLQ publish
        assert channel_mock.basic_publish.call_count >= 1

        # Check Redis storage
        retry_manager.redis.hset.assert_called()

        # Check cleanup
        assert sample_job.id not in retry_manager.failed_jobs
