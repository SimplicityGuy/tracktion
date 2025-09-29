"""Intelligent retry management for failed jobs."""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping
from urllib.parse import urlparse

import pika
from redis import Redis
from services.tracklist_service.src.queue.batch_queue import Job

# Import from shared resilience module
from shared.utils.resilience import (
    CircuitBreakerConfig,
    ServiceType,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    FIBONACCI = "fibonacci"  # Fibonacci sequence backoff
    FIXED = "fixed"  # Fixed delay between retries
    ADAPTIVE = "adaptive"  # Adaptive based on failure patterns


class FailureType(Enum):
    """Types of failures for categorization."""

    NETWORK = "network"  # Network-related errors
    TIMEOUT = "timeout"  # Timeouts
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    AUTH = "auth"  # Authentication/authorization errors
    SERVER = "server"  # Server errors (5xx)
    CLIENT = "client"  # Client errors (4xx)
    PARSE = "parse"  # Parsing errors
    UNKNOWN = "unknown"  # Unknown errors


@dataclass
class RetryPolicy:
    """Policy for job retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True  # Add random jitter to delays

    # Failure-specific overrides
    failure_policies: dict[FailureType, dict[str, Any]] = field(default_factory=dict)

    def get_delay(self, attempt: int, failure_type: FailureType = FailureType.UNKNOWN) -> float:
        """Calculate delay for retry attempt.

        Args:
            attempt: Current retry attempt number
            failure_type: Type of failure

        Returns:
            Delay in seconds
        """
        # Check for failure-specific policy
        if failure_type in self.failure_policies:
            policy = self.failure_policies[failure_type]
            base = policy.get("base_delay", self.base_delay)
            max_delay = policy.get("max_delay", self.max_delay)
            strategy = policy.get("strategy", self.strategy)
        else:
            base = self.base_delay
            max_delay = self.max_delay
            strategy = self.strategy

        # Calculate delay based on strategy
        if strategy == RetryStrategy.EXPONENTIAL:
            delay = base * (2**attempt)
        elif strategy == RetryStrategy.LINEAR:
            delay = base * attempt
        elif strategy == RetryStrategy.FIBONACCI:
            delay = self._fibonacci_delay(attempt, base)
        elif strategy == RetryStrategy.FIXED:
            delay = base
        else:  # ADAPTIVE
            delay = self._adaptive_delay(attempt, base, failure_type)

        # Apply maximum delay cap
        delay = min(delay, max_delay)

        # Add jitter if enabled
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return float(delay)

    def _fibonacci_delay(self, attempt: int, base: float) -> float:
        """Calculate Fibonacci-based delay."""
        if attempt <= 1:
            return base

        fib = [1, 1]
        for _ in range(2, attempt + 1):
            fib.append(fib[-1] + fib[-2])

        return base * fib[attempt]

    def _adaptive_delay(self, attempt: int, base: float, failure_type: FailureType) -> float:
        """Calculate adaptive delay based on failure patterns."""
        # Adapt based on failure type
        multipliers = {
            FailureType.NETWORK: 2.0,
            FailureType.TIMEOUT: 3.0,
            FailureType.RATE_LIMIT: 5.0,
            FailureType.AUTH: 10.0,
            FailureType.SERVER: 4.0,
            FailureType.CLIENT: 1.5,
            FailureType.PARSE: 1.0,
            FailureType.UNKNOWN: 2.0,
        }

        multiplier = multipliers.get(failure_type, 2.0)
        return base * (multiplier**attempt)


@dataclass
class FailedJob:
    """Information about a failed job."""

    job: Job
    error: str
    failure_type: FailureType
    failed_at: datetime
    retry_count: int = 0
    last_retry: datetime | None = None
    next_retry: datetime | None = None


class RetryManager:
    """Manages job retry logic and error recovery."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        rabbitmq_host: str = "localhost",
        default_policy: RetryPolicy | None = None,
    ):
        """Initialize retry manager.

        Args:
            redis_host: Redis host
            redis_port: Redis port
            rabbitmq_host: RabbitMQ host
            default_policy: Default retry policy
        """
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.rabbitmq_connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmq_host))
        self.rabbitmq_channel = self.rabbitmq_connection.channel()

        # Set up retry queue
        self.rabbitmq_channel.queue_declare(
            queue="retry_queue",
            durable=True,
            arguments={
                "x-message-ttl": 3600000,  # 1 hour TTL
                "x-max-length": 10000,  # Max 10k messages
            },
        )

        self.default_policy = default_policy or RetryPolicy()

        # Domain-specific policies
        self.domain_policies: dict[str, RetryPolicy] = {}

        # Circuit breakers per domain
        self.circuit_breakers: dict[str, Any] = {}

        # Failed job tracking
        self.failed_jobs: dict[str, FailedJob] = {}

        # Failure statistics
        self.failure_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _get_circuit_breaker(self, domain: str):
        """Get or create circuit breaker for domain."""
        if domain not in self.circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60.0,
                success_threshold=3,
                expected_exceptions=(Exception,),
            )
            self.circuit_breakers[domain] = get_circuit_breaker(
                name=f"retry_{domain}",
                config=config,
                service_type=ServiceType.EXTERNAL_SERVICE,
                domain=domain,
            )
        return self.circuit_breakers[domain]

    def classify_failure(self, error: str) -> FailureType:
        """Classify failure type from error message.

        Args:
            error: Error message

        Returns:
            Failure type
        """
        error_lower = error.lower()

        if any(keyword in error_lower for keyword in ["network", "connection", "dns", "ssl"]):
            return FailureType.NETWORK
        if any(keyword in error_lower for keyword in ["timeout", "timed out"]):
            return FailureType.TIMEOUT
        if any(keyword in error_lower for keyword in ["rate limit", "too many requests", "429"]):
            return FailureType.RATE_LIMIT
        if any(keyword in error_lower for keyword in ["unauthorized", "forbidden", "401", "403"]):
            return FailureType.AUTH
        if any(keyword in error_lower for keyword in ["server error", "internal error", "500", "502", "503"]):
            return FailureType.SERVER
        if any(keyword in error_lower for keyword in ["bad request", "not found", "400", "404"]):
            return FailureType.CLIENT
        if any(keyword in error_lower for keyword in ["parse", "json", "xml", "decode"]):
            return FailureType.PARSE
        return FailureType.UNKNOWN

    async def handle_failure(self, job: Job, error: str) -> bool:
        """Handle job failure and determine if retry is needed.

        Args:
            job: Failed job
            error: Error message

        Returns:
            True if job should be retried
        """
        failure_type = self.classify_failure(error)
        domain = self._extract_domain(job.url)

        # Update failure statistics
        self.failure_stats[domain][failure_type.value] += 1

        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(domain)
        if circuit_breaker.state.value == "open":
            logger.warning(f"Circuit breaker open for domain {domain}, not retrying")
            return False

        # Get or create failed job record
        if job.id not in self.failed_jobs:
            self.failed_jobs[job.id] = FailedJob(
                job=job,
                error=error,
                failure_type=failure_type,
                failed_at=datetime.now(UTC),
            )
        else:
            failed_job = self.failed_jobs[job.id]
            failed_job.retry_count += 1
            failed_job.error = error
            failed_job.failure_type = failure_type
            failed_job.last_retry = datetime.now(UTC)

        failed_job = self.failed_jobs[job.id]

        # Get retry policy
        policy = self.domain_policies.get(domain, self.default_policy)

        # Check if max retries reached
        if failed_job.retry_count >= policy.max_retries:
            logger.error(f"Job {job.id} exceeded max retries ({policy.max_retries})")
            await self._move_to_dlq(job, error)
            return False

        # Calculate retry delay
        delay = policy.get_delay(failed_job.retry_count, failure_type)
        failed_job.next_retry = datetime.now(UTC) + timedelta(seconds=delay)

        # Schedule retry
        await self.schedule_retry(job, delay)

        logger.info(f"Scheduled retry for job {job.id} in {delay:.1f} seconds (attempt {failed_job.retry_count + 1})")
        return True

    async def schedule_retry(self, job: Job, delay: float) -> None:
        """Schedule job for retry.

        Args:
            job: Job to retry
            delay: Delay in seconds
        """
        # Add to retry queue with delay
        retry_message = {
            "job_id": job.id,
            "batch_id": job.batch_id,
            "url": job.url,
            "priority": (job.priority.value if hasattr(job.priority, "value") else job.priority),
            "user_id": job.user_id,
            "retry_at": (datetime.now(UTC) + timedelta(seconds=delay)).isoformat(),
            "retry_count": (self.failed_jobs[job.id].retry_count if job.id in self.failed_jobs else 0),
        }

        # Publish to retry queue
        self.rabbitmq_channel.basic_publish(
            exchange="",
            routing_key="retry_queue",
            body=str(retry_message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                expiration=str(int(delay * 1000)),  # TTL in milliseconds
            ),
        )

        # Store in Redis for tracking
        retry_data = {
            "scheduled_at": datetime.now(UTC).isoformat(),
            "retry_at": retry_message["retry_at"],
            "retry_count": retry_message["retry_count"],
        }
        # Convert to proper types for Redis
        redis_retry_data: dict[str, str] = {k: str(v) for k, v in retry_data.items()}
        self.redis.hset(
            f"retry:{job.id}",
            mapping=cast("Mapping[str | bytes, bytes | float | int | str]", redis_retry_data),
        )
        self.redis.expire(f"retry:{job.id}", int(delay) + 3600)  # Expire after delay + 1 hour

    async def process_retry_queue(self) -> list[Job]:
        """Process jobs ready for retry.

        Returns:
            List of jobs ready for retry
        """
        ready_jobs = []
        now = datetime.now(UTC)

        # Check failed jobs for retry
        for job_id, failed_job in list(self.failed_jobs.items()):
            if failed_job.next_retry and failed_job.next_retry <= now:
                # Check circuit breaker
                domain = self._extract_domain(failed_job.job.url)
                if self.circuit_breakers[domain].state.value != "open":
                    ready_jobs.append(failed_job.job)
                    # Remove from failed jobs (will be re-added if fails again)
                    del self.failed_jobs[job_id]

        return ready_jobs

    async def _move_to_dlq(self, job: Job, error: str) -> None:
        """Move job to dead letter queue.

        Args:
            job: Failed job
            error: Final error message
        """
        # Declare DLQ
        self.rabbitmq_channel.queue_declare(queue="dead_letter_queue", durable=True)

        # Create DLQ message
        dlq_message = {
            "job_id": job.id,
            "batch_id": job.batch_id,
            "url": job.url,
            "priority": (job.priority.value if hasattr(job.priority, "value") else job.priority),
            "user_id": job.user_id,
            "failed_at": datetime.now(UTC).isoformat(),
            "final_error": error,
            "retry_count": (self.failed_jobs[job.id].retry_count if job.id in self.failed_jobs else 0),
        }

        # Publish to DLQ
        self.rabbitmq_channel.basic_publish(
            exchange="",
            routing_key="dead_letter_queue",
            body=str(dlq_message),
            properties=pika.BasicProperties(delivery_mode=2),  # Persistent
        )

        # Store in Redis for analysis
        # Convert to proper types for Redis
        redis_dlq_data: dict[str, str] = {k: str(v) for k, v in dlq_message.items()}
        self.redis.hset(
            f"dlq:{job.id}", mapping=cast("Mapping[str | bytes, bytes | float | int | str]", redis_dlq_data)
        )

        # Get retry count before cleanup
        retry_count = self.failed_jobs[job.id].retry_count if job.id in self.failed_jobs else 0

        # Clean up failed job tracking
        if job.id in self.failed_jobs:
            del self.failed_jobs[job.id]

        logger.error(f"Job {job.id} moved to dead letter queue after {retry_count} retries")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.

        Args:
            url: URL string

        Returns:
            Domain name
        """

        parsed = urlparse(url)
        return parsed.netloc or "unknown"

    def set_domain_policy(self, domain: str, policy: RetryPolicy) -> None:
        """Set retry policy for specific domain.

        Args:
            domain: Domain name
            policy: Retry policy
        """
        self.domain_policies[domain] = policy

    def get_failure_stats(self, domain: str | None = None) -> dict[str, Any]:
        """Get failure statistics.

        Args:
            domain: Optional domain filter

        Returns:
            Failure statistics
        """
        if domain:
            stats = self.failure_stats.get(domain, {})
            return {
                "domain": domain,
                "failures": stats,
                "circuit_breaker": (
                    self.circuit_breakers[domain].get_stats() if domain in self.circuit_breakers else None
                ),
            }
        return {
            domain: {
                "failures": stats,
                "circuit_breaker": self._get_circuit_breaker(domain).get_stats(),
            }
            for domain, stats in self.failure_stats.items()
        }

    def reset_circuit_breaker(self, domain: str) -> None:
        """Manually reset circuit breaker for domain.

        Args:
            domain: Domain name
        """
        if domain in self.circuit_breakers:
            self.circuit_breakers[domain].reset()
            logger.info(f"Circuit breaker for {domain} manually reset")

    async def recover_stalled_jobs(self, stall_timeout: int = 300) -> list[Job]:
        """Recover jobs that have been processing too long.

        Args:
            stall_timeout: Seconds before job is considered stalled

        Returns:
            List of recovered jobs
        """
        # This would integrate with the progress tracker to find stalled jobs
        # For now, return empty list
        return []
