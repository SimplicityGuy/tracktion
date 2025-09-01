"""Intelligent rate limiting for batch operations."""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED = "fixed"  # Fixed rate limiting
    ADAPTIVE = "adaptive"  # Adjust based on response times
    PROGRESSIVE = "progressive"  # Start slow, increase gradually
    EXPONENTIAL = "exponential"  # Exponential backoff on errors


@dataclass
class DomainMetrics:
    """Metrics for a specific domain."""

    domain: str
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    error_times: deque[datetime] = field(default_factory=lambda: deque(maxlen=50))
    last_request_time: float | None = None
    current_rate: float = 5.0  # Requests per second
    optimal_rate: float = 5.0
    backoff_until: datetime | None = None

    def add_response(self, response_time: float, success: bool = True) -> None:
        """Record a response."""
        self.response_times.append(response_time)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.error_times.append(datetime.now(UTC))

    def get_avg_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    def get_error_rate(self) -> float:
        """Get error rate in last minute."""
        if not self.error_times:
            return 0.0

        now = datetime.now(UTC)
        recent_errors = sum(1 for t in self.error_times if (now - t).total_seconds() < 60)
        total = self.successful_requests + self.failed_requests
        return recent_errors / max(total, 1)

    def is_backed_off(self) -> bool:
        """Check if domain is in backoff period."""
        if not self.backoff_until:
            return False
        return datetime.now(UTC) < self.backoff_until


@dataclass
class Request:
    """Represents a scheduled request."""

    url: str
    priority: int
    scheduled_time: float
    job_id: str


@dataclass
class ScheduledRequest:
    """Request with scheduled execution time."""

    request: Request
    execute_at: float


class BatchRateLimiter:
    """Intelligent rate limiter for batch operations."""

    def __init__(
        self,
        default_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 50.0,
        strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE,
    ):
        """Initialize rate limiter.

        Args:
            default_rate: Default requests per second
            min_rate: Minimum requests per second
            max_rate: Maximum requests per second
            strategy: Rate limiting strategy
        """
        self.default_rate = default_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.strategy = strategy

        self.domain_metrics: dict[str, DomainMetrics] = {}
        self.request_queue: dict[str, deque[Request]] = defaultdict(deque)
        self._lock = Lock()

        # Response time thresholds for adaptation
        self.target_response_time = 1.0  # seconds
        self.max_response_time = 5.0

        # Backpressure settings
        self.max_queue_depth = 1000
        self.backpressure_threshold = 0.8

    def calculate_optimal_rate(self, domain: str, metrics: DomainMetrics) -> float:
        """Calculate optimal request rate for domain.

        Args:
            domain: Domain name
            metrics: Domain metrics

        Returns:
            Optimal requests per second
        """
        if self.strategy == RateLimitStrategy.FIXED:
            return metrics.current_rate

        avg_response_time = metrics.get_avg_response_time()
        error_rate = metrics.get_error_rate()

        if self.strategy == RateLimitStrategy.ADAPTIVE:
            # Adjust based on response time and errors
            if avg_response_time > self.max_response_time:
                # Too slow, reduce rate
                new_rate = metrics.current_rate * 0.8
            elif avg_response_time > self.target_response_time:
                # Slightly slow, minor reduction
                new_rate = metrics.current_rate * 0.95
            elif error_rate > 0.1:
                # High error rate, reduce
                new_rate = metrics.current_rate * 0.7
            elif error_rate < 0.01 and avg_response_time < self.target_response_time * 0.5:
                # Very fast and stable, increase
                new_rate = metrics.current_rate * 1.2
            else:
                # Stable, slight increase
                new_rate = metrics.current_rate * 1.05

        elif self.strategy == RateLimitStrategy.PROGRESSIVE:
            # Start slow and gradually increase
            total_requests = metrics.successful_requests + metrics.failed_requests

            if total_requests < 10:
                new_rate = self.min_rate
            elif total_requests < 50:
                new_rate = self.min_rate * 2
            elif error_rate < 0.05:
                new_rate = min(metrics.current_rate * 1.1, self.max_rate)
            else:
                new_rate = metrics.current_rate

        elif self.strategy == RateLimitStrategy.EXPONENTIAL:
            # Exponential backoff on errors
            if error_rate > 0.2:
                # High errors, exponential backoff
                new_rate = max(metrics.current_rate * 0.5, self.min_rate)
                metrics.backoff_until = datetime.now(UTC) + timedelta(seconds=30)
            elif error_rate > 0.1:
                # Some errors, linear backoff
                new_rate = metrics.current_rate * 0.8
            elif metrics.is_backed_off():
                # In backoff, maintain current
                new_rate = metrics.current_rate
            else:
                # No errors, can increase
                new_rate = min(metrics.current_rate * 1.1, self.max_rate)
        else:
            new_rate = metrics.current_rate

        # Apply bounds
        return max(self.min_rate, min(new_rate, self.max_rate))

    def apply_backpressure(self, queue_depth: int) -> None:
        """Apply backpressure when queue is too deep.

        Args:
            queue_depth: Current queue depth
        """
        if queue_depth > self.max_queue_depth * self.backpressure_threshold:
            # Queue is getting full, slow down all domains
            with self._lock:
                for metrics in self.domain_metrics.values():
                    metrics.current_rate = max(metrics.current_rate * 0.5, self.min_rate)
            logger.warning(f"Backpressure applied: queue depth {queue_depth}")

    def schedule_requests(self, requests: list[Request]) -> list[ScheduledRequest]:
        """Schedule requests with optimal timing.

        Args:
            requests: List of requests to schedule

        Returns:
            List of scheduled requests
        """
        scheduled = []
        requests_by_domain: dict[str, list[Request]] = defaultdict(list)

        # Group by domain
        for request in requests:
            domain = self._extract_domain(request.url)
            requests_by_domain[domain].append(request)

        current_time = time.time()

        for domain, domain_requests in requests_by_domain.items():
            # Get or create metrics
            if domain not in self.domain_metrics:
                self.domain_metrics[domain] = DomainMetrics(domain=domain, current_rate=self.default_rate)

            metrics = self.domain_metrics[domain]

            # Check if backed off
            if metrics.is_backed_off():
                logger.warning(f"Domain {domain} is backed off")
                continue

            # Calculate timing
            delay_between = 1.0 / metrics.current_rate
            last_time = metrics.last_request_time or current_time

            # Sort by priority (lower number = higher priority)
            domain_requests.sort(key=lambda r: r.priority)

            # Schedule each request
            for i, request in enumerate(domain_requests):
                execute_at = max(current_time, last_time + delay_between * (i + 1))

                scheduled.append(ScheduledRequest(request=request, execute_at=execute_at))

            # Update last request time
            if domain_requests:
                metrics.last_request_time = scheduled[-1].execute_at

        # Sort by execution time
        scheduled.sort(key=lambda s: s.execute_at)

        return scheduled

    async def wait_for_slot(self, domain: str) -> bool:
        """Wait for next available slot for domain.

        Args:
            domain: Domain name

        Returns:
            True if slot acquired, False if backed off
        """
        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = DomainMetrics(domain=domain, current_rate=self.default_rate)

        metrics = self.domain_metrics[domain]

        # Check backoff
        if metrics.is_backed_off():
            return False

        # Calculate delay
        current_time = time.time()
        if metrics.last_request_time:
            delay_needed = 1.0 / metrics.current_rate
            time_since_last = current_time - metrics.last_request_time

            if time_since_last < delay_needed:
                await asyncio.sleep(delay_needed - time_since_last)

        # Update last request time
        metrics.last_request_time = time.time()
        return True

    def record_response(self, domain: str, response_time: float, success: bool = True) -> None:
        """Record response metrics.

        Args:
            domain: Domain name
            response_time: Response time in seconds
            success: Whether request was successful
        """
        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = DomainMetrics(domain=domain, current_rate=self.default_rate)

        metrics = self.domain_metrics[domain]
        metrics.add_response(response_time, success)

        # Update optimal rate
        metrics.optimal_rate = self.calculate_optimal_rate(domain, metrics)

        # Apply new rate gradually
        rate_diff = metrics.optimal_rate - metrics.current_rate
        metrics.current_rate += rate_diff * 0.3  # 30% adjustment

        logger.debug(
            f"Domain {domain}: rate={metrics.current_rate:.2f}, response_time={response_time:.2f}s, success={success}"
        )

    def get_domain_stats(self, domain: str) -> dict[str, Any]:
        """Get statistics for a domain.

        Args:
            domain: Domain name

        Returns:
            Domain statistics
        """
        if domain not in self.domain_metrics:
            return {
                "domain": domain,
                "current_rate": self.default_rate,
                "requests": 0,
                "error_rate": 0.0,
                "avg_response_time": 0.0,
            }

        metrics = self.domain_metrics[domain]
        total = metrics.successful_requests + metrics.failed_requests

        return {
            "domain": domain,
            "current_rate": metrics.current_rate,
            "optimal_rate": metrics.optimal_rate,
            "requests": total,
            "successful": metrics.successful_requests,
            "failed": metrics.failed_requests,
            "error_rate": metrics.get_error_rate(),
            "avg_response_time": metrics.get_avg_response_time(),
            "is_backed_off": metrics.is_backed_off(),
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all domains.

        Returns:
            Statistics by domain
        """
        return {domain: self.get_domain_stats(domain) for domain in self.domain_metrics}

    def reset_domain(self, domain: str) -> None:
        """Reset metrics for a domain.

        Args:
            domain: Domain name
        """
        if domain in self.domain_metrics:
            self.domain_metrics[domain] = DomainMetrics(domain=domain, current_rate=self.default_rate)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.

        Args:
            url: URL string

        Returns:
            Domain name
        """

        parsed = urlparse(url)
        return parsed.netloc or "unknown"
