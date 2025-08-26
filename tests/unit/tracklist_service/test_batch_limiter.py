"""Unit tests for batch rate limiting."""

import asyncio
import time
from datetime import datetime, UTC, timedelta
from unittest.mock import Mock, patch

import pytest

from services.tracklist_service.src.rate_limiting.batch_limiter import (
    BatchRateLimiter,
    RateLimitStrategy,
    DomainMetrics,
    Request,
    ScheduledRequest,
)


class TestDomainMetrics:
    """Test DomainMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = DomainMetrics(domain="example.com")
        
        assert metrics.domain == "example.com"
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.current_rate == 5.0
        assert metrics.backoff_until is None
    
    def test_add_response_success(self):
        """Test adding successful response."""
        metrics = DomainMetrics(domain="example.com")
        
        metrics.add_response(0.5, success=True)
        
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert len(metrics.response_times) == 1
        assert metrics.response_times[0] == 0.5
    
    def test_add_response_failure(self):
        """Test adding failed response."""
        metrics = DomainMetrics(domain="example.com")
        
        metrics.add_response(2.0, success=False)
        
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert len(metrics.error_times) == 1
    
    def test_get_avg_response_time(self):
        """Test average response time calculation."""
        metrics = DomainMetrics(domain="example.com")
        
        metrics.add_response(1.0)
        metrics.add_response(2.0)
        metrics.add_response(3.0)
        
        assert metrics.get_avg_response_time() == 2.0
    
    def test_get_avg_response_time_empty(self):
        """Test average response time with no data."""
        metrics = DomainMetrics(domain="example.com")
        
        assert metrics.get_avg_response_time() == 0.0
    
    def test_get_error_rate(self):
        """Test error rate calculation."""
        metrics = DomainMetrics(domain="example.com")
        
        # Add some successes and failures
        for _ in range(7):
            metrics.add_response(1.0, success=True)
        for _ in range(3):
            metrics.add_response(1.0, success=False)
        
        # Error rate should be based on recent errors
        error_rate = metrics.get_error_rate()
        assert 0.2 <= error_rate <= 0.4
    
    def test_is_backed_off(self):
        """Test backoff checking."""
        metrics = DomainMetrics(domain="example.com")
        
        assert metrics.is_backed_off() is False
        
        metrics.backoff_until = datetime.now(UTC) + timedelta(seconds=10)
        assert metrics.is_backed_off() is True
        
        metrics.backoff_until = datetime.now(UTC) - timedelta(seconds=10)
        assert metrics.is_backed_off() is False


class TestBatchRateLimiter:
    """Test BatchRateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = BatchRateLimiter(
            default_rate=15.0,
            min_rate=2.0,
            max_rate=30.0,
            strategy=RateLimitStrategy.ADAPTIVE
        )
        
        assert limiter.default_rate == 15.0
        assert limiter.min_rate == 2.0
        assert limiter.max_rate == 30.0
        assert limiter.strategy == RateLimitStrategy.ADAPTIVE
    
    def test_calculate_optimal_rate_fixed(self):
        """Test fixed rate strategy."""
        limiter = BatchRateLimiter(strategy=RateLimitStrategy.FIXED)
        metrics = DomainMetrics(domain="example.com", current_rate=10.0)
        
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        
        assert rate == 10.0  # Should not change
    
    def test_calculate_optimal_rate_adaptive_slow(self):
        """Test adaptive rate with slow responses."""
        limiter = BatchRateLimiter(strategy=RateLimitStrategy.ADAPTIVE)
        metrics = DomainMetrics(domain="example.com", current_rate=10.0)
        
        # Add slow responses
        for _ in range(10):
            metrics.add_response(6.0, success=True)
        
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        
        assert rate < 10.0  # Should reduce rate
    
    def test_calculate_optimal_rate_adaptive_fast(self):
        """Test adaptive rate with fast responses."""
        limiter = BatchRateLimiter(strategy=RateLimitStrategy.ADAPTIVE)
        metrics = DomainMetrics(domain="example.com", current_rate=10.0)
        
        # Add fast responses
        for _ in range(10):
            metrics.add_response(0.2, success=True)
        
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        
        assert rate > 10.0  # Should increase rate
    
    def test_calculate_optimal_rate_adaptive_errors(self):
        """Test adaptive rate with errors."""
        limiter = BatchRateLimiter(strategy=RateLimitStrategy.ADAPTIVE)
        metrics = DomainMetrics(domain="example.com", current_rate=10.0)
        
        # Add responses with high error rate
        for _ in range(5):
            metrics.add_response(1.0, success=True)
        for _ in range(2):
            metrics.add_response(1.0, success=False)
        
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        
        assert rate < 10.0  # Should reduce rate due to errors
    
    def test_calculate_optimal_rate_progressive(self):
        """Test progressive rate strategy."""
        limiter = BatchRateLimiter(
            strategy=RateLimitStrategy.PROGRESSIVE,
            min_rate=1.0
        )
        metrics = DomainMetrics(domain="example.com", current_rate=5.0)
        
        # Few requests - should start slow
        metrics.successful_requests = 5
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        assert rate == 1.0
        
        # More requests - should increase
        metrics.successful_requests = 25
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        assert rate > 1.0
    
    def test_calculate_optimal_rate_exponential(self):
        """Test exponential backoff strategy."""
        limiter = BatchRateLimiter(strategy=RateLimitStrategy.EXPONENTIAL)
        metrics = DomainMetrics(domain="example.com", current_rate=10.0)
        
        # High error rate - should back off
        metrics.successful_requests = 5
        metrics.failed_requests = 3
        for _ in range(3):
            metrics.error_times.append(datetime.now(UTC))
        
        rate = limiter.calculate_optimal_rate("example.com", metrics)
        
        assert rate < 10.0  # Should reduce rate
        assert metrics.backoff_until is not None
    
    def test_apply_backpressure(self):
        """Test backpressure application."""
        limiter = BatchRateLimiter()
        
        # Add some domain metrics
        limiter.domain_metrics["domain1"] = DomainMetrics(
            domain="domain1",
            current_rate=20.0
        )
        limiter.domain_metrics["domain2"] = DomainMetrics(
            domain="domain2",
            current_rate=15.0
        )
        
        # Apply backpressure
        limiter.apply_backpressure(850)  # Over threshold
        
        # Rates should be reduced
        assert limiter.domain_metrics["domain1"].current_rate < 20.0
        assert limiter.domain_metrics["domain2"].current_rate < 15.0
    
    def test_schedule_requests(self):
        """Test request scheduling."""
        limiter = BatchRateLimiter()
        
        requests = [
            Request(url="http://example.com/1", priority=1, scheduled_time=0, job_id="job1"),
            Request(url="http://example.com/2", priority=2, scheduled_time=0, job_id="job2"),
            Request(url="http://other.com/1", priority=1, scheduled_time=0, job_id="job3"),
        ]
        
        scheduled = limiter.schedule_requests(requests)
        
        assert len(scheduled) == 3
        assert all(isinstance(s, ScheduledRequest) for s in scheduled)
        
        # Should be sorted by execution time
        times = [s.execute_at for s in scheduled]
        assert times == sorted(times)
    
    def test_schedule_requests_with_priority(self):
        """Test request scheduling with priority."""
        limiter = BatchRateLimiter()
        
        requests = [
            Request(url="http://example.com/1", priority=3, scheduled_time=0, job_id="job1"),
            Request(url="http://example.com/2", priority=1, scheduled_time=0, job_id="job2"),
            Request(url="http://example.com/3", priority=2, scheduled_time=0, job_id="job3"),
        ]
        
        scheduled = limiter.schedule_requests(requests)
        
        # Higher priority (lower number) should be scheduled first
        example_scheduled = [s for s in scheduled if "example.com" in s.request.url]
        priorities = [s.request.priority for s in example_scheduled]
        assert priorities == [1, 2, 3]
    
    def test_schedule_requests_backed_off_domain(self):
        """Test scheduling with backed off domain."""
        limiter = BatchRateLimiter()
        
        # Set domain as backed off
        limiter.domain_metrics["example.com"] = DomainMetrics(
            domain="example.com",
            backoff_until=datetime.now(UTC) + timedelta(seconds=30)
        )
        
        requests = [
            Request(url="http://example.com/1", priority=1, scheduled_time=0, job_id="job1"),
        ]
        
        scheduled = limiter.schedule_requests(requests)
        
        assert len(scheduled) == 0  # Should not schedule backed off domain
    
    @pytest.mark.asyncio
    async def test_wait_for_slot(self):
        """Test waiting for rate limit slot."""
        limiter = BatchRateLimiter()
        
        # First request should go immediately
        start = time.time()
        result = await limiter.wait_for_slot("example.com")
        elapsed = time.time() - start
        
        assert result is True
        assert elapsed < 0.1
        
        # Second request should wait
        start = time.time()
        result = await limiter.wait_for_slot("example.com")
        elapsed = time.time() - start
        
        assert result is True
        # Should have some delay based on rate
        assert elapsed > 0.05
    
    @pytest.mark.asyncio
    async def test_wait_for_slot_backed_off(self):
        """Test waiting with backed off domain."""
        limiter = BatchRateLimiter()
        
        # Set domain as backed off
        limiter.domain_metrics["example.com"] = DomainMetrics(
            domain="example.com",
            backoff_until=datetime.now(UTC) + timedelta(seconds=1)
        )
        
        result = await limiter.wait_for_slot("example.com")
        
        assert result is False  # Should return False when backed off
    
    def test_record_response(self):
        """Test recording response metrics."""
        limiter = BatchRateLimiter()
        
        limiter.record_response("example.com", 0.5, success=True)
        
        assert "example.com" in limiter.domain_metrics
        metrics = limiter.domain_metrics["example.com"]
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert 0.5 in metrics.response_times
    
    def test_get_domain_stats(self):
        """Test getting domain statistics."""
        limiter = BatchRateLimiter()
        
        # Add some data
        limiter.record_response("example.com", 1.0, success=True)
        limiter.record_response("example.com", 2.0, success=True)
        limiter.record_response("example.com", 1.5, success=False)
        
        stats = limiter.get_domain_stats("example.com")
        
        assert stats["domain"] == "example.com"
        assert stats["requests"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
        assert stats["avg_response_time"] == 1.5
    
    def test_get_domain_stats_nonexistent(self):
        """Test getting stats for nonexistent domain."""
        limiter = BatchRateLimiter()
        
        stats = limiter.get_domain_stats("unknown.com")
        
        assert stats["domain"] == "unknown.com"
        assert stats["requests"] == 0
        assert stats["error_rate"] == 0.0
    
    def test_get_all_stats(self):
        """Test getting all domain statistics."""
        limiter = BatchRateLimiter()
        
        limiter.record_response("domain1.com", 1.0, success=True)
        limiter.record_response("domain2.com", 2.0, success=False)
        
        all_stats = limiter.get_all_stats()
        
        assert "domain1.com" in all_stats
        assert "domain2.com" in all_stats
        assert all_stats["domain1.com"]["successful"] == 1
        assert all_stats["domain2.com"]["failed"] == 1
    
    def test_reset_domain(self):
        """Test resetting domain metrics."""
        limiter = BatchRateLimiter()
        
        # Add some data
        limiter.record_response("example.com", 1.0, success=True)
        limiter.record_response("example.com", 2.0, success=False)
        
        # Reset
        limiter.reset_domain("example.com")
        
        metrics = limiter.domain_metrics["example.com"]
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert len(metrics.response_times) == 0
    
    def test_extract_domain(self):
        """Test domain extraction from URL."""
        limiter = BatchRateLimiter()
        
        assert limiter._extract_domain("http://example.com/path") == "example.com"
        assert limiter._extract_domain("https://sub.example.com/") == "sub.example.com"
        assert limiter._extract_domain("invalid-url") == "unknown"