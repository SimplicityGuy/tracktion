"""Unit tests for usage tracking and analytics."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from services.tracklist_service.src.analytics.usage_tracker import (
    AggregationPeriod,
    UsageRecord,
    UsageReport,
    UsageStats,
    UsageTracker,
)
from services.tracklist_service.src.auth.models import User, UserTier


class TestUsageRecord:
    """Test UsageRecord data class."""

    def test_initialization(self):
        """Test usage record creation."""
        timestamp = datetime.now(UTC)
        record = UsageRecord(
            user_id="user123",
            timestamp=timestamp,
            endpoint="/api/v1/search",
            method="GET",
            response_time=0.5,
            status_code=200,
            tokens_consumed=1,
            bytes_processed=1024,
        )

        assert record.user_id == "user123"
        assert record.timestamp == timestamp
        assert record.endpoint == "/api/v1/search"
        assert record.method == "GET"
        assert record.response_time == 0.5
        assert record.status_code == 200
        assert record.tokens_consumed == 1
        assert record.bytes_processed == 1024
        assert record.error_message is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2024, 1, 15, 12, 30, 45, tzinfo=UTC)
        record = UsageRecord(
            user_id="user123",
            timestamp=timestamp,
            endpoint="/api/v1/tracklist",
            method="POST",
            response_time=1.2,
            status_code=201,
            tokens_consumed=5,
            bytes_processed=2048,
            error_message=None,
        )

        result = record.to_dict()

        assert result["user_id"] == "user123"
        assert result["timestamp"] == "2024-01-15T12:30:45+00:00"
        assert result["endpoint"] == "/api/v1/tracklist"
        assert result["method"] == "POST"
        assert result["response_time"] == 1.2
        assert result["status_code"] == 201
        assert result["tokens_consumed"] == 5
        assert result["bytes_processed"] == 2048
        assert result["error_message"] is None

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "user_id": "user456",
            "timestamp": "2024-01-15T12:30:45+00:00",
            "endpoint": "/api/v1/batch",
            "method": "POST",
            "response_time": 2.1,
            "status_code": 400,
            "tokens_consumed": 10,
            "bytes_processed": 4096,
            "error_message": "Invalid request",
        }

        record = UsageRecord.from_dict(data)

        assert record.user_id == "user456"
        assert record.timestamp == datetime(2024, 1, 15, 12, 30, 45, tzinfo=UTC)
        assert record.endpoint == "/api/v1/batch"
        assert record.method == "POST"
        assert record.response_time == 2.1
        assert record.status_code == 400
        assert record.tokens_consumed == 10
        assert record.bytes_processed == 4096
        assert record.error_message == "Invalid request"


class TestUsageStats:
    """Test UsageStats data class."""

    def test_initialization(self):
        """Test usage stats creation."""
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(hours=1)

        stats = UsageStats(
            user_id="user123",
            period="hour",
            start_time=start_time,
            end_time=end_time,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_tokens=500,
            total_bytes=1024000,
            avg_response_time=0.8,
            endpoints={"/api/v1/search": 80, "/api/v1/tracklist": 20},
            error_rate=5.0,
        )

        assert stats.user_id == "user123"
        assert stats.period == "hour"
        assert stats.total_requests == 100
        assert stats.successful_requests == 95
        assert stats.failed_requests == 5
        assert stats.total_tokens == 500
        assert stats.error_rate == 5.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        start_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC)

        stats = UsageStats(
            user_id="user123",
            period="hour",
            start_time=start_time,
            end_time=end_time,
            total_requests=50,
            successful_requests=48,
            failed_requests=2,
            total_tokens=250,
            total_bytes=512000,
            avg_response_time=0.6,
            endpoints={"/api/v1/search": 50},
            error_rate=4.0,
        )

        result = stats.to_dict()

        assert result["user_id"] == "user123"
        assert result["period"] == "hour"
        assert result["total_requests"] == 50
        assert result["endpoints"] == {"/api/v1/search": 50}
        assert result["error_rate"] == 4.0


class TestUsageTracker:
    """Test UsageTracker class."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        return redis_mock

    @pytest.fixture
    def usage_tracker(self, mock_redis):
        """Create usage tracker instance."""
        return UsageTracker(mock_redis)

    @pytest.fixture
    def test_user_free(self):
        """Create free tier test user."""
        return User(id="user123", email="test@example.com", tier=UserTier.FREE)

    @pytest.fixture
    def test_user_premium(self):
        """Create premium tier test user."""
        return User(id="user456", email="premium@example.com", tier=UserTier.PREMIUM)

    def test_initialization(self, mock_redis):
        """Test usage tracker initialization."""
        tracker = UsageTracker(mock_redis)

        assert tracker.redis is mock_redis
        assert len(tracker.pricing) == 3
        assert tracker.pricing["free"] == Decimal("0.00")
        assert tracker.pricing["premium"] == Decimal("0.002")
        assert tracker.pricing["enterprise"] == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_track_request_success(self, usage_tracker, test_user_free, mock_redis):
        """Test successful request tracking."""
        mock_redis.zadd = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock(return_value=True)

        # Mock pipeline for real-time stats
        pipe_mock = Mock()
        pipe_mock.hincrby = Mock()
        pipe_mock.hincrbyfloat = Mock()
        pipe_mock.expire = Mock()
        pipe_mock.execute = AsyncMock(return_value=[1, 1, 1, 1, 1, 1, 1])
        mock_redis.pipeline = Mock(return_value=pipe_mock)

        await usage_tracker.track_request(
            user=test_user_free,
            endpoint="/api/v1/search",
            method="GET",
            response_time=0.5,
            status_code=200,
            tokens_consumed=1,
            bytes_processed=1024,
        )

        # Verify record was stored
        mock_redis.zadd.assert_called_once()
        mock_redis.expire.assert_called()

        # Verify real-time stats were updated
        assert pipe_mock.hincrby.called
        assert pipe_mock.execute.called

    @pytest.mark.asyncio
    async def test_track_request_error(self, usage_tracker, test_user_free, mock_redis):
        """Test error request tracking."""
        mock_redis.zadd = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock(return_value=True)

        # Mock pipeline for real-time stats
        pipe_mock = Mock()
        pipe_mock.hincrby = Mock()
        pipe_mock.hincrbyfloat = Mock()
        pipe_mock.expire = Mock()
        pipe_mock.execute = AsyncMock(return_value=[1, 1, 1, 1, 1, 1, 1])
        mock_redis.pipeline = Mock(return_value=pipe_mock)

        await usage_tracker.track_request(
            user=test_user_free,
            endpoint="/api/v1/tracklist",
            method="POST",
            response_time=2.1,
            status_code=500,
            tokens_consumed=5,
            error_message="Internal server error",
        )

        # Verify tracking was attempted
        mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage_stats_empty(self, usage_tracker, mock_redis):
        """Test getting usage stats with no data."""
        mock_redis.zrangebyscore.return_value = []

        stats = await usage_tracker.get_usage_stats("user123", AggregationPeriod.HOUR)

        assert stats.user_id == "user123"
        assert stats.period == "hour"
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_tokens == 0
        assert stats.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_get_usage_stats_with_data(self, usage_tracker, mock_redis):
        """Test getting usage stats with sample data."""
        # Mock Redis response with sample usage records
        sample_records = [
            UsageRecord(
                user_id="user123",
                timestamp=datetime.now(UTC),
                endpoint="/api/v1/search",
                method="GET",
                response_time=0.5,
                status_code=200,
                tokens_consumed=1,
                bytes_processed=1024,
            ),
            UsageRecord(
                user_id="user123",
                timestamp=datetime.now(UTC),
                endpoint="/api/v1/search",
                method="GET",
                response_time=0.8,
                status_code=200,
                tokens_consumed=2,
                bytes_processed=2048,
            ),
            UsageRecord(
                user_id="user123",
                timestamp=datetime.now(UTC),
                endpoint="/api/v1/tracklist",
                method="POST",
                response_time=1.2,
                status_code=500,
                tokens_consumed=5,
                bytes_processed=0,
                error_message="Server error",
            ),
        ]

        mock_redis.zrangebyscore.return_value = [json.dumps(record.to_dict()) for record in sample_records]

        stats = await usage_tracker.get_usage_stats("user123", AggregationPeriod.HOUR)

        assert stats.user_id == "user123"
        assert stats.total_requests == 3
        assert stats.successful_requests == 2
        assert stats.failed_requests == 1
        assert stats.total_tokens == 8
        assert stats.total_bytes == 3072
        assert stats.avg_response_time == (0.5 + 0.8 + 1.2) / 3
        assert stats.endpoints["/api/v1/search"] == 2
        assert stats.endpoints["/api/v1/tracklist"] == 1
        assert stats.error_rate == (1 / 3) * 100

    def test_calculate_cost_free_tier(self, usage_tracker):
        """Test cost calculation for free tier."""
        stats = UsageStats(
            user_id="user123",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_tokens=1000,
            total_bytes=1024000,
            avg_response_time=0.5,
            endpoints={"/api/v1/search": 100},
            error_rate=0.0,
        )

        cost_breakdown = usage_tracker.calculate_cost(stats, "free")

        # Free tier should have zero costs
        assert all(cost == Decimal("0") for cost in cost_breakdown.values())

    def test_calculate_cost_premium_tier(self, usage_tracker):
        """Test cost calculation for premium tier."""
        stats = UsageStats(
            user_id="user456",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=1000,
            successful_requests=1000,
            failed_requests=0,
            total_tokens=5000,  # 5000 tokens
            total_bytes=1024**3,  # 1 GB
            avg_response_time=0.5,
            endpoints={"/api/v1/search": 800, "/api/v1/tracklist": 200},
            error_rate=0.0,
        )

        cost_breakdown = usage_tracker.calculate_cost(stats, "premium")

        # Token cost: (5000 / 1000) * 0.002 = 0.01
        assert cost_breakdown["tokens"] == Decimal("0.010")

        # Data processing: 1 GB * 0.01 = 0.01
        assert cost_breakdown["data_processing"] == Decimal("0.01")

        # Request costs should be calculated
        assert "requests_search" in cost_breakdown
        assert "requests_tracklist" in cost_breakdown

    def test_get_endpoint_type(self, usage_tracker):
        """Test endpoint type determination."""
        assert usage_tracker._get_endpoint_type("/api/v1/search") == "search"
        assert usage_tracker._get_endpoint_type("/api/v1/tracklist/extract") == "tracklist"
        assert usage_tracker._get_endpoint_type("/api/v1/batch/process") == "batch"
        assert usage_tracker._get_endpoint_type("/api/v1/unknown") == "search"

    @pytest.mark.asyncio
    async def test_generate_usage_report(self, usage_tracker, test_user_premium, mock_redis):
        """Test usage report generation."""
        # Mock usage stats retrieval
        mock_redis.zrangebyscore.return_value = [
            json.dumps(
                UsageRecord(
                    user_id="user456",
                    timestamp=datetime.now(UTC),
                    endpoint="/api/v1/search",
                    method="GET",
                    response_time=0.5,
                    status_code=200,
                    tokens_consumed=1,
                    bytes_processed=1024,
                ).to_dict()
            )
        ]

        report = await usage_tracker.generate_usage_report(test_user_premium, AggregationPeriod.DAY)

        assert isinstance(report, UsageReport)
        assert report.user_id == "user456"
        assert report.period == "day"
        assert report.tier == "premium"
        assert isinstance(report.usage_stats, UsageStats)
        assert isinstance(report.total_cost, Decimal)
        assert isinstance(report.recommendations, list)

    def test_generate_recommendations_high_error_rate(self, usage_tracker):
        """Test recommendations for high error rate."""
        stats = UsageStats(
            user_id="user123",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=100,
            successful_requests=80,
            failed_requests=20,
            total_tokens=500,
            total_bytes=1024000,
            avg_response_time=0.5,
            endpoints={"/api/v1/search": 100},
            error_rate=20.0,  # High error rate
        )

        recommendations = usage_tracker._generate_recommendations(stats, "premium")

        assert any("High error rate" in rec for rec in recommendations)

    def test_generate_recommendations_slow_response(self, usage_tracker):
        """Test recommendations for slow response times."""
        stats = UsageStats(
            user_id="user123",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_tokens=500,
            total_bytes=1024000,
            avg_response_time=3.0,  # Slow response time
            endpoints={"/api/v1/search": 100},
            error_rate=0.0,
        )

        recommendations = usage_tracker._generate_recommendations(stats, "premium")

        assert any("response time" in rec for rec in recommendations)

    def test_generate_recommendations_upgrade_tier(self, usage_tracker):
        """Test recommendations for tier upgrades."""
        stats = UsageStats(
            user_id="user123",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=15000,  # High usage
            successful_requests=15000,
            failed_requests=0,
            total_tokens=50000,
            total_bytes=1024000,
            avg_response_time=0.5,
            endpoints={"/api/v1/search": 15000},
            error_rate=0.0,
        )

        # Free tier with high usage
        recommendations = usage_tracker._generate_recommendations(stats, "free")
        assert any("Premium tier" in rec for rec in recommendations)

        # Premium tier with very high usage
        stats.total_requests = 150000
        recommendations = usage_tracker._generate_recommendations(stats, "premium")
        assert any("Enterprise tier" in rec for rec in recommendations)

    def test_get_period_bounds_hour(self, usage_tracker):
        """Test period bounds calculation for hour."""
        start, end = usage_tracker._get_period_bounds(AggregationPeriod.HOUR)

        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        assert end > start

    def test_get_period_bounds_day(self, usage_tracker):
        """Test period bounds calculation for day."""
        start, end = usage_tracker._get_period_bounds(AggregationPeriod.DAY)

        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        assert end > start

    def test_get_period_bounds_month(self, usage_tracker):
        """Test period bounds calculation for month."""
        start, end = usage_tracker._get_period_bounds(AggregationPeriod.MONTH)

        assert start.day == 1
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        assert end > start

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, usage_tracker, mock_redis):
        """Test cleanup of old usage data."""

        # Mock Redis operations - create a proper async iterator
        async def mock_scan_iter(*args, **kwargs):
            for key in ["usage_records:user1", "usage_records:user2"]:
                yield key

        mock_redis.scan_iter = Mock(return_value=mock_scan_iter())
        mock_redis.zremrangebyscore.return_value = 50  # 50 records removed per user

        cleaned_count = await usage_tracker.cleanup_old_data(30)

        assert cleaned_count == 100  # 50 * 2 users
        assert mock_redis.zremrangebyscore.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_old_data_error(self, usage_tracker, mock_redis):
        """Test cleanup error handling."""
        mock_redis.scan_iter.side_effect = Exception("Redis error")

        cleaned_count = await usage_tracker.cleanup_old_data(30)

        assert cleaned_count == 0

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, usage_tracker, mock_redis):
        """Test health check when system is healthy."""
        mock_redis.ping.return_value = True

        # Mock Redis operations - create a proper async iterator
        async def mock_scan_iter(*args, **kwargs):
            for key in ["usage_records:user1", "usage_records:user2"]:
                yield key

        mock_redis.scan_iter = Mock(return_value=mock_scan_iter())
        mock_redis.zcard.return_value = 100  # 100 records per user

        health = await usage_tracker.health_check()

        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["total_records"] == 200  # 100 * 2 users
        assert "supported_periods" in health
        assert "pricing_tiers" in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, usage_tracker, mock_redis):
        """Test health check when Redis is down."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        health = await usage_tracker.health_check()

        assert health["status"] == "unhealthy"
        assert health["redis_connected"] is False
        assert "error" in health

    @pytest.mark.asyncio
    async def test_track_request_exception_handling(self, usage_tracker, test_user_free, mock_redis):
        """Test exception handling in request tracking."""
        mock_redis.zadd.side_effect = Exception("Redis error")

        # Should not raise exception, just log error
        await usage_tracker.track_request(
            user=test_user_free,
            endpoint="/api/v1/search",
            method="GET",
            response_time=0.5,
            status_code=200,
            tokens_consumed=1,
        )

        # Verify error was handled gracefully
        mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage_records_invalid_json(self, usage_tracker, mock_redis):
        """Test handling of invalid JSON in usage records."""
        mock_redis.zrangebyscore.return_value = [
            "invalid json",
            json.dumps(
                UsageRecord(
                    user_id="user123",
                    timestamp=datetime.now(UTC),
                    endpoint="/api/v1/search",
                    method="GET",
                    response_time=0.5,
                    status_code=200,
                    tokens_consumed=1,
                    bytes_processed=1024,
                ).to_dict()
            ),
        ]

        records = await usage_tracker._get_usage_records(
            "user123", datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC)
        )

        # Should only return valid records
        assert len(records) == 1
        assert records[0].user_id == "user123"

    def test_usage_report_to_dict(self, usage_tracker):
        """Test UsageReport to_dict conversion."""
        stats = UsageStats(
            user_id="user123",
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_tokens=500,
            total_bytes=1024000,
            avg_response_time=0.5,
            endpoints={"/api/v1/search": 100},
            error_rate=5.0,
        )

        cost_breakdown = {"tokens": Decimal("0.01"), "requests_total": Decimal("0.10")}

        report = UsageReport(
            user_id="user123",
            period="day",
            usage_stats=stats,
            cost_breakdown=cost_breakdown,
            total_cost=Decimal("0.11"),
            tier="premium",
            recommendations=["Optimize API usage"],
        )

        result = report.to_dict()

        assert result["user_id"] == "user123"
        assert result["period"] == "day"
        assert result["total_cost"] == "0.11"
        assert result["tier"] == "premium"
        assert result["recommendations"] == ["Optimize API usage"]
        assert "usage_stats" in result
        assert "cost_breakdown" in result
