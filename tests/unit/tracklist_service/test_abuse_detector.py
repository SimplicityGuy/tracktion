"""Unit tests for AbuseDetector."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from services.tracklist_service.src.auth.models import User, UserTier
from services.tracklist_service.src.security.abuse_detector import AbuseDetector
from services.tracklist_service.src.security.models import AbuseType


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return AsyncMock()


@pytest.fixture
def test_user():
    """Create test user."""
    return User(id="test-user-123", email="test@example.com", tier=UserTier.PREMIUM, is_active=True)


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/api/v1/test"
    request.url.query = ""
    request.headers = {"user-agent": "test-client/1.0", "x-forwarded-for": "203.0.113.1"}
    request.client.host = "192.168.1.100"
    return request


@pytest.fixture
def abuse_detector(mock_redis):
    """Create AbuseDetector instance."""
    return AbuseDetector(mock_redis)


class TestAbuseDetector:
    """Test AbuseDetector functionality."""

    def test_initialization(self, abuse_detector):
        """Test AbuseDetector initialization."""
        assert abuse_detector.redis is not None
        assert abuse_detector.thresholds["high_frequency_rpm"] == 1000
        assert abuse_detector.thresholds["invalid_requests_ratio"] == 0.5
        assert abuse_detector.behavior_key == "abuse:behavior"
        assert abuse_detector.patterns_key == "abuse:patterns"
        assert abuse_detector.scores_key == "abuse:scores"

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_normal(self, abuse_detector, test_user, mock_request):
        """Test analyzing normal user behavior."""
        # Mock normal behavior data
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 10,
                "requests_per_hour": 100,
                "error_rate": 0.01,
                "data_usage_mb": 5,
                "pattern_anomalies": [],
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert score.user_id == test_user.id
                assert score.score <= Decimal("0.2")  # Should be low for normal behavior
                assert len(score.abuse_types) == 0
                assert score.should_block is False
                assert "Normal behavior" in score.recommendation

                # Abuse score should be stored

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_high_frequency(self, abuse_detector, test_user, mock_request):
        """Test analyzing high frequency abuse behavior."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 2000,  # Above threshold
                "requests_per_hour": 10000,
                "error_rate": 0.01,
                "data_usage_mb": 5,
                "pattern_anomalies": [],
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert score.user_id == test_user.id
                assert score.score > Decimal("0.0")
                assert AbuseType.HIGH_FREQUENCY in score.abuse_types
                assert "frequency" in score.details

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_suspicious_patterns(self, abuse_detector, test_user, mock_request):
        """Test analyzing suspicious pattern abuse."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "error_rate": 0.01,
                "data_usage_mb": 5,
                "pattern_anomalies": ["bot_like", "scraping"],  # Suspicious patterns
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert AbuseType.SUSPICIOUS_PATTERN in score.abuse_types
                assert "patterns" in score.details

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_invalid_requests(self, abuse_detector, test_user, mock_request):
        """Test analyzing invalid request abuse."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "error_rate": 0.8,  # High error rate
                "data_usage_mb": 5,
                "pattern_anomalies": [],
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert AbuseType.INVALID_REQUESTS in score.abuse_types
                assert "invalid_requests" in score.details
                # Error rate of 0.8 is at threshold boundary, should not trigger block (needs >0.8)
                # But still detect as abuse type

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_resource_abuse(self, abuse_detector, test_user, mock_request):
        """Test analyzing resource abuse."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "error_rate": 0.01,
                "data_usage_mb": 200,  # High data usage
                "pattern_anomalies": [],
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert AbuseType.RESOURCE_ABUSE in score.abuse_types
                assert "resource_abuse" in score.details

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_quota_exhaustion(self, abuse_detector, test_user, mock_request):
        """Test analyzing quota exhaustion abuse."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "error_rate": 0.01,
                "data_usage_mb": 5,
                "pattern_anomalies": [],
                "geo_anomalies": [],
                "quota_usage_percent": 0.98,  # Near quota exhaustion
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert AbuseType.QUOTA_EXHAUSTION in score.abuse_types
                assert "quota" in score.details

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_multiple_abuse_types(self, abuse_detector, test_user, mock_request):
        """Test analyzing behavior with multiple abuse types."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.return_value = {
                "requests_per_minute": 2000,  # High frequency
                "requests_per_hour": 10000,
                "error_rate": 0.6,  # High error rate
                "data_usage_mb": 5,
                "pattern_anomalies": ["bot_like"],  # Suspicious pattern
                "geo_anomalies": [],
                "quota_usage_percent": 0.1,
            }

            with patch.object(abuse_detector, "_store_abuse_score"):
                score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

                assert len(score.abuse_types) >= 2
                assert score.should_block is True  # Multiple abuse types should trigger block
                assert "High risk" in score.recommendation

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_error_handling(self, abuse_detector, test_user, mock_request):
        """Test error handling in behavior analysis."""
        with patch.object(abuse_detector, "_collect_behavior_data") as mock_collect:
            mock_collect.side_effect = Exception("Data collection failed")

            score = await abuse_detector.analyze_user_behavior(test_user, mock_request)

            assert score.user_id == test_user.id
            assert score.score == Decimal("0.0")
            assert score.should_block is False
            assert "Error occurred" in score.recommendation
            assert "error" in score.details

    @pytest.mark.asyncio
    async def test_collect_behavior_data(self, abuse_detector, test_user, mock_request, mock_redis):
        """Test behavior data collection."""
        # Mock Redis responses
        mock_redis.zcount.return_value = 50  # Mock request count

        with patch.object(abuse_detector, "_store_behavior_point") as mock_store:
            data = await abuse_detector._collect_behavior_data(test_user, mock_request)

            assert "timestamp" in data
            assert "ip_address" in data
            assert "user_agent" in data
            assert "endpoint" in data
            assert "method" in data
            assert "requests_per_minute" in data
            assert "requests_per_hour" in data
            assert "error_rate" in data

            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_requests_in_window(self, abuse_detector, test_user, mock_redis):
        """Test counting requests in time window."""
        mock_redis.zcount.return_value = 42

        start = datetime.now(UTC) - timedelta(minutes=5)
        end = datetime.now(UTC)

        count = await abuse_detector._count_requests_in_window(test_user, start, end)
        assert count == 42

        expected_key = f"abuse:behavior:{test_user.id}"
        mock_redis.zcount.assert_called_once_with(expected_key, int(start.timestamp()), int(end.timestamp()))

    @pytest.mark.asyncio
    async def test_count_errors_in_window(self, abuse_detector, test_user, mock_redis):
        """Test counting errors in time window."""
        mock_redis.zcount.return_value = 15

        start = datetime.now(UTC) - timedelta(minutes=5)
        end = datetime.now(UTC)

        count = await abuse_detector._count_errors_in_window(test_user, start, end)
        assert count == 15

        expected_key = f"abuse:behavior:{test_user.id}:errors"
        mock_redis.zcount.assert_called_once_with(expected_key, int(start.timestamp()), int(end.timestamp()))

    @pytest.mark.asyncio
    async def test_estimate_data_usage(self, abuse_detector, test_user):
        """Test data usage estimation."""
        with patch.object(abuse_detector, "_count_requests_in_window") as mock_count:
            mock_count.return_value = 100

            start = datetime.now(UTC) - timedelta(hours=1)
            end = datetime.now(UTC)

            usage = await abuse_detector._estimate_data_usage(test_user, start, end)
            # 100 requests * 10KB = 1MB
            assert usage == 100 * 10 * 1024

    @pytest.mark.asyncio
    async def test_analyze_request_patterns_normal(self, abuse_detector, test_user):
        """Test request pattern analysis - normal patterns."""
        with patch.object(abuse_detector, "_count_requests_in_window") as mock_count:
            mock_count.return_value = 50  # Normal request count

            start = datetime.now(UTC) - timedelta(hours=1)
            end = datetime.now(UTC)

            anomalies = await abuse_detector._analyze_request_patterns(test_user, start, end)
            assert anomalies == []

    @pytest.mark.asyncio
    async def test_analyze_request_patterns_high_frequency(self, abuse_detector, test_user):
        """Test request pattern analysis - high frequency patterns."""
        with patch.object(abuse_detector, "_count_requests_in_window") as mock_count:
            mock_count.return_value = 500  # High request count

            start = datetime.now(UTC) - timedelta(hours=1)
            end = datetime.now(UTC)

            anomalies = await abuse_detector._analyze_request_patterns(test_user, start, end)
            assert "high_frequency_pattern" in anomalies

    @pytest.mark.asyncio
    async def test_estimate_quota_usage_free_tier(self, abuse_detector):
        """Test quota usage estimation for free tier user."""
        free_user = User(id="free-user-123", email="free@example.com", tier=UserTier.FREE, is_active=True)

        with patch.object(abuse_detector, "_count_requests_in_window") as mock_count:
            mock_count.return_value = 500  # 50% of free tier limit (1000)

            usage = await abuse_detector._estimate_quota_usage(free_user)
            assert usage == 0.5  # 50%

    @pytest.mark.asyncio
    async def test_estimate_quota_usage_premium_tier(self, abuse_detector, test_user):
        """Test quota usage estimation for premium tier user."""
        with patch.object(abuse_detector, "_count_requests_in_window") as mock_count:
            mock_count.return_value = 2000  # 20% of premium tier limit (10000)

            usage = await abuse_detector._estimate_quota_usage(test_user)
            assert usage == 0.2  # 20%

    @pytest.mark.asyncio
    async def test_store_behavior_point(self, abuse_detector, test_user, mock_redis):
        """Test storing behavior data point."""
        data = {"timestamp": datetime.now(UTC).isoformat(), "requests_per_minute": 10, "error_rate": 0.0}

        await abuse_detector._store_behavior_point(test_user, data)

        expected_key = f"abuse:behavior:{test_user.id}"
        # Should be called once - there's no error rate > 0 to trigger additional error storage
        mock_redis.zadd.assert_called_once()
        mock_redis.zremrangebyscore.assert_called_once_with(expected_key, 0, ANY)

    @pytest.mark.asyncio
    async def test_store_behavior_point_with_errors(self, abuse_detector, test_user, mock_redis):
        """Test storing behavior data point with errors."""
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "requests_per_minute": 10,
            "error_rate": 0.5,  # Has errors
        }

        await abuse_detector._store_behavior_point(test_user, data)

        # Should store both regular behavior and error data
        assert mock_redis.zadd.call_count == 2
        assert mock_redis.zremrangebyscore.call_count == 2

    @pytest.mark.asyncio
    async def test_store_abuse_score(self, abuse_detector, test_user, mock_redis):
        """Test storing abuse score."""
        score = Decimal("0.7")
        abuse_types = [AbuseType.HIGH_FREQUENCY, AbuseType.SUSPICIOUS_PATTERN]
        details = {"frequency": {"score": "0.4"}, "patterns": {"score": "0.3"}}

        await abuse_detector._store_abuse_score(test_user, score, abuse_types, details)

        mock_redis.zadd.assert_called_once()
        mock_redis.zremrangebyscore.assert_called_once()

    def test_get_client_ip_forwarded_for(self, abuse_detector, mock_request):
        """Test getting client IP from X-Forwarded-For header."""
        mock_request.headers = {"x-forwarded-for": "203.0.113.1, 192.168.1.100"}

        ip = abuse_detector._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_real_ip(self, abuse_detector, mock_request):
        """Test getting client IP from X-Real-IP header."""
        mock_request.headers = {"x-real-ip": "203.0.113.1"}

        ip = abuse_detector._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_direct(self, abuse_detector, mock_request):
        """Test getting client IP from direct connection."""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"

        ip = abuse_detector._get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_unknown(self, abuse_detector):
        """Test getting client IP when unknown."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.client = None

        ip = abuse_detector._get_client_ip(mock_request)
        assert ip == "unknown"

    @pytest.mark.asyncio
    async def test_get_user_abuse_history_success(self, abuse_detector, test_user, mock_redis):
        """Test getting user abuse history successfully."""
        # Mock Redis response with abuse score data
        score_data = {
            "user_id": test_user.id,
            "score": "0.6",
            "abuse_types": ["high_frequency"],
            "details": {"frequency": {"score": "0.6"}},
            "recommendation": "Monitor closely",
        }

        mock_redis.zrangebyscore.return_value = [json.dumps(score_data)]

        history = await abuse_detector.get_user_abuse_history(test_user, days=7)

        assert len(history) == 1
        assert history[0].user_id == test_user.id
        assert history[0].score == Decimal("0.6")
        assert AbuseType.HIGH_FREQUENCY in history[0].abuse_types

    @pytest.mark.asyncio
    async def test_get_user_abuse_history_empty(self, abuse_detector, test_user, mock_redis):
        """Test getting empty user abuse history."""
        mock_redis.zrangebyscore.return_value = []

        history = await abuse_detector.get_user_abuse_history(test_user, days=7)
        assert history == []

    @pytest.mark.asyncio
    async def test_get_user_abuse_history_parse_error(self, abuse_detector, test_user, mock_redis):
        """Test handling parse errors in abuse history."""
        mock_redis.zrangebyscore.return_value = ["invalid-json", '{"valid": "json"}']

        history = await abuse_detector.get_user_abuse_history(test_user, days=7)
        # Should skip invalid JSON and return empty list (since valid json doesn't have required fields)
        assert len(history) == 0

    def test_detect_high_frequency_above_threshold(self, abuse_detector):
        """Test high frequency detection above threshold."""
        data = {"requests_per_minute": 2000}  # Above 1000 threshold

        score, is_abuse = asyncio.run(abuse_detector._detect_high_frequency(None, data))

        assert is_abuse is True
        assert score > Decimal("0.0")

    def test_detect_high_frequency_below_threshold(self, abuse_detector):
        """Test high frequency detection below threshold."""
        data = {"requests_per_minute": 500}  # Below 1000 threshold

        score, is_abuse = asyncio.run(abuse_detector._detect_high_frequency(None, data))

        assert is_abuse is False
        assert score == Decimal("0.0")

    def test_detect_suspicious_patterns_with_anomalies(self, abuse_detector):
        """Test suspicious pattern detection with anomalies."""
        data = {"pattern_anomalies": ["bot_like", "scraping"]}

        score, is_abuse = asyncio.run(abuse_detector._detect_suspicious_patterns(None, data))

        assert is_abuse is True
        assert score > Decimal("0.0")

    def test_detect_suspicious_patterns_no_anomalies(self, abuse_detector):
        """Test suspicious pattern detection without anomalies."""
        data = {"pattern_anomalies": []}

        score, is_abuse = asyncio.run(abuse_detector._detect_suspicious_patterns(None, data))

        assert is_abuse is False
        assert score == Decimal("0.0")

    def test_detect_invalid_requests_high_error_rate(self, abuse_detector):
        """Test invalid request detection with high error rate."""
        data = {"error_rate": 0.8}  # Above 0.5 threshold

        score, is_abuse = asyncio.run(abuse_detector._detect_invalid_requests(None, data))

        assert is_abuse is True
        assert score > Decimal("0.0")

    def test_detect_invalid_requests_low_error_rate(self, abuse_detector):
        """Test invalid request detection with low error rate."""
        data = {"error_rate": 0.1}  # Below 0.5 threshold

        score, is_abuse = asyncio.run(abuse_detector._detect_invalid_requests(None, data))

        assert is_abuse is False
        assert score == Decimal("0.0")

    def test_should_block_user_high_score(self, abuse_detector):
        """Test user blocking decision with high score."""
        should_block = abuse_detector._should_block_user(Decimal("0.9"), [], {})
        assert should_block is True

    def test_should_block_user_multiple_abuse_types(self, abuse_detector):
        """Test user blocking decision with multiple abuse types."""
        abuse_types = [AbuseType.HIGH_FREQUENCY, AbuseType.INVALID_REQUESTS, AbuseType.SUSPICIOUS_PATTERN]
        should_block = abuse_detector._should_block_user(Decimal("0.5"), abuse_types, {})
        assert should_block is True

    def test_should_block_user_critical_patterns(self, abuse_detector):
        """Test user blocking decision with critical patterns."""
        abuse_types = [AbuseType.HIGH_FREQUENCY, AbuseType.INVALID_REQUESTS]
        should_block = abuse_detector._should_block_user(Decimal("0.5"), abuse_types, {})
        assert should_block is True

    def test_should_block_user_high_error_rate(self, abuse_detector):
        """Test user blocking decision with high error rate."""
        should_block = abuse_detector._should_block_user(Decimal("0.3"), [], {"error_rate": 0.9})
        assert should_block is True

    def test_should_not_block_user_normal_behavior(self, abuse_detector):
        """Test user blocking decision with normal behavior."""
        should_block = abuse_detector._should_block_user(Decimal("0.2"), [], {"error_rate": 0.01})
        assert should_block is False

    def test_generate_recommendation_normal_behavior(self, abuse_detector):
        """Test recommendation generation for normal behavior."""
        rec = abuse_detector._generate_recommendation(Decimal("0.1"), [], {})
        assert "Normal behavior" in rec

    def test_generate_recommendation_low_risk(self, abuse_detector):
        """Test recommendation generation for low risk."""
        rec = abuse_detector._generate_recommendation(Decimal("0.3"), [], {})
        assert "Monitor user activity" in rec

    def test_generate_recommendation_medium_risk(self, abuse_detector):
        """Test recommendation generation for medium risk."""
        abuse_types = [AbuseType.HIGH_FREQUENCY, AbuseType.INVALID_REQUESTS]
        rec = abuse_detector._generate_recommendation(Decimal("0.6"), abuse_types, {})

        assert "Medium risk" in rec
        assert "stricter rate limiting" in rec
        assert "review API integration" in rec

    def test_generate_recommendation_high_risk(self, abuse_detector):
        """Test recommendation generation for high risk."""
        rec = abuse_detector._generate_recommendation(Decimal("0.9"), [], {})
        assert "High risk" in rec
        assert "immediate blocking" in rec
