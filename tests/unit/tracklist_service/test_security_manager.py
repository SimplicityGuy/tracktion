"""Unit tests for SecurityManager."""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from services.tracklist_service.src.auth.models import User, UserTier
from services.tracklist_service.src.security.models import (
    AbuseScore,
    AbuseType,
    AccessRuleType,
    AuditEventType,
    AuditLog,
    IPAccessRule,
    SecurityConfig,
)
from services.tracklist_service.src.security.security_manager import SecurityManager


@pytest.fixture
def security_config():
    """Create test security configuration."""
    return SecurityConfig(
        hmac_secret_key="test-secret-key",
        max_signature_age=300,
        ip_whitelist_enabled=False,
        ip_blacklist_enabled=True,
        abuse_detection_enabled=True,
        audit_logging_enabled=True,
    )


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return AsyncMock()


@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        id="test-user-123",
        email="test@example.com",
        tier=UserTier.PREMIUM,
        is_active=True,
    )


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/api/v1/test"
    request.url.query = ""
    request.headers = {"user-agent": "test-client/1.0"}
    request.client.host = "192.168.1.100"
    return request


@pytest.fixture
def security_manager(mock_redis, security_config):
    """Create SecurityManager instance."""
    return SecurityManager(mock_redis, security_config)


class TestSecurityManager:
    """Test SecurityManager functionality."""

    def test_initialization(self, security_manager, security_config):
        """Test SecurityManager initialization."""
        assert security_manager.config == security_config
        assert security_manager.abuse_detector is not None
        assert security_manager.ip_rules_key == "security:ip_rules"
        assert security_manager.audit_logs_key == "security:audit_logs"
        assert security_manager.blocked_users_key == "security:blocked_users"

    @pytest.mark.asyncio
    async def test_verify_request_signature_valid(self, security_manager, mock_request):
        """Test valid HMAC signature verification."""
        # Setup request data
        timestamp = str(int(datetime.now(UTC).timestamp()))
        body = b'{"test": "data"}'

        # Calculate expected signature
        payload = f"GET\n/api/v1/test\n\n{timestamp}\n{body.decode('utf-8')}"
        expected_signature = hmac.new(b"test-secret-key", payload.encode("utf-8"), hashlib.sha256).hexdigest()

        # Setup mock request
        mock_request.headers = {
            "X-Timestamp": timestamp,
            "X-Signature": expected_signature,
            "user-agent": "test-client/1.0",
        }
        mock_request.body = AsyncMock(return_value=body)

        result = await security_manager.verify_request_signature(mock_request)
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_request_signature_invalid(self, security_manager, mock_request):
        """Test invalid HMAC signature verification."""
        timestamp = str(int(datetime.now(UTC).timestamp()))

        mock_request.headers = {
            "X-Timestamp": timestamp,
            "X-Signature": "invalid-signature",
            "user-agent": "test-client/1.0",
        }
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')

        with patch.object(security_manager, "log_security_violation") as mock_log:
            result = await security_manager.verify_request_signature(mock_request)
            assert result is False
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_request_signature_expired(self, security_manager, mock_request):
        """Test expired signature verification."""
        # Use timestamp from 10 minutes ago (expired)
        old_timestamp = int(datetime.now(UTC).timestamp()) - 600
        timestamp = str(old_timestamp)

        mock_request.headers = {
            "X-Timestamp": timestamp,
            "X-Signature": "some-signature",
            "user-agent": "test-client/1.0",
        }
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')

        result = await security_manager.verify_request_signature(mock_request)
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_request_signature_missing_headers(self, security_manager, mock_request):
        """Test signature verification with missing headers."""
        mock_request.headers = {"user-agent": "test-client/1.0"}
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')

        result = await security_manager.verify_request_signature(mock_request)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_ip_access_allowed_default(self, security_manager):
        """Test IP access check with default allow policy."""
        result = await security_manager.check_ip_access("192.168.1.100")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_ip_access_blacklisted(self, security_manager, mock_redis):
        """Test IP access check with blacklisted IP."""
        # Setup blacklist rule in Redis
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            reason="Test blacklist",
        )
        mock_redis.hgetall.return_value = rule.to_dict()

        result = await security_manager.check_ip_access("192.168.1.100")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_ip_access_whitelisted(self, security_manager, mock_redis):
        """Test IP access check with whitelisted IP."""
        # Enable whitelist mode
        security_manager.config.ip_whitelist_enabled = True
        security_manager.config.default_ip_access = False

        # Setup whitelist rule in Redis
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.WHITELIST,
            reason="Test whitelist",
        )
        mock_redis.hgetall.return_value = rule.to_dict()

        result = await security_manager.check_ip_access("192.168.1.100")
        assert result is True

    @pytest.mark.asyncio
    async def test_check_ip_access_expired_rule(self, security_manager, mock_redis):
        """Test IP access check with expired rule."""
        # Create expired rule
        expired_time = datetime.now(UTC) - timedelta(hours=1)
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            expires_at=expired_time,
        )
        mock_redis.hgetall.return_value = rule.to_dict()

        result = await security_manager.check_ip_access("192.168.1.100")
        assert result is True  # Rule expired, so allow access
        mock_redis.hdel.assert_called_once()  # Should clean up expired rule

    @pytest.mark.asyncio
    async def test_add_ip_rule_success(self, security_manager, mock_redis):
        """Test adding IP access rule."""
        rule = await security_manager.add_ip_rule(
            "192.168.1.100",
            AccessRuleType.BLACKLIST,
            user_id="admin-123",
            reason="Suspicious activity",
            expires_in_hours=24,
        )

        assert rule.ip_address == "192.168.1.100"
        assert rule.rule_type == AccessRuleType.BLACKLIST
        assert rule.user_id == "admin-123"
        assert rule.reason == "Suspicious activity"
        assert rule.expires_at is not None

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_ip_rule_invalid_ip(self, security_manager):
        """Test adding IP rule with invalid IP address."""
        with pytest.raises(ValueError, match="Invalid IP address format"):
            await security_manager.add_ip_rule("invalid-ip", AccessRuleType.BLACKLIST)

    @pytest.mark.asyncio
    async def test_remove_ip_rule_success(self, security_manager, mock_redis):
        """Test removing IP access rule."""
        mock_redis.delete.return_value = 1

        result = await security_manager.remove_ip_rule("192.168.1.100")
        assert result is True
        mock_redis.delete.assert_called_once_with("security:ip_rules:192.168.1.100")

    @pytest.mark.asyncio
    async def test_remove_ip_rule_not_found(self, security_manager, mock_redis):
        """Test removing non-existent IP rule."""
        mock_redis.delete.return_value = 0

        result = await security_manager.remove_ip_rule("192.168.1.100")
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_abuse(self, security_manager, test_user, mock_request):
        """Test abuse detection."""
        # Mock abuse detector
        expected_score = AbuseScore(
            user_id=test_user.id,
            score=0.3,
            abuse_types=[AbuseType.HIGH_FREQUENCY],
            details={"frequency": {"score": "0.3", "rpm": 150}},
            should_block=False,
            recommendation="Monitor user activity",
        )

        with patch.object(
            security_manager.abuse_detector,
            "analyze_user_behavior",
            return_value=expected_score,
        ):
            result = await security_manager.detect_abuse(test_user, mock_request)
            assert result == expected_score

    @pytest.mark.asyncio
    async def test_detect_abuse_disabled(self, security_manager, test_user, mock_request):
        """Test abuse detection when disabled."""
        security_manager.config.abuse_detection_enabled = False

        result = await security_manager.detect_abuse(test_user, mock_request)
        assert result.score == 0.0
        assert result.recommendation == "Abuse detection disabled"

    @pytest.mark.asyncio
    async def test_auto_block_user(self, security_manager, mock_redis, test_user):
        """Test automatic user blocking."""
        abuse_score = AbuseScore(
            user_id=test_user.id,
            score=0.9,
            abuse_types=[AbuseType.HIGH_FREQUENCY, AbuseType.INVALID_REQUESTS],
            details={},
            should_block=True,
            recommendation="Block user",
        )

        await security_manager.auto_block_user(test_user, "High abuse score", 24, abuse_score)

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_user_blocked_true(self, security_manager, mock_redis, test_user):
        """Test checking blocked user status - blocked."""
        future_time = datetime.now(UTC) + timedelta(hours=1)
        block_data = {
            "user_id": test_user.id,
            "reason": "Security violation",
            "blocked_until": future_time.isoformat(),
        }
        mock_redis.hgetall.return_value = block_data

        is_blocked, reason = await security_manager.is_user_blocked(test_user)
        assert is_blocked is True
        assert reason == "Security violation"

    @pytest.mark.asyncio
    async def test_is_user_blocked_false(self, security_manager, mock_redis, test_user):
        """Test checking blocked user status - not blocked."""
        mock_redis.hgetall.return_value = {}

        is_blocked, reason = await security_manager.is_user_blocked(test_user)
        assert is_blocked is False
        assert reason is None

    @pytest.mark.asyncio
    async def test_is_user_blocked_expired(self, security_manager, mock_redis, test_user):
        """Test checking blocked user status - block expired."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        block_data = {
            "user_id": test_user.id,
            "reason": "Security violation",
            "blocked_until": past_time.isoformat(),
        }
        mock_redis.hgetall.return_value = block_data

        is_blocked, reason = await security_manager.is_user_blocked(test_user)
        assert is_blocked is False
        assert reason is None
        mock_redis.delete.assert_called_once()  # Should clean up expired block

    @pytest.mark.asyncio
    async def test_log_api_access(self, security_manager, mock_redis, mock_request, test_user):
        """Test API access logging."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        await security_manager.log_api_access(mock_request, mock_response, test_user, "api-key-123", "jwt-token-456")

        # Should store audit log in Redis
        assert mock_redis.hset.call_count == 1
        assert mock_redis.expire.call_count == 2  # Log key and timeline key
        assert mock_redis.zadd.call_count == 1

    @pytest.mark.asyncio
    async def test_log_api_access_disabled(self, security_manager, mock_redis, mock_request, test_user):
        """Test API access logging when disabled."""
        security_manager.config.audit_logging_enabled = False
        mock_response = MagicMock()

        await security_manager.log_api_access(mock_request, mock_response, test_user)

        # Should not log anything
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_security_violation(self, security_manager, mock_redis, mock_request, test_user):
        """Test security violation logging."""
        await security_manager.log_security_violation(
            mock_request, test_user, "invalid_signature", {"timestamp": "123456789"}
        )

        # Should store violation in Redis
        assert mock_redis.hset.call_count == 1
        assert mock_redis.expire.call_count == 2  # Log key and timeline key
        assert mock_redis.zadd.call_count == 1

    @pytest.mark.asyncio
    async def test_get_audit_logs(self, security_manager, mock_redis):
        """Test retrieving audit logs."""
        # Mock Redis responses
        log_data = {
            "event_id": "test-event-123",
            "event_type": "api_access",
            "user_id": "user-123",
            "ip_address": "192.168.1.100",
            "endpoint": "/api/v1/test",
            "method": "GET",
            "status_code": "200",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        mock_redis.zrangebyscore.return_value = ["test-event-123"]
        mock_redis.hgetall.return_value = log_data

        start_date = datetime.now(UTC) - timedelta(hours=1)
        end_date = datetime.now(UTC)

        logs = await security_manager.get_audit_logs(start_date, end_date)

        assert len(logs) == 1
        assert logs[0].event_id == "test-event-123"
        assert logs[0].event_type == AuditEventType.API_ACCESS

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, security_manager, mock_redis):
        """Test health check - healthy state."""
        mock_redis.ping.return_value = True
        mock_redis.keys.side_effect = [["rule1", "rule2"], ["blocked1"]]

        health = await security_manager.health_check()

        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["stats"]["ip_rules"] == 2
        assert health["stats"]["blocked_users"] == 1

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, security_manager, mock_redis):
        """Test health check - unhealthy state."""
        mock_redis.ping.side_effect = ConnectionError("Redis connection failed")

        health = await security_manager.health_check()

        assert health["status"] == "unhealthy"
        assert health["redis_connected"] is False
        assert "error" in health

    def test_get_client_ip_forwarded_for(self, security_manager, mock_request):
        """Test getting client IP from X-Forwarded-For header."""
        mock_request.headers = {"x-forwarded-for": "203.0.113.1, 192.168.1.100"}

        ip = security_manager._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_real_ip(self, security_manager, mock_request):
        """Test getting client IP from X-Real-IP header."""
        mock_request.headers = {"x-real-ip": "203.0.113.1"}

        ip = security_manager._get_client_ip(mock_request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_direct(self, security_manager, mock_request):
        """Test getting client IP from direct connection."""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"

        ip = security_manager._get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_unknown(self, security_manager):
        """Test getting client IP when unknown."""
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.client = None

        ip = security_manager._get_client_ip(mock_request)
        assert ip == "unknown"


class TestSecurityModels:
    """Test security data models."""

    def test_security_config_defaults(self):
        """Test SecurityConfig default values."""
        config = SecurityConfig(hmac_secret_key="test-key")

        assert config.hmac_secret_key == "test-key"
        assert config.max_signature_age == 300
        assert config.ip_whitelist_enabled is False
        assert config.ip_blacklist_enabled is True
        assert config.abuse_detection_enabled is True
        assert config.audit_logging_enabled is True
        assert config.blocked_countries == []

    def test_ip_access_rule_creation(self):
        """Test IPAccessRule creation."""
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            reason="Test rule",
        )

        assert rule.ip_address == "192.168.1.100"
        assert rule.rule_type == AccessRuleType.BLACKLIST
        assert rule.reason == "Test rule"
        assert rule.is_active is True
        assert rule.created_at is not None

    def test_ip_access_rule_expired(self):
        """Test IPAccessRule expiration check."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            expires_at=past_time,
        )

        assert rule.is_expired() is True

    def test_ip_access_rule_not_expired(self):
        """Test IPAccessRule not expired."""
        future_time = datetime.now(UTC) + timedelta(hours=1)
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            expires_at=future_time,
        )

        assert rule.is_expired() is False

    def test_ip_access_rule_no_expiry(self):
        """Test IPAccessRule with no expiry."""
        rule = IPAccessRule(ip_address="192.168.1.100", rule_type=AccessRuleType.BLACKLIST)

        assert rule.is_expired() is False

    def test_ip_access_rule_serialization(self):
        """Test IPAccessRule to_dict and from_dict."""
        rule = IPAccessRule(
            ip_address="192.168.1.100",
            rule_type=AccessRuleType.BLACKLIST,
            reason="Test rule",
        )

        rule_dict = rule.to_dict()
        restored_rule = IPAccessRule.from_dict(rule_dict)

        assert restored_rule.ip_address == rule.ip_address
        assert restored_rule.rule_type == rule.rule_type
        assert restored_rule.reason == rule.reason

    def test_abuse_score_risk_levels(self):
        """Test AbuseScore risk level methods."""
        high_risk = AbuseScore(
            user_id="user1",
            score=0.9,
            abuse_types=[],
            details={},
            should_block=True,
            recommendation="Block",
        )

        medium_risk = AbuseScore(
            user_id="user2",
            score=0.6,
            abuse_types=[],
            details={},
            should_block=False,
            recommendation="Monitor",
        )

        low_risk = AbuseScore(
            user_id="user3",
            score=0.2,
            abuse_types=[],
            details={},
            should_block=False,
            recommendation="Normal",
        )

        assert high_risk.is_high_risk() is True
        assert high_risk.is_medium_risk() is False

        assert medium_risk.is_high_risk() is False
        assert medium_risk.is_medium_risk() is True

        assert low_risk.is_high_risk() is False
        assert low_risk.is_medium_risk() is False

    def test_audit_log_creation(self):
        """Test AuditLog creation."""
        log = AuditLog(
            event_id="test-event-123",
            event_type=AuditEventType.API_ACCESS,
            user_id="user-123",
            ip_address="192.168.1.100",
            user_agent="test-agent",
            endpoint="/api/v1/test",
            method="GET",
            status_code=200,
        )

        assert log.event_id == "test-event-123"
        assert log.event_type == AuditEventType.API_ACCESS
        assert log.user_id == "user-123"
        assert log.timestamp is not None
        assert log.tags == []
        assert log.details == {}

    def test_audit_log_add_tag(self):
        """Test adding tags to AuditLog."""
        log = AuditLog(
            event_id="test",
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=None,
            ip_address="192.168.1.100",
            user_agent="test-agent",
            endpoint="/api/v1/test",
            method="GET",
            status_code=400,
        )

        log.add_tag("security")
        log.add_tag("violation")
        log.add_tag("security")  # Duplicate

        assert len(log.tags) == 2
        assert "security" in log.tags
        assert "violation" in log.tags

    def test_audit_log_add_detail(self):
        """Test adding details to AuditLog."""
        log = AuditLog(
            event_id="test",
            event_type=AuditEventType.AUTHENTICATION,
            user_id="user-123",
            ip_address="192.168.1.100",
            user_agent="test-agent",
            endpoint="/api/v1/auth",
            method="POST",
            status_code=200,
        )

        log.add_detail("method", "api_key")
        log.add_detail("key_id", "key-123")

        assert log.details["method"] == "api_key"
        assert log.details["key_id"] == "key-123"

    def test_audit_log_serialization(self):
        """Test AuditLog to_dict and from_dict."""
        log = AuditLog(
            event_id="test-event-123",
            event_type=AuditEventType.API_ACCESS,
            user_id="user-123",
            ip_address="192.168.1.100",
            user_agent="test-agent",
            endpoint="/api/v1/test",
            method="GET",
            status_code=200,
        )

        log.add_tag("api")
        log.add_detail("response_time", 0.123)

        log_dict = log.to_dict()
        restored_log = AuditLog.from_dict(log_dict)

        assert restored_log.event_id == log.event_id
        assert restored_log.event_type == log.event_type
        assert restored_log.user_id == log.user_id
        assert restored_log.tags == log.tags
        assert restored_log.details == log.details
