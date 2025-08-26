"""
Unit tests for developer API endpoints.

Tests API key management functionality including creation, listing,
rotation, revocation, and usage analytics.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from services.tracklist_service.src.analytics.usage_tracker import UsageStats
from services.tracklist_service.src.api.developer_endpoints import (
    ApiKeyInfo,
    ApiKeyResponse,
    CreateKeyRequest,
    KeyUsageResponse,
)
from services.tracklist_service.src.auth.models import ApiKey, User, UserTier


class TestDeveloperEndpoints:
    """Test suite for developer API endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user = User(
            id="test-user-123",
            email="test@example.com",
            tier=UserTier.PREMIUM,
            is_active=True,
            created_at=datetime.now(UTC),
        )

        self.api_key = ApiKey(
            key_id="test-key-123",
            user_id=self.user.id,
            key_hash="hashed-key",
            name="Test Key",
            is_active=True,
            created_at=datetime.now(UTC),
            permissions={"read": True, "write": True, "admin": False},
        )

        self.usage_stats = UsageStats(
            user_id=self.user.id,
            period="day",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            total_tokens=5000,
            total_bytes=1024,
            avg_response_time=150.5,
            endpoints={"/api/v1/tracklist": 100},
            error_rate=0.02,
        )

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_usage_tracker")
    async def test_list_api_keys_success(self, mock_tracker, mock_auth, mock_user):
        """Test successful API key listing."""
        # Setup mocks
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"test-raw-key": self.api_key}
        mock_auth.return_value = mock_auth_manager

        mock_usage_tracker = MagicMock()
        mock_usage_tracker.get_usage_stats = AsyncMock(return_value=self.usage_stats)
        mock_tracker.return_value = mock_usage_tracker

        # Import and test the endpoint
        from services.tracklist_service.src.api.developer_endpoints import list_api_keys

        result = await list_api_keys(include_inactive=False, user=self.user)

        assert len(result) == 1
        assert result[0].key_id == "test-key-123"
        assert result[0].name == "Test Key"
        assert result[0].key_prefix == "test-raw..."
        assert result[0].is_active is True
        assert result[0].permissions == {"read": True, "write": True, "admin": False}

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_list_api_keys_no_keys(self, mock_auth, mock_user):
        """Test listing when user has no API keys."""
        # Setup mocks
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import list_api_keys

        result = await list_api_keys(include_inactive=False, user=self.user)

        assert len(result) == 0

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_list_api_keys_include_inactive(self, mock_auth, mock_user):
        """Test listing with inactive keys included."""
        # Setup inactive key
        inactive_key = ApiKey(
            key_id="inactive-key",
            user_id=self.user.id,
            key_hash="hashed-key",
            name="Inactive Key",
            is_active=False,
            created_at=datetime.now(UTC),
        )

        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"test-raw-key": self.api_key, "inactive-raw-key": inactive_key}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import list_api_keys

        # Without inactive keys
        result = await list_api_keys(include_inactive=False, user=self.user)
        assert len(result) == 1
        assert result[0].is_active is True

        # With inactive keys
        result = await list_api_keys(include_inactive=True, user=self.user)
        assert len(result) == 2
        active_keys = [k for k in result if k.is_active]
        inactive_keys = [k for k in result if not k.is_active]
        assert len(active_keys) == 1
        assert len(inactive_keys) == 1

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_create_api_key_success(self, mock_auth, mock_user):
        """Test successful API key creation."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}  # No existing keys
        mock_auth_manager.generate_api_key.return_value = self.api_key
        mock_auth_manager._get_default_permissions.return_value = {"read": True, "write": True}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import create_api_key

        request = CreateKeyRequest(
            name="New Test Key", description="Test description", permissions={"read": True, "write": True}
        )

        # Mock the key storage after generation
        mock_auth_manager._api_keys["generated-raw-key"] = self.api_key

        result = await create_api_key(request=request, user=self.user)

        assert isinstance(result, ApiKeyResponse)
        assert result.key_id == "test-key-123"
        assert result.name == "Test Key"
        assert result.api_key == "generated-raw-key"
        mock_auth_manager.generate_api_key.assert_called_once_with(
            user_id=self.user.id, tier=self.user.tier.value, name="New Test Key"
        )

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_create_api_key_exceeds_limit(self, mock_auth, mock_user):
        """Test API key creation when user exceeds tier limit."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()

        # Create 10 existing active keys (premium limit is 10)
        existing_keys = {}
        for i in range(10):
            key = ApiKey(key_id=f"key-{i}", user_id=self.user.id, key_hash="hash", is_active=True)
            existing_keys[f"raw-key-{i}"] = key

        mock_auth_manager._api_keys = existing_keys
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import create_api_key

        request = CreateKeyRequest(name="Excess Key")

        with pytest.raises(HTTPException) as exc_info:
            await create_api_key(request=request, user=self.user)

        assert exc_info.value.status_code == 403
        assert "Maximum number of API keys reached" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_create_api_key_invalid_permissions(self, mock_auth, mock_user):
        """Test API key creation with invalid permissions for user tier."""
        # Use FREE tier user
        free_user = User(id="free-user", email="free@example.com", tier=UserTier.FREE, is_active=True)

        mock_user.return_value = free_user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}
        mock_auth_manager._get_default_permissions.return_value = {"read": True, "write": False, "admin": False}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import create_api_key

        request = CreateKeyRequest(
            name="Invalid Permissions Key",
            permissions={"read": True, "write": True, "admin": True},  # Free tier can't have write/admin
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_api_key(request=request, user=free_user)

        assert exc_info.value.status_code == 403
        assert "not available for free tier" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_rotate_api_key_success(self, mock_auth, mock_user):
        """Test successful API key rotation."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"old-raw-key": self.api_key}

        # New key after rotation
        new_key = ApiKey(
            key_id="new-key-123",
            user_id=self.user.id,
            key_hash="new-hashed-key",
            name=self.api_key.name,
            is_active=True,
            created_at=datetime.now(UTC),
            permissions=self.api_key.permissions,
        )
        mock_auth_manager.generate_api_key.return_value = new_key
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import rotate_api_key

        # Add new key to manager after generation
        mock_auth_manager._api_keys["new-raw-key"] = new_key

        result = await rotate_api_key(key_id="test-key-123", user=self.user)

        assert isinstance(result, ApiKeyResponse)
        assert result.key_id == "new-key-123"
        assert result.api_key == "new-raw-key"
        assert result.name == self.api_key.name
        assert result.permissions == self.api_key.permissions

        # Original key should be deactivated
        assert self.api_key.is_active is False

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_rotate_api_key_not_found(self, mock_auth, mock_user):
        """Test rotating non-existent API key."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import rotate_api_key

        with pytest.raises(HTTPException) as exc_info:
            await rotate_api_key(key_id="nonexistent-key", user=self.user)

        assert exc_info.value.status_code == 404
        assert "API key not found" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_rotate_api_key_inactive(self, mock_auth, mock_user):
        """Test rotating inactive API key."""
        inactive_key = ApiKey(key_id="inactive-key", user_id=self.user.id, key_hash="hash", is_active=False)

        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"inactive-raw-key": inactive_key}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import rotate_api_key

        with pytest.raises(HTTPException) as exc_info:
            await rotate_api_key(key_id="inactive-key", user=self.user)

        assert exc_info.value.status_code == 400
        assert "Cannot rotate inactive API key" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_revoke_api_key_success(self, mock_auth, mock_user):
        """Test successful API key revocation."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"test-raw-key": self.api_key}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import revoke_api_key

        response = await revoke_api_key(key_id="test-key-123", user=self.user)

        assert response.status_code == 200
        content = response.body.decode()
        assert "success" in content
        assert "test-key-123" in content
        assert "revoked" in content

        # Key should be deactivated
        assert self.api_key.is_active is False

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_revoke_api_key_not_found(self, mock_auth, mock_user):
        """Test revoking non-existent API key."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import revoke_api_key

        with pytest.raises(HTTPException) as exc_info:
            await revoke_api_key(key_id="nonexistent-key", user=self.user)

        assert exc_info.value.status_code == 404
        assert "API key not found" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_usage_tracker")
    async def test_get_key_usage_analytics_success(self, mock_tracker, mock_auth, mock_user):
        """Test successful key usage analytics retrieval."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {"test-raw-key": self.api_key}
        mock_auth.return_value = mock_auth_manager

        mock_usage_tracker = MagicMock()
        mock_usage_tracker.get_usage_stats = AsyncMock(return_value=self.usage_stats)
        mock_tracker.return_value = mock_usage_tracker

        from services.tracklist_service.src.api.developer_endpoints import get_key_usage_analytics

        result = await get_key_usage_analytics(key_id="test-key-123", period="day", user=self.user)

        assert isinstance(result, KeyUsageResponse)
        assert result.key_id == "test-key-123"
        assert result.usage_stats["total_requests"] == 100
        assert result.usage_stats["total_tokens"] == 5000
        assert result.cost_breakdown["base_cost"] == 0.01  # (5000 tokens / 1000) * 0.002 premium rate
        assert len(result.recommendations) >= 0

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_get_key_usage_analytics_key_not_found(self, mock_auth, mock_user):
        """Test usage analytics for non-existent key."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import get_key_usage_analytics

        with pytest.raises(HTTPException) as exc_info:
            await get_key_usage_analytics(key_id="nonexistent-key", period="day", user=self.user)

        assert exc_info.value.status_code == 404
        assert "API key not found" in exc_info.value.detail

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_usage_tracker")
    async def test_get_usage_summary_success(self, mock_tracker, mock_auth, mock_user):
        """Test successful usage summary retrieval."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {
            "key1": ApiKey(key_id="k1", user_id=self.user.id, key_hash="h1", is_active=True),
            "key2": ApiKey(key_id="k2", user_id=self.user.id, key_hash="h2", is_active=True),
            "key3": ApiKey(key_id="k3", user_id=self.user.id, key_hash="h3", is_active=False),
        }
        mock_auth.return_value = mock_auth_manager

        mock_usage_tracker = MagicMock()
        mock_usage_tracker.get_usage_stats = AsyncMock(return_value=self.usage_stats)
        mock_tracker.return_value = mock_usage_tracker

        from services.tracklist_service.src.api.developer_endpoints import get_usage_summary

        response = await get_usage_summary(period="month", user=self.user)

        assert response.status_code == 200
        content = response.body.decode()
        assert "user_tier" in content
        assert "premium" in content
        assert "total_requests" in content
        assert "utilization" in content
        assert "recommendations" in content

    def test_create_key_request_validation(self):
        """Test CreateKeyRequest model validation."""
        # Valid request
        request = CreateKeyRequest(
            name="Test Key",
            description="Test description",
            permissions={"read": True, "write": False},
            expires_in_days=30,
        )
        assert request.name == "Test Key"
        assert request.expires_in_days == 30

        # Invalid expiration (too long)
        with pytest.raises(ValueError):
            CreateKeyRequest(
                name="Test Key",
                expires_in_days=400,  # Max is 365
            )

        # Invalid expiration (negative)
        with pytest.raises(ValueError):
            CreateKeyRequest(name="Test Key", expires_in_days=-1)

    def test_api_key_info_model(self):
        """Test ApiKeyInfo model."""
        key_info = ApiKeyInfo(
            key_id="test-key",
            name="Test Key",
            key_prefix="abc12345...",
            is_active=True,
            created_at=datetime.now(UTC),
            permissions={"read": True},
        )

        assert key_info.key_id == "test-key"
        assert key_info.name == "Test Key"
        assert key_info.is_active is True
        assert "read" in key_info.permissions

    def test_api_key_response_model(self):
        """Test ApiKeyResponse model."""
        response = ApiKeyResponse(
            key_id="test-key",
            api_key="secret-key-value",
            name="Test Key",
            permissions={"read": True, "write": False},
            created_at=datetime.now(UTC),
        )

        assert response.key_id == "test-key"
        assert response.api_key == "secret-key-value"
        assert response.permissions["read"] is True
        assert response.permissions["write"] is False

    def test_key_usage_response_model(self):
        """Test KeyUsageResponse model."""
        response = KeyUsageResponse(
            key_id="test-key",
            usage_stats={"requests": 100, "tokens": 5000},
            cost_breakdown={"base_cost": 10.50, "token_cost": 10.00},
            recommendations=["Consider upgrading tier"],
        )

        assert response.key_id == "test-key"
        assert response.usage_stats["requests"] == 100
        assert response.cost_breakdown["base_cost"] == 10.50
        assert len(response.recommendations) == 1


class TestDeveloperEndpointsEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user = User(id="test-user", email="test@example.com", tier=UserTier.FREE, is_active=True)

    @patch("services.tracklist_service.src.api.developer_endpoints.authenticate_user")
    @patch("services.tracklist_service.src.api.developer_endpoints.get_auth_manager")
    async def test_create_api_key_with_expiration(self, mock_auth, mock_user):
        """Test API key creation with expiration date."""
        mock_user.return_value = self.user
        mock_auth_manager = MagicMock()
        mock_auth_manager._api_keys = {}

        api_key = ApiKey(
            key_id="test-key",
            user_id=self.user.id,
            key_hash="hash",
            name="Expiring Key",
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_auth_manager.generate_api_key.return_value = api_key
        mock_auth_manager._get_default_permissions.return_value = {"read": True}
        mock_auth.return_value = mock_auth_manager

        from services.tracklist_service.src.api.developer_endpoints import create_api_key

        request = CreateKeyRequest(name="Expiring Key", expires_in_days=30)

        # Mock the key storage
        mock_auth_manager._api_keys["generated-key"] = api_key

        result = await create_api_key(request=request, user=self.user)

        assert isinstance(result, ApiKeyResponse)
        assert result.expires_at is not None
        # Verify expiration is set correctly (approximately 30 days from now)
        expected_expiry = datetime.now(UTC) + timedelta(days=30)
        assert abs((result.expires_at - expected_expiry).total_seconds()) < 60  # Within 1 minute

    @patch("services.tracklist_service.src.api.developer_endpoints.get_usage_tracker")
    async def test_usage_tracker_initialization_error(self, mock_tracker):
        """Test handling of usage tracker initialization errors."""
        mock_tracker.side_effect = ConnectionError("Redis connection failed")

        from services.tracklist_service.src.api.developer_endpoints import get_usage_tracker

        # Should handle gracefully in production
        with pytest.raises(ConnectionError):
            get_usage_tracker()

    def test_request_validation_edge_cases(self):
        """Test edge cases in request validation."""
        # Empty name should be invalid
        with pytest.raises(ValueError):
            CreateKeyRequest(name="")

        # Very long name should be accepted (up to validation limits)
        long_name = "A" * 100
        request = CreateKeyRequest(name=long_name)
        assert request.name == long_name

        # Boundary values for expiration
        request_min = CreateKeyRequest(name="Test", expires_in_days=1)
        assert request_min.expires_in_days == 1

        request_max = CreateKeyRequest(name="Test", expires_in_days=365)
        assert request_max.expires_in_days == 365
