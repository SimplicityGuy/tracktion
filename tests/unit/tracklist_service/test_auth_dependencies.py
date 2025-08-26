"""Unit tests for authentication dependencies."""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from services.tracklist_service.src.auth.authentication import AuthenticationManager
from services.tracklist_service.src.auth.dependencies import (
    authenticate_request,
    authenticate_user,
    get_auth_manager,
    require_permission,
    require_tier,
    set_auth_manager,
)
from services.tracklist_service.src.auth.models import UserTier


class TestAuthDependencies:
    """Test authentication dependency functions."""

    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager instance."""
        manager = AuthenticationManager(jwt_secret="test_secret")
        set_auth_manager(manager)
        return manager

    @pytest.fixture
    def test_user(self, auth_manager):
        """Create test user."""
        return auth_manager.create_user("test@example.com", UserTier.PREMIUM)

    @pytest.mark.asyncio
    async def test_authenticate_request_api_key(self, auth_manager, test_user):
        """Test request authentication with API key."""
        api_key = auth_manager.generate_api_key(test_user.id, "premium")

        # Find the raw key from storage
        raw_key = None
        for key, stored_key in auth_manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        user = await authenticate_request(x_api_key=raw_key)

        assert user is not None
        assert user.id == test_user.id

    @pytest.mark.asyncio
    async def test_authenticate_request_jwt(self, auth_manager, test_user):
        """Test request authentication with JWT token."""
        token_pair = auth_manager.generate_jwt_token(test_user)
        auth_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_pair.access_token)

        user = await authenticate_request(authorization=auth_credentials)

        assert user is not None
        assert user.id == test_user.id

    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_api_key(self, auth_manager):
        """Test request authentication with invalid API key."""
        with pytest.raises(HTTPException) as exc_info:
            await authenticate_request(x_api_key="invalid_key", authorization=None)

        assert exc_info.value.status_code == 401
        assert "Invalid or missing authentication" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_jwt(self, auth_manager):
        """Test request authentication with invalid JWT token."""
        auth_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")

        with pytest.raises(HTTPException) as exc_info:
            await authenticate_request(authorization=auth_credentials)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticate_request_no_credentials(self, auth_manager):
        """Test request authentication with no credentials."""
        with pytest.raises(HTTPException) as exc_info:
            await authenticate_request(x_api_key=None, authorization=None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticate_user_valid_jwt(self, auth_manager, test_user):
        """Test user authentication with valid JWT token."""
        token_pair = auth_manager.generate_jwt_token(test_user)
        auth_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_pair.access_token)

        user = await authenticate_user(authorization=auth_credentials)

        assert user is not None
        assert user.id == test_user.id

    @pytest.mark.asyncio
    async def test_authenticate_user_no_token(self, auth_manager):
        """Test user authentication with no token."""
        with pytest.raises(HTTPException) as exc_info:
            await authenticate_user(authorization=None)

        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_scheme(self, auth_manager):
        """Test user authentication with invalid auth scheme."""
        auth_credentials = HTTPAuthorizationCredentials(scheme="Basic", credentials="some_credentials")

        with pytest.raises(HTTPException) as exc_info:
            await authenticate_user(authorization=auth_credentials)

        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_token(self, auth_manager):
        """Test user authentication with invalid JWT token."""
        auth_credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")

        with pytest.raises(HTTPException) as exc_info:
            await authenticate_user(authorization=auth_credentials)

        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_tier_sufficient(self, auth_manager, test_user):
        """Test tier requirement with sufficient tier."""

        # Mock authenticate_request to return our test user
        async def mock_auth():
            return test_user

        check_tier = await require_tier("free")

        # Create a user with premium tier (higher than free)
        user = test_user  # Premium tier

        # This should pass since premium >= free
        result = check_tier(user)
        assert result.id == test_user.id

    @pytest.mark.asyncio
    async def test_require_tier_insufficient(self, auth_manager):
        """Test tier requirement with insufficient tier."""
        # Create free tier user
        free_user = auth_manager.create_user("free@example.com", UserTier.FREE)

        check_tier = await require_tier("enterprise")

        with pytest.raises(HTTPException) as exc_info:
            check_tier(free_user)

        assert exc_info.value.status_code == 403
        assert "Tier 'enterprise' or higher required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_permission(self, auth_manager, test_user):
        """Test permission requirement."""
        check_permission_func = await require_permission("read")

        # For now, this should just return the user since permissions are placeholder
        result = check_permission_func(test_user)
        assert result.id == test_user.id

    def test_get_auth_manager_not_set(self):
        """Test getting auth manager when not set."""
        # Reset the global manager
        set_auth_manager(None)

        with pytest.raises(HTTPException) as exc_info:
            get_auth_manager()

        assert exc_info.value.status_code == 500
        assert "Authentication manager not initialized" in exc_info.value.detail

    def test_set_get_auth_manager(self):
        """Test setting and getting auth manager."""
        manager = AuthenticationManager("test_secret")
        set_auth_manager(manager)

        retrieved_manager = get_auth_manager()
        assert retrieved_manager is manager
