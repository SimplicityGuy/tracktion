"""Unit tests for authentication system."""

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from services.tracklist_service.src.auth.authentication import AuthenticationManager
from services.tracklist_service.src.auth.models import TokenPair, UserTier


class TestAuthenticationManager:
    """Test AuthenticationManager class."""

    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager instance."""
        return AuthenticationManager(jwt_secret="test_secret_key_123")

    @pytest.fixture
    def test_user(self, auth_manager):
        """Create test user."""
        return auth_manager.create_user("test@example.com", UserTier.FREE)

    def test_initialization(self):
        """Test authentication manager initialization."""
        auth_manager = AuthenticationManager("secret", "HS256")

        assert auth_manager.jwt_secret == "secret"
        assert auth_manager.jwt_algorithm == "HS256"
        assert auth_manager._users == {}
        assert auth_manager._api_keys == {}

    def test_create_user(self, auth_manager):
        """Test user creation."""
        user = auth_manager.create_user("test@example.com", UserTier.PREMIUM)

        assert user.email == "test@example.com"
        assert user.tier == UserTier.PREMIUM
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)
        assert user.id in auth_manager._users

    def test_get_user(self, auth_manager, test_user):
        """Test user retrieval."""
        retrieved_user = auth_manager.get_user(test_user.id)

        assert retrieved_user is not None
        assert retrieved_user.id == test_user.id
        assert retrieved_user.email == test_user.email

    def test_get_user_not_found(self, auth_manager):
        """Test user retrieval for non-existent user."""
        user = auth_manager.get_user("nonexistent")

        assert user is None

    def test_update_user_tier(self, auth_manager, test_user):
        """Test user tier update."""
        success = auth_manager.update_user_tier(test_user.id, UserTier.ENTERPRISE)

        assert success is True
        updated_user = auth_manager.get_user(test_user.id)
        assert updated_user.tier == UserTier.ENTERPRISE

    def test_update_user_tier_not_found(self, auth_manager):
        """Test user tier update for non-existent user."""
        success = auth_manager.update_user_tier("nonexistent", UserTier.PREMIUM)

        assert success is False

    def test_generate_api_key(self, auth_manager, test_user):
        """Test API key generation."""
        api_key = auth_manager.generate_api_key(test_user.id, "free", "Test Key")

        assert api_key.user_id == test_user.id
        assert api_key.name == "Test Key"
        assert api_key.is_active is True
        assert isinstance(api_key.created_at, datetime)
        assert api_key.permissions is not None

    def test_generate_api_key_user_not_found(self, auth_manager):
        """Test API key generation for non-existent user."""
        with pytest.raises(ValueError, match="User nonexistent not found"):
            auth_manager.generate_api_key("nonexistent", "free")

    def test_validate_api_key_valid(self, auth_manager, test_user):
        """Test API key validation with valid key."""
        api_key = auth_manager.generate_api_key(test_user.id, "free")

        # Find the raw key from storage
        raw_key = None
        for key, stored_key in auth_manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        assert raw_key is not None
        validated_user = auth_manager.validate_api_key(raw_key)
        assert validated_user is not None
        assert validated_user.id == test_user.id

    def test_validate_api_key_invalid(self, auth_manager):
        """Test API key validation with invalid key."""
        user = auth_manager.validate_api_key("invalid_key")

        assert user is None

    def test_validate_api_key_inactive(self, auth_manager, test_user):
        """Test API key validation with inactive key."""
        api_key = auth_manager.generate_api_key(test_user.id, "free")

        # Find and deactivate the key
        for stored_key in auth_manager._api_keys.values():
            if stored_key.key_id == api_key.key_id:
                stored_key.is_active = False
                break

        # Find the raw key from storage
        raw_key = None
        for key, stored_key in auth_manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        user = auth_manager.validate_api_key(raw_key)
        assert user is None

    def test_revoke_api_key(self, auth_manager, test_user):
        """Test API key revocation."""
        api_key = auth_manager.generate_api_key(test_user.id, "free")

        # Find the raw key from storage
        raw_key = None
        for key, stored_key in auth_manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        success = auth_manager.revoke_api_key(raw_key)
        assert success is True

        # Verify key is inactive
        user = auth_manager.validate_api_key(raw_key)
        assert user is None

    def test_revoke_api_key_not_found(self, auth_manager):
        """Test API key revocation for non-existent key."""
        success = auth_manager.revoke_api_key("nonexistent_key")

        assert success is False

    def test_generate_jwt_token(self, auth_manager, test_user):
        """Test JWT token generation."""
        token_pair = auth_manager.generate_jwt_token(test_user)

        assert isinstance(token_pair, TokenPair)
        assert token_pair.access_token
        assert token_pair.refresh_token
        assert token_pair.token_type == "bearer"
        assert token_pair.expires_in == 3600

    def test_generate_jwt_token_custom_expiry(self, auth_manager, test_user):
        """Test JWT token generation with custom expiry."""
        custom_delta = timedelta(hours=2)
        token_pair = auth_manager.generate_jwt_token(test_user, custom_delta)

        assert token_pair.expires_in == 7200  # 2 hours

    def test_validate_jwt_token_valid(self, auth_manager, test_user):
        """Test JWT token validation with valid token."""
        token_pair = auth_manager.generate_jwt_token(test_user)
        validated_user = auth_manager.validate_jwt_token(token_pair.access_token)

        assert validated_user is not None
        assert validated_user.id == test_user.id

    def test_validate_jwt_token_invalid(self, auth_manager):
        """Test JWT token validation with invalid token."""
        user = auth_manager.validate_jwt_token("invalid_token")

        assert user is None

    def test_validate_jwt_token_expired(self, auth_manager, test_user):
        """Test JWT token validation with expired token."""
        # Create expired token
        expired_payload = {
            "sub": test_user.id,
            "exp": datetime.now(UTC) - timedelta(hours=1),  # Expired 1 hour ago
            "iat": datetime.now(UTC) - timedelta(hours=2),
            "type": "access",
        }
        expired_token = jwt.encode(expired_payload, auth_manager.jwt_secret, algorithm="HS256")

        user = auth_manager.validate_jwt_token(expired_token)
        assert user is None

    def test_refresh_jwt_token_valid(self, auth_manager, test_user):
        """Test JWT token refresh with valid refresh token."""

        token_pair = auth_manager.generate_jwt_token(test_user)
        time.sleep(1.1)  # Wait just over a second to ensure different timestamps in JWT
        new_token_pair = auth_manager.refresh_jwt_token(token_pair.refresh_token)

        assert new_token_pair is not None
        assert isinstance(new_token_pair, TokenPair)
        # Tokens should be different due to different timestamps
        assert new_token_pair.access_token != token_pair.access_token

    def test_refresh_jwt_token_invalid(self, auth_manager):
        """Test JWT token refresh with invalid token."""
        new_token_pair = auth_manager.refresh_jwt_token("invalid_refresh_token")

        assert new_token_pair is None

    def test_refresh_jwt_token_wrong_type(self, auth_manager, test_user):
        """Test JWT token refresh with access token (wrong type)."""
        token_pair = auth_manager.generate_jwt_token(test_user)
        new_token_pair = auth_manager.refresh_jwt_token(token_pair.access_token)

        assert new_token_pair is None

    def test_get_default_permissions_free(self, auth_manager):
        """Test default permissions for free tier."""
        permissions = auth_manager._get_default_permissions("free")

        assert permissions["read"] is True
        assert permissions["write"] is False
        assert permissions["admin"] is False

    def test_get_default_permissions_premium(self, auth_manager):
        """Test default permissions for premium tier."""
        permissions = auth_manager._get_default_permissions("premium")

        assert permissions["read"] is True
        assert permissions["write"] is True
        assert permissions["admin"] is False

    def test_get_default_permissions_enterprise(self, auth_manager):
        """Test default permissions for enterprise tier."""
        permissions = auth_manager._get_default_permissions("enterprise")

        assert permissions["read"] is True
        assert permissions["write"] is True
        assert permissions["admin"] is True

    def test_setup_oauth2_provider(self, auth_manager):
        """Test OAuth2 provider setup."""
        config = auth_manager.setup_oauth2_provider("google")

        assert config["provider"] == "google"
        assert "client_id" in config
        assert "redirect_uri" in config
        assert "scopes" in config

    def test_api_key_last_used_update(self, auth_manager, test_user):
        """Test that last_used_at is updated on API key validation."""
        api_key = auth_manager.generate_api_key(test_user.id, "free")

        # Find the raw key from storage
        raw_key = None
        for key, stored_key in auth_manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        # Validate key twice to check timestamp update
        original_last_used = auth_manager._api_keys[raw_key].last_used_at
        auth_manager.validate_api_key(raw_key)
        updated_last_used = auth_manager._api_keys[raw_key].last_used_at

        assert updated_last_used is not None
        if original_last_used:
            assert updated_last_used >= original_last_used
