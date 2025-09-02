"""Tests for admin authentication system."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException

from services.tracklist_service.src.auth.admin_auth import (
    ALGORITHM,
    SECRET_KEY,
    AdminUser,
    TokenResponse,
    authenticate_admin,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_tokens,
    get_current_admin,
    get_password_hash,
    require_parser_admin,
    require_readonly,
    require_super_admin,
    verify_password,
)


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_password_hash_and_verify(self):
        """Test password hashing and verification."""
        plain_password = "test_password_123"

        # Hash password
        hashed = get_password_hash(plain_password)

        # Verify correct password
        assert verify_password(plain_password, hashed) is True

        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False

        # Verify hash is different each time (due to salt)
        hashed2 = get_password_hash(plain_password)
        assert hashed != hashed2
        assert verify_password(plain_password, hashed2) is True


class TestAuthentication:
    """Test admin authentication."""

    def test_authenticate_admin_success(self):
        """Test successful admin authentication."""
        with patch(
            "services.tracklist_service.src.auth.admin_auth.ADMIN_USERS",
            {
                "test_admin": {
                    "username": "test_admin",
                    "hashed_password": get_password_hash("test_password"),
                    "role": "parser_admin",
                    "is_active": True,
                }
            },
        ):
            user = authenticate_admin("test_admin", "test_password")

            assert user is not None
            assert user.username == "test_admin"
            assert user.role == "parser_admin"
            assert user.is_active is True

    def test_authenticate_admin_wrong_password(self):
        """Test authentication with wrong password."""
        with patch(
            "services.tracklist_service.src.auth.admin_auth.ADMIN_USERS",
            {
                "test_admin": {
                    "username": "test_admin",
                    "hashed_password": get_password_hash("test_password"),
                    "role": "parser_admin",
                    "is_active": True,
                }
            },
        ):
            user = authenticate_admin("test_admin", "wrong_password")
            assert user is None

    def test_authenticate_admin_user_not_found(self):
        """Test authentication with non-existent user."""
        user = authenticate_admin("non_existent_user", "any_password")
        assert user is None

    def test_authenticate_admin_inactive_user(self):
        """Test authentication with inactive user."""
        with patch(
            "services.tracklist_service.src.auth.admin_auth.ADMIN_USERS",
            {
                "inactive_admin": {
                    "username": "inactive_admin",
                    "hashed_password": get_password_hash("test_password"),
                    "role": "parser_admin",
                    "is_active": False,
                }
            },
        ):
            user = authenticate_admin("inactive_admin", "test_password")
            assert user is None


class TestTokenGeneration:
    """Test JWT token generation and validation."""

    def test_create_access_token(self):
        """Test access token creation."""
        data = {"sub": "test_user", "role": "parser_admin"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))

        assert token is not None
        assert isinstance(token, str)

        # Decode to verify contents using the actual SECRET_KEY from the module
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        assert payload["sub"] == "test_user"
        assert payload["role"] == "parser_admin"
        assert "exp" in payload

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        data = {"sub": "test_user", "role": "parser_admin"}
        token = create_refresh_token(data)

        assert token is not None
        assert isinstance(token, str)

        # Decode to verify contents using the actual SECRET_KEY from the module
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        assert payload["sub"] == "test_user"
        assert payload["role"] == "parser_admin"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    def test_decode_token_valid(self):
        """Test decoding a valid token."""
        # Create a valid token
        data = {"sub": "test_user", "role": "parser_admin"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))

        # Decode it
        token_data = decode_token(token)

        assert token_data.username == "test_user"
        assert token_data.role == "parser_admin"
        assert token_data.exp is not None

    def test_decode_token_expired(self):
        """Test decoding an expired token."""
        # Create an expired token
        data = {"sub": "test_user", "role": "parser_admin"}
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_decode_token_invalid(self):
        """Test decoding an invalid token."""
        # Invalid token
        token = "invalid.token.string"

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)

        assert exc_info.value.status_code == 401
        assert "validate" in exc_info.value.detail.lower()

    def test_decode_token_missing_username(self):
        """Test decoding a token without username."""
        # Create token without 'sub' field
        payload = {"role": "parser_admin", "exp": datetime.now(UTC) + timedelta(minutes=30)}
        token = jwt.encode(
            payload,
            "your-secret-key-change-in-production",
            algorithm="HS256",
        )

        # Should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)

        assert exc_info.value.status_code == 401

    def test_generate_tokens(self):
        """Test generating both access and refresh tokens."""
        response = generate_tokens("test_user", "parser_admin")

        assert isinstance(response, TokenResponse)
        assert response.access_token is not None
        assert response.refresh_token is not None
        assert response.token_type == "bearer"
        assert response.expires_in == 30 * 60  # 30 minutes in seconds


class TestRoleBasedAccess:
    """Test role-based access control."""

    @pytest.mark.asyncio
    async def test_require_super_admin_success(self):
        """Test super admin role requirement with valid role."""
        admin_user = AdminUser(username="admin", role="super_admin", is_active=True)

        result = await require_super_admin(admin_user)
        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_super_admin_failure(self):
        """Test super admin role requirement with insufficient role."""
        admin_user = AdminUser(username="user", role="parser_admin", is_active=True)

        with pytest.raises(HTTPException) as exc_info:
            await require_super_admin(admin_user)

        assert exc_info.value.status_code == 403
        assert "Super admin" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_parser_admin_with_super_admin(self):
        """Test parser admin requirement accepts super admin."""
        admin_user = AdminUser(username="admin", role="super_admin", is_active=True)

        result = await require_parser_admin(admin_user)
        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_parser_admin_with_parser_admin(self):
        """Test parser admin requirement accepts parser admin."""
        admin_user = AdminUser(username="parser", role="parser_admin", is_active=True)

        result = await require_parser_admin(admin_user)
        assert result == admin_user

    @pytest.mark.asyncio
    async def test_require_parser_admin_with_readonly(self):
        """Test parser admin requirement rejects readonly."""
        admin_user = AdminUser(username="reader", role="readonly", is_active=True)

        with pytest.raises(HTTPException) as exc_info:
            await require_parser_admin(admin_user)

        assert exc_info.value.status_code == 403
        assert "Parser admin" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_readonly_accepts_all_roles(self):
        """Test readonly requirement accepts all authenticated users."""
        for role in ["super_admin", "parser_admin", "readonly"]:
            admin_user = AdminUser(username=f"user_{role}", role=role, is_active=True)

            result = await require_readonly(admin_user)
            assert result == admin_user


class TestGetCurrentAdmin:
    """Test getting current admin from token."""

    @pytest.mark.asyncio
    async def test_get_current_admin_valid_token(self):
        """Test getting current admin with valid token."""
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = create_access_token(
            {"sub": "test_admin", "role": "parser_admin"},
            expires_delta=timedelta(minutes=30),
        )

        with patch(
            "services.tracklist_service.src.auth.admin_auth.ADMIN_USERS",
            {
                "test_admin": {
                    "username": "test_admin",
                    "hashed_password": get_password_hash("test_password"),
                    "role": "parser_admin",
                    "is_active": True,
                }
            },
        ):
            admin = await get_current_admin(mock_credentials)

            assert admin.username == "test_admin"
            assert admin.role == "parser_admin"
            assert admin.is_active is True

    @pytest.mark.asyncio
    async def test_get_current_admin_user_not_found(self):
        """Test getting current admin when user not found in database."""
        # Mock credentials with valid token but non-existent user
        mock_credentials = Mock()
        mock_credentials.credentials = create_access_token(
            {"sub": "non_existent", "role": "parser_admin"},
            expires_delta=timedelta(minutes=30),
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin(mock_credentials)

        assert exc_info.value.status_code == 401
        assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_get_current_admin_inactive_user(self):
        """Test getting current admin when user is inactive."""
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = create_access_token(
            {"sub": "inactive_admin", "role": "parser_admin"},
            expires_delta=timedelta(minutes=30),
        )

        with patch(
            "services.tracklist_service.src.auth.admin_auth.ADMIN_USERS",
            {
                "inactive_admin": {
                    "username": "inactive_admin",
                    "hashed_password": get_password_hash("test_password"),
                    "role": "parser_admin",
                    "is_active": False,
                }
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_admin(mock_credentials)

            assert exc_info.value.status_code == 403
            assert "Inactive" in exc_info.value.detail
