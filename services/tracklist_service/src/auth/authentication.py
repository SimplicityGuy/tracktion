"""Authentication manager for API key and JWT token handling."""

import secrets
import logging
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext  # type: ignore[import-untyped]

from .models import User, ApiKey, TokenPair, UserTier

logger = logging.getLogger(__name__)


class AuthenticationManager:
    """Manages API keys, JWT tokens, and user authentication."""

    def __init__(self, jwt_secret: str, jwt_algorithm: str = "HS256"):
        """Initialize authentication manager.

        Args:
            jwt_secret: Secret key for JWT token signing
            jwt_algorithm: Algorithm for JWT token signing
        """
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # In-memory storage for demo - replace with database
        self._users: Dict[str, User] = {}
        self._api_keys: Dict[str, ApiKey] = {}

    def generate_api_key(self, user_id: str, tier: str, name: Optional[str] = None) -> ApiKey:
        """Generate a new API key for a user.

        Args:
            user_id: User ID to associate the key with
            tier: User tier for rate limiting
            name: Optional name for the key

        Returns:
            Generated API key with metadata

        Raises:
            ValueError: If user doesn't exist
        """
        if user_id not in self._users:
            raise ValueError(f"User {user_id} not found")

        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_id = secrets.token_urlsafe(16)

        # Hash the key for storage
        key_hash = self.pwd_context.hash(raw_key)

        # Create API key record
        api_key = ApiKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            is_active=True,
            created_at=datetime.now(UTC),
            permissions=self._get_default_permissions(tier),
        )

        # Store the key (using raw key as lookup, but storing hash)
        self._api_keys[raw_key] = api_key

        logger.info(f"Generated API key {key_id} for user {user_id}")
        return api_key

    def validate_api_key(self, key: str) -> Optional[User]:
        """Validate an API key and return associated user.

        Args:
            key: API key to validate

        Returns:
            User if key is valid, None otherwise
        """
        # First try direct lookup (for newly created keys)
        if key in self._api_keys:
            api_key = self._api_keys[key]
            if api_key.is_active and api_key.user_id in self._users:
                user = self._users[api_key.user_id]
                if user.is_active:
                    # Update last used timestamp
                    api_key.last_used_at = datetime.now(UTC)
                    return user

        # Try hash verification for existing keys
        for stored_key, api_key in self._api_keys.items():
            if api_key.is_active and self.pwd_context.verify(key, api_key.key_hash):
                if api_key.user_id in self._users:
                    user = self._users[api_key.user_id]
                    if user.is_active:
                        api_key.last_used_at = datetime.now(UTC)
                        return user

        return None

    def generate_jwt_token(self, user: User, expires_delta: Optional[timedelta] = None) -> TokenPair:
        """Generate JWT token pair for a user.

        Args:
            user: User to generate tokens for
            expires_delta: Custom expiration time

        Returns:
            Token pair with access and refresh tokens
        """
        if expires_delta is None:
            expires_delta = timedelta(hours=1)

        # Create access token
        access_payload = {
            "sub": user.id,
            "email": user.email,
            "tier": user.tier.value,
            "exp": datetime.now(UTC) + expires_delta,
            "iat": datetime.now(UTC),
            "type": "access",
        }
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Create refresh token (longer lived)
        refresh_payload = {
            "sub": user.id,
            "exp": datetime.now(UTC) + timedelta(days=7),
            "iat": datetime.now(UTC),
            "type": "refresh",
        }
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        return TokenPair(
            access_token=access_token, refresh_token=refresh_token, expires_in=int(expires_delta.total_seconds())
        )

    def validate_jwt_token(self, token: str) -> Optional[User]:
        """Validate a JWT token and return associated user.

        Args:
            token: JWT token to validate

        Returns:
            User if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("sub")

            if user_id and user_id in self._users:
                user = self._users[user_id]
                if user.is_active:
                    return user

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")

        return None

    def refresh_jwt_token(self, refresh_token: str) -> Optional[TokenPair]:
        """Refresh JWT tokens using refresh token.

        Args:
            refresh_token: Refresh token to use

        Returns:
            New token pair if refresh is valid, None otherwise
        """
        try:
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("sub")
            token_type = payload.get("type")

            if token_type != "refresh":
                return None

            if user_id and user_id in self._users:
                user = self._users[user_id]
                if user.is_active:
                    return self.generate_jwt_token(user)

        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid refresh token")

        return None

    def create_user(
        self, email: str, tier: UserTier = UserTier.FREE, metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user account.

        Args:
            email: User email address
            tier: User tier level
            metadata: Additional user metadata

        Returns:
            Created user
        """
        user_id = secrets.token_urlsafe(16)
        user = User(
            id=user_id,
            email=email,
            tier=tier,
            is_active=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            metadata=metadata or {},
        )

        self._users[user_id] = user
        logger.info(f"Created user {user_id} with email {email}")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User ID to lookup

        Returns:
            User if found, None otherwise
        """
        return self._users.get(user_id)

    def update_user_tier(self, user_id: str, new_tier: UserTier) -> bool:
        """Update user's tier level.

        Args:
            user_id: User ID to update
            new_tier: New tier level

        Returns:
            True if updated successfully, False otherwise
        """
        if user_id in self._users:
            self._users[user_id].tier = new_tier
            self._users[user_id].updated_at = datetime.now(UTC)
            logger.info(f"Updated user {user_id} to tier {new_tier.value}")
            return True
        return False

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key.

        Args:
            key: API key to revoke

        Returns:
            True if revoked successfully, False otherwise
        """
        if key in self._api_keys:
            self._api_keys[key].is_active = False
            logger.info(f"Revoked API key {self._api_keys[key].key_id}")
            return True

        # Try hash verification for existing keys
        for stored_key, api_key in self._api_keys.items():
            if self.pwd_context.verify(key, api_key.key_hash):
                api_key.is_active = False
                logger.info(f"Revoked API key {api_key.key_id}")
                return True

        return False

    def _get_default_permissions(self, tier: str) -> Dict[str, bool]:
        """Get default permissions for a tier.

        Args:
            tier: User tier level

        Returns:
            Default permissions dictionary
        """
        permissions = {"read": True, "write": False, "admin": False}

        if tier in ["premium", "enterprise"]:
            permissions["write"] = True

        if tier == "enterprise":
            permissions["admin"] = True

        return permissions

    def setup_oauth2_provider(self, provider: str) -> Dict[str, Any]:
        """Setup OAuth2 provider configuration.

        Args:
            provider: OAuth2 provider name (google, github, etc.)

        Returns:
            Provider configuration
        """
        # This would integrate with actual OAuth2 providers
        # For now, return placeholder configuration
        config = {
            "provider": provider,
            "client_id": f"placeholder_{provider}_client_id",
            "redirect_uri": f"https://api.tracktion.com/auth/{provider}/callback",
            "scopes": ["openid", "email", "profile"],
        }

        logger.info(f"Setup OAuth2 provider: {provider}")
        return config
