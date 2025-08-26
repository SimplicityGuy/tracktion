"""FastAPI dependencies for authentication."""

import logging
from typing import Optional, Callable
from fastapi import Header, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .authentication import AuthenticationManager
from .models import User

logger = logging.getLogger(__name__)

# Global authentication manager instance
# In production, this would be injected via dependency injection
auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get authentication manager instance."""
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication manager not initialized"
        )
    return auth_manager


def set_auth_manager(manager: AuthenticationManager) -> None:
    """Set authentication manager instance."""
    global auth_manager
    auth_manager = manager


# Security scheme for Bearer token
security = HTTPBearer(auto_error=False)


async def authenticate_request(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """Authenticate API request via API key or JWT token.

    Args:
        x_api_key: API key from X-API-Key header
        authorization: Authorization header with Bearer token

    Returns:
        Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    auth_mgr = get_auth_manager()

    # Try API key authentication first
    if x_api_key:
        user = auth_mgr.validate_api_key(x_api_key)
        if user:
            return user
        logger.warning("Invalid API key provided")

    # Try JWT authentication
    if authorization and authorization.scheme.lower() == "bearer":
        user = auth_mgr.validate_jwt_token(authorization.credentials)
        if user:
            return user
        logger.warning("Invalid JWT token provided")

    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def authenticate_user(authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Authenticate user via JWT token only (for web dashboard).

    Args:
        authorization: Authorization header with Bearer token

    Returns:
        Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    if not authorization or authorization.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_jwt_token(authorization.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def require_tier(required_tier: str) -> Callable[[User], User]:
    """Require specific user tier for access.

    Args:
        required_tier: Required tier level

    Returns:
        Dependency function that checks user tier
    """

    def check_tier(user: User = Depends(authenticate_request)) -> User:
        tier_hierarchy = {"free": 0, "premium": 1, "enterprise": 2}
        user_level = tier_hierarchy.get(user.tier.value, 0)
        required_level = tier_hierarchy.get(required_tier, 999)

        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Tier '{required_tier}' or higher required"
            )
        return user

    return check_tier


async def require_permission(permission: str) -> Callable[[User], User]:
    """Require specific permission for access.

    Args:
        permission: Required permission name

    Returns:
        Dependency function that checks user permission
    """

    def check_permission(user: User = Depends(authenticate_request)) -> User:
        # For now, we'll check permissions when API keys are implemented with them
        # This would integrate with the permissions system
        return user

    return check_permission
