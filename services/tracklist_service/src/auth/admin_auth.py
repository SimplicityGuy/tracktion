"""
Admin authentication module for parser administration.

This module provides JWT-based authentication for admin operations
with role-based access control and token validation.
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration
# JWT secret key - MUST be set in production via environment variable
_secret_key = os.getenv("ADMIN_JWT_SECRET")
if not _secret_key:
    # Only use a default in development - log a warning
    import logging

    logging.warning("ADMIN_JWT_SECRET not set - using insecure default for development only!")
    _secret_key = "INSECURE-DEVELOPMENT-KEY-CHANGE-IN-PRODUCTION"

# Type-safe SECRET_KEY - guaranteed to be a string after the above logic
SECRET_KEY: str = _secret_key

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ADMIN_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("ADMIN_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class TokenData(BaseModel):
    """Token data model."""

    username: str | None = None
    role: str | None = None
    exp: datetime | None = None


class AdminUser(BaseModel):
    """Admin user model."""

    username: str
    role: str
    is_active: bool = True


class AdminCredentials(BaseModel):
    """Admin login credentials."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


# Admin users configuration - passwords should be set via environment variables
# In production, this should be stored in a database with proper user management
def _get_admin_users() -> dict[str, dict[str, Any]]:
    """
    Get admin users configuration.

    Passwords are loaded from environment variables for security.
    Falls back to secure defaults only for development.
    """
    return {
        "admin": {
            "username": "admin",
            "hashed_password": pwd_context.hash(os.getenv("ADMIN_PASSWORD", "change-me-in-production-admin")),
            "role": "super_admin",
            "is_active": True,
        },
        "parser_admin": {
            "username": "parser_admin",
            "hashed_password": pwd_context.hash(os.getenv("PARSER_ADMIN_PASSWORD", "change-me-in-production-parser")),
            "role": "parser_admin",
            "is_active": True,
        },
        "readonly_admin": {
            "username": "readonly_admin",
            "hashed_password": pwd_context.hash(
                os.getenv("READONLY_ADMIN_PASSWORD", "change-me-in-production-readonly")
            ),
            "role": "readonly",
            "is_active": True,
        },
    }


# Initialize admin users from environment configuration
ADMIN_USERS = _get_admin_users()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bool(pwd_context.verify(plain_password, hashed_password))


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return str(pwd_context.hash(password))


def authenticate_admin(username: str, password: str) -> AdminUser | None:
    """
    Authenticate admin user.

    Args:
        username: Admin username
        password: Admin password

    Returns:
        AdminUser if authentication successful, None otherwise
    """
    user_data = ADMIN_USERS.get(username)
    if not user_data:
        return None

    if not verify_password(password, user_data["hashed_password"]):
        return None

    if not user_data["is_active"]:
        return None

    return AdminUser(
        username=user_data["username"],
        role=user_data["role"],
        is_active=user_data["is_active"],
    )


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Token payload data
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt: str = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict[str, Any]) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Token payload data

    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt: str = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(
            username=username,
            role=role,
            exp=datetime.fromtimestamp(payload.get("exp"), tz=UTC),
        )

    except jwt.ExpiredSignatureError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err
    except jwt.PyJWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """
    Get current authenticated admin user from JWT token.

    Args:
        credentials: HTTP authorization credentials with bearer token

    Returns:
        Current admin user

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    token_data = decode_token(token)

    # Get user from "database"
    user_data = ADMIN_USERS.get(token_data.username) if token_data.username else None
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return AdminUser(
        username=user_data["username"],
        role=user_data["role"],
        is_active=user_data["is_active"],
    )


async def require_role(required_role: str):
    """
    Dependency to require a specific role.

    Args:
        required_role: Required role name

    Returns:
        Dependency function that validates role
    """

    async def role_checker(current_user: AdminUser = Depends(get_current_admin)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
        """Check if user has required role."""
        # Define role hierarchy
        role_hierarchy = {
            "super_admin": ["super_admin", "parser_admin", "readonly"],
            "parser_admin": ["parser_admin", "readonly"],
            "readonly": ["readonly"],
        }

        allowed_roles = role_hierarchy.get(current_user.role, [])

        if required_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )

        return current_user

    return role_checker


# Convenience dependency functions for specific roles
async def require_super_admin(current_user: AdminUser = Depends(get_current_admin)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """Require super admin role."""
    if current_user.role != "super_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required",
        )
    return current_user


async def require_parser_admin(current_user: AdminUser = Depends(get_current_admin)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """Require parser admin role or higher."""
    if current_user.role not in ["super_admin", "parser_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Parser admin access required",
        )
    return current_user


async def require_readonly(current_user: AdminUser = Depends(get_current_admin)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """Require readonly access or higher (any authenticated admin)."""
    # Any authenticated admin has at least readonly access
    return current_user


def generate_tokens(username: str, role: str) -> TokenResponse:
    """
    Generate access and refresh tokens for admin user.

    Args:
        username: Admin username
        role: Admin role

    Returns:
        Token response with access and refresh tokens
    """
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username, "role": role},
        expires_delta=access_token_expires,
    )

    # Create refresh token
    refresh_token = create_refresh_token(data={"sub": username, "role": role})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
