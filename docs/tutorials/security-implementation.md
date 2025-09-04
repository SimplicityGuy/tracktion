# Security Implementation Examples

This document provides comprehensive security implementation examples for Tracktion services, covering authentication, authorization, data protection, secure communication, and security monitoring.

## Table of Contents

1. [Authentication Systems](#authentication-systems)
2. [Authorization and Access Control](#authorization-and-access-control)
3. [Data Protection and Encryption](#data-protection-and-encryption)
4. [Secure Communication](#secure-communication)
5. [Input Validation and Sanitization](#input-validation-and-sanitization)
6. [Security Headers and CORS](#security-headers-and-cors)
7. [Audit Logging and Security Monitoring](#audit-logging-and-security-monitoring)
8. [Secrets Management](#secrets-management)
9. [Security Testing](#security-testing)
10. [Vulnerability Management](#vulnerability-management)

## Authentication Systems

### 1. JWT-Based Authentication

```python
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib
from dataclasses import dataclass

@dataclass
class SecurityConfig:
    """Security configuration settings."""

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_DELTA: int = 3600  # 1 hour
    REFRESH_TOKEN_EXPIRATION_DELTA: int = 604800  # 1 week
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_SPECIAL: bool = True
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION: int = 900  # 15 minutes

class SecureAuthenticationManager:
    """Secure JWT-based authentication with comprehensive security features."""

    def __init__(self, config: SecurityConfig, redis_client=None):
        self.config = config
        self.redis = redis_client
        self.security = HTTPBearer(auto_error=False)

    async def create_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Create new user with secure password handling."""

        # Validate password strength
        if not self._validate_password_strength(password):
            raise HTTPException(
                status_code=400,
                detail="Password does not meet security requirements"
            )

        # Check if user already exists
        existing_user = await self._get_user_by_email(email)
        if existing_user:
            raise HTTPException(status_code=409, detail="User already exists")

        # Hash password with salt
        password_hash = self._hash_password(password)

        # Create user record
        user_data = {
            'username': username,
            'email': email.lower(),
            'password_hash': password_hash,
            'is_active': True,
            'created_at': datetime.utcnow(),
            'last_login': None,
            'login_attempts': 0,
            'locked_until': None
        }

        # Save user to database
        user_id = await self._save_user(user_data)

        # Generate verification token
        verification_token = self._generate_verification_token(user_id)

        return {
            'user_id': user_id,
            'verification_token': verification_token,
            'message': 'User created successfully. Please verify your email.'
        }

    async def authenticate_user(self, email: str, password: str,
                              client_ip: str = None) -> Dict[str, Any]:
        """Authenticate user with security measures."""

        email = email.lower()

        # Check for account lockout
        if await self._is_account_locked(email):
            raise HTTPException(
                status_code=423,
                detail="Account temporarily locked due to too many failed attempts"
            )

        # Get user from database
        user = await self._get_user_by_email(email)
        if not user:
            await self._record_failed_attempt(email, client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify password
        if not self._verify_password(password, user['password_hash']):
            await self._record_failed_attempt(email, client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if account is active
        if not user['is_active']:
            raise HTTPException(status_code=403, detail="Account is inactive")

        # Reset login attempts on successful authentication
        await self._reset_login_attempts(email)

        # Generate JWT tokens
        access_token = self._generate_access_token(user)
        refresh_token = self._generate_refresh_token(user)

        # Update last login
        await self._update_last_login(user['id'], client_ip)

        # Store refresh token (for revocation capability)
        await self._store_refresh_token(user['id'], refresh_token)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': self.config.JWT_EXPIRATION_DELTA,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        }

    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        if len(password) < self.config.PASSWORD_MIN_LENGTH:
            return False

        if self.config.PASSWORD_REQUIRE_SPECIAL:
            has_upper = any(c.isupper() for c in password)
            has_lower = any(c.islower() for c in password)
            has_digit = any(c.isdigit() for c in password)
            has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

            return all([has_upper, has_lower, has_digit, has_special])

        return True

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)  # Strong cost factor
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, hash_str: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hash_str.encode('utf-8'))

    def _generate_access_token(self, user: Dict[str, Any]) -> str:
        """Generate JWT access token."""
        now = datetime.utcnow()
        payload = {
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'iat': now,
            'exp': now + timedelta(seconds=self.config.JWT_EXPIRATION_DELTA),
            'type': 'access'
        }

        return jwt.encode(payload, self.config.JWT_SECRET_KEY, algorithm=self.config.JWT_ALGORITHM)

    def _generate_refresh_token(self, user: Dict[str, Any]) -> str:
        """Generate JWT refresh token."""
        now = datetime.utcnow()
        payload = {
            'user_id': user['id'],
            'iat': now,
            'exp': now + timedelta(seconds=self.config.REFRESH_TOKEN_EXPIRATION_DELTA),
            'type': 'refresh',
            'jti': secrets.token_urlsafe(32)  # Unique token ID for revocation
        }

        return jwt.encode(payload, self.config.JWT_SECRET_KEY, algorithm=self.config.JWT_ALGORITHM)

    async def verify_token(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        try:
            payload = jwt.decode(
                credentials.credentials,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM]
            )

            # Verify token type
            if payload.get('type') != 'access':
                raise HTTPException(status_code=401, detail="Invalid token type")

            # Check if token is blacklisted
            if await self._is_token_blacklisted(credentials.credentials):
                raise HTTPException(status_code=401, detail="Token has been revoked")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is temporarily locked."""
        if not self.redis:
            return False

        lock_key = f"account_lock:{hashlib.md5(email.encode()).hexdigest()}"
        return await self.redis.exists(lock_key)

    async def _record_failed_attempt(self, email: str, client_ip: str = None):
        """Record failed login attempt."""
        if not self.redis:
            return

        # Record attempt with IP
        attempt_key = f"login_attempts:{hashlib.md5(email.encode()).hexdigest()}"
        attempts = await self.redis.incr(attempt_key)
        await self.redis.expire(attempt_key, 900)  # 15 minutes

        # Lock account if too many attempts
        if attempts >= self.config.MAX_LOGIN_ATTEMPTS:
            lock_key = f"account_lock:{hashlib.md5(email.encode()).hexdigest()}"
            await self.redis.setex(lock_key, self.config.LOCKOUT_DURATION, "locked")

        # Log security event
        await self._log_security_event("failed_login", {
            'email': email,
            'client_ip': client_ip,
            'attempts': attempts
        })

# Dependencies for FastAPI
auth_manager = SecureAuthenticationManager(security_config, redis_client)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_manager.security)):
    """FastAPI dependency to get current authenticated user."""
    return await auth_manager.verify_token(credentials)

# Usage in FastAPI routes
from fastapi import FastAPI, Depends

app = FastAPI()

@app.post("/auth/login")
async def login(login_data: dict):
    """Secure login endpoint."""
    return await auth_manager.authenticate_user(
        email=login_data['email'],
        password=login_data['password'],
        client_ip=login_data.get('client_ip')
    )

@app.get("/auth/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile - requires authentication."""
    return {
        'user_id': current_user['user_id'],
        'username': current_user['username'],
        'email': current_user['email']
    }
```

### 2. Multi-Factor Authentication (MFA)

```python
import pyotp
import qrcode
from io import BytesIO
import base64
from typing import Optional

class MFAManager:
    """Multi-Factor Authentication manager."""

    def __init__(self, app_name: str = "Tracktion"):
        self.app_name = app_name

    async def setup_totp(self, user_id: int, email: str) -> Dict[str, Any]:
        """Setup TOTP for user."""

        # Generate secret key
        secret = pyotp.random_base32()

        # Create TOTP instance
        totp = pyotp.TOTP(secret)

        # Generate provisioning URI for QR code
        provisioning_uri = totp.provisioning_uri(
            name=email,
            issuer_name=self.app_name
        )

        # Generate QR code
        qr_code_data = self._generate_qr_code(provisioning_uri)

        # Store secret in database (encrypted)
        await self._store_mfa_secret(user_id, secret)

        return {
            'secret': secret,
            'qr_code': qr_code_data,
            'backup_codes': await self._generate_backup_codes(user_id)
        }

    async def verify_totp(self, user_id: int, token: str) -> bool:
        """Verify TOTP token."""

        # Get user's secret
        secret = await self._get_mfa_secret(user_id)
        if not secret:
            return False

        # Create TOTP instance
        totp = pyotp.TOTP(secret)

        # Verify token (with window for clock drift)
        return totp.verify(token, window=1)

    async def verify_backup_code(self, user_id: int, code: str) -> bool:
        """Verify and consume backup code."""

        # Get unused backup codes
        backup_codes = await self._get_backup_codes(user_id)

        # Check if code is valid
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash in backup_codes:
            # Mark code as used
            await self._mark_backup_code_used(user_id, code_hash)
            return True

        return False

    def _generate_qr_code(self, provisioning_uri: str) -> str:
        """Generate QR code as base64 string."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode()

    async def _generate_backup_codes(self, user_id: int, count: int = 10) -> List[str]:
        """Generate backup codes for user."""
        codes = []
        code_hashes = []

        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8 character hex codes
            codes.append(code)
            code_hashes.append(hashlib.sha256(code.encode()).hexdigest())

        # Store hashed codes in database
        await self._store_backup_codes(user_id, code_hashes)

        return codes

# Enhanced authentication with MFA
class MFAAuthenticationManager(SecureAuthenticationManager):
    """Authentication manager with MFA support."""

    def __init__(self, config: SecurityConfig, redis_client=None):
        super().__init__(config, redis_client)
        self.mfa_manager = MFAManager()

    async def authenticate_with_mfa(self, email: str, password: str,
                                  mfa_token: Optional[str] = None,
                                  client_ip: str = None) -> Dict[str, Any]:
        """Authenticate user with MFA requirement."""

        # First stage: password authentication
        user = await self._get_user_by_email(email.lower())
        if not user or not self._verify_password(password, user['password_hash']):
            await self._record_failed_attempt(email, client_ip)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if MFA is enabled for user
        if await self._is_mfa_enabled(user['id']):
            if not mfa_token:
                # Return partial token requiring MFA
                partial_token = self._generate_partial_token(user)
                return {
                    'requires_mfa': True,
                    'partial_token': partial_token,
                    'message': 'MFA token required'
                }

            # Verify MFA token
            if not (await self.mfa_manager.verify_totp(user['id'], mfa_token) or
                   await self.mfa_manager.verify_backup_code(user['id'], mfa_token)):
                await self._record_failed_attempt(email, client_ip)
                raise HTTPException(status_code=401, detail="Invalid MFA token")

        # Complete authentication
        return await self._complete_authentication(user, client_ip)
```

## Authorization and Access Control

### 1. Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import List, Set, Optional, Dict, Any
from functools import wraps

class Permission(Enum):
    """System permissions."""

    # Track management
    TRACK_READ = "track:read"
    TRACK_WRITE = "track:write"
    TRACK_DELETE = "track:delete"
    TRACK_ANALYZE = "track:analyze"

    # Playlist management
    PLAYLIST_READ = "playlist:read"
    PLAYLIST_WRITE = "playlist:write"
    PLAYLIST_DELETE = "playlist:delete"
    PLAYLIST_SHARE = "playlist:share"

    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    USER_ADMIN = "user:admin"

    # System administration
    SYSTEM_CONFIG = "system:config"
    SYSTEM_LOGS = "system:logs"
    SYSTEM_METRICS = "system:metrics"

    # Analytics
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_EXPORT = "analytics:export"

class Role(Enum):
    """System roles with associated permissions."""

    GUEST = "guest"
    USER = "user"
    PREMIUM_USER = "premium_user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self):
        self.role_permissions = {
            Role.GUEST: {
                Permission.TRACK_READ,
            },
            Role.USER: {
                Permission.TRACK_READ,
                Permission.PLAYLIST_READ,
                Permission.PLAYLIST_WRITE,
            },
            Role.PREMIUM_USER: {
                Permission.TRACK_READ,
                Permission.TRACK_ANALYZE,
                Permission.PLAYLIST_READ,
                Permission.PLAYLIST_WRITE,
                Permission.PLAYLIST_SHARE,
                Permission.ANALYTICS_READ,
            },
            Role.MODERATOR: {
                Permission.TRACK_READ,
                Permission.TRACK_WRITE,
                Permission.TRACK_ANALYZE,
                Permission.PLAYLIST_READ,
                Permission.PLAYLIST_WRITE,
                Permission.PLAYLIST_DELETE,
                Permission.PLAYLIST_SHARE,
                Permission.USER_READ,
                Permission.ANALYTICS_READ,
            },
            Role.ADMIN: {
                Permission.TRACK_READ,
                Permission.TRACK_WRITE,
                Permission.TRACK_DELETE,
                Permission.TRACK_ANALYZE,
                Permission.PLAYLIST_READ,
                Permission.PLAYLIST_WRITE,
                Permission.PLAYLIST_DELETE,
                Permission.PLAYLIST_SHARE,
                Permission.USER_READ,
                Permission.USER_WRITE,
                Permission.USER_DELETE,
                Permission.SYSTEM_CONFIG,
                Permission.SYSTEM_LOGS,
                Permission.ANALYTICS_READ,
                Permission.ANALYTICS_EXPORT,
            },
            Role.SUPER_ADMIN: set(Permission),  # All permissions
        }

        # Resource-based permissions cache
        self.resource_cache = {}

    def get_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self.role_permissions.get(role, set())

    def has_permission(self, user_role: Role, required_permission: Permission) -> bool:
        """Check if role has specific permission."""
        user_permissions = self.get_permissions(user_role)
        return required_permission in user_permissions

    def can_access_resource(self, user_id: int, resource_type: str,
                          resource_id: int, action: str) -> bool:
        """Check resource-level permissions."""

        # Convert action to permission
        permission = Permission(f"{resource_type}:{action}")

        # Get user role
        user_role = self._get_user_role(user_id)

        # Check basic permission
        if not self.has_permission(user_role, permission):
            return False

        # Check resource ownership for non-admin users
        if user_role not in [Role.ADMIN, Role.SUPER_ADMIN]:
            return self._check_resource_ownership(user_id, resource_type, resource_id)

        return True

    async def _get_user_role(self, user_id: int) -> Role:
        """Get user's role from database."""
        # This would query the database
        # Simplified for example
        user_data = await get_user_by_id(user_id)
        return Role(user_data.get('role', 'user'))

    async def _check_resource_ownership(self, user_id: int, resource_type: str,
                                      resource_id: int) -> bool:
        """Check if user owns the resource."""
        cache_key = f"{user_id}:{resource_type}:{resource_id}"

        if cache_key in self.resource_cache:
            return self.resource_cache[cache_key]

        # Query database for ownership
        is_owner = await self._query_resource_ownership(user_id, resource_type, resource_id)

        # Cache result
        self.resource_cache[cache_key] = is_owner

        return is_owner

# Authorization decorators
rbac_manager = RBACManager()

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from context
            current_user = kwargs.get('current_user') or args[0] if args else None

            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            user_role = Role(current_user.get('role', 'user'))

            if not rbac_manager.has_permission(user_role, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permission.value}"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_resource_access(resource_type: str, action: str):
    """Decorator to require resource-level access."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            resource_id = kwargs.get('resource_id') or kwargs.get('id')

            if not current_user or not resource_id:
                raise HTTPException(status_code=400, detail="Invalid request")

            if not rbac_manager.can_access_resource(
                current_user['user_id'], resource_type, resource_id, action
            ):
                raise HTTPException(status_code=403, detail="Access denied")

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage in FastAPI routes
@app.get("/tracks/{track_id}")
@require_permission(Permission.TRACK_READ)
async def get_track(track_id: int, current_user: dict = Depends(get_current_user)):
    """Get track with permission check."""
    return await get_track_by_id(track_id)

@app.delete("/playlists/{playlist_id}")
@require_resource_access("playlist", "delete")
async def delete_playlist(playlist_id: int, current_user: dict = Depends(get_current_user)):
    """Delete playlist with ownership check."""
    return await delete_playlist_by_id(playlist_id)
```

### 2. Resource-Based Authorization

```python
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List

class ResourcePermissionChecker(Protocol):
    """Protocol for resource permission checking."""

    async def can_read(self, user_id: int, resource_id: int) -> bool:
        """Check if user can read resource."""
        ...

    async def can_write(self, user_id: int, resource_id: int) -> bool:
        """Check if user can modify resource."""
        ...

    async def can_delete(self, user_id: int, resource_id: int) -> bool:
        """Check if user can delete resource."""
        ...

class PlaylistPermissionChecker:
    """Permission checker for playlist resources."""

    async def can_read(self, user_id: int, playlist_id: int) -> bool:
        """Check playlist read permission."""
        playlist = await get_playlist_by_id(playlist_id)

        if not playlist:
            return False

        # Owner can always read
        if playlist['owner_id'] == user_id:
            return True

        # Check if playlist is public
        if playlist['is_public']:
            return True

        # Check if user has been granted access
        return await check_playlist_sharing(playlist_id, user_id)

    async def can_write(self, user_id: int, playlist_id: int) -> bool:
        """Check playlist write permission."""
        playlist = await get_playlist_by_id(playlist_id)

        if not playlist:
            return False

        # Owner can always write
        if playlist['owner_id'] == user_id:
            return True

        # Check if user has collaborator access
        return await check_playlist_collaborator(playlist_id, user_id)

    async def can_delete(self, user_id: int, playlist_id: int) -> bool:
        """Check playlist delete permission."""
        playlist = await get_playlist_by_id(playlist_id)

        if not playlist:
            return False

        # Only owner can delete
        return playlist['owner_id'] == user_id

class ResourceAuthorizationManager:
    """Centralized resource authorization management."""

    def __init__(self):
        self.permission_checkers = {
            'playlist': PlaylistPermissionChecker(),
            'track': TrackPermissionChecker(),
            'user': UserPermissionChecker(),
        }

    async def authorize_action(self, user_id: int, resource_type: str,
                             resource_id: int, action: str) -> bool:
        """Authorize action on resource."""

        checker = self.permission_checkers.get(resource_type)
        if not checker:
            return False

        action_method = getattr(checker, f"can_{action}", None)
        if not action_method:
            return False

        return await action_method(user_id, resource_id)

    def require_resource_permission(self, resource_type: str, action: str):
        """Decorator for resource permission checking."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = kwargs.get('current_user')
                resource_id = kwargs.get(f'{resource_type}_id')

                if not current_user or not resource_id:
                    raise HTTPException(status_code=400, detail="Invalid request")

                authorized = await self.authorize_action(
                    current_user['user_id'], resource_type, resource_id, action
                )

                if not authorized:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Not authorized to {action} {resource_type}"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Usage
resource_auth = ResourceAuthorizationManager()

@app.put("/playlists/{playlist_id}")
@resource_auth.require_resource_permission("playlist", "write")
async def update_playlist(playlist_id: int, playlist_data: dict,
                         current_user: dict = Depends(get_current_user)):
    """Update playlist with authorization check."""
    return await update_playlist_data(playlist_id, playlist_data)
```

## Data Protection and Encryption

### 1. Field-Level Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os
import base64
from typing import Optional, Union

class FieldEncryption:
    """Field-level encryption for sensitive data."""

    def __init__(self, master_key: str):
        self.master_key = master_key.encode()
        self._derive_keys()

    def _derive_keys(self):
        """Derive encryption keys from master key."""
        # Derive key for field encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'tracktion_field_salt',  # Use proper random salt in production
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.field_cipher = Fernet(key)

    def encrypt_field(self, value: str) -> str:
        """Encrypt a field value."""
        if not value:
            return value

        encrypted_bytes = self.field_cipher.encrypt(value.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')

    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a field value."""
        if not encrypted_value:
            return encrypted_value

        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
            decrypted_bytes = self.field_cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception:
            # Handle decryption errors gracefully
            return "[DECRYPTION_ERROR]"

class SecureDataModel:
    """Base model with encryption support."""

    def __init__(self, encryptor: FieldEncryption):
        self.encryptor = encryptor
        self.encrypted_fields = set()

    def mark_encrypted(self, *field_names):
        """Mark fields as encrypted."""
        self.encrypted_fields.update(field_names)

    def encrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data."""
        encrypted_data = data.copy()

        for field in self.encrypted_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encryptor.encrypt_field(
                    str(encrypted_data[field])
                )

        return encrypted_data

    def decrypt_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data."""
        decrypted_data = data.copy()

        for field in self.encrypted_fields:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.encryptor.decrypt_field(
                    decrypted_data[field]
                )

        return decrypted_data

# Example usage
encryptor = FieldEncryption(master_key=os.getenv('ENCRYPTION_MASTER_KEY'))

class UserDataModel(SecureDataModel):
    """User data model with field encryption."""

    def __init__(self, encryptor: FieldEncryption):
        super().__init__(encryptor)
        self.mark_encrypted('email', 'phone', 'real_name')

    async def create_user(self, user_data: Dict[str, Any]) -> int:
        """Create user with encrypted sensitive fields."""
        encrypted_data = self.encrypt_data(user_data)

        # Save to database
        user_id = await save_user_to_database(encrypted_data)
        return user_id

    async def get_user(self, user_id: int) -> Dict[str, Any]:
        """Get user and decrypt sensitive fields."""
        encrypted_user = await get_user_from_database(user_id)

        if not encrypted_user:
            return None

        return self.decrypt_data(encrypted_user)
```

### 2. Database Encryption at Rest

```python
import sqlalchemy as sa
from sqlalchemy import TypeDecorator, String
from sqlalchemy.ext.declarative import declarative_base

class EncryptedType(TypeDecorator):
    """SQLAlchemy type for automatic field encryption."""

    impl = String
    cache_ok = True

    def __init__(self, encryptor: FieldEncryption, *args, **kwargs):
        self.encryptor = encryptor
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value, dialect):
        """Encrypt value before storing in database."""
        if value is not None:
            return self.encryptor.encrypt_field(value)
        return value

    def process_result_value(self, value, dialect):
        """Decrypt value after retrieving from database."""
        if value is not None:
            return self.encryptor.decrypt_field(value)
        return value

# Database models with encryption
Base = declarative_base()

class User(Base):
    """User model with encrypted fields."""
    __tablename__ = 'users'

    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String(50), unique=True, nullable=False)

    # Encrypted fields
    email = sa.Column(EncryptedType(encryptor), nullable=False)
    phone = sa.Column(EncryptedType(encryptor), nullable=True)
    real_name = sa.Column(EncryptedType(encryptor), nullable=True)

    # Non-encrypted fields
    created_at = sa.Column(sa.DateTime, default=sa.func.now())
    is_active = sa.Column(sa.Boolean, default=True)

class UserPreferences(Base):
    """User preferences with selective encryption."""
    __tablename__ = 'user_preferences'

    id = sa.Column(sa.Integer, primary_key=True)
    user_id = sa.Column(sa.Integer, sa.ForeignKey('users.id'))

    # Potentially sensitive preference data
    listening_history = sa.Column(EncryptedType(encryptor), nullable=True)
    favorite_genres = sa.Column(sa.Text, nullable=True)  # Not encrypted

    created_at = sa.Column(sa.DateTime, default=sa.func.now())
```

## Secure Communication

### 1. API Security Headers and Middleware

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
import secrets
from typing import Dict, Any

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    def __init__(self, app, csp_policy: str = None):
        super().__init__(app)
        self.csp_policy = csp_policy or (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'"
        )

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp_policy

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HTTPS enforcement
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

    def __init__(self, app, redis_client, default_limit: int = 1000):
        super().__init__(app)
        self.redis = redis_client
        self.default_limit = default_limit

        # Different limits for different endpoints
        self.endpoint_limits = {
            '/auth/login': 5,  # 5 attempts per minute
            '/auth/register': 3,  # 3 registrations per minute
            '/api/tracks': 100,  # 100 requests per minute
            '/api/analyze': 10,  # 10 analysis requests per minute
        }

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # Get limit for endpoint
        limit = self.endpoint_limits.get(endpoint, self.default_limit)

        # Create rate limit key
        key = f"rate_limit:{client_ip}:{endpoint}"

        # Check current count
        current_requests = await self.redis.get(key)
        if current_requests and int(current_requests) >= limit:
            response = Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0"
                }
            )
            return response

        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 1 minute window
        await pipe.execute()

        # Continue with request
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, limit - int(current_requests or 0) - 1)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers (behind proxy)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip

        return request.client.host

def create_secure_app() -> FastAPI:
    """Create FastAPI app with security middleware."""

    app = FastAPI(
        title="Tracktion API",
        docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
        redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    )

    # Trusted hosts (prevent Host header injection)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "localhost",
            "127.0.0.1",
            "tracktion.example.com",
            "api.tracktion.example.com"
        ]
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://tracktion.example.com",
            "https://app.tracktion.example.com"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "X-CSRF-Token"
        ],
    )

    # Session middleware with secure settings
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("SESSION_SECRET_KEY"),
        https_only=True,
        same_site="strict"
    )

    # Custom security middleware
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware, redis_client=redis_client)

    return app
```

### 2. TLS/SSL Configuration

```python
import ssl
from pathlib import Path

class TLSConfig:
    """TLS/SSL configuration for secure communication."""

    @staticmethod
    def create_ssl_context(cert_file: str, key_file: str,
                          ca_file: str = None) -> ssl.SSLContext:
        """Create secure SSL context."""

        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # Load certificate and private key
        context.load_cert_chain(cert_file, key_file)

        # Load CA certificate if provided
        if ca_file:
            context.load_verify_locations(ca_file)

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')

        return context

    @staticmethod
    def get_production_ssl_config() -> Dict[str, Any]:
        """Get production SSL configuration."""
        return {
            "certfile": "/etc/ssl/certs/tracktion.crt",
            "keyfile": "/etc/ssl/private/tracktion.key",
            "ssl_version": ssl.PROTOCOL_TLSv1_2,
            "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
            "ssl_verify_mode": ssl.CERT_REQUIRED,
        }

# HTTPS redirect middleware
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP requests to HTTPS."""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        """Redirect HTTP to HTTPS if enabled."""
        if (self.enabled and
            request.headers.get("x-forwarded-proto") == "http" and
            request.url.hostname not in ["localhost", "127.0.0.1"]):

            # Redirect to HTTPS
            https_url = request.url.replace(scheme="https")
            return Response(
                status_code=301,
                headers={"Location": str(https_url)}
            )

        return await call_next(request)
```

## Input Validation and Sanitization

### 1. Comprehensive Input Validation

```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
import re
from datetime import datetime
from enum import Enum

class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"

class InputSanitizer:
    """Input sanitization utilities."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove directory separators and null bytes
        sanitized = re.sub(r'[/\\:*?"<>|\x00]', '_', filename)

        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')

        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext

        return sanitized

    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Sanitize text input."""
        if not text:
            return ""

        # Remove control characters except newline and tab
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # Limit length
        return sanitized[:max_length]

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

class UserRegistrationModel(BaseModel):
    """User registration with comprehensive validation."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)

    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        v = v.lower().strip()
        if not InputSanitizer.validate_email(v):
            raise ValueError('Invalid email format')
        return v

    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')

        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')

        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')

        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')

        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', v):
            raise ValueError('Password must contain at least one special character')

        return v

    @validator('full_name')
    def sanitize_full_name(cls, v):
        """Sanitize full name input."""
        if v:
            return InputSanitizer.sanitize_text(v, max_length=100)
        return v

class TrackUploadModel(BaseModel):
    """Track upload with file validation."""

    filename: str = Field(..., max_length=255)
    file_size: int = Field(..., gt=0, le=500*1024*1024)  # Max 500MB
    format: AudioFormat
    title: Optional[str] = Field(None, max_length=200)
    artist: Optional[str] = Field(None, max_length=200)

    @validator('filename')
    def sanitize_filename(cls, v):
        """Sanitize filename."""
        return InputSanitizer.sanitize_filename(v)

    @validator('format')
    def validate_audio_format(cls, v, values):
        """Validate audio format matches filename."""
        if 'filename' in values:
            file_ext = Path(values['filename']).suffix.lower()[1:]  # Remove dot
            if file_ext != v.value:
                raise ValueError(f'File extension {file_ext} does not match format {v.value}')
        return v

    @validator('title', 'artist')
    def sanitize_metadata(cls, v):
        """Sanitize metadata fields."""
        if v:
            return InputSanitizer.sanitize_text(v, max_length=200)
        return v

class PlaylistCreateModel(BaseModel):
    """Playlist creation with validation."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    is_public: bool = False
    tags: Optional[List[str]] = Field(None, max_items=20)

    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize playlist name."""
        return InputSanitizer.sanitize_text(v.strip(), max_length=100)

    @validator('description')
    def sanitize_description(cls, v):
        """Sanitize description."""
        if v:
            return InputSanitizer.sanitize_text(v.strip(), max_length=500)
        return v

    @validator('tags')
    def validate_tags(cls, v):
        """Validate and sanitize tags."""
        if v:
            sanitized_tags = []
            for tag in v[:20]:  # Limit to 20 tags
                sanitized_tag = InputSanitizer.sanitize_text(tag.strip(), max_length=30)
                if sanitized_tag and sanitized_tag not in sanitized_tags:
                    sanitized_tags.append(sanitized_tag)
            return sanitized_tags
        return v

# Request validation middleware
class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for additional request validation."""

    def __init__(self, app):
        super().__init__(app)

        # Dangerous patterns to block
        self.dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'<iframe',
            r'<object',
            r'<embed',
        ]

    async def dispatch(self, request: Request, call_next):
        """Validate request for dangerous content."""

        # Check URL for dangerous patterns
        url_str = str(request.url).lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, url_str):
                return Response(
                    content="Malicious content detected",
                    status_code=400
                )

        # Check headers for dangerous content
        for header_value in request.headers.values():
            header_lower = header_value.lower()
            for pattern in self.dangerous_patterns:
                if re.search(pattern, header_lower):
                    return Response(
                        content="Malicious content detected",
                        status_code=400
                    )

        return await call_next(request)

# Usage in FastAPI routes
@app.post("/auth/register")
async def register_user(user_data: UserRegistrationModel):
    """Register new user with validated input."""
    # Input is automatically validated by Pydantic
    return await create_user_account(user_data.dict())

@app.post("/tracks/upload")
async def upload_track(track_data: TrackUploadModel,
                      current_user: dict = Depends(get_current_user)):
    """Upload track with validated metadata."""
    return await process_track_upload(track_data.dict(), current_user['user_id'])
```

This comprehensive security implementation guide covers the essential aspects of securing Tracktion services. Each example includes production-ready code that can be adapted to your specific security requirements and infrastructure.
