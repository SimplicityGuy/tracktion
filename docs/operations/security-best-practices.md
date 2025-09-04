# Security Best Practices

## Table of Contents

1. [Overview](#overview)
2. [Security Framework](#security-framework)
3. [Authentication and Authorization](#authentication-and-authorization)
4. [Data Protection](#data-protection)
5. [Network Security](#network-security)
6. [Container Security](#container-security)
7. [Application Security](#application-security)
8. [Infrastructure Security](#infrastructure-security)
9. [Compliance Requirements](#compliance-requirements)
10. [Security Monitoring](#security-monitoring)
11. [Incident Response](#incident-response)
12. [Security Testing](#security-testing)
13. [Developer Security Guidelines](#developer-security-guidelines)
14. [Operational Security](#operational-security)

## Overview

This document establishes comprehensive security best practices for the Tracktion audio analysis platform. These practices ensure the confidentiality, integrity, and availability of data and services while maintaining compliance with industry standards and regulatory requirements.

### Security Principles

- **Defense in Depth**: Multiple layers of security controls
- **Least Privilege**: Minimal access rights necessary for operations
- **Zero Trust**: Never trust, always verify
- **Security by Design**: Security integrated throughout development lifecycle
- **Continuous Monitoring**: Real-time threat detection and response
- **Regular Assessment**: Ongoing security evaluations and improvements

### Threat Landscape

#### Primary Threats
1. **Data Breaches**: Unauthorized access to sensitive audio files and user data
2. **Ransomware**: Malicious encryption of critical systems and data
3. **API Attacks**: Exploitation of service interfaces and endpoints
4. **Insider Threats**: Malicious or negligent actions by authorized users
5. **Supply Chain Attacks**: Compromised dependencies and third-party services
6. **Infrastructure Attacks**: Cloud service and container vulnerabilities

#### Attack Vectors
- Web application vulnerabilities
- Unpatched systems and dependencies
- Weak authentication mechanisms
- Misconfigured cloud services
- Social engineering and phishing
- Physical security breaches

## Security Framework

### Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Internet/Users                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 Web Application Firewall (WAF)                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Load Balancer/API Gateway                   │
│                  (Rate Limiting, SSL/TLS)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    Application Layer                           │
│    ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│    │ Analysis    │ File        │ Tracklist   │ Notification│    │
│    │ Service     │ Watcher     │ Service     │ Service     │    │
│    └─────────────┴─────────────┴─────────────┴─────────────┘    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     Data Layer                                 │
│    ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│    │ PostgreSQL  │ Redis       │ File        │ Backup      │    │
│    │ (Encrypted) │ (Auth)      │ Storage     │ Storage     │    │
│    └─────────────┴─────────────┴─────────────┴─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Security Controls Matrix

| Layer | Control Type | Implementation | Monitoring |
|-------|--------------|----------------|------------|
| **Network** | Firewall | iptables, Security Groups | Network flow logs |
| **Application** | Authentication | JWT, OAuth 2.0 | Auth logs, failed attempts |
| **Data** | Encryption | AES-256, TLS 1.3 | Key rotation, cert expiry |
| **Infrastructure** | Access Control | IAM, RBAC | Access logs, privilege changes |
| **Monitoring** | SIEM | ELK Stack, alerts | Security dashboard |

### Compliance Framework

#### Standards Adherence
- **OWASP Top 10**: Web application security risks mitigation
- **ISO 27001**: Information security management system
- **SOC 2 Type II**: Security, availability, and confidentiality controls
- **GDPR**: Data privacy and protection regulations
- **NIST Cybersecurity Framework**: Identify, Protect, Detect, Respond, Recover

#### Security Governance
```yaml
security_governance:
  policies:
    - Information Security Policy
    - Data Classification Policy
    - Access Control Policy
    - Incident Response Policy
    - Vendor Security Policy

  procedures:
    - Security Risk Assessment
    - Vulnerability Management
    - Security Training Program
    - Security Audit Process
    - Business Continuity Planning

  standards:
    - Secure Coding Standards
    - Encryption Standards
    - Password Policy
    - Network Security Standards
    - Cloud Security Standards
```

## Authentication and Authorization

### Multi-Factor Authentication (MFA)

#### Implementation
```python
# mfa_implementation.py
import pyotp
import qrcode
from cryptography.fernet import Fernet
import hashlib
import secrets

class MFAManager:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def generate_secret(self, user_id: str) -> tuple:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()

        # Store encrypted secret
        encrypted_secret = self.cipher.encrypt(secret.encode())

        # Generate QR code for user setup
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            user_id,
            issuer_name="Tracktion Audio Analysis"
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        return encrypted_secret, qr

    def verify_totp(self, user_secret: bytes, token: str) -> bool:
        """Verify TOTP token"""
        try:
            secret = self.cipher.decrypt(user_secret).decode()
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # Allow 30s window
        except Exception:
            return False

    def generate_backup_codes(self, user_id: str, count: int = 10) -> list:
        """Generate backup recovery codes"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(8).upper()
            codes.append(code)

        # Hash codes for storage
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]

        return codes, hashed_codes

# FastAPI MFA middleware
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()
mfa_manager = MFAManager()

async def verify_mfa_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mfa_token: str = None
):
    """Middleware to verify MFA token"""
    if not mfa_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="MFA token required"
        )

    # Get user's MFA secret from database
    user_secret = get_user_mfa_secret(credentials.credentials)

    if not mfa_manager.verify_totp(user_secret, mfa_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA token"
        )

    return True
```

#### MFA Configuration
```yaml
# mfa_config.yml
mfa:
  required_for:
    - admin_users: true
    - api_access: true
    - sensitive_operations: true
    - production_environment: true

  methods:
    totp:
      enabled: true
      algorithm: "SHA1"
      digits: 6
      period: 30
      issuer: "Tracktion"

    backup_codes:
      enabled: true
      count: 10
      single_use: true
      expiry_days: 90

    sms:
      enabled: false  # For future implementation
      provider: "twilio"

    email:
      enabled: true
      fallback_only: true
      code_length: 8
      expiry_minutes: 15

  enforcement:
    grace_period_days: 7
    max_attempts: 3
    lockout_duration_minutes: 15
    remember_device_days: 30
```

### JWT Token Security

#### Secure JWT Implementation
```python
# jwt_security.py
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import secrets
import redis

class JWTManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.access_token_expiry = timedelta(minutes=15)
        self.refresh_token_expiry = timedelta(days=7)

        # Generate RSA key pair for JWT signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()

        # Serialize keys
        self.private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        self.public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def create_access_token(self, user_id: str, roles: list = None) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        token_id = secrets.token_urlsafe(32)

        payload = {
            'iss': 'tracktion-auth-service',
            'sub': user_id,
            'aud': 'tracktion-api',
            'iat': now,
            'exp': now + self.access_token_expiry,
            'jti': token_id,
            'type': 'access',
            'roles': roles or [],
            'permissions': self.get_user_permissions(user_id, roles)
        }

        token = jwt.encode(payload, self.private_pem, algorithm='RS256')

        # Store token ID in Redis for revocation capability
        self.redis_client.setex(
            f"access_token:{token_id}",
            int(self.access_token_expiry.total_seconds()),
            user_id
        )

        return token

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        token_id = secrets.token_urlsafe(32)

        payload = {
            'iss': 'tracktion-auth-service',
            'sub': user_id,
            'aud': 'tracktion-auth',
            'iat': now,
            'exp': now + self.refresh_token_expiry,
            'jti': token_id,
            'type': 'refresh'
        }

        token = jwt.encode(payload, self.private_pem, algorithm='RS256')

        # Store refresh token with longer expiry
        self.redis_client.setex(
            f"refresh_token:{token_id}",
            int(self.refresh_token_expiry.total_seconds()),
            user_id
        )

        return token

    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.public_pem,
                algorithms=['RS256'],
                audience=['tracktion-api', 'tracktion-auth'],
                issuer='tracktion-auth-service'
            )

            # Check if token is revoked
            token_id = payload.get('jti')
            if not self.redis_client.exists(f"{payload['type']}_token:{token_id}"):
                raise jwt.InvalidTokenError("Token has been revoked")

            return payload

        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")

    def revoke_token(self, token: str):
        """Revoke a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.public_pem,
                algorithms=['RS256'],
                options={"verify_exp": False}  # Allow expired tokens for revocation
            )

            token_id = payload.get('jti')
            token_type = payload.get('type')

            # Remove from Redis
            self.redis_client.delete(f"{token_type}_token:{token_id}")

        except jwt.InvalidTokenError:
            pass  # Token was already invalid

    def get_user_permissions(self, user_id: str, roles: list) -> list:
        """Get user permissions based on roles"""
        # This would typically query a database
        permission_map = {
            'admin': ['*'],
            'user': ['read:own_data', 'write:own_data'],
            'analyst': ['read:all_data', 'analyze:audio'],
            'operator': ['read:system', 'manage:services']
        }

        permissions = set()
        for role in roles or []:
            permissions.update(permission_map.get(role, []))

        return list(permissions)
```

### Role-Based Access Control (RBAC)

#### RBAC Implementation
```python
# rbac_system.py
from enum import Enum
from typing import List, Set
from dataclasses import dataclass
import functools

class Permission(Enum):
    # Audio file permissions
    READ_AUDIO_FILES = "read:audio_files"
    WRITE_AUDIO_FILES = "write:audio_files"
    DELETE_AUDIO_FILES = "delete:audio_files"

    # Analysis permissions
    RUN_ANALYSIS = "run:analysis"
    VIEW_ANALYSIS_RESULTS = "view:analysis_results"
    EXPORT_ANALYSIS_DATA = "export:analysis_data"

    # System permissions
    MANAGE_USERS = "manage:users"
    MANAGE_SYSTEM = "manage:system"
    VIEW_LOGS = "view:logs"

    # Admin permissions
    FULL_ACCESS = "*"

class Role(Enum):
    GUEST = "guest"
    USER = "user"
    ANALYST = "analyst"
    OPERATOR = "operator"
    ADMIN = "admin"

@dataclass
class User:
    id: str
    username: str
    roles: List[Role]
    permissions: Set[Permission]

class RBACManager:
    def __init__(self):
        self.role_permissions = {
            Role.GUEST: {
                Permission.READ_AUDIO_FILES,
            },
            Role.USER: {
                Permission.READ_AUDIO_FILES,
                Permission.WRITE_AUDIO_FILES,
                Permission.RUN_ANALYSIS,
                Permission.VIEW_ANALYSIS_RESULTS,
            },
            Role.ANALYST: {
                Permission.READ_AUDIO_FILES,
                Permission.WRITE_AUDIO_FILES,
                Permission.RUN_ANALYSIS,
                Permission.VIEW_ANALYSIS_RESULTS,
                Permission.EXPORT_ANALYSIS_DATA,
            },
            Role.OPERATOR: {
                Permission.READ_AUDIO_FILES,
                Permission.VIEW_ANALYSIS_RESULTS,
                Permission.VIEW_LOGS,
                Permission.MANAGE_SYSTEM,
            },
            Role.ADMIN: {
                Permission.FULL_ACCESS,
            }
        }

    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user based on their roles"""
        permissions = set()

        for role in user.roles:
            permissions.update(self.role_permissions.get(role, set()))

        # Add any additional user-specific permissions
        permissions.update(user.permissions)

        return permissions

    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(user)

        # Check for full access
        if Permission.FULL_ACCESS in user_permissions:
            return True

        return permission in user_permissions

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user from request context
                user = get_current_user()  # Implementation depends on your auth system

                if not self.has_permission(user, permission):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Insufficient permissions. Required: {permission.value}"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Usage example
rbac = RBACManager()

@app.post("/analyze")
@rbac.require_permission(Permission.RUN_ANALYSIS)
async def analyze_audio(file: UploadFile):
    # Analysis logic here
    pass

@app.get("/users")
@rbac.require_permission(Permission.MANAGE_USERS)
async def list_users():
    # User management logic here
    pass
```

### API Authentication

#### API Key Management
```python
# api_key_management.py
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict
import uuid

class APIKeyManager:
    def __init__(self):
        self.key_prefix = "tk_"  # Tracktion key prefix
        self.key_length = 32

    def generate_api_key(self,
                        user_id: str,
                        name: str,
                        permissions: List[str] = None,
                        expires_in_days: int = 365) -> Dict:
        """Generate a new API key"""

        # Generate random key
        key_id = str(uuid.uuid4())
        key_secret = secrets.token_urlsafe(self.key_length)
        key_full = f"{self.key_prefix}{key_id}_{key_secret}"

        # Hash the secret for storage
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        # Set expiry
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Store key metadata
        key_metadata = {
            'id': key_id,
            'user_id': user_id,
            'name': name,
            'key_hash': key_hash,
            'permissions': permissions or [],
            'created_at': datetime.utcnow(),
            'expires_at': expires_at,
            'last_used': None,
            'usage_count': 0,
            'is_active': True
        }

        # Store in database (implementation depends on your DB)
        store_api_key(key_metadata)

        return {
            'key': key_full,
            'id': key_id,
            'expires_at': expires_at.isoformat()
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key and return key metadata"""

        if not api_key.startswith(self.key_prefix):
            return None

        try:
            # Extract key ID and secret
            key_part = api_key[len(self.key_prefix):]
            key_id, key_secret = key_part.split('_', 1)

            # Get key metadata from database
            key_metadata = get_api_key_metadata(key_id)

            if not key_metadata or not key_metadata['is_active']:
                return None

            # Verify key secret
            provided_hash = hashlib.sha256(key_secret.encode()).hexdigest()
            if not hmac.compare_digest(provided_hash, key_metadata['key_hash']):
                return None

            # Check expiry
            if datetime.utcnow() > key_metadata['expires_at']:
                return None

            # Update usage statistics
            update_api_key_usage(key_id)

            return key_metadata

        except (ValueError, KeyError):
            return None

    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key"""
        return deactivate_api_key(key_id, user_id)

    def list_user_api_keys(self, user_id: str) -> List[Dict]:
        """List all API keys for a user (without secrets)"""
        return get_user_api_keys(user_id)

# FastAPI middleware for API key authentication
from fastapi import HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param

api_key_manager = APIKeyManager()

async def api_key_auth(request: Request):
    """API key authentication middleware"""

    # Check Authorization header
    authorization = request.headers.get("Authorization")
    if authorization:
        scheme, param = get_authorization_scheme_param(authorization)
        if scheme.lower() == "bearer":
            key_metadata = api_key_manager.validate_api_key(param)
            if key_metadata:
                request.state.user_id = key_metadata['user_id']
                request.state.permissions = key_metadata['permissions']
                return

    # Check API key in headers
    api_key = request.headers.get("X-API-Key")
    if api_key:
        key_metadata = api_key_manager.validate_api_key(api_key)
        if key_metadata:
            request.state.user_id = key_metadata['user_id']
            request.state.permissions = key_metadata['permissions']
            return

    # Check API key in query parameters (less secure, for convenience)
    api_key = request.query_params.get("api_key")
    if api_key:
        key_metadata = api_key_manager.validate_api_key(api_key)
        if key_metadata:
            request.state.user_id = key_metadata['user_id']
            request.state.permissions = key_metadata['permissions']
            return

    raise HTTPException(
        status_code=401,
        detail="Valid API key required"
    )
```

## Data Protection

### Data Classification

#### Data Classification Framework
```yaml
# data_classification.yml
data_classification:
  levels:
    public:
      description: "Data that can be freely shared"
      examples: ["marketing materials", "public documentation"]
      security_controls: ["basic_access_logging"]

    internal:
      description: "Data for internal use only"
      examples: ["system logs", "internal documentation"]
      security_controls: ["access_logging", "employee_access_only"]

    confidential:
      description: "Sensitive business data"
      examples: ["user_data", "business_metrics", "audio_files"]
      security_controls: ["encryption_at_rest", "encryption_in_transit", "access_logging", "role_based_access"]

    restricted:
      description: "Highly sensitive data requiring special handling"
      examples: ["authentication_credentials", "encryption_keys", "financial_data"]
      security_controls: ["strong_encryption", "multi_factor_auth", "audit_logging", "data_loss_prevention"]

  handling_requirements:
    confidential:
      encryption:
        at_rest: "AES-256"
        in_transit: "TLS 1.3"
      access_control: "RBAC with MFA"
      retention: "7 years"
      backup: "encrypted, offsite"

    restricted:
      encryption:
        at_rest: "AES-256 with HSM"
        in_transit: "TLS 1.3 with mutual auth"
      access_control: "RBAC with MFA and approval workflow"
      retention: "7 years"
      backup: "encrypted, air-gapped"
```

### Encryption Implementation

#### Database Encryption
```python
# database_encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

class DatabaseEncryption:
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)

    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or key management service"""
        key_material = os.environ.get('DB_ENCRYPTION_KEY')
        if not key_material:
            # In production, this should come from AWS KMS, Azure Key Vault, etc.
            raise ValueError("DB_ENCRYPTION_KEY environment variable not set")

        # Derive key from password using PBKDF2
        salt = b'stable_salt_for_db'  # In production, use random salt per key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material.encode()))
        return key

    def encrypt_field(self, plaintext: str) -> str:
        """Encrypt a database field"""
        if plaintext is None:
            return None
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_field(self, ciphertext: str) -> str:
        """Decrypt a database field"""
        if ciphertext is None:
            return None
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()

# SQLAlchemy model with encrypted fields
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()
db_encryption = DatabaseEncryption()

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True)
    username = Column(String, nullable=False)
    _email = Column('email', Text)  # Stored encrypted
    _phone = Column('phone', Text)  # Stored encrypted
    created_at = Column(DateTime)

    @hybrid_property
    def email(self):
        return db_encryption.decrypt_field(self._email) if self._email else None

    @email.setter
    def email(self, value):
        self._email = db_encryption.encrypt_field(value) if value else None

    @hybrid_property
    def phone(self):
        return db_encryption.decrypt_field(self._phone) if self._phone else None

    @phone.setter
    def phone(self, value):
        self._phone = db_encryption.encrypt_field(value) if value else None

# PostgreSQL Transparent Data Encryption (TDE) configuration
# This would be configured at the database level
postgresql_tde_config = """
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'

# Enable encryption for specific tablespaces
# CREATE TABLESPACE encrypted_ts
# LOCATION '/encrypted_data'
# WITH (encryption_key_id = 'key1');
"""
```

#### File Storage Encryption
```python
# file_encryption.py
import os
import boto3
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

class FileEncryption:
    def __init__(self):
        self.kms_client = boto3.client('kms')
        self.key_id = os.environ.get('AWS_KMS_KEY_ID')

    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """Encrypt a file using AWS KMS and AES-256-GCM"""

        # Generate data encryption key (DEK)
        response = self.kms_client.generate_data_key(
            KeyId=self.key_id,
            KeySpec='AES_256'
        )

        plaintext_key = response['Plaintext']
        encrypted_key = response['CiphertextBlob']

        # Generate random IV
        iv = secrets.token_bytes(12)  # 96 bits for GCM

        # Encrypt file content
        cipher = Cipher(algorithms.AES(plaintext_key), modes.GCM(iv))
        encryptor = cipher.encryptor()

        output_path = output_path or f"{file_path}.encrypted"

        with open(file_path, 'rb') as infile, open(output_path, 'wb') as outfile:
            # Write header: encrypted_key_length + encrypted_key + iv
            outfile.write(len(encrypted_key).to_bytes(4, 'big'))
            outfile.write(encrypted_key)
            outfile.write(iv)

            # Encrypt and write file content
            while chunk := infile.read(8192):
                outfile.write(encryptor.update(chunk))

            # Write authentication tag
            outfile.write(encryptor.finalize())
            outfile.write(encryptor.tag)

        # Securely clear the plaintext key
        plaintext_key = b'\x00' * len(plaintext_key)

        return output_path

    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """Decrypt a file encrypted with encrypt_file"""

        output_path = output_path or encrypted_file_path.replace('.encrypted', '')

        with open(encrypted_file_path, 'rb') as infile:
            # Read header
            key_length = int.from_bytes(infile.read(4), 'big')
            encrypted_key = infile.read(key_length)
            iv = infile.read(12)

            # Decrypt the data encryption key
            response = self.kms_client.decrypt(CiphertextBlob=encrypted_key)
            plaintext_key = response['Plaintext']

            # Read encrypted content (all except last 16 bytes which is the tag)
            encrypted_content = infile.read()[:-16]
            infile.seek(-16, 2)  # Seek to last 16 bytes
            tag = infile.read(16)

            # Decrypt content
            cipher = Cipher(algorithms.AES(plaintext_key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()

            with open(output_path, 'wb') as outfile:
                decrypted = decryptor.update(encrypted_content)
                outfile.write(decrypted)
                decryptor.finalize()  # Verify authentication

            # Securely clear the plaintext key
            plaintext_key = b'\x00' * len(plaintext_key)

        return output_path

# S3 server-side encryption configuration
s3_encryption_config = {
    "Rules": [
        {
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms",
                "KMSMasterKeyID": "arn:aws:kms:region:account:key/key-id"
            },
            "BucketKeyEnabled": True
        }
    ]
}
```

### Data Loss Prevention (DLP)

#### DLP Implementation
```python
# dlp_scanner.py
import re
import hashlib
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class SensitivityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DLPRule:
    name: str
    pattern: str
    sensitivity: SensitivityLevel
    description: str
    action: str  # log, block, redact

class DLPScanner:
    def __init__(self):
        self.rules = [
            # Credit card numbers
            DLPRule(
                name="credit_card",
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                sensitivity=SensitivityLevel.CRITICAL,
                description="Credit card number detected",
                action="block"
            ),

            # Social Security Numbers
            DLPRule(
                name="ssn",
                pattern=r'\b(?!000|666|9\d{2})\d{3}[-.]?(?!00)\d{2}[-.]?(?!0000)\d{4}\b',
                sensitivity=SensitivityLevel.CRITICAL,
                description="Social Security Number detected",
                action="block"
            ),

            # Email addresses
            DLPRule(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                sensitivity=SensitivityLevel.MEDIUM,
                description="Email address detected",
                action="log"
            ),

            # Phone numbers
            DLPRule(
                name="phone",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                sensitivity=SensitivityLevel.MEDIUM,
                description="Phone number detected",
                action="redact"
            ),

            # API keys (generic pattern)
            DLPRule(
                name="api_key",
                pattern=r'\b[A-Za-z0-9]{32,}\b',
                sensitivity=SensitivityLevel.HIGH,
                description="Potential API key detected",
                action="block"
            ),

            # Private keys
            DLPRule(
                name="private_key",
                pattern=r'-----BEGIN (?:RSA )?PRIVATE KEY-----',
                sensitivity=SensitivityLevel.CRITICAL,
                description="Private key detected",
                action="block"
            )
        ]

    def scan_text(self, text: str, context: str = "") -> List[Dict]:
        """Scan text for sensitive data"""
        findings = []

        for rule in self.rules:
            matches = re.finditer(rule.pattern, text, re.IGNORECASE)

            for match in matches:
                finding = {
                    'rule_name': rule.name,
                    'sensitivity': rule.sensitivity.value,
                    'description': rule.description,
                    'action': rule.action,
                    'match': match.group(),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'context': context,
                    'hash': hashlib.sha256(match.group().encode()).hexdigest()[:16]
                }
                findings.append(finding)

        return findings

    def scan_file(self, file_path: str) -> List[Dict]:
        """Scan file for sensitive data"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return self.scan_text(content, context=file_path)
        except Exception as e:
            return [{
                'rule_name': 'scan_error',
                'sensitivity': 'high',
                'description': f'Error scanning file: {str(e)}',
                'action': 'log',
                'context': file_path
            }]

    def redact_text(self, text: str) -> str:
        """Redact sensitive information from text"""
        redacted_text = text

        for rule in self.rules:
            if rule.action == "redact":
                # Replace with asterisks, keeping first and last character
                def redact_match(match):
                    matched_text = match.group()
                    if len(matched_text) <= 2:
                        return '*' * len(matched_text)
                    return matched_text[0] + '*' * (len(matched_text) - 2) + matched_text[-1]

                redacted_text = re.sub(rule.pattern, redact_match, redacted_text, flags=re.IGNORECASE)

        return redacted_text

# FastAPI middleware for DLP scanning
from fastapi import Request, HTTPException
import json

dlp_scanner = DLPScanner()

async def dlp_middleware(request: Request, call_next):
    """DLP middleware to scan request data"""

    # Skip DLP for certain endpoints or content types
    if request.url.path in ['/health', '/metrics']:
        return await call_next(request)

    # Scan request body for sensitive data
    if request.method in ['POST', 'PUT', 'PATCH']:
        body = await request.body()

        if body:
            try:
                # Try to decode as JSON
                text_content = body.decode('utf-8')
                findings = dlp_scanner.scan_text(text_content, context=f"{request.method} {request.url.path}")

                # Check for critical findings that should block the request
                critical_findings = [f for f in findings if f['action'] == 'block']

                if critical_findings:
                    # Log the incident
                    log_dlp_incident(request, critical_findings)

                    raise HTTPException(
                        status_code=400,
                        detail="Request contains sensitive data that cannot be processed"
                    )

                # Log other findings
                if findings:
                    log_dlp_findings(request, findings)

            except UnicodeDecodeError:
                # Binary content, skip DLP scanning
                pass

    response = await call_next(request)
    return response

def log_dlp_incident(request: Request, findings: List[Dict]):
    """Log DLP security incident"""
    incident_log = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': 'dlp_violation',
        'severity': 'high',
        'source_ip': request.client.host,
        'user_agent': request.headers.get('User-Agent'),
        'endpoint': f"{request.method} {request.url.path}",
        'findings': findings
    }

    # Send to SIEM system
    send_to_siem(incident_log)

    # Alert security team
    send_security_alert(incident_log)
```

## Network Security

### Firewall Configuration

#### iptables Rules
```bash
#!/bin/bash
# firewall_config.sh

# Clear existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback traffic
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j REJECT --reject-with tcp-reset
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application ports (with rate limiting)
# Analysis Service
iptables -A INPUT -p tcp --dport 8000 -m limit --limit 100/sec --limit-burst 200 -j ACCEPT

# File Watcher Service
iptables -A INPUT -p tcp --dport 8001 -m limit --limit 50/sec --limit-burst 100 -j ACCEPT

# Tracklist Service
iptables -A INPUT -p tcp --dport 8002 -m limit --limit 100/sec --limit-burst 200 -j ACCEPT

# Notification Service
iptables -A INPUT -p tcp --dport 8003 -m limit --limit 25/sec --limit-burst 50 -j ACCEPT

# Allow database connections (internal only)
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -s 172.16.0.0/12 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -s 192.168.0.0/16 -j ACCEPT

# Allow Redis connections (internal only)
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -s 172.16.0.0/12 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -s 192.168.0.0/16 -j ACCEPT

# Allow ICMP (ping) but rate limit
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 5/sec -j ACCEPT

# Log and drop all other traffic
iptables -A INPUT -m limit --limit 3/min --limit-burst 3 -j LOG --log-prefix "[FIREWALL BLOCK] "
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4

echo "Firewall configuration applied successfully"
```

#### Network Segmentation
```yaml
# network_segmentation.yml
network_architecture:
  dmz:
    description: "Public-facing services"
    cidr: "10.0.1.0/24"
    services:
      - load_balancer
      - web_application_firewall
      - reverse_proxy

  application_tier:
    description: "Application services"
    cidr: "10.0.2.0/24"
    services:
      - analysis_service
      - file_watcher
      - tracklist_service
      - notification_service

  data_tier:
    description: "Database and storage"
    cidr: "10.0.3.0/24"
    services:
      - postgresql
      - redis
      - file_storage

  management:
    description: "Administrative and monitoring"
    cidr: "10.0.4.0/24"
    services:
      - monitoring
      - logging
      - backup_systems

security_groups:
  web_tier:
    ingress:
      - protocol: tcp
        port: 80
        source: "0.0.0.0/0"
      - protocol: tcp
        port: 443
        source: "0.0.0.0/0"
    egress:
      - protocol: tcp
        port: 8000-8003
        destination: "application_tier"

  app_tier:
    ingress:
      - protocol: tcp
        port: 8000-8003
        source: "dmz"
      - protocol: tcp
        port: 22
        source: "management"
    egress:
      - protocol: tcp
        port: 5432
        destination: "data_tier"
      - protocol: tcp
        port: 6379
        destination: "data_tier"

  data_tier:
    ingress:
      - protocol: tcp
        port: 5432
        source: "application_tier"
      - protocol: tcp
        port: 6379
        source: "application_tier"
    egress:
      - protocol: tcp
        port: 443
        destination: "0.0.0.0/0"  # For backups to cloud storage
```

### TLS/SSL Configuration

#### TLS Best Practices
```nginx
# nginx_ssl_config.conf
server {
    listen 443 ssl http2;
    server_name tracktion.com;

    # TLS Configuration
    ssl_certificate /etc/ssl/certs/tracktion.crt;
    ssl_certificate_key /etc/ssl/private/tracktion.key;

    # TLS Protocol Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;

    # Strong Cipher Suites
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;

    # SSL Session Configuration
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_session_tickets off;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-certs.pem;
    resolver 8.8.8.8 8.8.4.4 valid=300s;
    resolver_timeout 5s;

    # Security Headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; child-src 'none'; worker-src 'none'; frame-ancestors 'none'; form-action 'self'; base-uri 'self';" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;

    # Main application proxy
    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://app_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Proxy timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffer configuration
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # API endpoints with stricter rate limiting
    location /api/auth/login {
        limit_req zone=login burst=3 nodelay;
        proxy_pass http://app_backend;
    }

    # File upload with size limits
    location /api/upload {
        client_max_body_size 100M;
        proxy_pass http://app_backend;
        proxy_request_buffering off;
    }

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://app_backend;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name tracktion.com;
    return 301 https://$server_name$request_uri;
}
```

#### Certificate Management
```bash
#!/bin/bash
# certificate_management.sh

# Certificate renewal with Let's Encrypt
renew_certificates() {
    echo "Renewing SSL certificates..."

    # Stop services that use port 80
    docker-compose stop nginx

    # Renew certificates
    certbot renew --standalone --non-interactive

    if [ $? -eq 0 ]; then
        echo "Certificates renewed successfully"

        # Copy new certificates to application directory
        cp /etc/letsencrypt/live/tracktion.com/fullchain.pem /opt/tracktion/ssl/
        cp /etc/letsencrypt/live/tracktion.com/privkey.pem /opt/tracktion/ssl/

        # Restart services
        docker-compose start nginx

        # Test SSL configuration
        openssl s_client -connect tracktion.com:443 -servername tracktion.com < /dev/null

        echo "Certificate renewal completed"
    else
        echo "Certificate renewal failed"

        # Restart services even if renewal failed
        docker-compose start nginx

        exit 1
    fi
}

# Check certificate expiry
check_certificate_expiry() {
    echo "Checking certificate expiry..."

    CERT_FILE="/etc/letsencrypt/live/tracktion.com/cert.pem"

    if [ -f "$CERT_FILE" ]; then
        EXPIRY_DATE=$(openssl x509 -enddate -noout -in "$CERT_FILE" | cut -d= -f2)
        EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
        CURRENT_EPOCH=$(date +%s)
        DAYS_UNTIL_EXPIRY=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))

        echo "Certificate expires in $DAYS_UNTIL_EXPIRY days"

        if [ $DAYS_UNTIL_EXPIRY -lt 30 ]; then
            echo "WARNING: Certificate expires in less than 30 days"

            # Send alert
            curl -X POST -H 'Content-type: application/json' \
                --data '{"text":"⚠️ SSL Certificate expires in '"$DAYS_UNTIL_EXPIRY"' days for tracktion.com"}' \
                "$SLACK_WEBHOOK_URL"
        fi

        if [ $DAYS_UNTIL_EXPIRY -lt 7 ]; then
            echo "CRITICAL: Certificate expires in less than 7 days - attempting renewal"
            renew_certificates
        fi
    else
        echo "Certificate file not found: $CERT_FILE"
        exit 1
    fi
}

# Generate Certificate Signing Request (CSR) for manual certificates
generate_csr() {
    DOMAIN="$1"

    if [ -z "$DOMAIN" ]; then
        echo "Usage: generate_csr <domain>"
        exit 1
    fi

    # Create private key
    openssl genrsa -out "${DOMAIN}.key" 2048

    # Create CSR configuration
    cat > "${DOMAIN}.conf" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Tracktion Inc
CN = $DOMAIN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = www.$DOMAIN
EOF

    # Generate CSR
    openssl req -new -key "${DOMAIN}.key" -out "${DOMAIN}.csr" -config "${DOMAIN}.conf"

    echo "Private key: ${DOMAIN}.key"
    echo "CSR: ${DOMAIN}.csr"
    echo "Submit the CSR to your Certificate Authority"
}

# Main execution
case "$1" in
    "renew")
        renew_certificates
        ;;
    "check")
        check_certificate_expiry
        ;;
    "csr")
        generate_csr "$2"
        ;;
    *)
        echo "Usage: $0 {renew|check|csr <domain>}"
        exit 1
        ;;
esac
```

## Container Security

### Docker Security Best Practices

#### Dockerfile Security
```dockerfile
# Secure Dockerfile for Analysis Service
FROM python:3.11-slim

# Create non-root user
RUN groupadd --gid 1000 tracktion && \
    useradd --uid 1000 --gid tracktion --shell /bin/bash --create-home tracktion

# Install security updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=tracktion:tracktion . .

# Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -exec rm -rf {} + || true

# Set file permissions
RUN chmod -R 755 /app && \
    chmod 644 /app/config/*.conf

# Switch to non-root user
USER tracktion

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set security-related environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose Security
```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  analysis-service:
    build: ./services/analysis_service
    restart: unless-stopped
    networks:
      - app_network
    environment:
      - PYTHONPATH=/app
    volumes:
      - analysis_data:/app/data:rw
      - analysis_logs:/app/logs:rw
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:exec,nosuid,nodev,size=100m
      - /var/tmp:exec,nosuid,nodev,size=100m
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  database:
    image: postgres:15-alpine
    restart: unless-stopped
    networks:
      - db_network
    environment:
      - POSTGRES_DB_FILE=/run/secrets/postgres_db
      - POSTGRES_USER_FILE=/run/secrets/postgres_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    volumes:
      - db_data:/var/lib/postgresql/data:rw
      - ./database/init:/docker-entrypoint-initdb.d:ro
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - SETGID
      - SETUID
    secrets:
      - postgres_db
      - postgres_user
      - postgres_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  app_network:
    driver: bridge
    internal: false
    attachable: false
    ipam:
      config:
        - subnet: 172.20.0.0/16

  db_network:
    driver: bridge
    internal: true
    attachable: false

volumes:
  analysis_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/tracktion/data/analysis
      o: bind,uid=1000,gid=1000,mode=0755

  db_data:
    driver: local
    driver_opts:
      type: none
      device: /opt/tracktion/data/postgres
      o: bind,uid=999,gid=999,mode=0700

secrets:
  postgres_db:
    file: ./secrets/postgres_db.txt
  postgres_user:
    file: ./secrets/postgres_user.txt
  postgres_password:
    file: ./secrets/postgres_password.txt
```

### Container Runtime Security

#### Container Security Scanning
```bash
#!/bin/bash
# container_security_scan.sh

set -e

SCAN_RESULTS_DIR="/var/log/security_scans"
mkdir -p "$SCAN_RESULTS_DIR"

# Function to scan image with Trivy
scan_with_trivy() {
    local image="$1"
    local report_file="$SCAN_RESULTS_DIR/trivy_$(echo "$image" | tr '/' '_' | tr ':' '_')_$(date +%Y%m%d).json"

    echo "Scanning $image with Trivy..."

    trivy image \
        --format json \
        --output "$report_file" \
        --severity HIGH,CRITICAL \
        --exit-code 1 \
        "$image"

    if [ $? -eq 0 ]; then
        echo "✅ $image: No high/critical vulnerabilities found"
    else
        echo "❌ $image: High/critical vulnerabilities detected"

        # Extract vulnerability summary
        CRITICAL_COUNT=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$report_file")
        HIGH_COUNT=$(jq '[.Results[].Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$report_file")

        echo "  Critical: $CRITICAL_COUNT, High: $HIGH_COUNT"

        # Send alert if critical vulnerabilities found
        if [ "$CRITICAL_COUNT" -gt 0 ]; then
            send_security_alert "$image" "$CRITICAL_COUNT" "$HIGH_COUNT"
        fi

        return 1
    fi
}

# Function to scan image with Docker Scout (if available)
scan_with_scout() {
    local image="$1"

    if command -v docker &> /dev/null && docker scout version &> /dev/null; then
        echo "Scanning $image with Docker Scout..."

        docker scout cves "$image" --format json > "$SCAN_RESULTS_DIR/scout_$(echo "$image" | tr '/' '_' | tr ':' '_')_$(date +%Y%m%d).json"

        # Check for critical vulnerabilities
        docker scout cves "$image" --exit-code --only-severity critical

        if [ $? -ne 0 ]; then
            echo "❌ $image: Critical vulnerabilities found by Docker Scout"
            return 1
        else
            echo "✅ $image: No critical vulnerabilities found by Docker Scout"
        fi
    else
        echo "Docker Scout not available, skipping..."
    fi
}

# Function to check image configuration
check_image_config() {
    local image="$1"

    echo "Checking configuration for $image..."

    # Get image configuration
    CONFIG=$(docker inspect "$image" | jq '.[0].Config')

    # Check if running as root
    USER=$(echo "$CONFIG" | jq -r '.User // "root"')
    if [ "$USER" = "root" ] || [ "$USER" = "" ]; then
        echo "⚠️ $image: Running as root user"
    else
        echo "✅ $image: Running as non-root user ($USER)"
    fi

    # Check for exposed ports
    EXPOSED_PORTS=$(echo "$CONFIG" | jq -r '.ExposedPorts // {} | keys[]')
    if [ -n "$EXPOSED_PORTS" ]; then
        echo "📢 $image: Exposed ports: $EXPOSED_PORTS"
    fi

    # Check environment variables for secrets
    ENV_VARS=$(echo "$CONFIG" | jq -r '.Env[]? // empty')
    if echo "$ENV_VARS" | grep -i -E "(password|secret|key|token)" > /dev/null; then
        echo "⚠️ $image: Potential secrets in environment variables"
        echo "$ENV_VARS" | grep -i -E "(password|secret|key|token)"
    fi
}

# Function to send security alert
send_security_alert() {
    local image="$1"
    local critical_count="$2"
    local high_count="$3"

    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 Security Alert: Image '"$image"' has '"$critical_count"' critical and '"$high_count"' high severity vulnerabilities"}' \
        "$SLACK_WEBHOOK_URL" || true
}

# Main scanning logic
echo "Starting container security scan..."

# Get list of images to scan
IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>")

FAILED_SCANS=0
TOTAL_SCANS=0

for image in $IMAGES; do
    echo "----------------------------------------"
    echo "Scanning: $image"

    ((TOTAL_SCANS++))

    # Skip scanning base images that we don't control
    if [[ "$image" =~ ^(postgres|redis|nginx|alpine): ]]; then
        echo "Skipping base image: $image"
        continue
    fi

    # Scan with Trivy
    if ! scan_with_trivy "$image"; then
        ((FAILED_SCANS++))
    fi

    # Scan with Docker Scout
    scan_with_scout "$image" || true

    # Check image configuration
    check_image_config "$image"
done

echo "----------------------------------------"
echo "Scan Summary:"
echo "Total images scanned: $TOTAL_SCANS"
echo "Images with issues: $FAILED_SCANS"
echo "Clean images: $((TOTAL_SCANS - FAILED_SCANS))"

# Generate HTML report
generate_html_report() {
    cat > "$SCAN_RESULTS_DIR/security_report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Container Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 10px; }
        .summary { margin: 20px 0; }
        .critical { color: #d32f2f; }
        .high { color: #f57c00; }
        .medium { color: #fbc02d; }
        .low { color: #388e3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Container Security Scan Report</h1>
        <p>Generated: $(date)</p>
    </div>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total images scanned: $TOTAL_SCANS</p>
        <p>Images with vulnerabilities: $FAILED_SCANS</p>
        <p>Clean images: $((TOTAL_SCANS - FAILED_SCANS))</p>
    </div>
    <div class="details">
        <h2>Detailed Results</h2>
        <!-- Detailed results would be populated here -->
    </div>
</body>
</html>
EOF
}

generate_html_report

echo "Security scan completed. Results in: $SCAN_RESULTS_DIR"

# Exit with error if any scans failed
if [ $FAILED_SCANS -gt 0 ]; then
    exit 1
fi
```

### Runtime Security Monitoring

#### Container Runtime Monitoring with Falco
```yaml
# falco_rules.yml
- rule: Unauthorized Process in Container
  desc: Detect processes not expected to run in containers
  condition: >
    spawned_process and container and
    not proc.name in (python, uvicorn, gunicorn, sh, bash, curl, postgres, redis-server)
  output: >
    Unauthorized process started in container (user=%user.name command=%proc.cmdline
    container=%container.name image=%container.image.repository:%container.image.tag)
  priority: WARNING

- rule: Container Drift Detection
  desc: Detect file system changes in containers
  condition: >
    open_write and container and
    fd.typechar='f' and
    not fd.name startswith /tmp and
    not fd.name startswith /var/tmp and
    not proc.name in (python, postgres, redis-server)
  output: >
    File modified in container (user=%user.name command=%proc.cmdline
    file=%fd.name container=%container.name image=%container.image.repository:%container.image.tag)
  priority: INFO

- rule: Sensitive File Access in Container
  desc: Detect access to sensitive files
  condition: >
    open_read and container and
    (fd.name startswith /etc/passwd or
     fd.name startswith /etc/shadow or
     fd.name contains secret or
     fd.name contains key)
  output: >
    Sensitive file accessed in container (user=%user.name command=%proc.cmdline
    file=%fd.name container=%container.name)
  priority: WARNING

- rule: Network Connection from Container
  desc: Monitor network connections from containers
  condition: >
    outbound and container and
    not fd.sip in (private_ip_ranges) and
    not fd.sport in (53, 80, 443)
  output: >
    Outbound connection from container (user=%user.name command=%proc.cmdline
    connection=%fd.sip:%fd.sport->%fd.cip:%fd.cport container=%container.name)
  priority: INFO

- rule: Privileged Container Started
  desc: Detect privileged containers
  condition: >
    container_started and
    container.privileged=true
  output: >
    Privileged container started (user=%user.name command=%proc.cmdline
    container=%container.name image=%container.image.repository:%container.image.tag)
  priority: WARNING
```

This comprehensive security best practices guide provides detailed implementation guidance for securing all aspects of the Tracktion system, from authentication and authorization to container security and network protection. Regular review and updates of these practices ensure the system remains secure against evolving threats.
