# API Authentication Guide

This guide covers all authentication methods supported by the Tracktion API, including API keys, JWT tokens, OAuth2, and security measures.

## Table of Contents
- [Authentication Methods](#authentication-methods)
- [API Key Authentication](#api-key-authentication)
- [JWT Token Authentication](#jwt-token-authentication)
- [OAuth2 Authentication](#oauth2-authentication)
- [Rate Limiting](#rate-limiting)
- [Security Measures](#security-measures)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Authentication Methods

Tracktion API supports multiple authentication methods:

1. **API Key Authentication** - For server-to-server communication
2. **JWT Token Authentication** - For web applications and short-lived sessions
3. **OAuth2 Authentication** - For third-party integrations

## API Key Authentication

### Overview
API keys provide secure, long-lived authentication for server-to-server communication. Each API key has specific permissions and rate limits based on your subscription tier.

### Creating API Keys

**Endpoint**: `POST /api/v1/developer/keys`

**Request**:
```json
{
  "name": "My Application Key",
  "permissions": ["read", "write"],
  "scopes": ["search", "tracklists"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

**Response**:
```json
{
  "id": "key_1234567890abcdef",
  "name": "My Application Key",
  "key": "tk_live_1234567890abcdef...",
  "permissions": ["read", "write"],
  "scopes": ["search", "tracklists"],
  "created_at": "2024-01-01T00:00:00Z",
  "expires_at": "2024-12-31T23:59:59Z",
  "last_used": null,
  "is_active": true
}
```

### Using API Keys

Include the API key in the `Authorization` header:

```http
GET /api/v1/search/tracks?q=electronic
Authorization: Bearer tk_live_1234567890abcdef...
Content-Type: application/json
```

### API Key Management

**List API Keys**:
```http
GET /api/v1/developer/keys
Authorization: Bearer <your-jwt-token>
```

**Rotate API Key**:
```http
POST /api/v1/developer/keys/{key_id}/rotate
Authorization: Bearer <your-jwt-token>
```

**Revoke API Key**:
```http
DELETE /api/v1/developer/keys/{key_id}
Authorization: Bearer <your-jwt-token>
```

### API Key Scopes

| Scope | Description | Permissions |
|-------|-------------|-------------|
| `search` | Search tracks and tracklists | Read-only access to search endpoints |
| `tracklists` | Access tracklist data | Read/write access to tracklist endpoints |
| `analytics` | View usage analytics | Read access to analytics data |
| `batch` | Batch processing | Access to batch processing endpoints |
| `admin` | Administrative functions | Full access (enterprise tier only) |

## JWT Token Authentication

### Overview
JWT tokens provide secure, short-lived authentication for web applications and mobile apps. Tokens contain user information and permissions.

### Obtaining JWT Tokens

**Endpoint**: `POST /api/v1/auth/token`

**Request**:
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "tier": "premium"
  }
}
```

### Using JWT Tokens

Include the JWT token in the `Authorization` header:

```http
GET /api/v1/user/profile
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

### Refreshing JWT Tokens

**Endpoint**: `POST /api/v1/auth/refresh`

**Request**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### JWT Token Structure

JWT tokens contain the following claims:

```json
{
  "sub": "user_123",
  "email": "user@example.com",
  "tier": "premium",
  "permissions": ["read", "write"],
  "iat": 1640995200,
  "exp": 1640998800
}
```

## OAuth2 Authentication

### Overview
OAuth2 provides secure authentication for third-party applications without sharing credentials.

### Supported Flows
- **Authorization Code Flow** - For web applications
- **Client Credentials Flow** - For server-to-server communication

### Authorization Code Flow

**Step 1: Authorization Request**
```http
GET /api/v1/oauth/authorize?
  response_type=code&
  client_id=your_client_id&
  redirect_uri=https://yourapp.com/callback&
  scope=search+tracklists&
  state=random_state_string
```

**Step 2: Exchange Code for Token**
```http
POST /api/v1/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&
code=AUTH_CODE&
client_id=your_client_id&
client_secret=your_client_secret&
redirect_uri=https://yourapp.com/callback
```

**Response**:
```json
{
  "access_token": "oauth_token_123...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_456...",
  "scope": "search tracklists"
}
```

### Client Credentials Flow

**Request**:
```http
POST /api/v1/oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret&
scope=search
```

## Rate Limiting

### Rate Limit Tiers

| Tier | Requests/Minute | Burst Allowance | Quota/Month |
|------|-----------------|-----------------|-------------|
| Free | 10 | 12 | 1,000 |
| Premium | 100 | 120 | 10,000 |
| Enterprise | 1,000 | 1,200 | 100,000 |

### Rate Limit Headers

All API responses include rate limiting information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640998800
X-RateLimit-Retry-After: 60
```

### Rate Limit Exceeded

When rate limits are exceeded, the API returns a 429 status:

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640998800
X-RateLimit-Retry-After: 60

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Please wait 60 seconds before making another request.",
  "retry_after": 60
}
```

## Security Measures

### HMAC Request Signing

For sensitive operations, requests must include HMAC-SHA256 signatures:

```http
POST /api/v1/sensitive-operation
Authorization: Bearer tk_live_1234567890abcdef...
X-Timestamp: 1640995200
X-Signature: sha256=abc123def456...
Content-Type: application/json

{
  "data": "sensitive information"
}
```

**Signature Calculation**:
```python
import hmac
import hashlib
import time

def generate_signature(secret_key, timestamp, method, path, body):
    message = f"{timestamp}.{method}.{path}.{body}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"
```

### IP Whitelist/Blacklist

Configure IP access control through the developer dashboard:

**Add IP to Whitelist**:
```http
POST /api/v1/security/ip-whitelist
Authorization: Bearer <admin-token>

{
  "ip_address": "203.0.113.1",
  "reason": "Production server",
  "expires_at": "2024-12-31T23:59:59Z"
}
```

### Abuse Detection

The API automatically detects and blocks abusive behavior:

- **High frequency requests** - Unusual request patterns
- **Suspicious patterns** - Bot-like behavior detection
- **Invalid requests** - Repeated authentication failures
- **Resource abuse** - Excessive data consumption
- **Quota exhaustion** - Rapid quota consumption

## Error Handling

### Authentication Errors

**Invalid API Key**:
```json
{
  "error": "invalid_api_key",
  "message": "The provided API key is invalid or has been revoked.",
  "code": 401
}
```

**Expired JWT Token**:
```json
{
  "error": "token_expired",
  "message": "The JWT token has expired. Please refresh your token.",
  "code": 401
}
```

**Insufficient Permissions**:
```json
{
  "error": "insufficient_permissions",
  "message": "Your API key does not have permission to access this resource.",
  "code": 403,
  "required_scopes": ["admin"]
}
```

### Security Errors

**IP Address Blocked**:
```json
{
  "error": "ip_blocked",
  "message": "Your IP address has been blocked due to suspicious activity.",
  "code": 403
}
```

**Invalid Signature**:
```json
{
  "error": "invalid_signature",
  "message": "The request signature is invalid or expired.",
  "code": 401
}
```

## Best Practices

### API Key Security
1. **Store securely** - Never expose API keys in client-side code
2. **Rotate regularly** - Rotate API keys every 90 days
3. **Use minimal scopes** - Only request necessary permissions
4. **Monitor usage** - Regularly check API key usage in dashboard
5. **Revoke unused keys** - Remove API keys that are no longer needed

### Rate Limiting
1. **Implement exponential backoff** - Handle 429 errors gracefully
2. **Cache responses** - Reduce API calls by caching results
3. **Batch requests** - Use batch endpoints when available
4. **Monitor quotas** - Track monthly usage to avoid limits
5. **Upgrade tier** - Consider higher tiers for increased limits

### JWT Tokens
1. **Store securely** - Use secure storage (HttpOnly cookies, secure keychain)
2. **Handle expiry** - Implement automatic token refresh
3. **Validate tokens** - Verify token signature and claims
4. **Use HTTPS** - Always use HTTPS for token transmission
5. **Implement logout** - Properly invalidate tokens on logout

### HMAC Signatures
1. **Protect secret keys** - Store HMAC secrets securely
2. **Use current timestamp** - Include recent timestamp in signature
3. **Validate server-side** - Always verify signatures on server
4. **Handle replay attacks** - Implement timestamp validation
5. **Rotate secrets** - Regularly rotate HMAC secret keys

### Error Handling
1. **Check status codes** - Always check HTTP status codes
2. **Parse error responses** - Handle structured error responses
3. **Implement retries** - Retry transient errors with backoff
4. **Log errors** - Log authentication errors for debugging
5. **User feedback** - Provide meaningful error messages to users

## Testing Authentication

### Test API Keys
Use test API keys in development:
```
tk_test_1234567890abcdef...
```

Test keys have the same functionality but don't count against quotas.

### Webhooks Testing
Test webhook authentication using the provided test endpoint:
```http
POST /api/v1/test/webhook
Authorization: Bearer tk_test_...
```

## Support

For authentication issues:
- **Documentation**: [https://docs.tracktion.com/api](https://docs.tracktion.com/api)
- **Support**: [support@tracktion.com](mailto:support@tracktion.com)
- **Status Page**: [https://status.tracktion.com](https://status.tracktion.com)
- **Community**: [https://community.tracktion.com](https://community.tracktion.com)

## Migration Guide

### From v1 to v2 Authentication
If you're migrating from v1 API keys:

1. **Generate new API keys** through the developer dashboard
2. **Update endpoints** to use v2 authentication headers
3. **Implement proper error handling** for new error response format
4. **Test thoroughly** with new authentication flow

### Breaking Changes
- v1 API keys are not compatible with v2 endpoints
- JWT token structure has changed
- Rate limiting is now enforced more strictly
- HMAC signatures are required for sensitive operations

For detailed migration assistance, contact our support team.
