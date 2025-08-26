# Python API Examples

This guide provides Python examples for authenticating with and using the Tracktion API.

## Installation

```bash
pip install requests PyJWT cryptography
```

## Basic Setup

```python
import requests
import hmac
import hashlib
import time
import json
from typing import Optional, Dict, Any

class TracktionClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.tracktion.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('X-RateLimit-Retry-After', 60))
            raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        
        response.raise_for_status()
        return response
    
    def get(self, endpoint: str, **kwargs) -> Dict[Any, Any]:
        response = self._make_request('GET', endpoint, **kwargs)
        return response.json()
    
    def post(self, endpoint: str, data: Dict[Any, Any], **kwargs) -> Dict[Any, Any]:
        response = self._make_request('POST', endpoint, json=data, **kwargs)
        return response.json()

class RateLimitError(Exception):
    pass
```

## API Key Authentication

```python
# Initialize client with API key
client = TracktionClient(api_key="tk_live_1234567890abcdef...")

# Search for tracks
def search_tracks(query: str, limit: int = 10):
    try:
        response = client.get(f"/api/v1/search/tracks", params={
            'q': query,
            'limit': limit
        })
        return response['tracks']
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        return None

# Example usage
tracks = search_tracks("electronic music", limit=5)
if tracks:
    for track in tracks:
        print(f"{track['title']} by {track['artist']}")
```

## JWT Token Authentication

```python
import jwt
from datetime import datetime, timedelta

class JWTAuthClient:
    def __init__(self, base_url: str = "https://api.tracktion.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None
        self.refresh_token = None
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate with username/password and get JWT tokens."""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/auth/token", json={
                'username': username,
                'password': password
            })
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            self.refresh_token = data['refresh_token']
            
            # Update session headers
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            })
            
            return True
        except requests.exceptions.HTTPError:
            return False
    
    def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.refresh_token:
            return False
        
        try:
            response = self.session.post(f"{self.base_url}/api/v1/auth/refresh", json={
                'refresh_token': self.refresh_token
            })
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            
            # Update session headers
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}'
            })
            
            return True
        except requests.exceptions.HTTPError:
            return False
    
    def is_token_expired(self) -> bool:
        """Check if the current access token is expired."""
        if not self.access_token:
            return True
        
        try:
            # Decode without verification to check expiry
            decoded = jwt.decode(self.access_token, options={"verify_signature": False})
            exp = decoded.get('exp')
            if exp and datetime.fromtimestamp(exp) <= datetime.utcnow():
                return True
            return False
        except jwt.InvalidTokenError:
            return True
    
    def make_authenticated_request(self, method: str, endpoint: str, **kwargs):
        """Make an authenticated request, handling token refresh if needed."""
        if self.is_token_expired():
            if not self.refresh_access_token():
                raise Exception("Unable to refresh access token")
        
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        # If token expired during request, try to refresh once
        if response.status_code == 401:
            if self.refresh_access_token():
                response = self.session.request(method, url, **kwargs)
        
        response.raise_for_status()
        return response

# Example usage
jwt_client = JWTAuthClient()
if jwt_client.login("user@example.com", "password"):
    response = jwt_client.make_authenticated_request('GET', '/api/v1/user/profile')
    user_data = response.json()
    print(f"Welcome, {user_data['email']}!")
```

## HMAC Signature Authentication

```python
def generate_hmac_signature(secret_key: str, timestamp: int, method: str, path: str, body: str = "") -> str:
    """Generate HMAC-SHA256 signature for request authentication."""
    message = f"{timestamp}.{method}.{path}.{body}"
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"

class SecureTracktionClient(TracktionClient):
    def __init__(self, api_key: str, hmac_secret: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.hmac_secret = hmac_secret
    
    def make_secure_request(self, method: str, endpoint: str, data: Optional[Dict] = None):
        """Make a request with HMAC signature for sensitive operations."""
        timestamp = int(time.time())
        body = json.dumps(data) if data else ""
        
        signature = generate_hmac_signature(
            self.hmac_secret, timestamp, method, endpoint, body
        )
        
        headers = {
            'X-Timestamp': str(timestamp),
            'X-Signature': signature
        }
        
        if method.upper() == 'GET':
            response = self._make_request(method, endpoint, headers=headers)
        else:
            response = self._make_request(method, endpoint, json=data, headers=headers)
        
        return response.json()

# Example usage
secure_client = SecureTracktionClient(
    api_key="tk_live_1234567890abcdef...",
    hmac_secret="your_hmac_secret"
)

# Make a secure API call
result = secure_client.make_secure_request('POST', '/api/v1/sensitive-operation', {
    'action': 'delete_user_data',
    'user_id': '12345'
})
```

## Rate Limiting and Retry Logic

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for implementing exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt == max_retries:
                        raise
                    
                    # Extract retry_after from error message or use exponential backoff
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited, retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429 and attempt < max_retries:
                        retry_after = int(e.response.headers.get('X-RateLimit-Retry-After', 60))
                        print(f"Rate limited, retrying in {retry_after} seconds...")
                        time.sleep(retry_after)
                    else:
                        raise
        return wrapper
    return decorator

class RobustTracktionClient(TracktionClient):
    @retry_with_backoff(max_retries=3)
    def search_tracks_with_retry(self, query: str, **params):
        """Search tracks with automatic retry on rate limiting."""
        response = self.get("/api/v1/search/tracks", params={'q': query, **params})
        return response['tracks']
    
    def batch_search_tracks(self, queries: list[str], delay: float = 0.1):
        """Perform multiple searches with rate limiting consideration."""
        results = []
        
        for i, query in enumerate(queries):
            try:
                tracks = self.search_tracks_with_retry(query)
                results.append({'query': query, 'tracks': tracks})
                
                # Add delay between requests to avoid rate limiting
                if i < len(queries) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                results.append({'query': query, 'error': str(e)})
        
        return results

# Example usage
robust_client = RobustTracktionClient(api_key="tk_live_1234567890abcdef...")

# Batch search multiple queries
queries = ["electronic", "house music", "techno", "ambient"]
results = robust_client.batch_search_tracks(queries)

for result in results:
    if 'tracks' in result:
        print(f"Found {len(result['tracks'])} tracks for '{result['query']}'")
    else:
        print(f"Error for '{result['query']}'': {result['error']}")
```

## API Key Management

```python
class APIKeyManager:
    def __init__(self, jwt_token: str, base_url: str = "https://api.tracktion.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {jwt_token}',
            'Content-Type': 'application/json'
        })
    
    def create_api_key(self, name: str, permissions: list[str], 
                      scopes: list[str], expires_at: Optional[str] = None):
        """Create a new API key."""
        data = {
            'name': name,
            'permissions': permissions,
            'scopes': scopes
        }
        if expires_at:
            data['expires_at'] = expires_at
        
        response = self.session.post(f"{self.base_url}/api/v1/developer/keys", json=data)
        response.raise_for_status()
        return response.json()
    
    def list_api_keys(self):
        """List all API keys."""
        response = self.session.get(f"{self.base_url}/api/v1/developer/keys")
        response.raise_for_status()
        return response.json()
    
    def rotate_api_key(self, key_id: str):
        """Rotate an existing API key."""
        response = self.session.post(f"{self.base_url}/api/v1/developer/keys/{key_id}/rotate")
        response.raise_for_status()
        return response.json()
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key."""
        response = self.session.delete(f"{self.base_url}/api/v1/developer/keys/{key_id}")
        response.raise_for_status()
        return response.json()

# Example usage
key_manager = APIKeyManager(jwt_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

# Create a new API key
new_key = key_manager.create_api_key(
    name="Production Server Key",
    permissions=["read", "write"],
    scopes=["search", "tracklists"],
    expires_at="2024-12-31T23:59:59Z"
)
print(f"Created API key: {new_key['key']}")

# List all keys
keys = key_manager.list_api_keys()
for key in keys['keys']:
    print(f"Key: {key['name']} - Active: {key['is_active']}")
```

## Error Handling

```python
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TracktionAPIError(Exception):
    def __init__(self, message: str, status_code: int = None, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)

class RobustAPIHandler:
    def __init__(self, client: TracktionClient):
        self.client = client
    
    def handle_api_call(self, func, *args, **kwargs):
        """Handle API calls with comprehensive error handling."""
        try:
            return func(*args, **kwargs)
        
        except requests.exceptions.HTTPError as e:
            response = e.response
            
            if response.status_code == 401:
                error_data = response.json() if response.content else {}
                error_code = error_data.get('error', 'authentication_failed')
                
                if error_code == 'token_expired':
                    raise TracktionAPIError("JWT token expired", 401, error_code)
                elif error_code == 'invalid_api_key':
                    raise TracktionAPIError("Invalid API key", 401, error_code)
                else:
                    raise TracktionAPIError("Authentication failed", 401, error_code)
            
            elif response.status_code == 403:
                error_data = response.json() if response.content else {}
                error_code = error_data.get('error', 'access_denied')
                
                if error_code == 'insufficient_permissions':
                    required_scopes = error_data.get('required_scopes', [])
                    raise TracktionAPIError(
                        f"Insufficient permissions. Required scopes: {', '.join(required_scopes)}", 
                        403, error_code
                    )
                elif error_code == 'ip_blocked':
                    raise TracktionAPIError("IP address blocked", 403, error_code)
                else:
                    raise TracktionAPIError("Access denied", 403, error_code)
            
            elif response.status_code == 429:
                retry_after = response.headers.get('X-RateLimit-Retry-After', '60')
                raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds.")
            
            else:
                raise TracktionAPIError(f"HTTP {response.status_code}: {response.text}", response.status_code)
        
        except requests.exceptions.ConnectionError:
            raise TracktionAPIError("Connection error. Please check your internet connection.")
        
        except requests.exceptions.Timeout:
            raise TracktionAPIError("Request timeout. Please try again.")
        
        except Exception as e:
            logger.exception("Unexpected error during API call")
            raise TracktionAPIError(f"Unexpected error: {str(e)}")

# Example usage with error handling
client = TracktionClient(api_key="tk_live_1234567890abcdef...")
handler = RobustAPIHandler(client)

def safe_search_tracks(query: str):
    try:
        return handler.handle_api_call(client.search_tracks, query)
    except TracktionAPIError as e:
        print(f"API Error ({e.error_code}): {e.message}")
        return None
    except RateLimitError as e:
        print(f"Rate Limit Error: {e}")
        return None

# Safe API call
tracks = safe_search_tracks("electronic music")
if tracks:
    print(f"Found {len(tracks)} tracks")
```

## Complete Example Application

```python
#!/usr/bin/env python3
"""
Complete example application demonstrating Tracktion API usage.
"""

import os
import sys
import json
from datetime import datetime, timedelta

def main():
    # Get API key from environment variable
    api_key = os.getenv('TRACKTION_API_KEY')
    if not api_key:
        print("Please set TRACKTION_API_KEY environment variable")
        sys.exit(1)
    
    # Initialize client
    client = RobustTracktionClient(api_key=api_key)
    
    print("Tracktion API Example")
    print("====================")
    
    # Search for tracks
    print("\\nSearching for electronic music tracks...")
    tracks = client.search_tracks_with_retry("electronic music", limit=5)
    
    if tracks:
        print(f"Found {len(tracks)} tracks:")
        for i, track in enumerate(tracks, 1):
            print(f"{i}. {track['title']} by {track['artist']}")
            print(f"   Genre: {track.get('genre', 'Unknown')}")
            print(f"   Duration: {track.get('duration', 'Unknown')}")
    else:
        print("No tracks found or error occurred")
    
    # Demonstrate batch searching
    print("\\nBatch searching multiple genres...")
    genres = ["house", "techno", "ambient", "drum and bass"]
    batch_results = client.batch_search_tracks(genres)
    
    for result in batch_results:
        if 'tracks' in result:
            print(f"{result['query']}: {len(result['tracks'])} tracks found")
        else:
            print(f"{result['query']}: Error - {result['error']}")
    
    print("\\nAPI example completed successfully!")

if __name__ == "__main__":
    main()
```

## Installation and Setup Script

```python
#!/usr/bin/env python3
"""
Setup script for Tracktion API Python client.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    requirements = [
        'requests>=2.28.0',
        'PyJWT>=2.6.0',
        'cryptography>=3.4.8'
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
            return False
    
    return True

def create_config_file():
    """Create a sample configuration file."""
    config = {
        "api_key": "your_api_key_here",
        "base_url": "https://api.tracktion.com",
        "rate_limit_strategy": "exponential_backoff",
        "max_retries": 3,
        "timeout": 30
    }
    
    with open('tracktion_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created tracktion_config.json")
    print("  Please update the api_key field with your actual API key")

if __name__ == "__main__":
    print("Tracktion API Python Setup")
    print("===========================")
    
    print("\\n1. Installing Python packages...")
    if install_requirements():
        print("✓ All packages installed successfully")
    else:
        print("✗ Package installation failed")
        sys.exit(1)
    
    print("\\n2. Creating configuration file...")
    create_config_file()
    
    print("\\n3. Setup completed!")
    print("\\nNext steps:")
    print("1. Update tracktion_config.json with your API key")
    print("2. Set TRACKTION_API_KEY environment variable")
    print("3. Run the example script to test your setup")
```

This comprehensive Python guide covers all aspects of using the Tracktion API, including authentication methods, error handling, rate limiting, and best practices.