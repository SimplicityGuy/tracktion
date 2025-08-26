# cURL API Examples

This guide provides cURL examples for authenticating with and using the Tracktion API from the command line.

## Basic Setup

```bash
# Set environment variables
export TRACKTION_API_KEY="tk_live_1234567890abcdef..."
export TRACKTION_BASE_URL="https://api.tracktion.com"
```

## API Key Authentication

### Basic Search Request

```bash
# Search for tracks using API key
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/search/tracks?q=electronic+music&limit=5" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "tracks": [
    {
      "id": "track_123",
      "title": "Electronic Dreams",
      "artist": "DJ Producer",
      "genre": "Electronic",
      "duration": 240,
      "url": "https://example.com/track"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 5,
    "total": 150
  }
}
```

### Search with Parameters

```bash
# Search with additional parameters
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/search/tracks" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -G \
  -d "q=house music" \
  -d "genre=house" \
  -d "limit=10" \
  -d "offset=0" \
  -d "sort=popularity"
```

### Check Rate Limits

```bash
# Check current rate limit status
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/user/rate-limit" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -i  # Include headers in output
```

**Response Headers:**
```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640998800
X-RateLimit-Retry-After: 60
```

## JWT Token Authentication

### Login and Get JWT Token

```bash
# Login with username/password to get JWT token
JWT_RESPONSE=$(curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/auth/token" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "secure_password"
  }' \
  --silent)

# Extract tokens from response
ACCESS_TOKEN=$(echo $JWT_RESPONSE | jq -r '.access_token')
REFRESH_TOKEN=$(echo $JWT_RESPONSE | jq -r '.refresh_token')

echo "Access Token: $ACCESS_TOKEN"
echo "Refresh Token: $REFRESH_TOKEN"
```

### Use JWT Token for Authenticated Requests

```bash
# Make authenticated request with JWT token
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/user/profile" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

### Refresh JWT Token

```bash
# Refresh the access token
REFRESH_RESPONSE=$(curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"${REFRESH_TOKEN}\"}" \
  --silent)

# Extract new access token
NEW_ACCESS_TOKEN=$(echo $REFRESH_RESPONSE | jq -r '.access_token')
ACCESS_TOKEN=$NEW_ACCESS_TOKEN
```

## HMAC Signature Authentication

### Generate HMAC Signature (Bash Function)

```bash
# Function to generate HMAC signature
generate_hmac_signature() {
    local secret_key="$1"
    local timestamp="$2"
    local method="$3"
    local path="$4"
    local body="$5"
    
    local message="${timestamp}.${method}.${path}.${body}"
    local signature=$(echo -n "$message" | openssl dgst -sha256 -hmac "$secret_key" -binary | base64)
    echo "sha256=${signature}"
}

# Set HMAC secret (normally stored securely)
HMAC_SECRET="your_hmac_secret_key"
```

### Make Secure Request with HMAC

```bash
# Prepare request data
TIMESTAMP=$(date +%s)
METHOD="POST"
PATH="/api/v1/sensitive-operation"
BODY='{"action": "delete_user_data", "user_id": "12345"}'

# Generate HMAC signature
SIGNATURE=$(generate_hmac_signature "$HMAC_SECRET" "$TIMESTAMP" "$METHOD" "$PATH" "$BODY")

# Make secure request
curl -X POST \
  "${TRACKTION_BASE_URL}${PATH}" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: ${TIMESTAMP}" \
  -H "X-Signature: ${SIGNATURE}" \
  -d "$BODY"
```

### Alternative HMAC Generation (using jq and xxd)

```bash
# Alternative HMAC generation method
generate_hmac_alt() {
    local secret="$1"
    local message="$2"
    
    echo -n "$message" | openssl dgst -sha256 -hmac "$secret" | sed 's/^.* //'
}

# Usage
TIMESTAMP=$(date +%s)
MESSAGE="${TIMESTAMP}.POST./api/v1/secure-endpoint.{\"data\":\"test\"}"
SIGNATURE="sha256=$(generate_hmac_alt "$HMAC_SECRET" "$MESSAGE")"
```

## API Key Management

### Create New API Key

```bash
# Create a new API key (requires JWT authentication)
curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/developer/keys" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Server Key",
    "permissions": ["read", "write"],
    "scopes": ["search", "tracklists"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

### List API Keys

```bash
# List all API keys
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/developer/keys" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

### Rotate API Key

```bash
# Rotate an existing API key
KEY_ID="key_1234567890abcdef"
curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/developer/keys/${KEY_ID}/rotate" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

### Revoke API Key

```bash
# Revoke an API key
curl -X DELETE \
  "${TRACKTION_BASE_URL}/api/v1/developer/keys/${KEY_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

### Get API Key Usage Statistics

```bash
# Get usage statistics for an API key
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/developer/keys/${KEY_ID}/stats" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

## OAuth2 Authentication

### Authorization Code Flow - Step 1

```bash
# Step 1: Get authorization URL (typically done in browser)
AUTH_URL="${TRACKTION_BASE_URL}/api/v1/oauth/authorize"
CLIENT_ID="your_client_id"
REDIRECT_URI="https://yourapp.com/callback"
STATE="random_state_string"

echo "Visit this URL to authorize:"
echo "${AUTH_URL}?response_type=code&client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&scope=search+tracklists&state=${STATE}"
```

### Authorization Code Flow - Step 2

```bash
# Step 2: Exchange authorization code for tokens
AUTH_CODE="received_from_callback"
CLIENT_SECRET="your_client_secret"

curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=${AUTH_CODE}" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "redirect_uri=${REDIRECT_URI}"
```

### Client Credentials Flow

```bash
# Client credentials flow (for server-to-server)
curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=${CLIENT_ID}" \
  -d "client_secret=${CLIENT_SECRET}" \
  -d "scope=search"
```

## Batch Operations

### Batch Track Search

```bash
# Batch search multiple queries
curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/batch/search" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"q": "electronic music", "limit": 5},
      {"q": "house music", "limit": 5},
      {"q": "techno", "limit": 5}
    ]
  }'
```

### Batch Status Check

```bash
# Check batch operation status
BATCH_ID="batch_1234567890"
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/batch/${BATCH_ID}/status" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json"
```

## Error Handling Examples

### Handling Rate Limits

```bash
# Function to handle rate limits with retry
make_request_with_retry() {
    local url="$1"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        response=$(curl -w "HTTPSTATUS:%{http_code}" -s \
          "$url" \
          -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
          -H "Content-Type: application/json")
        
        http_status=$(echo $response | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        body=$(echo $response | sed -E 's/HTTPSTATUS:[0-9]*$//')
        
        if [ "$http_status" -eq 429 ]; then
            retry_after=$(curl -I "$url" -H "Authorization: Bearer ${TRACKTION_API_KEY}" 2>/dev/null | grep -i "x-ratelimit-retry-after" | cut -d: -f2 | tr -d ' \r')
            retry_after=${retry_after:-60}
            
            echo "Rate limited. Waiting ${retry_after} seconds..."
            sleep $retry_after
            retry_count=$((retry_count + 1))
        else
            echo "$body"
            return
        fi
    done
    
    echo "Max retries exceeded"
}

# Usage
make_request_with_retry "${TRACKTION_BASE_URL}/api/v1/search/tracks?q=electronic"
```

### Handling Authentication Errors

```bash
# Function to check and handle authentication errors
check_auth_error() {
    local response="$1"
    local http_status=$(echo $response | jq -r '.status // empty')
    local error_code=$(echo $response | jq -r '.error // empty')
    
    if [ "$http_status" = "401" ]; then
        case "$error_code" in
            "token_expired")
                echo "JWT token expired. Refreshing..."
                # Refresh token logic here
                ;;
            "invalid_api_key")
                echo "Invalid API key. Please check your credentials."
                exit 1
                ;;
            *)
                echo "Authentication failed: $error_code"
                exit 1
                ;;
        esac
    fi
}
```

## Security Best Practices

### Secure API Key Storage

```bash
# Store API key in environment file (not in code)
echo "TRACKTION_API_KEY=tk_live_1234567890abcdef..." > .env
echo ".env" >> .gitignore

# Load from environment
source .env
```

### IP Whitelist Management

```bash
# Add IP to whitelist
curl -X POST \
  "${TRACKTION_BASE_URL}/api/v1/security/ip-whitelist" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "ip_address": "203.0.113.1",
    "reason": "Production server",
    "expires_at": "2024-12-31T23:59:59Z"
  }'

# List whitelisted IPs
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/security/ip-whitelist" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"

# Remove IP from whitelist
IP_RULE_ID="rule_123456"
curl -X DELETE \
  "${TRACKTION_BASE_URL}/api/v1/security/ip-whitelist/${IP_RULE_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json"
```

## Complete Shell Script Example

```bash
#!/bin/bash

# Tracktion API CLI Script
# Usage: ./tracktion-cli.sh search "electronic music"
#        ./tracktion-cli.sh batch "house" "techno" "ambient"

set -e  # Exit on error

# Configuration
TRACKTION_BASE_URL="${TRACKTION_BASE_URL:-https://api.tracktion.com}"
TRACKTION_API_KEY="${TRACKTION_API_KEY}"

# Check if API key is set
if [ -z "$TRACKTION_API_KEY" ]; then
    echo "Error: TRACKTION_API_KEY environment variable not set"
    exit 1
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    echo "Install with: apt-get install jq  # or  brew install jq"
    exit 1
fi

# Function to make API requests
api_request() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local url="${TRACKTION_BASE_URL}${endpoint}"
    local curl_opts=(
        -X "$method"
        -H "Authorization: Bearer ${TRACKTION_API_KEY}"
        -H "Content-Type: application/json"
        --silent
        --show-error
    )
    
    if [ -n "$data" ]; then
        curl_opts+=(-d "$data")
    fi
    
    curl "${curl_opts[@]}" "$url"
}

# Function to search tracks
search_tracks() {
    local query="$1"
    local limit="${2:-10}"
    
    echo "Searching for: $query"
    
    local encoded_query=$(printf '%s' "$query" | jq -sRr @uri)
    local response=$(api_request "GET" "/api/v1/search/tracks?q=${encoded_query}&limit=${limit}")
    
    if [ $? -eq 0 ]; then
        local track_count=$(echo "$response" | jq '.tracks | length')
        
        if [ "$track_count" -gt 0 ]; then
            echo "Found $track_count tracks:"
            echo "$response" | jq -r '.tracks[] | "  \\(.title) by \\(.artist)"'
        else
            echo "No tracks found"
        fi
    else
        echo "Error: Search request failed"
        echo "$response" | jq -r '.message // "Unknown error"'
    fi
}

# Function to perform batch search
batch_search() {
    local queries=("$@")
    
    echo "Performing batch search for ${#queries[@]} queries..."
    
    for i in "${!queries[@]}"; do
        local query="${queries[$i]}"
        echo ""
        echo "[$((i+1))/${#queries[@]}] $query"
        search_tracks "$query" 3
        
        # Add delay to avoid rate limiting
        if [ $i -lt $((${#queries[@]} - 1)) ]; then
            sleep 1
        fi
    done
}

# Function to show usage
show_usage() {
    echo "Tracktion API CLI"
    echo ""
    echo "Usage:"
    echo "  $0 search \"query\" [limit]     Search for tracks"
    echo "  $0 batch \"query1\" \"query2\"    Batch search multiple queries"
    echo "  $0 profile                    Show user profile"
    echo "  $0 keys                       List API keys"
    echo ""
    echo "Environment Variables:"
    echo "  TRACKTION_API_KEY             Your Tracktion API key"
    echo "  TRACKTION_BASE_URL            API base URL (default: https://api.tracktion.com)"
}

# Function to show user profile
show_profile() {
    echo "Fetching user profile..."
    local response=$(api_request "GET" "/api/v1/user/profile")
    
    if [ $? -eq 0 ]; then
        echo "User Profile:"
        echo "$response" | jq -r '"  Email: \\(.email)"'
        echo "$response" | jq -r '"  Tier: \\(.tier)"'
        echo "$response" | jq -r '"  Created: \\(.created_at)"'
    else
        echo "Error: Failed to fetch profile"
    fi
}

# Function to list API keys
list_keys() {
    echo "Listing API keys..."
    local response=$(api_request "GET" "/api/v1/developer/keys")
    
    if [ $? -eq 0 ]; then
        local key_count=$(echo "$response" | jq '.keys | length')
        
        if [ "$key_count" -gt 0 ]; then
            echo "Found $key_count API keys:"
            echo "$response" | jq -r '.keys[] | "  \\(.name) (\\(.id)) - Active: \\(.is_active)"'
        else
            echo "No API keys found"
        fi
    else
        echo "Error: Failed to list keys"
    fi
}

# Main script logic
case "${1:-}" in
    "search")
        if [ $# -lt 2 ]; then
            echo "Error: Search query required"
            show_usage
            exit 1
        fi
        search_tracks "$2" "$3"
        ;;
    "batch")
        if [ $# -lt 2 ]; then
            echo "Error: At least one search query required"
            show_usage
            exit 1
        fi
        batch_search "${@:2}"
        ;;
    "profile")
        show_profile
        ;;
    "keys")
        list_keys
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "Error: Unknown command '${1:-}'"
        show_usage
        exit 1
        ;;
esac
```

### Make the Script Executable

```bash
chmod +x tracktion-cli.sh

# Example usage
export TRACKTION_API_KEY="tk_live_1234567890abcdef..."
./tracktion-cli.sh search "electronic music"
./tracktion-cli.sh batch "house" "techno" "ambient"
```

## Testing and Debugging

### Verbose Output

```bash
# Enable verbose output for debugging
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/search/tracks?q=test" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -v  # Verbose output
```

### Save Response to File

```bash
# Save response to file for analysis
curl -X GET \
  "${TRACKTION_BASE_URL}/api/v1/search/tracks?q=electronic" \
  -H "Authorization: Bearer ${TRACKTION_API_KEY}" \
  -H "Content-Type: application/json" \
  -o response.json

# Pretty print JSON
jq . response.json
```

### Test API Key Validity

```bash
# Test if API key is valid
test_api_key() {
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" \
      "${TRACKTION_BASE_URL}/api/v1/user/profile" \
      -H "Authorization: Bearer ${TRACKTION_API_KEY}")
    
    local http_status=$(echo $response | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    
    if [ "$http_status" -eq 200 ]; then
        echo "✓ API key is valid"
    elif [ "$http_status" -eq 401 ]; then
        echo "✗ API key is invalid"
    else
        echo "? API key test returned status: $http_status"
    fi
}

test_api_key
```

This comprehensive cURL guide provides command-line examples for all major Tracktion API operations, including authentication, security, error handling, and automation scripts.