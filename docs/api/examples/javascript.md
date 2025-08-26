# JavaScript/Node.js API Examples

This guide provides JavaScript and Node.js examples for authenticating with and using the Tracktion API.

## Installation

```bash
npm install axios jsonwebtoken crypto-js
```

## Basic Setup

```javascript
const axios = require('axios');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');

class TracktionClient {
  constructor(apiKey = null, baseURL = 'https://api.tracktion.com') {
    this.apiKey = apiKey;
    this.baseURL = baseURL;
    
    // Create axios instance with default config
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (apiKey) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      response => response,
      error => {
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['x-ratelimit-retry-after'] || 60;
          throw new RateLimitError(`Rate limit exceeded. Retry after ${retryAfter} seconds.`);
        }
        throw error;
      }
    );
  }
  
  async get(endpoint, params = {}) {
    const response = await this.client.get(endpoint, { params });
    return response.data;
  }
  
  async post(endpoint, data = {}) {
    const response = await this.client.post(endpoint, data);
    return response.data;
  }
  
  async put(endpoint, data = {}) {
    const response = await this.client.put(endpoint, data);
    return response.data;
  }
  
  async delete(endpoint) {
    const response = await this.client.delete(endpoint);
    return response.data;
  }
}

class RateLimitError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RateLimitError';
  }
}

module.exports = { TracktionClient, RateLimitError };
```

## API Key Authentication

```javascript
// Initialize client with API key
const client = new TracktionClient('tk_live_1234567890abcdef...');

// Search for tracks
async function searchTracks(query, limit = 10) {
  try {
    const response = await client.get('/api/v1/search/tracks', {
      q: query,
      limit: limit
    });
    return response.tracks;
  } catch (error) {
    if (error instanceof RateLimitError) {
      console.error('Rate limit error:', error.message);
      return null;
    } else if (error.response) {
      console.error('API error:', error.response.data);
      return null;
    } else {
      console.error('Network error:', error.message);
      return null;
    }
  }
}

// Example usage
async function main() {
  const tracks = await searchTracks('electronic music', 5);
  if (tracks) {
    tracks.forEach(track => {
      console.log(`${track.title} by ${track.artist}`);
    });
  }
}

main();
```

## JWT Token Authentication

```javascript
class JWTAuthClient {
  constructor(baseURL = 'https://api.tracktion.com') {
    this.baseURL = baseURL;
    this.accessToken = null;
    this.refreshToken = null;
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Add request interceptor to handle token refresh
    this.client.interceptors.request.use(async (config) => {
      if (this.isTokenExpired() && this.refreshToken) {
        await this.refreshAccessToken();
      }
      
      if (this.accessToken) {
        config.headers['Authorization'] = `Bearer ${this.accessToken}`;
      }
      
      return config;
    });
    
    // Add response interceptor for 401 errors
    this.client.interceptors.response.use(
      response => response,
      async (error) => {
        if (error.response?.status === 401 && this.refreshToken) {
          try {
            await this.refreshAccessToken();
            // Retry the original request
            return this.client.request(error.config);
          } catch (refreshError) {
            // Refresh failed, redirect to login
            this.clearTokens();
            throw refreshError;
          }
        }
        throw error;
      }
    );
  }
  
  async login(username, password) {
    try {
      const response = await this.client.post('/api/v1/auth/token', {
        username,
        password
      });
      
      this.accessToken = response.data.access_token;
      this.refreshToken = response.data.refresh_token;
      
      return {
        success: true,
        user: response.data.user
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.message || 'Login failed'
      };
    }
  }
  
  async refreshAccessToken() {
    if (!this.refreshToken) {
      throw new Error('No refresh token available');
    }
    
    const response = await this.client.post('/api/v1/auth/refresh', {
      refresh_token: this.refreshToken
    });
    
    this.accessToken = response.data.access_token;
    
    // Update refresh token if provided
    if (response.data.refresh_token) {
      this.refreshToken = response.data.refresh_token;
    }
  }
  
  isTokenExpired() {
    if (!this.accessToken) {
      return true;
    }
    
    try {
      const decoded = jwt.decode(this.accessToken);
      const currentTime = Math.floor(Date.now() / 1000);
      return decoded.exp <= currentTime;
    } catch (error) {
      return true;
    }
  }
  
  clearTokens() {
    this.accessToken = null;
    this.refreshToken = null;
  }
  
  async makeAuthenticatedRequest(method, endpoint, data = null) {
    const config = {
      method,
      url: endpoint
    };
    
    if (data) {
      config.data = data;
    }
    
    const response = await this.client.request(config);
    return response.data;
  }
}

// Example usage
async function jwtExample() {
  const jwtClient = new JWTAuthClient();
  
  const loginResult = await jwtClient.login('user@example.com', 'password');
  if (loginResult.success) {
    console.log(`Welcome, ${loginResult.user.email}!`);
    
    // Make authenticated requests
    try {
      const profile = await jwtClient.makeAuthenticatedRequest('GET', '/api/v1/user/profile');
      console.log('User profile:', profile);
    } catch (error) {
      console.error('Error fetching profile:', error.message);
    }
  } else {
    console.error('Login failed:', loginResult.error);
  }
}
```

## HMAC Signature Authentication

```javascript
const crypto = require('crypto');

function generateHMACSignature(secretKey, timestamp, method, path, body = '') {
  const message = `${timestamp}.${method}.${path}.${body}`;
  const signature = crypto
    .createHmac('sha256', secretKey)
    .update(message)
    .digest('hex');
  return `sha256=${signature}`;
}

class SecureTracktionClient extends TracktionClient {
  constructor(apiKey, hmacSecret, baseURL) {
    super(apiKey, baseURL);
    this.hmacSecret = hmacSecret;
  }
  
  async makeSecureRequest(method, endpoint, data = null) {
    const timestamp = Math.floor(Date.now() / 1000);
    const body = data ? JSON.stringify(data) : '';
    
    const signature = generateHMACSignature(
      this.hmacSecret,
      timestamp,
      method.toUpperCase(),
      endpoint,
      body
    );
    
    const config = {
      method,
      url: endpoint,
      headers: {
        'X-Timestamp': timestamp.toString(),
        'X-Signature': signature
      }
    };
    
    if (data) {
      config.data = data;
    }
    
    const response = await this.client.request(config);
    return response.data;
  }
}

// Example usage
async function secureExample() {
  const secureClient = new SecureTracktionClient(
    'tk_live_1234567890abcdef...',
    'your_hmac_secret'
  );
  
  try {
    const result = await secureClient.makeSecureRequest('POST', '/api/v1/sensitive-operation', {
      action: 'delete_user_data',
      user_id: '12345'
    });
    console.log('Secure operation result:', result);
  } catch (error) {
    console.error('Secure operation failed:', error.message);
  }
}
```

## Rate Limiting and Retry Logic

```javascript
class RobustTracktionClient extends TracktionClient {
  constructor(apiKey, baseURL, options = {}) {
    super(apiKey, baseURL);
    this.maxRetries = options.maxRetries || 3;
    this.baseDelay = options.baseDelay || 1000; // milliseconds
  }
  
  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  async retryWithBackoff(fn, maxRetries = this.maxRetries) {
    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        if (attempt === maxRetries) {
          throw error;
        }
        
        let delay;
        if (error instanceof RateLimitError) {
          // Extract retry-after from error message or use exponential backoff
          delay = this.baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
        } else if (error.response?.status === 429) {
          const retryAfter = parseInt(error.response.headers['x-ratelimit-retry-after']) || 60;
          delay = retryAfter * 1000;
        } else if (error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT') {
          // Network errors - retry with exponential backoff
          delay = this.baseDelay * Math.pow(2, attempt);
        } else {
          // Don't retry for other errors
          throw error;
        }
        
        console.log(`Attempt ${attempt + 1} failed, retrying in ${delay}ms...`);
        await this.sleep(delay);
      }
    }
    
    throw lastError;
  }
  
  async searchTracksWithRetry(query, options = {}) {
    return this.retryWithBackoff(async () => {
      const response = await this.get('/api/v1/search/tracks', {
        q: query,
        ...options
      });
      return response.tracks;
    });
  }
  
  async batchSearchTracks(queries, delay = 100) {
    const results = [];
    
    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      try {
        const tracks = await this.searchTracksWithRetry(query);
        results.push({ query, tracks });
        
        // Add delay between requests to avoid rate limiting
        if (i < queries.length - 1) {
          await this.sleep(delay);
        }
      } catch (error) {
        console.error(`Error searching for '${query}':`, error.message);
        results.push({ query, error: error.message });
      }
    }
    
    return results;
  }
}

// Example usage
async function robustExample() {
  const robustClient = new RobustTracktionClient('tk_live_1234567890abcdef...');
  
  // Batch search multiple queries
  const queries = ['electronic', 'house music', 'techno', 'ambient'];
  const results = await robustClient.batchSearchTracks(queries);
  
  results.forEach(result => {
    if (result.tracks) {
      console.log(`Found ${result.tracks.length} tracks for '${result.query}'`);
    } else {
      console.log(`Error for '${result.query}': ${result.error}`);
    }
  });
}
```

## API Key Management

```javascript
class APIKeyManager {
  constructor(jwtToken, baseURL = 'https://api.tracktion.com') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Authorization': `Bearer ${jwtToken}`,
        'Content-Type': 'application/json'
      }
    });
  }
  
  async createAPIKey(name, permissions, scopes, expiresAt = null) {
    const data = {
      name,
      permissions,
      scopes
    };
    
    if (expiresAt) {
      data.expires_at = expiresAt;
    }
    
    const response = await this.client.post('/api/v1/developer/keys', data);
    return response.data;
  }
  
  async listAPIKeys() {
    const response = await this.client.get('/api/v1/developer/keys');
    return response.data;
  }
  
  async rotateAPIKey(keyId) {
    const response = await this.client.post(`/api/v1/developer/keys/${keyId}/rotate`);
    return response.data;
  }
  
  async revokeAPIKey(keyId) {
    const response = await this.client.delete(`/api/v1/developer/keys/${keyId}`);
    return response.data;
  }
  
  async getKeyUsageStats(keyId) {
    const response = await this.client.get(`/api/v1/developer/keys/${keyId}/stats`);
    return response.data;
  }
}

// Example usage
async function keyManagementExample() {
  const keyManager = new APIKeyManager('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...');
  
  try {
    // Create a new API key
    const newKey = await keyManager.createAPIKey(
      'Production Server Key',
      ['read', 'write'],
      ['search', 'tracklists'],
      '2024-12-31T23:59:59Z'
    );
    console.log('Created API key:', newKey.key);
    
    // List all keys
    const keys = await keyManager.listAPIKeys();
    keys.keys.forEach(key => {
      console.log(`Key: ${key.name} - Active: ${key.is_active}`);
    });
    
    // Get usage stats
    const stats = await keyManager.getKeyUsageStats(newKey.id);
    console.log('Usage stats:', stats);
    
  } catch (error) {
    console.error('Key management error:', error.response?.data || error.message);
  }
}
```

## Browser Usage (Frontend)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Tracktion API Browser Example</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
</head>
<body>
    <h1>Tracktion API Search</h1>
    <input type="text" id="searchInput" placeholder="Search for tracks...">
    <button onclick="searchTracks()">Search</button>
    <div id="results"></div>

    <script>
    class BrowserTracktionClient {
      constructor(baseURL = 'https://api.tracktion.com') {
        this.baseURL = baseURL;
        this.accessToken = localStorage.getItem('tracktion_access_token');
        this.refreshToken = localStorage.getItem('tracktion_refresh_token');
        
        // Configure axios
        axios.defaults.baseURL = this.baseURL;
        axios.defaults.headers.common['Content-Type'] = 'application/json';
        
        if (this.accessToken) {
          axios.defaults.headers.common['Authorization'] = `Bearer ${this.accessToken}`;
        }
        
        // Set up axios interceptors
        axios.interceptors.response.use(
          response => response,
          async error => {
            if (error.response?.status === 401 && this.refreshToken) {
              try {
                await this.refreshAccessToken();
                // Retry the original request
                return axios.request(error.config);
              } catch (refreshError) {
                this.logout();
                throw refreshError;
              }
            }
            throw error;
          }
        );
      }
      
      async login(username, password) {
        try {
          const response = await axios.post('/api/v1/auth/token', {
            username,
            password
          });
          
          this.accessToken = response.data.access_token;
          this.refreshToken = response.data.refresh_token;
          
          // Store tokens in localStorage
          localStorage.setItem('tracktion_access_token', this.accessToken);
          localStorage.setItem('tracktion_refresh_token', this.refreshToken);
          
          // Set default header
          axios.defaults.headers.common['Authorization'] = `Bearer ${this.accessToken}`;
          
          return response.data;
        } catch (error) {
          throw error;
        }
      }
      
      async refreshAccessToken() {
        const response = await axios.post('/api/v1/auth/refresh', {
          refresh_token: this.refreshToken
        });
        
        this.accessToken = response.data.access_token;
        localStorage.setItem('tracktion_access_token', this.accessToken);
        axios.defaults.headers.common['Authorization'] = `Bearer ${this.accessToken}`;
      }
      
      logout() {
        this.accessToken = null;
        this.refreshToken = null;
        localStorage.removeItem('tracktion_access_token');
        localStorage.removeItem('tracktion_refresh_token');
        delete axios.defaults.headers.common['Authorization'];
      }
      
      async searchTracks(query, limit = 10) {
        const response = await axios.get('/api/v1/search/tracks', {
          params: { q: query, limit }
        });
        return response.data.tracks;
      }
    }
    
    const client = new BrowserTracktionClient();
    
    async function searchTracks() {
      const query = document.getElementById('searchInput').value;
      const resultsDiv = document.getElementById('results');
      
      if (!query.trim()) {
        resultsDiv.innerHTML = '<p>Please enter a search query</p>';
        return;
      }
      
      try {
        resultsDiv.innerHTML = '<p>Searching...</p>';
        
        const tracks = await client.searchTracks(query, 10);
        
        if (tracks && tracks.length > 0) {
          let html = '<h2>Search Results:</h2><ul>';
          tracks.forEach(track => {
            html += `<li><strong>${track.title}</strong> by ${track.artist}</li>`;
          });
          html += '</ul>';
          resultsDiv.innerHTML = html;
        } else {
          resultsDiv.innerHTML = '<p>No tracks found</p>';
        }
      } catch (error) {
        console.error('Search error:', error);
        if (error.response?.status === 401) {
          resultsDiv.innerHTML = '<p>Please log in to search tracks</p>';
        } else if (error.response?.status === 429) {
          resultsDiv.innerHTML = '<p>Rate limit exceeded. Please wait and try again.</p>';
        } else {
          resultsDiv.innerHTML = '<p>Error occurred while searching. Please try again.</p>';
        }
      }
    }
    
    // Example login function
    async function login() {
      try {
        const result = await client.login('user@example.com', 'password');
        console.log('Login successful:', result);
      } catch (error) {
        console.error('Login failed:', error);
      }
    }
    </script>
</body>
</html>
```

## Error Handling

```javascript
class TracktionAPIError extends Error {
  constructor(message, statusCode = null, errorCode = null) {
    super(message);
    this.name = 'TracktionAPIError';
    this.statusCode = statusCode;
    this.errorCode = errorCode;
  }
}

class ErrorHandler {
  static handle(error) {
    if (error.response) {
      const { status, data } = error.response;
      const errorCode = data?.error || 'unknown_error';
      const message = data?.message || error.message;
      
      switch (status) {
        case 401:
          if (errorCode === 'token_expired') {
            throw new TracktionAPIError('JWT token expired', 401, errorCode);
          } else if (errorCode === 'invalid_api_key') {
            throw new TracktionAPIError('Invalid API key', 401, errorCode);
          } else {
            throw new TracktionAPIError('Authentication failed', 401, errorCode);
          }
        
        case 403:
          if (errorCode === 'insufficient_permissions') {
            const requiredScopes = data?.required_scopes || [];
            throw new TracktionAPIError(
              `Insufficient permissions. Required scopes: ${requiredScopes.join(', ')}`,
              403,
              errorCode
            );
          } else if (errorCode === 'ip_blocked') {
            throw new TracktionAPIError('IP address blocked', 403, errorCode);
          } else {
            throw new TracktionAPIError('Access denied', 403, errorCode);
          }
        
        case 429:
          const retryAfter = error.response.headers['x-ratelimit-retry-after'] || 60;
          throw new RateLimitError(`Rate limit exceeded. Retry after ${retryAfter} seconds.`);
        
        default:
          throw new TracktionAPIError(`HTTP ${status}: ${message}`, status, errorCode);
      }
    } else if (error.code === 'ECONNABORTED') {
      throw new TracktionAPIError('Request timeout. Please try again.');
    } else if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      throw new TracktionAPIError('Connection error. Please check your internet connection.');
    } else {
      throw new TracktionAPIError(`Unexpected error: ${error.message}`);
    }
  }
}

// Example usage with comprehensive error handling
async function safeSearchTracks(query) {
  try {
    const client = new TracktionClient('tk_live_1234567890abcdef...');
    const tracks = await client.get('/api/v1/search/tracks', { q: query });
    return tracks;
  } catch (error) {
    try {
      ErrorHandler.handle(error);
    } catch (apiError) {
      console.error(`API Error (${apiError.errorCode}):`, apiError.message);
      return null;
    }
  }
}
```

## Complete Example Application

```javascript
#!/usr/bin/env node

const { TracktionClient, RateLimitError } = require('./tracktion-client');

class TracktionCLI {
  constructor() {
    this.apiKey = process.env.TRACKTION_API_KEY;
    if (!this.apiKey) {
      console.error('Please set TRACKTION_API_KEY environment variable');
      process.exit(1);
    }
    
    this.client = new TracktionClient(this.apiKey);
  }
  
  async searchCommand(query, options = {}) {
    try {
      console.log(`Searching for: ${query}`);
      
      const response = await this.client.get('/api/v1/search/tracks', {
        q: query,
        limit: options.limit || 10
      });
      
      const tracks = response.tracks;
      if (tracks && tracks.length > 0) {
        console.log(`\\nFound ${tracks.length} tracks:`);
        tracks.forEach((track, index) => {
          console.log(`${index + 1}. ${track.title} by ${track.artist}`);
          if (track.genre) console.log(`   Genre: ${track.genre}`);
          if (track.duration) console.log(`   Duration: ${track.duration}`);
          console.log('');
        });
      } else {
        console.log('No tracks found');
      }
    } catch (error) {
      if (error instanceof RateLimitError) {
        console.error('Rate limit exceeded:', error.message);
      } else if (error.response) {
        console.error('API error:', error.response.data);
      } else {
        console.error('Error:', error.message);
      }
    }
  }
  
  async batchCommand(queries) {
    console.log('Performing batch search...');
    
    for (let i = 0; i < queries.length; i++) {
      const query = queries[i];
      console.log(`\\n[${i + 1}/${queries.length}] Searching: ${query}`);
      
      try {
        const response = await this.client.get('/api/v1/search/tracks', {
          q: query,
          limit: 3
        });
        
        console.log(`Found ${response.tracks.length} tracks`);
        
        // Add delay to avoid rate limiting
        if (i < queries.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      } catch (error) {
        console.error(`Error for "${query}":`, error.message);
      }
    }
  }
  
  printUsage() {
    console.log('Tracktion API CLI');
    console.log('Usage:');
    console.log('  node cli.js search "electronic music"');
    console.log('  node cli.js batch "house" "techno" "ambient"');
  }
  
  async run() {
    const args = process.argv.slice(2);
    
    if (args.length === 0) {
      this.printUsage();
      return;
    }
    
    const command = args[0];
    
    switch (command) {
      case 'search':
        if (args.length < 2) {
          console.error('Please provide a search query');
          return;
        }
        await this.searchCommand(args[1]);
        break;
      
      case 'batch':
        if (args.length < 2) {
          console.error('Please provide search queries');
          return;
        }
        await this.batchCommand(args.slice(1));
        break;
      
      default:
        console.error('Unknown command:', command);
        this.printUsage();
    }
  }
}

// Run the CLI
if (require.main === module) {
  const cli = new TracktionCLI();
  cli.run().catch(console.error);
}

module.exports = TracktionCLI;
```

## Package.json Setup

```json
{
  "name": "tracktion-api-client",
  "version": "1.0.0",
  "description": "Tracktion API client for JavaScript/Node.js",
  "main": "index.js",
  "scripts": {
    "start": "node cli.js",
    "test": "jest",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "dependencies": {
    "axios": "^1.4.0",
    "jsonwebtoken": "^9.0.0",
    "crypto-js": "^4.1.1"
  },
  "devDependencies": {
    "jest": "^29.5.0",
    "eslint": "^8.42.0",
    "prettier": "^2.8.8"
  },
  "keywords": [
    "tracktion",
    "api",
    "music",
    "tracks",
    "client"
  ],
  "author": "Your Name",
  "license": "MIT"
}
```

This comprehensive JavaScript guide covers all aspects of using the Tracktion API in both Node.js and browser environments, including authentication, error handling, rate limiting, and practical examples.