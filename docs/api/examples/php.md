# PHP API Examples

This guide provides PHP examples for authenticating with and using the Tracktion API.

## Installation

```bash
composer require guzzlehttp/guzzle firebase/php-jwt
```

## Basic Setup

```php
<?php

require_once 'vendor/autoload.php';

use GuzzleHttp\Client;
use GuzzleHttp\Exception\ClientException;
use GuzzleHttp\Exception\ServerException;
use Firebase\JWT\JWT;
use Firebase\JWT\Key;

class TracktionClient
{
    private $client;
    private $apiKey;
    private $baseURL;
    
    public function __construct($apiKey = null, $baseURL = 'https://api.tracktion.com')
    {
        $this->apiKey = $apiKey;
        $this->baseURL = $baseURL;
        
        $headers = ['Content-Type' => 'application/json'];
        if ($apiKey) {
            $headers['Authorization'] = 'Bearer ' . $apiKey;
        }
        
        $this->client = new Client([
            'base_uri' => $this->baseURL,
            'timeout' => 30,
            'headers' => $headers
        ]);
    }
    
    public function get($endpoint, $params = [])
    {
        try {
            $response = $this->client->get($endpoint, [
                'query' => $params
            ]);
            
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            $this->handleClientException($e);
        } catch (ServerException $e) {
            throw new Exception('Server error: ' . $e->getMessage());
        }
    }
    
    public function post($endpoint, $data = [])
    {
        try {
            $response = $this->client->post($endpoint, [
                'json' => $data
            ]);
            
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            $this->handleClientException($e);
        } catch (ServerException $e) {
            throw new Exception('Server error: ' . $e->getMessage());
        }
    }
    
    public function put($endpoint, $data = [])
    {
        try {
            $response = $this->client->put($endpoint, [
                'json' => $data
            ]);
            
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            $this->handleClientException($e);
        } catch (ServerException $e) {
            throw new Exception('Server error: ' . $e->getMessage());
        }
    }
    
    public function delete($endpoint)
    {
        try {
            $response = $this->client->delete($endpoint);
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            $this->handleClientException($e);
        } catch (ServerException $e) {
            throw new Exception('Server error: ' . $e->getMessage());
        }
    }
    
    private function handleClientException(ClientException $e)
    {
        $response = $e->getResponse();
        $statusCode = $response->getStatusCode();
        $body = json_decode($response->getBody()->getContents(), true);
        
        if ($statusCode === 429) {
            $retryAfter = $response->getHeader('X-RateLimit-Retry-After')[0] ?? 60;
            throw new RateLimitException("Rate limit exceeded. Retry after {$retryAfter} seconds.");
        }
        
        $message = $body['message'] ?? $e->getMessage();
        throw new TracktionAPIException($message, $statusCode, $body['error'] ?? null);
    }
}

class TracktionAPIException extends Exception
{
    private $errorCode;
    
    public function __construct($message, $statusCode, $errorCode = null)
    {
        parent::__construct($message, $statusCode);
        $this->errorCode = $errorCode;
    }
    
    public function getErrorCode()
    {
        return $this->errorCode;
    }
}

class RateLimitException extends Exception {}
```

## API Key Authentication

```php
<?php

// Initialize client with API key
$client = new TracktionClient('tk_live_1234567890abcdef...');

// Search for tracks
function searchTracks($client, $query, $limit = 10)
{
    try {
        $response = $client->get('/api/v1/search/tracks', [
            'q' => $query,
            'limit' => $limit
        ]);
        
        return $response['tracks'] ?? [];
    } catch (RateLimitException $e) {
        error_log('Rate limit error: ' . $e->getMessage());
        return null;
    } catch (TracktionAPIException $e) {
        error_log('API error: ' . $e->getMessage());
        return null;
    } catch (Exception $e) {
        error_log('Unexpected error: ' . $e->getMessage());
        return null;
    }
}

// Example usage
$tracks = searchTracks($client, 'electronic music', 5);
if ($tracks) {
    foreach ($tracks as $track) {
        echo "{$track['title']} by {$track['artist']}\n";
    }
}
```

## JWT Token Authentication

```php
<?php

class JWTAuthClient
{
    private $client;
    private $baseURL;
    private $accessToken;
    private $refreshToken;
    
    public function __construct($baseURL = 'https://api.tracktion.com')
    {
        $this->baseURL = $baseURL;
        
        $this->client = new Client([
            'base_uri' => $this->baseURL,
            'timeout' => 30,
            'headers' => ['Content-Type' => 'application/json']
        ]);
    }
    
    public function login($username, $password)
    {
        try {
            $response = $this->client->post('/api/v1/auth/token', [
                'json' => [
                    'username' => $username,
                    'password' => $password
                ]
            ]);
            
            $data = json_decode($response->getBody()->getContents(), true);
            
            $this->accessToken = $data['access_token'];
            $this->refreshToken = $data['refresh_token'];
            
            return [
                'success' => true,
                'user' => $data['user']
            ];
        } catch (ClientException $e) {
            return [
                'success' => false,
                'error' => 'Login failed: ' . $e->getMessage()
            ];
        }
    }
    
    public function refreshAccessToken()
    {
        if (!$this->refreshToken) {
            throw new Exception('No refresh token available');
        }
        
        try {
            $response = $this->client->post('/api/v1/auth/refresh', [
                'json' => ['refresh_token' => $this->refreshToken]
            ]);
            
            $data = json_decode($response->getBody()->getContents(), true);
            $this->accessToken = $data['access_token'];
            
            // Update refresh token if provided
            if (isset($data['refresh_token'])) {
                $this->refreshToken = $data['refresh_token'];
            }
            
            return true;
        } catch (ClientException $e) {
            return false;
        }
    }
    
    public function isTokenExpired()
    {
        if (!$this->accessToken) {
            return true;
        }
        
        try {
            $parts = explode('.', $this->accessToken);
            $payload = json_decode(base64_decode($parts[1]), true);
            
            return isset($payload['exp']) && $payload['exp'] <= time();
        } catch (Exception $e) {
            return true;
        }
    }
    
    public function makeAuthenticatedRequest($method, $endpoint, $data = null)
    {
        // Refresh token if expired
        if ($this->isTokenExpired() && $this->refreshToken) {
            $this->refreshAccessToken();
        }
        
        if (!$this->accessToken) {
            throw new Exception('No valid access token available');
        }
        
        $options = [
            'headers' => [
                'Authorization' => 'Bearer ' . $this->accessToken,
                'Content-Type' => 'application/json'
            ]
        ];
        
        if ($data) {
            $options['json'] = $data;
        }
        
        try {
            $response = $this->client->request($method, $endpoint, $options);
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            // If 401, try to refresh token and retry once
            if ($e->getResponse()->getStatusCode() === 401 && $this->refreshToken) {
                if ($this->refreshAccessToken()) {
                    $options['headers']['Authorization'] = 'Bearer ' . $this->accessToken;
                    $response = $this->client->request($method, $endpoint, $options);
                    return json_decode($response->getBody()->getContents(), true);
                }
            }
            throw $e;
        }
    }
    
    public function clearTokens()
    {
        $this->accessToken = null;
        $this->refreshToken = null;
    }
}

// Example usage
$jwtClient = new JWTAuthClient();

$loginResult = $jwtClient->login('user@example.com', 'password');
if ($loginResult['success']) {
    echo "Welcome, {$loginResult['user']['email']}!\n";
    
    try {
        $profile = $jwtClient->makeAuthenticatedRequest('GET', '/api/v1/user/profile');
        echo "User profile: " . json_encode($profile, JSON_PRETTY_PRINT) . "\n";
    } catch (Exception $e) {
        echo "Error fetching profile: " . $e->getMessage() . "\n";
    }
} else {
    echo "Login failed: {$loginResult['error']}\n";
}
```

## HMAC Signature Authentication

```php
<?php

class SecureTracktionClient extends TracktionClient
{
    private $hmacSecret;
    
    public function __construct($apiKey, $hmacSecret, $baseURL = 'https://api.tracktion.com')
    {
        parent::__construct($apiKey, $baseURL);
        $this->hmacSecret = $hmacSecret;
    }
    
    public function generateHMACSignature($timestamp, $method, $path, $body = '')
    {
        $message = $timestamp . '.' . strtoupper($method) . '.' . $path . '.' . $body;
        $signature = hash_hmac('sha256', $message, $this->hmacSecret);
        return 'sha256=' . $signature;
    }
    
    public function makeSecureRequest($method, $endpoint, $data = null)
    {
        $timestamp = time();
        $body = $data ? json_encode($data) : '';
        
        $signature = $this->generateHMACSignature($timestamp, $method, $endpoint, $body);
        
        $options = [
            'headers' => [
                'X-Timestamp' => (string)$timestamp,
                'X-Signature' => $signature,
                'Authorization' => 'Bearer ' . $this->apiKey,
                'Content-Type' => 'application/json'
            ]
        ];
        
        if ($data) {
            $options['json'] = $data;
        }
        
        try {
            $response = $this->client->request($method, $endpoint, $options);
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            $this->handleClientException($e);
        } catch (ServerException $e) {
            throw new Exception('Server error: ' . $e->getMessage());
        }
    }
}

// Example usage
$secureClient = new SecureTracktionClient(
    'tk_live_1234567890abcdef...',
    'your_hmac_secret'
);

try {
    $result = $secureClient->makeSecureRequest('POST', '/api/v1/sensitive-operation', [
        'action' => 'delete_user_data',
        'user_id' => '12345'
    ]);
    echo "Secure operation result: " . json_encode($result, JSON_PRETTY_PRINT) . "\n";
} catch (Exception $e) {
    echo "Secure operation failed: " . $e->getMessage() . "\n";
}
```

## Rate Limiting and Retry Logic

```php
<?php

class RobustTracktionClient extends TracktionClient
{
    private $maxRetries;
    private $baseDelay;
    
    public function __construct($apiKey, $baseURL = 'https://api.tracktion.com', $options = [])
    {
        parent::__construct($apiKey, $baseURL);
        $this->maxRetries = $options['maxRetries'] ?? 3;
        $this->baseDelay = $options['baseDelay'] ?? 1; // seconds
    }
    
    public function retryWithBackoff(callable $fn, $maxRetries = null)
    {
        $maxRetries = $maxRetries ?? $this->maxRetries;
        $lastException = null;
        
        for ($attempt = 0; $attempt <= $maxRetries; $attempt++) {
            try {
                return $fn();
            } catch (Exception $e) {
                $lastException = $e;
                
                if ($attempt === $maxRetries) {
                    throw $e;
                }
                
                $delay = $this->calculateDelay($e, $attempt);
                if ($delay > 0) {
                    error_log("Attempt " . ($attempt + 1) . " failed, retrying in {$delay} seconds...");
                    sleep($delay);
                } else {
                    // Don't retry for non-retryable errors
                    throw $e;
                }
            }
        }
        
        throw $lastException;
    }
    
    private function calculateDelay(Exception $e, $attempt)
    {
        if ($e instanceof RateLimitException) {
            // Extract retry-after from message or use exponential backoff
            $delay = $this->baseDelay * pow(2, $attempt) + rand(0, 1000) / 1000;
            return (int)$delay;
        } elseif ($e instanceof TracktionAPIException && $e->getCode() === 429) {
            // Use retry-after header if available
            return 60; // Default retry after
        } elseif ($e->getMessage() && (
            strpos($e->getMessage(), 'Connection') !== false ||
            strpos($e->getMessage(), 'timeout') !== false
        )) {
            // Network errors - retry with exponential backoff
            return $this->baseDelay * pow(2, $attempt);
        }
        
        // Don't retry for other errors
        return 0;
    }
    
    public function searchTracksWithRetry($query, $options = [])
    {
        return $this->retryWithBackoff(function () use ($query, $options) {
            $response = $this->get('/api/v1/search/tracks', array_merge(['q' => $query], $options));
            return $response['tracks'] ?? [];
        });
    }
    
    public function batchSearchTracks(array $queries, $delay = 0.1)
    {
        $results = [];
        
        foreach ($queries as $i => $query) {
            try {
                $tracks = $this->searchTracksWithRetry($query);
                $results[] = ['query' => $query, 'tracks' => $tracks];
                
                // Add delay between requests to avoid rate limiting
                if ($i < count($queries) - 1) {
                    usleep($delay * 1000000); // Convert to microseconds
                }
            } catch (Exception $e) {
                error_log("Error searching for '{$query}': " . $e->getMessage());
                $results[] = ['query' => $query, 'error' => $e->getMessage()];
            }
        }
        
        return $results;
    }
}

// Example usage
$robustClient = new RobustTracktionClient('tk_live_1234567890abcdef...');

// Batch search multiple queries
$queries = ['electronic', 'house music', 'techno', 'ambient'];
$results = $robustClient->batchSearchTracks($queries);

foreach ($results as $result) {
    if (isset($result['tracks'])) {
        echo "Found " . count($result['tracks']) . " tracks for '{$result['query']}'\n";
    } else {
        echo "Error for '{$result['query']}': {$result['error']}\n";
    }
}
```

## API Key Management

```php
<?php

class APIKeyManager
{
    private $client;
    
    public function __construct($jwtToken, $baseURL = 'https://api.tracktion.com')
    {
        $this->client = new Client([
            'base_uri' => $baseURL,
            'timeout' => 30,
            'headers' => [
                'Authorization' => 'Bearer ' . $jwtToken,
                'Content-Type' => 'application/json'
            ]
        ]);
    }
    
    public function createAPIKey($name, array $permissions, array $scopes, $expiresAt = null)
    {
        $data = [
            'name' => $name,
            'permissions' => $permissions,
            'scopes' => $scopes
        ];
        
        if ($expiresAt) {
            $data['expires_at'] = $expiresAt;
        }
        
        $response = $this->client->post('/api/v1/developer/keys', ['json' => $data]);
        return json_decode($response->getBody()->getContents(), true);
    }
    
    public function listAPIKeys()
    {
        $response = $this->client->get('/api/v1/developer/keys');
        return json_decode($response->getBody()->getContents(), true);
    }
    
    public function rotateAPIKey($keyId)
    {
        $response = $this->client->post("/api/v1/developer/keys/{$keyId}/rotate");
        return json_decode($response->getBody()->getContents(), true);
    }
    
    public function revokeAPIKey($keyId)
    {
        $response = $this->client->delete("/api/v1/developer/keys/{$keyId}");
        return json_decode($response->getBody()->getContents(), true);
    }
    
    public function getKeyUsageStats($keyId)
    {
        $response = $this->client->get("/api/v1/developer/keys/{$keyId}/stats");
        return json_decode($response->getBody()->getContents(), true);
    }
}

// Example usage
try {
    $keyManager = new APIKeyManager('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...');
    
    // Create a new API key
    $newKey = $keyManager->createAPIKey(
        'Production Server Key',
        ['read', 'write'],
        ['search', 'tracklists'],
        '2024-12-31T23:59:59Z'
    );
    echo "Created API key: " . $newKey['key'] . "\n";
    
    // List all keys
    $keys = $keyManager->listAPIKeys();
    foreach ($keys['keys'] as $key) {
        echo "Key: {$key['name']} - Active: " . ($key['is_active'] ? 'Yes' : 'No') . "\n";
    }
    
} catch (Exception $e) {
    echo "Key management error: " . $e->getMessage() . "\n";
}
```

## Laravel Integration

```php
<?php

// config/tracktion.php
return [
    'api_key' => env('TRACKTION_API_KEY'),
    'base_url' => env('TRACKTION_BASE_URL', 'https://api.tracktion.com'),
    'hmac_secret' => env('TRACKTION_HMAC_SECRET'),
];

// app/Services/TracktionService.php
namespace App\Services;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\ClientException;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;

class TracktionService
{
    private $client;
    private $apiKey;
    private $hmacSecret;
    
    public function __construct()
    {
        $this->apiKey = config('tracktion.api_key');
        $this->hmacSecret = config('tracktion.hmac_secret');
        
        $this->client = new Client([
            'base_uri' => config('tracktion.base_url'),
            'timeout' => 30,
            'headers' => [
                'Authorization' => 'Bearer ' . $this->apiKey,
                'Content-Type' => 'application/json'
            ]
        ]);
    }
    
    public function searchTracks($query, $limit = 10, $cacheTTL = 300)
    {
        $cacheKey = "tracktion_search_" . md5($query . $limit);
        
        return Cache::remember($cacheKey, $cacheTTL, function () use ($query, $limit) {
            try {
                $response = $this->client->get('/api/v1/search/tracks', [
                    'query' => [
                        'q' => $query,
                        'limit' => $limit
                    ]
                ]);
                
                $data = json_decode($response->getBody()->getContents(), true);
                return $data['tracks'] ?? [];
            } catch (ClientException $e) {
                Log::error('Tracktion API error', [
                    'query' => $query,
                    'error' => $e->getMessage(),
                    'status' => $e->getResponse()->getStatusCode()
                ]);
                
                return [];
            } catch (Exception $e) {
                Log::error('Tracktion service error', [
                    'query' => $query,
                    'error' => $e->getMessage()
                ]);
                
                return [];
            }
        });
    }
    
    public function getUserProfile($userId)
    {
        try {
            $response = $this->client->get("/api/v1/users/{$userId}/profile");
            return json_decode($response->getBody()->getContents(), true);
        } catch (ClientException $e) {
            if ($e->getResponse()->getStatusCode() === 404) {
                return null;
            }
            throw $e;
        }
    }
    
    public function makeSecureRequest($method, $endpoint, $data = null)
    {
        $timestamp = time();
        $body = $data ? json_encode($data) : '';
        
        $message = $timestamp . '.' . strtoupper($method) . '.' . $endpoint . '.' . $body;
        $signature = 'sha256=' . hash_hmac('sha256', $message, $this->hmacSecret);
        
        $options = [
            'headers' => [
                'X-Timestamp' => (string)$timestamp,
                'X-Signature' => $signature
            ]
        ];
        
        if ($data) {
            $options['json'] = $data;
        }
        
        $response = $this->client->request($method, $endpoint, $options);
        return json_decode($response->getBody()->getContents(), true);
    }
}

// app/Http/Controllers/TrackController.php
namespace App\Http\Controllers;

use App\Services\TracktionService;
use Illuminate\Http\Request;

class TrackController extends Controller
{
    private $tracktionService;
    
    public function __construct(TracktionService $tracktionService)
    {
        $this->tracktionService = $tracktionService;
    }
    
    public function search(Request $request)
    {
        $query = $request->get('q');
        $limit = $request->get('limit', 10);
        
        if (!$query) {
            return response()->json(['error' => 'Query parameter required'], 400);
        }
        
        $tracks = $this->tracktionService->searchTracks($query, $limit);
        
        return response()->json([
            'query' => $query,
            'tracks' => $tracks,
            'count' => count($tracks)
        ]);
    }
}

// routes/api.php
Route::get('/search/tracks', [TrackController::class, 'search']);
```

## Symfony Integration

```php
<?php

// config/packages/tracktion.yaml
tracktion:
    api_key: '%env(TRACKTION_API_KEY)%'
    base_url: '%env(TRACKTION_BASE_URL)%'
    hmac_secret: '%env(TRACKTION_HMAC_SECRET)%'

// src/Service/TracktionClient.php
namespace App\Service;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\ClientException;
use Psr\Log\LoggerInterface;
use Symfony\Contracts\Cache\CacheInterface;

class TracktionClient
{
    private $client;
    private $cache;
    private $logger;
    private $hmacSecret;
    
    public function __construct(
        string $apiKey,
        string $baseUrl,
        string $hmacSecret,
        CacheInterface $cache,
        LoggerInterface $logger
    ) {
        $this->cache = $cache;
        $this->logger = $logger;
        $this->hmacSecret = $hmacSecret;
        
        $this->client = new Client([
            'base_uri' => $baseUrl,
            'timeout' => 30,
            'headers' => [
                'Authorization' => 'Bearer ' . $apiKey,
                'Content-Type' => 'application/json'
            ]
        ]);
    }
    
    public function searchTracks(string $query, int $limit = 10): array
    {
        $cacheKey = 'tracktion_search_' . md5($query . $limit);
        
        return $this->cache->get($cacheKey, function () use ($query, $limit) {
            try {
                $response = $this->client->get('/api/v1/search/tracks', [
                    'query' => [
                        'q' => $query,
                        'limit' => $limit
                    ]
                ]);
                
                $data = json_decode($response->getBody()->getContents(), true);
                return $data['tracks'] ?? [];
            } catch (ClientException $e) {
                $this->logger->error('Tracktion API error', [
                    'query' => $query,
                    'error' => $e->getMessage(),
                    'status' => $e->getResponse()->getStatusCode()
                ]);
                
                return [];
            }
        });
    }
}

// src/Controller/TrackController.php
namespace App\Controller;

use App\Service\TracktionClient;
use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\Routing\Annotation\Route;

class TrackController extends AbstractController
{
    private $tracktionClient;
    
    public function __construct(TracktionClient $tracktionClient)
    {
        $this->tracktionClient = $tracktionClient;
    }
    
    /**
     * @Route("/api/search/tracks", methods={"GET"})
     */
    public function search(Request $request): JsonResponse
    {
        $query = $request->query->get('q');
        $limit = $request->query->getInt('limit', 10);
        
        if (!$query) {
            return new JsonResponse(['error' => 'Query parameter required'], 400);
        }
        
        $tracks = $this->tracktionClient->searchTracks($query, $limit);
        
        return new JsonResponse([
            'query' => $query,
            'tracks' => $tracks,
            'count' => count($tracks)
        ]);
    }
}
```

## Complete Example Application

```php
<?php

require_once 'vendor/autoload.php';

/**
 * Complete Tracktion API example application
 */
class TracktionApp
{
    private $client;
    
    public function __construct()
    {
        $apiKey = $_ENV['TRACKTION_API_KEY'] ?? null;
        
        if (!$apiKey) {
            die("Please set TRACKTION_API_KEY environment variable\n");
        }
        
        $this->client = new RobustTracktionClient($apiKey);
    }
    
    public function run()
    {
        echo "Tracktion API PHP Example\n";
        echo "========================\n\n";
        
        // Search for tracks
        echo "Searching for electronic music tracks...\n";
        $tracks = $this->client->searchTracksWithRetry('electronic music', ['limit' => 5]);
        
        if ($tracks) {
            echo "Found " . count($tracks) . " tracks:\n";
            foreach ($tracks as $i => $track) {
                echo ($i + 1) . ". {$track['title']} by {$track['artist']}\n";
                if (isset($track['genre'])) {
                    echo "   Genre: {$track['genre']}\n";
                }
                if (isset($track['duration'])) {
                    echo "   Duration: {$track['duration']}\n";
                }
            }
        } else {
            echo "No tracks found or error occurred\n";
        }
        
        // Demonstrate batch searching
        echo "\nBatch searching multiple genres...\n";
        $genres = ['house', 'techno', 'ambient', 'drum and bass'];
        $batchResults = $this->client->batchSearchTracks($genres);
        
        foreach ($batchResults as $result) {
            if (isset($result['tracks'])) {
                echo "{$result['query']}: " . count($result['tracks']) . " tracks found\n";
            } else {
                echo "{$result['query']}: Error - {$result['error']}\n";
            }
        }
        
        echo "\nAPI example completed successfully!\n";
    }
}

// Load environment variables from .env file
if (file_exists('.env')) {
    $lines = file('.env', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    foreach ($lines as $line) {
        if (strpos($line, '=') !== false) {
            list($key, $value) = explode('=', $line, 2);
            $_ENV[trim($key)] = trim($value);
        }
    }
}

try {
    $app = new TracktionApp();
    $app->run();
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    exit(1);
}
```

## Composer.json

```json
{
    "name": "your-company/tracktion-api-client",
    "description": "Tracktion API client for PHP",
    "type": "library",
    "require": {
        "php": ">=7.4",
        "guzzlehttp/guzzle": "^7.0",
        "firebase/php-jwt": "^6.0"
    },
    "require-dev": {
        "phpunit/phpunit": "^9.0",
        "squizlabs/php_codesniffer": "^3.6"
    },
    "autoload": {
        "psr-4": {
            "Tracktion\\": "src/"
        }
    },
    "autoload-dev": {
        "psr-4": {
            "Tracktion\\Tests\\": "tests/"
        }
    },
    "scripts": {
        "test": "phpunit",
        "cs": "phpcs --standard=PSR12 src/",
        "cbf": "phpcbf --standard=PSR12 src/"
    },
    "config": {
        "preferred-install": "dist",
        "sort-packages": true
    }
}
```

This comprehensive PHP guide provides complete examples for using the Tracktion API in PHP applications, including framework integrations for Laravel and Symfony, error handling, caching, and best practices.