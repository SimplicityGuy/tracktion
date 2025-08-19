# BPM Detection API Documentation

## Overview

The BPM Detection API provides accurate tempo detection and temporal analysis for audio files. It supports multiple detection algorithms, caching for performance optimization, and comprehensive metadata extraction.

## Base URL

```
POST /api/v1/analysis/bpm
```

## Authentication

All API requests require authentication using API keys:

```http
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

## Endpoints

### 1. Detect BPM

Analyzes an audio file and returns BPM detection results.

#### Request

```http
POST /api/v1/analysis/bpm/detect
Content-Type: application/json

{
  "recording_id": "uuid",
  "file_path": "/path/to/audio/file.mp3",
  "options": {
    "enable_temporal_analysis": true,
    "cache_results": true,
    "confidence_threshold": 0.7
  }
}
```

#### Response

```json
{
  "status": "success",
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "results": {
    "bpm_data": {
      "bpm": 128.5,
      "confidence": 0.92,
      "algorithm": "primary",
      "beats": [0.0, 0.468, 0.937, 1.406],
      "needs_review": false
    },
    "temporal_data": {
      "average_bpm": 128.3,
      "start_bpm": 127.8,
      "end_bpm": 129.1,
      "stability_score": 0.94,
      "is_variable_tempo": false,
      "tempo_changes": [],
      "temporal_windows": [
        {
          "start_time": 0.0,
          "end_time": 10.0,
          "bpm": 127.8,
          "confidence": 0.89
        }
      ]
    },
    "metadata": {
      "processing_time_ms": 1250,
      "file_duration_seconds": 180.5,
      "sample_rate": 44100,
      "cache_hit": false
    }
  }
}
```

#### Error Response

```json
{
  "status": "error",
  "error_code": "INVALID_AUDIO_FILE",
  "message": "Unable to load audio file: unsupported format",
  "details": {
    "file_path": "/path/to/invalid/file.xyz",
    "supported_formats": [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
  }
}
```

### 2. Batch BPM Detection

Process multiple audio files in a single request.

#### Request

```http
POST /api/v1/analysis/bpm/batch
Content-Type: application/json

{
  "files": [
    {
      "recording_id": "uuid1",
      "file_path": "/path/to/file1.mp3"
    },
    {
      "recording_id": "uuid2",
      "file_path": "/path/to/file2.wav"
    }
  ],
  "options": {
    "parallel_processing": true,
    "max_workers": 4,
    "enable_temporal_analysis": false
  }
}
```

#### Response

```json
{
  "status": "success",
  "processed_count": 2,
  "results": [
    {
      "recording_id": "uuid1",
      "status": "success",
      "bpm_data": { /* BPM results */ }
    },
    {
      "recording_id": "uuid2",
      "status": "error",
      "error": "File not found"
    }
  ],
  "summary": {
    "successful": 1,
    "failed": 1,
    "total_processing_time_ms": 2100
  }
}
```

### 3. Get Cached Results

Retrieve previously calculated BPM results from cache.

#### Request

```http
GET /api/v1/analysis/bpm/cache/{file_hash}
```

#### Response

```json
{
  "status": "success",
  "cache_hit": true,
  "results": {
    "bpm_data": { /* Cached BPM results */ },
    "temporal_data": { /* Cached temporal results */ },
    "cached_at": "2024-01-15T10:30:00Z",
    "expires_at": "2024-02-14T10:30:00Z"
  }
}
```

### 4. Health Check

Check the health of the BPM detection service.

#### Request

```http
GET /api/v1/analysis/bpm/health
```

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "essentia": {
      "status": "available",
      "version": "2.1_beta6"
    },
    "redis_cache": {
      "status": "connected",
      "ping_ms": 2
    },
    "database": {
      "status": "connected",
      "postgresql": true,
      "neo4j": true
    }
  },
  "performance": {
    "memory_usage_mb": 125.5,
    "cpu_percent": 15.2,
    "cache_hit_rate": 0.78
  }
}
```

## Request Options

### BPM Detection Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 0.7 | Minimum confidence for primary algorithm |
| `enable_temporal_analysis` | boolean | true | Enable temporal BPM analysis |
| `cache_results` | boolean | true | Cache results in Redis |
| `force_recalculation` | boolean | false | Ignore cached results |
| `algorithm_preference` | string | "auto" | "primary", "fallback", "consensus", "auto" |

### Performance Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_processing` | boolean | true | Enable parallel processing for batch |
| `max_workers` | integer | 4 | Maximum worker threads |
| `enable_streaming` | boolean | true | Stream large files |
| `memory_limit_mb` | integer | 1000 | Memory limit per worker |

## Response Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Invalid or missing API key |
| 404 | Not Found | Audio file not found |
| 413 | Payload Too Large | File exceeds size limit |
| 422 | Unprocessable Entity | Unsupported audio format |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server processing error |
| 503 | Service Unavailable | Service temporarily unavailable |

## Error Codes

### BPM Detection Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `INVALID_AUDIO_FILE` | File cannot be loaded | Check file format and integrity |
| `UNSUPPORTED_FORMAT` | Audio format not supported | Use supported formats: MP3, WAV, FLAC, M4A, OGG |
| `FILE_TOO_LARGE` | File exceeds size limit | Compress file or use streaming |
| `FILE_TOO_SHORT` | Audio too short for analysis | Minimum duration: 5 seconds |
| `PROCESSING_TIMEOUT` | Analysis timed out | Reduce file size or increase timeout |
| `LOW_CONFIDENCE` | All algorithms returned low confidence | Manual review recommended |
| `CACHE_ERROR` | Cache operation failed | Check Redis connection |

### System Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `MEMORY_LIMIT_EXCEEDED` | Process exceeded memory limit | Reduce parallel workers or increase limit |
| `SERVICE_UNAVAILABLE` | Required service not available | Check service dependencies |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Implement request throttling |
| `DATABASE_ERROR` | Database operation failed | Check database connectivity |

## Rate Limits

| Tier | Requests/Minute | Concurrent Requests | File Size Limit |
|------|-----------------|-------------------|-----------------|
| Basic | 60 | 2 | 50 MB |
| Standard | 300 | 5 | 200 MB |
| Premium | 1000 | 10 | 500 MB |
| Enterprise | Unlimited | 25 | 2 GB |

## SDKs and Examples

### Python SDK

```python
import tracktion

client = tracktion.Client(api_key="your_key")

# Single file analysis
result = client.bpm.detect(
    recording_id="uuid",
    file_path="/path/to/audio.mp3",
    enable_temporal_analysis=True
)

print(f"BPM: {result.bpm_data.bpm}")
print(f"Confidence: {result.bpm_data.confidence}")

# Batch processing
results = client.bmp.batch([
    {"recording_id": "uuid1", "file_path": "/path/to/file1.mp3"},
    {"recording_id": "uuid2", "file_path": "/path/to/file2.wav"}
])

for result in results:
    if result.status == "success":
        print(f"File {result.recording_id}: {result.bpm_data.bpm} BPM")
```

### JavaScript SDK

```javascript
const Tracktion = require('@tracktion/sdk');

const client = new Tracktion({ apiKey: 'your_key' });

// Single file analysis
const result = await client.bpm.detect({
  recordingId: 'uuid',
  filePath: '/path/to/audio.mp3',
  options: {
    enableTemporalAnalysis: true,
    confidenceThreshold: 0.8
  }
});

console.log(`BPM: ${result.bpmData.bpm}`);
console.log(`Confidence: ${result.bpmData.confidence}`);

// Batch processing
const results = await client.bpm.batch([
  { recordingId: 'uuid1', filePath: '/path/to/file1.mp3' },
  { recordingId: 'uuid2', filePath: '/path/to/file2.wav' }
]);

results.forEach(result => {
  if (result.status === 'success') {
    console.log(`File ${result.recordingId}: ${result.bmpData.bpm} BPM`);
  }
});
```

### cURL Examples

#### Basic BPM Detection

```bash
curl -X POST https://api.tracktion.com/v1/analysis/bpm/detect \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "123e4567-e89b-12d3-a456-426614174000",
    "file_path": "/path/to/audio.mp3",
    "options": {
      "enable_temporal_analysis": true,
      "confidence_threshold": 0.7
    }
  }'
```

#### Batch Processing

```bash
curl -X POST https://api.tracktion.com/v1/analysis/bpm/batch \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "files": [
      {
        "recording_id": "uuid1",
        "file_path": "/path/to/file1.mp3"
      },
      {
        "recording_id": "uuid2",
        "file_path": "/path/to/file2.wav"
      }
    ],
    "options": {
      "parallel_processing": true,
      "max_workers": 4
    }
  }'
```

## Webhooks

### Configuration

Configure webhooks to receive asynchronous notifications when BPM analysis completes.

```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/bmp-complete",
  "events": ["bpm.analysis.complete", "bpm.analysis.failed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Payload

```json
{
  "event": "bpm.analysis.complete",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "recording_id": "123e4567-e89b-12d3-a456-426614174000",
    "bpm_data": {
      "bmp": 128.5,
      "confidence": 0.92,
      "algorithm": "primary"
    },
    "processing_time_ms": 1250
  }
}
```

## Best Practices

### Performance Optimization

1. **Use Caching**: Enable caching for repeated analysis
2. **Batch Processing**: Process multiple files together for efficiency
3. **Streaming**: Enable streaming for large files (>50MB)
4. **Parallel Processing**: Use appropriate worker count for your system

### Error Handling

1. **Retry Logic**: Implement exponential backoff for transient errors
2. **Graceful Degradation**: Handle low confidence results appropriately
3. **Validation**: Validate file formats before API calls
4. **Monitoring**: Monitor error rates and performance metrics

### Security

1. **API Key Protection**: Store API keys securely
2. **File Path Validation**: Validate file paths to prevent security issues
3. **Rate Limiting**: Implement client-side rate limiting
4. **HTTPS Only**: Always use HTTPS for API communication

This API documentation provides comprehensive guidance for integrating with the BPM detection service, including detailed examples, error handling, and best practices.
