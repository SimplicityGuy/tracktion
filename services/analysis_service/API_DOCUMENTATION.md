# Analysis Service REST API Documentation

## Overview

The Analysis Service provides a RESTful API for managing audio file analysis, metadata extraction, tracklist generation, and real-time streaming capabilities. This service processes music files to extract information like BPM, key detection, mood analysis, and generates visualizations like waveforms and spectrograms.

**Base URL**: `http://localhost:8000` (development)
**API Version**: v1
**Documentation URLs**:
- Swagger UI: `/v1/docs`
- ReDoc: `/v1/redoc`
- OpenAPI JSON: `/v1/openapi.json`

## Authentication

The API currently uses API key authentication via headers:

```http
X-API-Key: your-api-key-here
```

**Authentication Errors**:
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions

## Rate Limiting

The API implements comprehensive rate limiting with the following default limits:

- **Per Second**: 10 requests
- **Per Minute**: 100 requests
- **Per Hour**: 1000 requests
- **Burst Size**: 20 requests
- **Max Concurrent Connections**: 1000
- **Max Connections per IP**: 10

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

**Rate Limit Errors**:
- `429 Too Many Requests`: Rate limit exceeded

## Common Response Headers

All responses include these standard headers:

```http
X-Request-ID: uuid4-generated-id
X-Process-Time: 123.45ms
Cache-Control: no-cache
Connection: keep-alive
```

## Error Response Format

All errors follow a consistent JSON structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "status": 400,
    "request_id": "uuid4-request-id",
    "details": [...] // Optional additional details
  }
}
```

**Common Error Codes**:
- `RESOURCE_NOT_FOUND` (404)
- `VALIDATION_ERROR` (422)
- `AUTHENTICATION_REQUIRED` (401)
- `INSUFFICIENT_PERMISSIONS` (403)
- `RATE_LIMIT_EXCEEDED` (429)
- `SERVICE_UNAVAILABLE` (503)
- `INTERNAL_ERROR` (500)

---

## Health Check Endpoints

### GET /v1/health

Basic health check to verify service is running.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "service": "analysis_service"
}
```

### GET /v1/health/ready

Kubernetes readiness probe to check if service is ready to accept requests.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "ready": true,
  "checks": {
    "database": "ready",
    "message_queue": "ready",
    "cache": "ready"
  },
  "service": "analysis_service"
}
```

### GET /v1/health/live

Kubernetes liveness probe to verify service is alive.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "alive",
  "service": "analysis_service"
}
```

---

## Recording Management Endpoints

### POST /v1/recordings

Submit a recording for analysis.

**Request Body**:
```json
{
  "file_path": "/path/to/audio/file.mp3",
  "priority": 5,
  "metadata": {
    "custom_field": "value"
  }
}
```

**Response**:
```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Recording submitted for analysis",
  "correlation_id": "correlation-uuid"
}
```

**Errors**:
- `404 Not Found`: Audio file not found
- `422 Unprocessable Entity`: Invalid request data

### GET /v1/recordings/{recording_id}

Get status and metadata for a recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "file_path": "/path/to/audio/file.mp3",
  "status": "completed",
  "priority": 5,
  "metadata": {
    "file_size": 15728640
  }
}
```

### GET /v1/recordings

List recordings with optional filtering and pagination.

**Query Parameters**:
- `status` (string, optional): Filter by status
- `limit` (integer, 1-100): Maximum results per page (default: 10)
- `offset` (integer, ≥0): Pagination offset (default: 0)

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "file_path": "/path/to/audio/file1.mp3",
    "status": "completed",
    "priority": 5,
    "metadata": {
      "file_size": 15728640
    }
  },
  {
    "id": "456e7890-e89b-12d3-a456-426614174001",
    "file_path": "/path/to/audio/file2.wav",
    "status": "processing",
    "priority": 3,
    "metadata": {
      "file_size": 52428800
    }
  }
]
```

### DELETE /v1/recordings/{recording_id}

Cancel a recording analysis.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "cancelled",
  "message": "Recording analysis cancelled",
  "correlation_id": "correlation-uuid"
}
```

---

## Analysis Endpoints

### POST /v1/analysis

Start analysis for a recording.

**Request Body**:
```json
{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_types": ["bpm", "key", "mood", "energy"],
  "priority": 5
}
```

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "task_id": "analysis-task-uuid",
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Analysis started",
  "analysis_types": ["bpm", "key", "mood", "energy"],
  "correlation_id": "correlation-uuid"
}
```

**Errors**:
- `404 Not Found`: Recording or audio file not found
- `400 Bad Request`: Recording has no file path

### GET /v1/analysis/{recording_id}

Get analysis status and results for a recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 1.0,
  "results": [
    {
      "type": "bpm",
      "value": 128.5,
      "confidence": 0.95,
      "metadata": {
        "processing_time_ms": 1250,
        "created_at": "2023-12-01T10:30:00Z"
      }
    },
    {
      "type": "key",
      "value": "C major",
      "confidence": 0.87,
      "metadata": {
        "processing_time_ms": 2100,
        "created_at": "2023-12-01T10:30:05Z"
      }
    }
  ],
  "started_at": "2023-12-01T10:29:45Z",
  "completed_at": "2023-12-01T10:30:10Z"
}
```

### GET /v1/analysis/{recording_id}/bpm

Get BPM analysis results for a specific recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_type": "bpm",
  "result": 128.5,
  "confidence": 0.95,
  "status": "completed",
  "created_at": "2023-12-01T10:30:00Z",
  "processing_time_ms": 1250
}
```

### GET /v1/analysis/{recording_id}/key

Get key detection results for a recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_type": "key",
  "result": "C major",
  "confidence": 0.87,
  "status": "completed",
  "created_at": "2023-12-01T10:30:05Z",
  "processing_time_ms": 2100
}
```

### GET /v1/analysis/{recording_id}/mood

Get mood analysis results for a recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_type": "mood",
  "result": {
    "valence": 0.75,
    "arousal": 0.65,
    "mood": "happy"
  },
  "confidence": 0.82,
  "status": "completed",
  "created_at": "2023-12-01T10:30:08Z",
  "processing_time_ms": 3200
}
```

### POST /v1/analysis/{recording_id}/waveform

Generate waveform visualization for a recording.

**Query Parameters**:
- `width` (integer): Image width in pixels (default: 1920)
- `height` (integer): Image height in pixels (default: 256)
- `color` (string): Waveform color in hex (default: "#00ff00")

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "task_id": "waveform-task-uuid",
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "generating",
  "message": "Waveform generation started",
  "parameters": {
    "width": 1920,
    "height": 256,
    "color": "#00ff00"
  },
  "correlation_id": "correlation-uuid"
}
```

### POST /v1/analysis/{recording_id}/spectrogram

Generate spectrogram for a recording.

**Query Parameters**:
- `fft_size` (integer): FFT window size (default: 2048)
- `hop_size` (integer): Hop size between windows (default: 512)
- `color_map` (string): Color map for visualization (default: "viridis")

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "task_id": "spectrogram-task-uuid",
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "generating",
  "message": "Spectrogram generation started",
  "parameters": {
    "fft_size": 2048,
    "hop_size": 512,
    "color_map": "viridis"
  },
  "correlation_id": "correlation-uuid"
}
```

---

## Metadata Management Endpoints

### GET /v1/metadata/{recording_id}

Get metadata for a recording.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Sample Song",
  "artist": "Artist Name",
  "album": "Album Title",
  "genre": "Electronic",
  "year": 2023,
  "duration": 245.5,
  "format": "mp3",
  "bitrate": 320,
  "sample_rate": 44100,
  "channels": 2,
  "custom_fields": {
    "label": "Record Label",
    "bpm": "128"
  }
}
```

### PUT /v1/metadata/{recording_id}

Update metadata for a recording.

**Request Body**:
```json
{
  "title": "Updated Song Title",
  "artist": "Updated Artist",
  "album": "Updated Album",
  "genre": "House",
  "year": 2024,
  "track_number": 3,
  "custom_fields": {
    "label": "New Label",
    "remix": "Extended Mix"
  }
}
```

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "updated",
  "message": "Metadata updated successfully (7 fields)"
}
```

### POST /v1/metadata/{recording_id}/extract

Trigger metadata extraction from the audio file.

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "extracting",
  "message": "Metadata extraction started",
  "correlation_id": "extraction-uuid"
}
```

### POST /v1/metadata/{recording_id}/enrich

Enrich metadata using external sources (MusicBrainz, Last.fm).

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "enriching",
  "message": "Metadata enrichment started",
  "correlation_id": "enrichment-uuid"
}
```

---

## Tracklist Management Endpoints

### GET /v1/tracklist/{recording_id}

Get tracklist for a recording (for DJ mixes, albums, etc.).

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "format": "cue",
  "total_tracks": 3,
  "total_duration": 612.5,
  "tracks": [
    {
      "index": 1,
      "title": "Track 1",
      "artist": "Artist 1",
      "start_time": 0.0,
      "end_time": 203.2,
      "duration": 203.2,
      "file_path": "/path/to/track1.flac"
    },
    {
      "index": 2,
      "title": "Track 2",
      "artist": "Artist 2",
      "start_time": 203.2,
      "end_time": 408.7,
      "duration": 205.5,
      "file_path": "/path/to/track2.flac"
    },
    {
      "index": 3,
      "title": "Track 3",
      "artist": "Artist 3",
      "start_time": 408.7,
      "end_time": 612.5,
      "duration": 203.8,
      "file_path": "/path/to/track3.flac"
    }
  ]
}
```

### POST /v1/tracklist/{recording_id}/detect

Detect tracks in a recording using silence detection.

**Query Parameters**:
- `min_duration` (float): Minimum track duration in seconds (default: 30.0)
- `sensitivity` (float, 0.0-1.0): Detection sensitivity (default: 0.5)

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "detecting",
  "message": "Track detection started",
  "parameters": {
    "min_duration": 30.0,
    "sensitivity": 0.5
  },
  "correlation_id": "detection-uuid"
}
```

### POST /v1/tracklist/{recording_id}/split

Split recording into individual track files.

**Query Parameters**:
- `output_format` (string): Format for split tracks (default: "flac")

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "splitting",
  "message": "Track splitting started",
  "output_format": "flac",
  "correlation_id": "splitting-uuid"
}
```

**Errors**:
- `400 Bad Request`: No tracklist found - run track detection first

### POST /v1/tracklist/parse-cue

Parse a CUE sheet and extract tracklist information.

**Request Body**:
```json
{
  "cue_content": "FILE \"album.flac\" WAVE\n  TRACK 01 AUDIO\n    TITLE \"Track 1\"\n    PERFORMER \"Artist 1\"\n    INDEX 01 00:00:00\n  TRACK 02 AUDIO\n    TITLE \"Track 2\"\n    PERFORMER \"Artist 2\"\n    INDEX 01 03:23:15",
  "audio_file_path": "/path/to/album.flac",
  "validate_cue": true
}
```

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "parsing",
  "format": "cue",
  "recording_id": "123e4567-e89b-12d3-a456-426614174000",
  "audio_file": "/path/to/album.flac",
  "message": "CUE sheet parsing started",
  "correlation_id": "parsing-uuid"
}
```

### PUT /v1/tracklist/{recording_id}/tracks

Update tracklist for a recording.

**Request Body**:
```json
[
  {
    "index": 1,
    "title": "Updated Track 1",
    "artist": "Updated Artist 1",
    "start_time": 0.0,
    "end_time": 205.0,
    "duration": 205.0,
    "file_path": "/updated/path/track1.flac"
  },
  {
    "index": 2,
    "title": "Updated Track 2",
    "artist": "Updated Artist 2",
    "start_time": 205.0,
    "end_time": 410.0,
    "duration": 205.0,
    "file_path": "/updated/path/track2.flac"
  }
]
```

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "updated",
  "message": "Tracklist updated with 2 tracks"
}
```

---

## Streaming Endpoints

### GET /v1/streaming/audio/{recording_id}

Stream audio file with range request support.

**Query Parameters**:
- `chunk_size` (integer): Chunk size in bytes (default: 8192)
- `start_byte` (integer, optional): Start byte for range request
- `end_byte` (integer, optional): End byte for range request

**Headers**:
- `Range: bytes=0-1023` (optional): HTTP range header

**Response**:
```http
HTTP/1.1 206 Partial Content
Content-Type: audio/mpeg
Content-Length: 1024
Content-Range: bytes 0-1023/15728640
Accept-Ranges: bytes
Cache-Control: public, max-age=3600
X-Recording-ID: 123e4567-e89b-12d3-a456-426614174000

[binary audio data]
```

**Supported Formats**:
- MP3 (`audio/mpeg`)
- WAV (`audio/wav`)
- FLAC (`audio/flac`)
- OGG (`audio/ogg`)
- M4A (`audio/mp4`)
- AAC (`audio/aac`)

### GET /v1/streaming/events/{recording_id}

Stream analysis progress using Server-Sent Events (SSE).

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Recording-ID: 123e4567-e89b-12d3-a456-426614174000

event: started
data: {"recording_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Starting analysis monitoring", "timestamp": 1640995200.123}

event: progress
data: {"recording_id": "123e4567-e89b-12d3-a456-426614174000", "stage": "bpm_detection", "progress": 25.0, "message": "Detecting BPM (processing)", "status": "processing", "completed_analyses": 0, "total_analyses": 4, "timestamp": 1640995205.456}

event: progress
data: {"recording_id": "123e4567-e89b-12d3-a456-426614174000", "stage": "key_detection", "progress": 50.0, "message": "Detecting key (processing)", "status": "processing", "completed_analyses": 1, "total_analyses": 4, "timestamp": 1640995210.789}

event: complete
data: {"recording_id": "123e4567-e89b-12d3-a456-426614174000", "results": {...}, "message": "Analysis completed successfully", "timestamp": 1640995220.012}
```

**Event Types**:
- `started`: Analysis monitoring started
- `progress`: Progress update with percentage and stage
- `complete`: Analysis completed successfully
- `failed`: Analysis failed with error details
- `timeout`: Monitoring timed out (5 minutes)
- `error`: Invalid recording ID or other error

### POST /v1/streaming/batch-process

Stream batch processing results as NDJSON.

**Request Body**:
```json
["recording-id-1", "recording-id-2", "recording-id-3"]
```

**Query Parameters**:
- `batch_size` (integer): Batch size for processing (default: 5)

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: application/x-ndjson
X-Total-Count: 3
X-Batch-Size: 5

{"recording_id": "recording-id-1", "status": "processed", "timestamp": 1640995200.123}
{"recording_id": "recording-id-2", "status": "processed", "timestamp": 1640995200.678}
{"recording_id": "recording-id-3", "status": "processed", "timestamp": 1640995201.234}
```

### GET /v1/streaming/logs/{recording_id}

Stream processing logs for a recording.

**Query Parameters**:
- `follow` (boolean): Whether to follow new logs (default: false)

**Response**:
```http
HTTP/1.1 200 OK
Content-Type: text/plain
X-Recording-ID: 123e4567-e89b-12d3-a456-426614174000
X-Follow: true

[INFO] Starting log streaming for recording 123e4567-e89b-12d3-a456-426614174000
[INFO] File: /path/to/audio/file.mp3
[INFO] Status: processing
[10:30:00] [COMPLETED] bpm analysis - Result: 128.5 (confidence: 0.95)
[10:30:05] [COMPLETED] key analysis - Result: C major (confidence: 0.87)
[10:30:08] [COMPLETED] mood analysis - Processing time: 3200ms
[INFO] Following analysis progress...
[001s] [STATUS] Changed from 'processing' to 'completed'
[001s] [SUCCESS] Analysis completed successfully
```

---

## WebSocket Endpoints

### WS /v1/ws

Real-time WebSocket connection for live updates.

**Query Parameters**:
- `client_id` (string, optional): Client identifier (auto-generated if not provided)

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8000/v1/ws?client_id=my-client');
```

**Welcome Message**:
```json
{
  "type": "welcome",
  "client_id": "my-client",
  "message": "Connected to Analysis Service WebSocket"
}
```

**Client Messages**:

**Ping/Pong**:
```json
// Send
{"type": "ping"}

// Receive
{"type": "pong", "timestamp": 1640995200.123}
```

**Subscribe to Recording**:
```json
// Send
{"type": "subscribe", "recording_id": "123e4567-e89b-12d3-a456-426614174000"}

// Receive
{"type": "subscribed", "recording_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Subscribed to updates for recording 123e4567-e89b-12d3-a456-426614174000"}
```

**Unsubscribe**:
```json
// Send
{"type": "unsubscribe", "recording_id": "123e4567-e89b-12d3-a456-426614174000"}

// Receive
{"type": "unsubscribed", "recording_id": "123e4567-e89b-12d3-a456-426614174000", "message": "Unsubscribed from updates for recording 123e4567-e89b-12d3-a456-426614174000"}
```

**Get Status**:
```json
// Send
{"type": "get_status"}

// Receive
{
  "type": "status",
  "client_id": "my-client",
  "subscriptions": ["123e4567-e89b-12d3-a456-426614174000"],
  "connected_at": "2023-12-01T10:00:00Z",
  "total_connections": 15
}
```

**Server-Sent Updates**:

**Progress Update**:
```json
{
  "type": "progress",
  "progress": 0.75,
  "status": "processing",
  "message": "BPM analysis in progress",
  "timestamp": 1640995200.123
}
```

**Analysis Complete**:
```json
{
  "type": "analysis_complete",
  "results": {
    "bpm": 128.5,
    "key": "C major",
    "mood": "happy"
  },
  "timestamp": 1640995220.456
}
```

**Error**:
```json
{
  "type": "error",
  "error": "Analysis failed",
  "details": {"error_code": "PROCESSING_ERROR"},
  "timestamp": 1640995215.789
}
```

---

## Response Status Codes

### Success Codes
- `200 OK`: Request successful
- `202 Accepted`: Request accepted for processing
- `206 Partial Content`: Partial content for range requests

### Client Error Codes
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded

### Server Error Codes
- `500 Internal Server Error`: Unexpected server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## Request/Response Examples

### Complete Analysis Workflow

**1. Submit Recording**:
```bash
curl -X POST "http://localhost:8000/v1/recordings" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "file_path": "/path/to/song.mp3",
    "priority": 5,
    "metadata": {"source": "upload"}
  }'
```

**2. Start Analysis**:
```bash
curl -X POST "http://localhost:8000/v1/analysis" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "recording_id": "123e4567-e89b-12d3-a456-426614174000",
    "analysis_types": ["bpm", "key", "mood", "energy"]
  }'
```

**3. Monitor Progress via SSE**:
```bash
curl -N "http://localhost:8000/v1/streaming/events/123e4567-e89b-12d3-a456-426614174000" \
  -H "Accept: text/event-stream" \
  -H "X-API-Key: your-api-key"
```

**4. Get Final Results**:
```bash
curl "http://localhost:8000/v1/analysis/123e4567-e89b-12d3-a456-426614174000" \
  -H "X-API-Key: your-api-key"
```

### Streaming Audio with Range Requests

```bash
# Stream entire file
curl "http://localhost:8000/v1/streaming/audio/123e4567-e89b-12d3-a456-426614174000" \
  -H "X-API-Key: your-api-key"

# Stream with range header
curl "http://localhost:8000/v1/streaming/audio/123e4567-e89b-12d3-a456-426614174000" \
  -H "X-API-Key: your-api-key" \
  -H "Range: bytes=0-1023"
```

### Metadata Management

```bash
# Extract metadata from file
curl -X POST "http://localhost:8000/v1/metadata/123e4567-e89b-12d3-a456-426614174000/extract" \
  -H "X-API-Key: your-api-key"

# Update metadata
curl -X PUT "http://localhost:8000/v1/metadata/123e4567-e89b-12d3-a456-426614174000" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "title": "New Title",
    "artist": "New Artist",
    "custom_fields": {"label": "Record Label"}
  }'
```

---

## Development Information

**Framework**: FastAPI 0.100+
**Python**: 3.11+
**Database**: PostgreSQL (async)
**Message Queue**: RabbitMQ
**Caching**: Redis (optional)
**Documentation**: Auto-generated OpenAPI 3.0

**Key Dependencies**:
- FastAPI (web framework)
- Pydantic (data validation)
- SQLAlchemy (ORM with async support)
- Alembic (database migrations)
- aio-pika (async RabbitMQ)
- structlog (structured logging)

**Production Considerations**:
- Configure appropriate CORS origins
- Set up proper authentication/authorization
- Implement database connection pooling
- Configure Redis for caching and sessions
- Set up proper logging and monitoring
- Configure rate limiting based on your needs
- Set up SSL/TLS termination
- Implement request timeout handling

This API documentation covers all REST endpoints with comprehensive examples, error handling, and authentication details for the Analysis Service.
