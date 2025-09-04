# Tracklist Service API Documentation

## Overview

The Tracklist Service provides comprehensive REST API endpoints for managing tracklists from 1001tracklists.com, CUE file generation, manual tracklist creation, search functionality, and developer tools. This service supports both synchronous and asynchronous processing with comprehensive caching, rate limiting, and authentication.

**Base URL**: `http://localhost:8000` (default)
**API Version**: v1
**API Prefix**: `/api/v1`

## Authentication

The service supports multiple authentication methods:

### API Key Authentication
- Header: `X-API-Key: your_api_key_here`
- Keys are managed through developer endpoints
- Different user tiers (Free, Premium, Enterprise) with varying rate limits

### JWT Token Authentication
- Header: `Authorization: Bearer your_jwt_token`
- Access tokens expire in 1 hour
- Refresh tokens available for 7 days

### OAuth2 Support
- Google, GitHub providers supported
- Redirect URI: `https://api.tracktion.com/auth/{provider}/callback`

## Rate Limiting

Rate limits vary by user tier:

- **Free Tier**: 1,000 requests/day, 25,000 tokens/month
- **Premium Tier**: 10,000 requests/day, 250,000 tokens/month
- **Enterprise Tier**: 100,000 requests/day, 2,500,000 tokens/month

Rate limit headers:
- `X-RateLimit-Limit`: Request limit per time window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets

## API Endpoints

### Health Check

#### Get Service Health
```http
GET /health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "service": "tracklist_service",
  "version": "0.1.0"
}
```

---

## Search API

### Search 1001tracklists

#### Search Tracklists
```http
GET /api/v1/search?query={query}&search_type={type}&page={page}&limit={limit}
```

**Parameters**:
- `query` (required): Search query string
- `search_type` (optional): `dj`, `event`, or `track` (default: `dj`)
- `page` (optional): Page number, default 1
- `limit` (optional): Results per page (1-100), default 20
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)

**Example Request**:
```http
GET /api/v1/search?query=Armin%20van%20Buuren&search_type=dj&page=1&limit=10
```

**Response (200 OK)**:
```json
{
  "results": [
    {
      "dj_name": "Armin van Buuren",
      "event_name": "A State of Trance 1000",
      "date": "2024-01-15",
      "venue": "Club Example",
      "set_type": "DJ Set",
      "url": "https://1001tracklists.com/tracklist/12345/armin-asot-1000",
      "duration": "2:00:00",
      "track_count": 30,
      "genre": "Trance",
      "scraped_at": "2024-01-15T10:00:00Z",
      "source_url": "https://1001tracklists.com/search?query=armin"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total_pages": 5,
    "total_items": 50,
    "has_next": true,
    "has_previous": false
  },
  "query_info": {
    "query": "Armin van Buuren",
    "search_type": "dj",
    "filters_applied": []
  },
  "cache_hit": false,
  "response_time_ms": 250.5,
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Get DJ Tracklists
```http
GET /api/v1/search/dj/{dj_slug}?page={page}&limit={limit}
```

**Parameters**:
- `dj_slug` (required): DJ identifier/slug
- `page` (optional): Page number, default 1
- `limit` (optional): Results per page, default 20

**Response**: Same structure as search results

#### Get Event Tracklists
```http
GET /api/v1/search/event/{event_slug}?page={page}&limit={limit}
```

**Parameters**:
- `event_slug` (required): Event identifier/slug
- `page` (optional): Page number, default 1
- `limit` (optional): Results per page, default 20

**Response**: Same structure as search results

### Advanced Search
```http
GET /api/v1/tracklists/search/1001tracklists
```

**Parameters**:
- `query` (optional): General search query
- `artist` (optional): DJ/Artist name
- `title` (optional): Tracklist title
- `genre` (optional): Genre filter
- `date_from` (optional): Start date (YYYY-MM-DD)
- `date_to` (optional): End date (YYYY-MM-DD)
- `page` (optional): Page number, default 1
- `page_size` (optional): Results per page (1-100), default 20
- `force_refresh` (optional): Force re-search, default false

**Response (200 OK)**:
```json
{
  "success": true,
  "results": [
    {
      "id": "12345",
      "url": "https://1001tracklists.com/tracklist/12345/amazing-trance-set",
      "title": "Amazing Trance Set",
      "dj_name": "DJ Example",
      "date": "2024-01-15",
      "event_name": "Winter Festival 2024",
      "track_count": 25,
      "duration": "1:30:00",
      "genre": "Trance",
      "confidence": 0.95
    }
  ],
  "total_count": 100,
  "page": 1,
  "page_size": 20,
  "has_more": true,
  "error": null,
  "cached": false,
  "processing_time_ms": 450,
  "correlation_id": "abc-123-def"
}
```

---

## Tracklist API

### Import Tracklist from 1001tracklists

#### Import Tracklist
```http
POST /api/v1/tracklists/import/1001tracklists
```

**Request Body**:
```json
{
  "url": "https://1001tracklists.com/tracklist/12345/example-set",
  "audio_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "force_refresh": false,
  "cue_format": "standard"
}
```

**Parameters**:
- `url` (required): 1001tracklists URL to import
- `audio_file_id` (required): Associated audio file ID
- `force_refresh` (optional): Force re-import, default false
- `cue_format` (optional): CUE format (`standard`, `cdj`, `traktor`), default `standard`
- `async_processing` (query param): Process asynchronously, default false

**Response (201 Created)**:
```json
{
  "success": true,
  "tracklist": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "audio_file_id": "123e4567-e89b-12d3-a456-426614174000",
    "source": "1001tracklists",
    "url": "https://1001tracklists.com/tracklist/12345/example-set",
    "dj_name": "Example DJ",
    "event_name": "Example Event",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z",
    "tracks": [
      {
        "position": 1,
        "artist": "Artist Name",
        "title": "Track Title",
        "start_time": "00:00:00",
        "end_time": "00:03:30",
        "remix": "Extended Mix",
        "label": "Record Label",
        "catalog_track_id": null,
        "transition_type": null,
        "is_manual_entry": false,
        "bpm": 128,
        "key": "Fm"
      }
    ],
    "confidence_score": 0.95,
    "is_draft": false
  },
  "cue_file_path": "/path/to/generated.cue",
  "error": null,
  "cached": false,
  "processing_time_ms": 5000,
  "correlation_id": "abc-123-def",
  "message": null
}
```

#### Get Import Status
```http
GET /api/v1/tracklists/import/status/{correlation_id}
```

**Response (200 OK)**:
```json
{
  "status": "completed",
  "progress": 100,
  "result": {
    "tracklist_id": "123e4567-e89b-12d3-a456-426614174000",
    "cue_file_path": "/path/to/generated.cue"
  },
  "error": null,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:05:00Z"
}
```

### Retrieve Tracklist

#### Get Tracklist by URL
```http
POST /api/v1/tracklist
```

**Request Body**:
```json
{
  "url": "https://1001tracklists.com/tracklist/12345/example-set",
  "force_refresh": false,
  "include_transitions": true,
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Response**: Same structure as import response

#### Get Tracklist by ID
```http
GET /api/v1/tracklist/{tracklist_id}
```

**Parameters**:
- `force_refresh` (optional): Force re-scraping, default false
- `include_transitions` (optional): Include transitions, default true

**Response**: Currently returns 501 Not Implemented

### Cache Management

#### Clear Tracklist Cache
```http
DELETE /api/v1/tracklist/cache?url={url}
```

**Parameters**:
- `url` (optional): Specific URL to clear, otherwise clears all

**Response (200 OK)**:
```json
{
  "success": true,
  "message": "Cleared cache for URL: https://1001tracklists.com/...",
  "entries_cleared": 1
}
```

---

## CUE Generation API

### Generate CUE File

#### Generate CUE File from Tracklist Data
```http
POST /api/v1/cue/generate
```

**Request Body**:
```json
{
  "format": "standard",
  "options": {
    "include_metadata": true,
    "gap_handling": "auto"
  },
  "validate_audio": false,
  "audio_file_path": "/path/to/audio.wav"
}
```

**Parameters**:
- `async_processing` (query param): Process asynchronously, default false

**Response (200 OK)**:
```json
{
  "success": true,
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "cue_file_id": "456e7890-e89b-12d3-a456-426614174000",
  "file_path": "/path/to/generated.cue",
  "validation_report": {
    "valid": true,
    "error": null,
    "warnings": [],
    "metadata": {}
  },
  "error": null,
  "processing_time_ms": 1500
}
```

#### Generate CUE for Existing Tracklist
```http
POST /api/v1/cue/generate/{tracklist_id}
```

**Request Body**:
```json
{
  "format": "cdj",
  "options": {
    "cdj_version": "3000",
    "memory_cue_points": true
  },
  "validate_audio": true,
  "audio_file_path": "/path/to/audio.wav"
}
```

**Response**: Same structure as generate endpoint

#### Batch CUE Generation
```http
POST /api/v1/cue/generate/batch
```

**Request Body**:
```json
{
  "formats": ["standard", "cdj", "traktor"],
  "options": {
    "include_metadata": true
  },
  "validate_audio": false,
  "audio_file_path": "/path/to/audio.wav"
}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "results": [
    {
      "format": "standard",
      "success": true,
      "cue_file_id": "123-456-789",
      "file_path": "/path/to/standard.cue",
      "error": null
    },
    {
      "format": "cdj",
      "success": true,
      "cue_file_id": "234-567-890",
      "file_path": "/path/to/cdj.cue",
      "error": null
    }
  ],
  "total_files": 3,
  "successful_files": 2,
  "failed_files": 1,
  "processing_time_ms": 3500
}
```

### CUE Format Information

#### Get Supported Formats
```http
GET /api/v1/cue/formats
```

**Response (200 OK)**:
```json
[
  "standard",
  "cdj",
  "traktor",
  "serato",
  "rekordbox",
  "kodi"
]
```

#### Get Format Capabilities
```http
GET /api/v1/cue/formats/{format}/capabilities
```

**Response (200 OK)**:
```json
{
  "format": "cdj",
  "features": [
    "memory_cue_points",
    "hot_cues",
    "beat_grid",
    "waveform_data"
  ],
  "limitations": [
    "max_tracks_99",
    "filename_length_64"
  ],
  "metadata_support": {
    "artist": true,
    "title": true,
    "remix": true,
    "bpm": true,
    "key": true
  }
}
```

#### Get Conversion Preview
```http
GET /api/v1/cue/formats/conversion-preview?source_format=standard&target_format=cdj
```

**Response (200 OK)**:
```json
[
  "Memory cue points will be generated at track boundaries",
  "BPM information may be lost if not available",
  "Some metadata fields may be truncated"
]
```

### Job Management

#### Get Job Status
```http
GET /api/v1/cue/jobs/{job_id}/status
```

**Response (200 OK)**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "job_type": "cue_generation",
  "service_name": "tracklist_service",
  "progress": 100,
  "total_items": 1,
  "result": {
    "cue_file_id": "456-789-012",
    "file_path": "/path/to/generated.cue"
  },
  "error_message": null,
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:01:30Z",
  "created_at": "2024-01-15T09:59:45Z",
  "timestamp": "2024-01-15T10:01:30Z"
}
```

### File Management

#### Download CUE File
```http
GET /api/v1/cue/download/{cue_file_id}
```

**Response (200 OK)**:
```
Content-Type: application/x-cue
Content-Disposition: attachment; filename=standard_12345678_v1.cue
Cache-Control: no-cache
X-File-Version: 1
X-File-Format: standard

FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track Title"
    PERFORMER "Artist Name"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track Title 2"
    PERFORMER "Artist Name 2"
    INDEX 01 03:30:00
```

#### Get CUE File Info
```http
GET /api/v1/cue/files/{cue_file_id}
```

**Response (200 OK)**:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "tracklist_id": "456e7890-e89b-12d3-a456-426614174000",
  "format": "standard",
  "file_path": "/path/to/file.cue",
  "file_size": 2048,
  "checksum": "sha256:abc123...",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "version": 1,
  "is_active": true,
  "metadata": {
    "total_tracks": 15,
    "total_duration": "01:30:00"
  },
  "storage_info": {
    "backend": "filesystem",
    "accessible": true
  }
}
```

#### List CUE Files
```http
GET /api/v1/cue/files?tracklist_id={id}&format={format}&limit={limit}&offset={offset}
```

**Parameters**:
- `tracklist_id` (optional): Filter by tracklist ID
- `format` (optional): Filter by CUE format
- `limit` (optional): Number of files (1-100), default 20
- `offset` (optional): Number to skip, default 0

**Response (200 OK)**:
```json
{
  "files": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "tracklist_id": "456e7890-e89b-12d3-a456-426614174000",
      "format": "standard",
      "file_path": "/path/to/file.cue",
      "file_size": 2048,
      "checksum": "sha256:abc123...",
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T10:00:00Z",
      "version": 1,
      "is_active": true,
      "metadata": {}
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 1,
    "has_more": false,
    "returned": 1
  },
  "filters": {
    "tracklist_id": null,
    "format": null
  }
}
```

#### Delete CUE File
```http
DELETE /api/v1/cue/files/{cue_file_id}?soft_delete=true
```

**Parameters**:
- `soft_delete` (optional): Soft delete vs permanent, default true

**Response (200 OK)**:
```json
{
  "success": true,
  "cue_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "deletion_type": "soft",
  "deleted_at": "2024-01-15T10:00:00Z",
  "message": "CUE file soft deleted successfully"
}
```

#### Regenerate CUE File
```http
POST /api/v1/cue/files/{cue_file_id}/regenerate
```

**Request Body**:
```json
{
  "include_metadata": true,
  "gap_handling": "preserve"
}
```

**Response**: Same structure as generate endpoint

### Validation

#### Validate CUE File
```http
POST /api/v1/cue/files/{cue_file_id}/validate
```

**Request Body**:
```json
{
  "audio_file_path": "/path/to/audio.wav",
  "validation_options": {
    "strict_timing": false,
    "check_file_references": true
  }
}
```

**Response (200 OK)**:
```json
{
  "cue_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "format": "standard",
  "file_path": "/path/to/file.cue",
  "validation_timestamp": "2024-01-15T10:00:00Z",
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "type": "timing_warning",
      "message": "Small gap detected between tracks 3 and 4",
      "severity": "warning"
    }
  ],
  "metadata": {
    "file_size": 2048,
    "checksum": "sha256:abc123...",
    "version": 1,
    "audio_duration": 5400,
    "tracklist_duration": 5395
  },
  "processing_time_ms": 150,
  "recommendations": [
    "Review warnings for potential quality improvements"
  ]
}
```

#### Validate Tracklist for CUE
```http
POST /api/v1/cue/validate
```

**Request Body**:
```json
{
  "tracklist_id": "123e4567-e89b-12d3-a456-426614174000",
  "format": "standard",
  "audio_file_path": "/path/to/audio.wav",
  "validation_options": {
    "strict_mode": false
  }
}
```

**Response**: Same structure as file validation

### Format Conversion

#### Convert CUE File Format
```http
POST /api/v1/cue/files/{cue_file_id}/convert
```

**Request Body**:
```json
{
  "target_format": "cdj",
  "preserve_metadata": true,
  "conversion_options": {
    "cdj_version": "3000",
    "generate_memory_cues": true
  }
}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "cue_file_id": "new-file-id-123",
  "source_cue_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_format": "standard",
  "target_format": "cdj",
  "conversion_needed": true,
  "warnings": [
    {
      "type": "conversion_warning",
      "message": "Memory cue points generated automatically",
      "severity": "warning"
    }
  ],
  "errors": [],
  "metadata": {
    "preserve_metadata": true,
    "conversion_options": {
      "cdj_version": "3000",
      "generate_memory_cues": true
    }
  },
  "file_path": "/path/to/converted.cue",
  "file_size": 2150,
  "checksum": "sha256:def456...",
  "version": 1,
  "processing_time_ms": 800,
  "recommendations": [
    "Successfully converted from standard to cdj"
  ]
}
```

### Cache Management

#### Get Cache Stats
```http
GET /api/v1/cue/cache/stats
```

**Response (200 OK)**:
```json
{
  "hit_rate": 0.85,
  "total_requests": 1000,
  "cache_hits": 850,
  "cache_misses": 150,
  "cached_items": 250,
  "cache_size_mb": 45.6,
  "memory_usage": {
    "used_mb": 45.6,
    "available_mb": 954.4,
    "utilization": 0.046
  },
  "popular_formats": {
    "standard": 120,
    "cdj": 85,
    "traktor": 45
  }
}
```

#### Warm Cache
```http
POST /api/v1/cue/cache/warm
```

**Request Body**:
```json
{
  "tracklist_ids": [
    "123e4567-e89b-12d3-a456-426614174000",
    "234e5678-e89b-12d3-a456-426614174000"
  ],
  "formats": ["standard", "cdj"]
}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "tracklists_processed": 2,
  "formats_cached": 2,
  "total_entries_created": 4,
  "processing_time_ms": 2500,
  "cache_hit_improvement": 0.15
}
```

#### Clear Cache
```http
DELETE /api/v1/cue/cache/clear?pattern=cue:*
```

**Parameters**:
- `pattern` (optional): Redis pattern to match keys

**Response (200 OK)**:
```json
{
  "success": true,
  "cleared_entries": 25,
  "pattern": "cue:*",
  "message": "Cleared 25 cache entries",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

---

## Manual Tracklist API

### Create Manual Tracklist

#### Create Manual Tracklist
```http
POST /api/v1/tracklists/manual
```

**Request Body**:
```json
{
  "audio_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "tracks": [
    {
      "position": 1,
      "artist": "Artist Name",
      "title": "Track Title",
      "start_time": "00:00:00",
      "end_time": "00:03:30",
      "remix": "Extended Mix",
      "label": "Label Name"
    }
  ],
  "is_draft": true
}
```

**Response (201 Created)**:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "audio_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "source": "manual",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "tracks": [
    {
      "position": 1,
      "artist": "Artist Name",
      "title": "Track Title",
      "start_time": "00:00:00",
      "end_time": "00:03:30",
      "remix": "Extended Mix",
      "label": "Label Name",
      "catalog_track_id": null,
      "transition_type": null,
      "is_manual_entry": true,
      "bpm": null,
      "key": null
    }
  ],
  "confidence_score": 1.0,
  "draft_version": 1,
  "is_draft": true,
  "parent_tracklist_id": null,
  "default_cue_format": null
}
```

### Track Management

#### Add Track
```http
POST /api/v1/tracklists/{tracklist_id}/tracks
```

**Request Body**:
```json
{
  "position": 2,
  "artist": "New Artist",
  "title": "New Track",
  "start_time": "03:30:00",
  "end_time": "06:45:00",
  "remix": "Radio Edit",
  "label": "New Label"
}
```

**Response (201 Created)**:
```json
{
  "position": 2,
  "artist": "New Artist",
  "title": "New Track",
  "start_time": "03:30:00",
  "end_time": "06:45:00",
  "remix": "Radio Edit",
  "label": "New Label",
  "catalog_track_id": null,
  "transition_type": null,
  "is_manual_entry": true,
  "bpm": null,
  "key": null
}
```

#### Update Track
```http
PUT /api/v1/tracklists/{tracklist_id}/tracks/{position}
```

**Request Body**:
```json
{
  "artist": "Updated Artist",
  "title": "Updated Title",
  "start_time": "03:35:00"
}
```

**Response (200 OK)**: Same structure as track object

#### Delete Track
```http
DELETE /api/v1/tracklists/{tracklist_id}/tracks/{position}
```

**Response (204 No Content)**

#### Update Track Timing
```http
PUT /api/v1/tracklists/{tracklist_id}/tracks/{position}/timing
```

**Request Body**:
```json
{
  "start_time": "03:40:00",
  "end_time": "06:50:00"
}
```

**Response (200 OK)**: Track object with updated timing

### Bulk Operations

#### Bulk Update Tracks
```http
PUT /api/v1/tracklists/{tracklist_id}/tracks/bulk
```

**Request Body**:
```json
{
  "tracks": [
    {
      "position": 1,
      "artist": "Artist 1",
      "title": "Track 1",
      "start_time": "00:00:00",
      "end_time": "03:30:00"
    },
    {
      "position": 2,
      "artist": "Artist 2",
      "title": "Track 2",
      "start_time": "03:30:00",
      "end_time": "07:00:00"
    }
  ]
}
```

**Response (200 OK)**: Array of updated track objects

#### Reorder Track
```http
POST /api/v1/tracklists/{tracklist_id}/tracks/reorder
```

**Request Body**:
```json
{
  "from_position": 3,
  "to_position": 1
}
```

**Response (200 OK)**: Array of reordered track objects

### Timing Operations

#### Auto-Calculate End Times
```http
POST /api/v1/tracklists/{tracklist_id}/tracks/auto-calculate-end-times
```

**Request Body**:
```json
{
  "audio_duration": "01:30:00"
}
```

**Response (200 OK)**: Array of tracks with calculated end times

#### Get Timing Suggestions
```http
GET /api/v1/tracklists/{tracklist_id}/tracks/timing-suggestions?target_duration=01:25:00
```

**Response (200 OK)**:
```json
[
  {
    "track_position": 3,
    "suggestion": "reduce_by_10_seconds",
    "current_duration": "04:30:00",
    "suggested_duration": "04:20:00",
    "reason": "Track appears to have extended outro"
  }
]
```

#### Validate Timing
```http
POST /api/v1/tracklists/{tracklist_id}/tracks/validate-timing
```

**Request Body**:
```json
{
  "audio_duration": "01:30:00",
  "allow_gaps": true
}
```

**Response (200 OK)**:
```json
{
  "is_valid": false,
  "issues": [
    "Track 2 overlaps with track 3 by 5.0 seconds"
  ],
  "track_count": 15
}
```

### Draft Management

#### List Drafts
```http
GET /api/v1/tracklists/{audio_file_id}/drafts?include_versions=false
```

**Response (200 OK)**: Array of draft tracklist objects

#### Publish Draft
```http
POST /api/v1/tracklists/{tracklist_id}/publish?validate_before_publish=true&generate_cue_async=true
```

**Response (200 OK)**: Published tracklist object

### Catalog Integration

#### Search Catalog
```http
GET /api/v1/tracklists/catalog/search?query=example&artist=Artist&title=Title&limit=10
```

**Response (200 OK)**:
```json
[
  {
    "catalog_track_id": "456e7890-e89b-12d3-a456-426614174000",
    "artist": "Artist Name",
    "title": "Track Title",
    "album": "Album Name",
    "genre": "Electronic",
    "bpm": 128.0,
    "key": "Am",
    "confidence": 0.95
  }
]
```

#### Match Tracks to Catalog
```http
POST /api/v1/tracklists/catalog/match
```

**Request Body**:
```json
{
  "tracks": [
    {
      "position": 1,
      "artist": "Artist Name",
      "title": "Track Title",
      "start_time": "00:00:00"
    }
  ],
  "threshold": 0.7
}
```

**Response (200 OK)**: Array of tracks with populated `catalog_track_id` fields

#### Match All Tracks
```http
POST /api/v1/tracklists/{tracklist_id}/tracks/match-to-catalog?threshold=0.8
```

**Response (200 OK)**: Array of matched track objects

### CUE Generation

#### Generate CUE for Manual Tracklist
```http
POST /api/v1/tracklists/{tracklist_id}/generate-cue
```

**Request Body**:
```json
{
  "audio_file_path": "/path/to/audio.wav",
  "cue_format": "standard"
}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "cue_file_path": null,
  "cue_file_id": null,
  "format": "standard"
}
```

---

## Batch Processing API

### Batch Operations

#### Create Batch Job
```http
POST /api/v1/batch
```

**Request Body**:
```json
{
  "urls": [
    "https://1001tracklists.com/tracklist/12345/set-1",
    "https://1001tracklists.com/tracklist/12346/set-2"
  ],
  "priority": "normal",
  "user_id": "user-123",
  "template": "dj_set",
  "options": {
    "generate_cue": true,
    "cue_format": "standard"
  }
}
```

**Response (200 OK)**:
```json
{
  "batch_id": "batch-123e4567-e89b-12d3",
  "total_jobs": 2,
  "priority": "normal",
  "status": "queued",
  "estimated_completion": "2024-01-15T10:15:00Z",
  "message": "Batch successfully queued"
}
```

#### Get Batch Status
```http
GET /api/v1/batch/{batch_id}/status
```

**Response (200 OK)**:
```json
{
  "batch_id": "batch-123e4567-e89b-12d3",
  "status": "processing",
  "total_jobs": 2,
  "jobs_status": {
    "queued": 0,
    "processing": 1,
    "completed": 1,
    "failed": 0
  },
  "progress_percentage": 50.0,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:07:30Z",
  "error": null
}
```

#### Control Batch
```http
POST /api/v1/batch/{batch_id}/cancel
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "message": "Batch batch-123e4567-e89b-12d3 cancelled"
}
```

#### Schedule Batch
```http
POST /api/v1/batch/schedule
```

**Request Body**:
```json
{
  "urls": [
    "https://1001tracklists.com/tracklist/12345/weekly-set"
  ],
  "cron_expression": "0 2 * * 1",
  "user_id": "user-123",
  "name": "Weekly Set Import"
}
```

**Response (200 OK)**:
```json
{
  "schedule_id": "schedule-789",
  "message": "Batch scheduled successfully",
  "cron": "0 2 * * 1"
}
```

### Real-time Updates

#### WebSocket Progress Updates
```websocket
WS /api/v1/batch/{batch_id}/progress
```

**Message Format**:
```json
{
  "batch_id": "batch-123e4567-e89b-12d3",
  "status": "processing",
  "progress": 75,
  "jobs_status": {
    "completed": 3,
    "processing": 1,
    "failed": 0
  },
  "timestamp": "2024-01-15T10:12:30Z"
}
```

---

## Developer API

### API Key Management

#### List API Keys
```http
GET /api/v1/developer/keys?include_inactive=false
```

**Response (200 OK)**:
```json
[
  {
    "key_id": "key-123e4567-e89b",
    "name": "Production API Key",
    "key_prefix": "tk_12345...",
    "is_active": true,
    "created_at": "2024-01-15T10:00:00Z",
    "expires_at": null,
    "last_used_at": "2024-01-15T09:45:00Z",
    "permissions": {
      "read": true,
      "write": true,
      "admin": false
    },
    "usage_stats": {
      "requests_today": 150,
      "tokens_used_today": 5000,
      "average_response_time": 250
    }
  }
]
```

#### Create API Key
```http
POST /api/v1/developer/keys
```

**Request Body**:
```json
{
  "name": "Development Key",
  "description": "Key for development environment",
  "permissions": {
    "read": true,
    "write": false,
    "admin": false
  },
  "expires_in_days": 90
}
```

**Response (201 Created)**:
```json
{
  "key_id": "key-234e5678-e89b",
  "api_key": "tk_123456789abcdef...",
  "name": "Development Key",
  "permissions": {
    "read": true,
    "write": false,
    "admin": false
  },
  "expires_at": "2024-04-15T10:00:00Z",
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### Rotate API Key
```http
POST /api/v1/developer/keys/{key_id}/rotate
```

**Response (200 OK)**:
```json
{
  "key_id": "key-345e6789-e89b",
  "api_key": "tk_newkey123456789...",
  "name": "Development Key",
  "permissions": {
    "read": true,
    "write": false,
    "admin": false
  },
  "expires_at": "2024-04-15T10:00:00Z",
  "created_at": "2024-01-15T10:05:00Z"
}
```

#### Revoke API Key
```http
DELETE /api/v1/developer/keys/{key_id}
```

**Response (200 OK)**:
```json
{
  "success": true,
  "message": "API key key-123e4567-e89b has been revoked",
  "revoked_at": "2024-01-15T10:00:00Z"
}
```

### Usage Analytics

#### Get Key Usage Analytics
```http
GET /api/v1/developer/keys/{key_id}/usage?period=month
```

**Response (200 OK)**:
```json
{
  "key_id": "key-123e4567-e89b",
  "usage_stats": {
    "total_requests": 15000,
    "total_tokens": 450000,
    "total_cost": 0.90,
    "average_response_time": 275,
    "error_rate": 0.02,
    "top_endpoints": [
      "/api/v1/search",
      "/api/v1/cue/generate"
    ],
    "period": "month",
    "key_id": "key-123e4567-e89b",
    "key_name": "Production API Key",
    "key_created": "2024-01-15T10:00:00Z",
    "key_last_used": "2024-01-15T09:45:00Z"
  },
  "cost_breakdown": {
    "base_cost": 0.90,
    "token_cost": 0.90,
    "request_cost": 0.0,
    "tier": "premium",
    "period": "month"
  },
  "recommendations": [
    "API response times are high - consider optimizing queries"
  ]
}
```

#### Get Usage Summary
```http
GET /api/v1/developer/usage/summary?period=month
```

**Response (200 OK)**:
```json
{
  "user_id": "user-123",
  "user_tier": "premium",
  "period": "month",
  "usage": {
    "total_requests": 25000,
    "total_tokens": 750000,
    "total_cost": 1.50,
    "average_response_time": 280,
    "error_rate": 0.015
  },
  "limits": {
    "requests_per_day": 10000,
    "tokens_per_month": 250000,
    "max_keys": 10
  },
  "utilization": {
    "requests_percent": 8.33,
    "tokens_percent": 300.0,
    "keys_percent": 30.0
  },
  "keys": {
    "active": 3,
    "total": 5
  },
  "recommendations": [
    "Token usage is high - consider upgrading tier"
  ]
}
```

---

## Error Responses

### Standard Error Format

All API endpoints follow a consistent error response format:

```json
{
  "error": "Error message description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
  "details": {
    "field": "field_name",
    "value": "invalid_value"
  },
  "retry_after": 30
}
```

### HTTP Status Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **204 No Content**: Request successful, no content returned
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required or invalid
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (e.g., duplicate)
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **501 Not Implemented**: Feature not yet implemented
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Codes

#### Authentication & Authorization
- `INVALID_API_KEY`: API key is invalid or revoked
- `EXPIRED_TOKEN`: JWT token has expired
- `INSUFFICIENT_PERMISSIONS`: Operation requires higher permissions
- `RATE_LIMIT_EXCEEDED`: Request rate limit exceeded

#### Validation Errors
- `VALIDATION_ERROR`: Request validation failed
- `INVALID_FORMAT`: Invalid data format
- `REQUIRED_FIELD_MISSING`: Required field not provided
- `INVALID_URL`: Provided URL is invalid or unsupported

#### Service Errors
- `SEARCH_FAILED`: Search operation failed
- `IMPORT_FAILED`: Tracklist import failed
- `CUE_GENERATION_FAILED`: CUE file generation failed
- `CACHE_ERROR`: Cache operation failed

#### Resource Errors
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `RESOURCE_CONFLICT`: Resource already exists or conflicts
- `STORAGE_ERROR`: File storage operation failed

## Request/Response Examples

### Example: Complete Tracklist Import Flow

1. **Search for tracklist**:
```http
GET /api/v1/search?query=Armin%20van%20Buuren%20ASOT&search_type=dj&limit=5
```

2. **Import selected tracklist**:
```http
POST /api/v1/tracklists/import/1001tracklists
Content-Type: application/json
X-API-Key: tk_your_api_key_here

{
  "url": "https://1001tracklists.com/tracklist/12345/armin-asot-1000",
  "audio_file_id": "123e4567-e89b-12d3-a456-426614174000",
  "cue_format": "standard"
}
```

3. **Generate additional CUE formats**:
```http
POST /api/v1/cue/generate/batch
Content-Type: application/json
X-API-Key: tk_your_api_key_here

{
  "formats": ["cdj", "traktor"],
  "options": {
    "include_metadata": true
  },
  "audio_file_path": "/path/to/audio.wav"
}
```

4. **Download generated CUE file**:
```http
GET /api/v1/cue/download/456e7890-e89b-12d3-a456-426614174000
X-API-Key: tk_your_api_key_here
```

## Rate Limiting Examples

### Rate Limit Headers in Response
```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642291200
Content-Type: application/json

{
  "status": "success"
}
```

### Rate Limit Exceeded Response
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642291200
Retry-After: 60
Content-Type: application/json

{
  "error": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60,
  "details": {
    "limit": 1000,
    "window": "1 hour",
    "upgrade_url": "/api/v1/developer/usage/summary"
  }
}
```

---

This documentation covers all REST API endpoints available in the Tracklist Service. For WebSocket connections, SDK usage, or additional integration examples, please refer to the service's README.md and example files.
