# WebSocket Connections and Message Queue Contracts

This document provides comprehensive documentation for WebSocket endpoints and RabbitMQ message queue contracts across all Tracktion services.

## WebSocket Endpoints

### Analysis Service WebSocket API

#### Connection Endpoint
- **URL**: `ws://localhost:8000/v1/ws`
- **Protocol**: WebSocket
- **Query Parameters**:
  - `client_id` (optional): Unique client identifier. Generated automatically if not provided.

#### Connection Flow
1. Client connects to WebSocket endpoint
2. Server assigns or uses provided client_id
3. Server sends welcome message
4. Client can send subscription requests
5. Server broadcasts real-time updates to subscribed clients

#### Message Types

##### Client-to-Server Messages

**Ping Message**
```json
{
  "type": "ping"
}
```
Response: Server responds with pong message including timestamp.

**Subscribe to Recording Updates**
```json
{
  "type": "subscribe",
  "recording_id": "uuid-string"
}
```
Response: Confirmation message with subscription details.

**Unsubscribe from Recording Updates**
```json
{
  "type": "unsubscribe",
  "recording_id": "uuid-string"
}
```
Response: Confirmation message with unsubscription details.

**Get Connection Status**
```json
{
  "type": "get_status"
}
```
Response: Current connection status including subscriptions and metadata.

##### Server-to-Client Messages

**Welcome Message**
```json
{
  "type": "welcome",
  "client_id": "uuid-string",
  "message": "Connected to Analysis Service WebSocket"
}
```

**Pong Response**
```json
{
  "type": "pong",
  "timestamp": 1234567890.123
}
```

**Subscription Confirmation**
```json
{
  "type": "subscribed",
  "recording_id": "uuid-string",
  "message": "Subscribed to updates for recording {recording_id}"
}
```

**Progress Update**
```json
{
  "recording_id": "uuid-string",
  "data": {
    "type": "progress",
    "progress": 0.75,
    "status": "analyzing",
    "message": "Processing audio analysis",
    "timestamp": 1234567890.123
  }
}
```

**Analysis Complete**
```json
{
  "recording_id": "uuid-string",
  "data": {
    "type": "analysis_complete",
    "results": {
      "bpm": 128.5,
      "key": "C major",
      "energy": 0.8
    },
    "timestamp": 1234567890.123
  }
}
```

**Error Notification**
```json
{
  "recording_id": "uuid-string",
  "data": {
    "type": "error",
    "error": "Analysis failed",
    "details": {
      "error_code": "ANALYSIS_ERROR",
      "file_path": "/path/to/file.mp3"
    },
    "timestamp": 1234567890.123
  }
}
```

**Connection Status**
```json
{
  "type": "status",
  "client_id": "uuid-string",
  "subscriptions": ["recording-uuid-1", "recording-uuid-2"],
  "connected_at": 1234567890.123,
  "total_connections": 5
}
```

**Generic Error**
```json
{
  "type": "error",
  "message": "Error description"
}
```

#### WebSocket Manager Features

- **Connection Management**: Automatic client ID assignment, connection tracking
- **Subscription Management**: Subscribe/unsubscribe to specific recording updates
- **Broadcasting**: Targeted broadcasts to subscribed clients only
- **Error Handling**: Graceful handling of disconnections and errors
- **Metadata Tracking**: Connection timestamps and subscription tracking

## RabbitMQ Message Queue Contracts

### Exchange and Queue Architecture

#### Primary Exchanges

**File Events Exchange**
- **Name**: `file_events`
- **Type**: Topic
- **Durability**: Durable
- **Description**: Handles all file lifecycle events (create, modify, delete, move, rename)

**Tracktion Events Exchange**
- **Name**: `tracktion.events` or `tracktion_exchange`
- **Type**: Topic
- **Durability**: Durable
- **Description**: General application events and task requests

**CUE Generation Exchange**
- **Name**: `cue.direct`
- **Type**: Direct
- **Durability**: Durable
- **Description**: CUE file generation and processing messages

**File Rename Exchange**
- **Name**: Configurable via `RABBITMQ_EXCHANGE` environment variable
- **Type**: Topic
- **Durability**: Durable
- **Description**: File rename operations and feedback

### Message Schemas and Routing

#### File Events (File Watcher Service)

**File Discovery Message**
- **Routing Key**: `file.discovered`
- **Exchange**: `file_events`
- **Queue**: Consumer-defined

```json
{
  "correlation_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z",
  "event_type": "file_discovered",
  "file_info": {
    "path": "/data/music/track.mp3",
    "extension": ".mp3",
    "size_bytes": 5242880,
    "sha256_hash": "sha256-hash",
    "xxh128_hash": "xxhash"
  },
  "file_type": "mp3",
  "instance_id": "watcher-instance-1",
  "watched_directory": "/data/music",
  "format_family": "mp3"
}
```

**File Lifecycle Events**
- **Routing Keys**: `file.created`, `file.modified`, `file.deleted`, `file.moved`, `file.renamed`
- **Exchange**: `file_events`

```json
{
  "correlation_id": "uuid-string",
  "timestamp": "2024-01-01T00:00:00Z",
  "event_type": "created|modified|deleted|moved|renamed",
  "file_path": "/data/music/track.mp3",
  "old_path": "/data/music/old-track.mp3",
  "size_bytes": 5242880,
  "sha256_hash": "sha256-hash",
  "xxh128_hash": "xxhash",
  "file_type": "mp3",
  "instance_id": "watcher-instance-1",
  "watched_directory": "/data/music"
}
```

#### Analysis Service Messages

**Analysis Request**
- **Routing Key**: `analysis.request`
- **Exchange**: `tracktion.events`

```json
{
  "recording_id": "uuid-string",
  "file_path": "/data/music/track.mp3",
  "analysis_types": ["bpm", "key", "energy"],
  "metadata": {},
  "correlation_id": "uuid-string"
}
```

**Metadata Extraction Request**
- **Routing Key**: `metadata.extract`
- **Exchange**: `tracktion.events`

```json
{
  "recording_id": "uuid-string",
  "extraction_types": ["id3_tags", "audio_analysis"],
  "correlation_id": "uuid-string"
}
```

**Tracklist Generation Request**
- **Routing Key**: `tracklist.generate`
- **Exchange**: `tracktion.events`

```json
{
  "recording_id": "uuid-string",
  "source_hint": "auto",
  "correlation_id": "uuid-string"
}
```

**Analysis Cancellation**
- **Routing Key**: `analysis.cancel`
- **Exchange**: `tracktion.events`
- **Priority**: High (8)

```json
{
  "recording_id": "uuid-string",
  "action": "cancel",
  "correlation_id": "uuid-string"
}
```

#### CUE Generation Messages (Tracklist Service)

**Single CUE Generation Request**
- **Routing Key**: `cue.generation.single`
- **Exchange**: `cue.direct`
- **Queue**: `cue.generation`

```json
{
  "message_id": "uuid-string",
  "message_type": "cue_generation",
  "timestamp": "2024-01-01T00:00:00Z",
  "correlation_id": "uuid-string",
  "retry_count": 0,
  "priority": 5,
  "metadata": {},
  "tracklist_id": "uuid-string",
  "format": "standard",
  "options": {},
  "validate_audio": true,
  "audio_file_path": "/path/to/audio.mp3",
  "job_id": "uuid-string",
  "requested_by": "user-id"
}
```

**CUE Generation Complete**
- **Routing Key**: `cue.generation.complete`
- **Exchange**: `cue.direct`
- **Queue**: `cue.generation.complete`

```json
{
  "message_id": "uuid-string",
  "message_type": "cue_generation_complete",
  "timestamp": "2024-01-01T00:00:00Z",
  "original_message_id": "uuid-string",
  "job_id": "uuid-string",
  "tracklist_id": "uuid-string",
  "success": true,
  "cue_file_id": "uuid-string",
  "file_path": "/path/to/generated.cue",
  "file_size": 2048,
  "checksum": "sha256-hash",
  "validation_report": {
    "is_valid": true,
    "errors": []
  },
  "processing_time_ms": 1500.0,
  "queue_time_ms": 250.0
}
```

**Batch CUE Generation Request**
- **Routing Key**: `cue.generation.batch`
- **Exchange**: `cue.direct`
- **Queue**: `cue.generation.batch`

```json
{
  "message_id": "uuid-string",
  "message_type": "batch_cue_generation",
  "timestamp": "2024-01-01T00:00:00Z",
  "tracklist_id": "uuid-string",
  "formats": ["standard", "serato", "traktor"],
  "options": {},
  "validate_audio": false,
  "batch_job_id": "uuid-string",
  "requested_by": "user-id"
}
```

**CUE Validation Request**
- **Routing Key**: `cue.validation`
- **Exchange**: `cue.direct`
- **Queue**: `cue.validation`

```json
{
  "message_id": "uuid-string",
  "message_type": "cue_validation",
  "timestamp": "2024-01-01T00:00:00Z",
  "cue_file_id": "uuid-string",
  "audio_file_path": "/path/to/audio.mp3",
  "validation_options": {},
  "validation_job_id": "uuid-string",
  "requested_by": "user-id"
}
```

**CUE Conversion Request**
- **Routing Key**: `cue.conversion`
- **Exchange**: `cue.direct`
- **Queue**: `cue.conversion`

```json
{
  "message_id": "uuid-string",
  "message_type": "cue_conversion",
  "timestamp": "2024-01-01T00:00:00Z",
  "source_cue_file_id": "uuid-string",
  "target_format": "serato",
  "preserve_metadata": true,
  "conversion_options": {},
  "conversion_job_id": "uuid-string",
  "requested_by": "user-id"
}
```

#### File Rename Service Messages

**Rename Request**
- **Routing Key**: `rename.request`
- **Queue**: `file_rename.request`

```json
{
  "file_path": "/data/music/track.mp3",
  "suggested_name": "Artist - Track Title.mp3",
  "options": {
    "preserve_extension": true,
    "backup_original": false
  },
  "correlation_id": "uuid-string"
}
```

**Rename Response**
- **Routing Key**: `rename.response`
- **Queue**: `file_rename.response`

```json
{
  "original_path": "/data/music/track.mp3",
  "new_path": "/data/music/Artist - Track Title.mp3",
  "success": true,
  "error": null,
  "correlation_id": "uuid-string"
}
```

**Pattern Analysis Request**
- **Routing Key**: `rename.pattern.analyze`
- **Queue**: `file_rename.pattern`

```json
{
  "file_paths": ["/data/music/track1.mp3", "/data/music/track2.mp3"],
  "options": {
    "detect_patterns": true,
    "suggest_improvements": true
  },
  "correlation_id": "uuid-string"
}
```

**Rename Feedback**
- **Routing Key**: `rename.feedback`
- **Queue**: `file_rename.feedback`

```json
{
  "rename_id": "uuid-string",
  "feedback_type": "success|failure|improvement",
  "message": "Rename completed successfully",
  "user_rating": 5,
  "correlation_id": "uuid-string"
}
```

### Queue Configurations

#### Analysis Service Queues

| Queue Name | Routing Key | Durable | Priority | Dead Letter |
|------------|-------------|---------|----------|-------------|
| `analysis_queue` | `file.analyze` | Yes | 1-10 | Yes |
| `analysis.lifecycle.events` | `file.#` | Yes | Standard | Yes |

#### Cataloging Service Queues

| Queue Name | Routing Key | Durable | TTL | Dead Letter |
|------------|-------------|---------|-----|-------------|
| `cataloging.file.events` | `file.created`, `file.modified`, `file.deleted`, `file.moved`, `file.renamed` | Yes | 24h | Yes |

#### Tracklist Service Queues

| Queue Name | Routing Key | Durable | Description |
|------------|-------------|---------|-------------|
| `cue.generation` | `cue.generation.single` | Yes | Single CUE file generation |
| `cue.generation.batch` | `cue.generation.batch` | Yes | Batch CUE file generation |
| `cue.generation.complete` | `cue.generation.complete` | Yes | Generation completion notifications |
| `cue.generation.batch.complete` | `cue.generation.batch.complete` | Yes | Batch completion notifications |
| `cue.validation` | `cue.validation` | Yes | CUE file validation |
| `cue.conversion` | `cue.conversion` | Yes | CUE format conversion |

#### File Rename Service Queues

| Queue Name | Routing Key | Durable | Description |
|------------|-------------|---------|-------------|
| `file_rename.request` | `rename.request` | Yes | Rename operation requests |
| `file_rename.response` | `rename.response` | Yes | Rename operation results |
| `file_rename.feedback` | `rename.feedback` | Yes | User feedback on renames |
| `file_rename.pattern` | `rename.pattern.analyze` | Yes | Pattern analysis requests |

## Authentication and Security

### WebSocket Security

#### Connection Security
- **Transport**: WebSocket over HTTP (upgradable to WSS for production)
- **Origin Validation**: Implement origin checking in production
- **Rate Limiting**: Connection and message rate limiting per client
- **Authentication**: Currently no authentication (implement JWT/session-based auth for production)

#### Message Validation
- **JSON Schema Validation**: All incoming messages validated against schemas
- **Size Limits**: Message size restrictions to prevent abuse
- **Type Checking**: Strict message type validation
- **Sanitization**: Input sanitization for all user-provided data

### RabbitMQ Security

#### Connection Security
- **Authentication**: Username/password authentication (configurable)
- **Virtual Hosts**: Logical separation of message flows
- **TLS Support**: Available for encrypted connections
- **Connection Limits**: Configurable connection and channel limits

#### Message Security
- **Persistent Messages**: Critical messages marked as persistent
- **Dead Letter Exchanges**: Failed messages routed to dead letter queues
- **Message TTL**: Time-to-live configured for message expiration
- **Prefetch Limits**: Consumer prefetch limits to prevent overwhelming

#### Access Control
- **Queue Permissions**: Read/write permissions per queue
- **Exchange Permissions**: Publishing permissions per exchange
- **User Isolation**: Separate users for different services
- **Audit Logging**: Message processing and error logging

## Error Handling

### WebSocket Error Handling

#### Connection Errors
- **Reconnection Logic**: Automatic reconnection with exponential backoff
- **Connection State Tracking**: Monitor connection health
- **Graceful Degradation**: Continue operation when WebSocket unavailable

#### Message Errors
- **Validation Errors**: Return descriptive error messages for invalid requests
- **Processing Errors**: Notify clients of processing failures
- **Timeout Handling**: Handle long-running operations with proper timeout responses

### RabbitMQ Error Handling

#### Message Processing Errors
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Dead Letter Queues**: Failed messages routed to DLQ after max retries
- **Error Classification**: Distinguish between transient and permanent errors
- **Circuit Breaker**: Prevent cascade failures during service outages

#### Connection Errors
- **Robust Connections**: Automatic reconnection on connection loss
- **Heartbeat Monitoring**: Connection health monitoring
- **Failover Support**: Multiple broker support for high availability

## Performance Considerations

### WebSocket Performance

- **Connection Pooling**: Efficient connection management
- **Message Batching**: Batch multiple updates for efficiency
- **Subscription Management**: Efficient client subscription tracking
- **Memory Management**: Proper cleanup of disconnected clients

### RabbitMQ Performance

- **Message Persistence**: Balance durability vs performance
- **Prefetch Configuration**: Optimize consumer prefetch counts
- **Queue Design**: Avoid anti-patterns like very long queues
- **Connection Management**: Reuse connections and channels efficiently
- **Monitoring**: Track queue depths and processing rates

## Monitoring and Observability

### WebSocket Metrics
- Active connection count
- Messages sent/received per second
- Connection duration statistics
- Error rates by message type
- Client subscription patterns

### RabbitMQ Metrics
- Queue depths and processing rates
- Message publish/consume rates
- Consumer utilization
- Error and retry rates
- Connection and channel counts
- Dead letter queue monitoring

## Development and Testing

### Local Development
- Use provided Docker Compose configuration for RabbitMQ
- WebSocket endpoint available at `ws://localhost:8000/v1/ws`
- RabbitMQ management UI at `http://localhost:15672`

### Testing Tools
- WebSocket testing with browser developer tools or Postman
- RabbitMQ message testing with management UI or CLI tools
- Unit tests cover message schema validation
- Integration tests verify end-to-end message flows

### Configuration
- Environment variables for connection strings and credentials
- Configurable queue names, exchange names, and routing keys
- Development vs production configuration profiles
- Health check endpoints for monitoring service status

---

*This documentation covers WebSocket and RabbitMQ implementations as of the current codebase. For implementation details, refer to the respective service source code.*
