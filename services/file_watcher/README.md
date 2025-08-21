# File Watcher Service

Audio file discovery and monitoring service for the Tracktion project.

## Features

### File Discovery
- **Recursive Directory Scanning**: Discovers audio files in configured directories
- **File Format Support**: MP3, FLAC, WAV, M4A, OGG Vorbis (.ogg, .oga)
- **Duplicate Detection**: SHA-256 and xxHash128 hashing for duplicate prevention
- **Change Monitoring**: Tracks file modifications based on size and modification time

### Message Publishing
- **RabbitMQ Integration**: Publishes file discovery events to message queue
- **Correlation IDs**: Unique IDs for message tracing throughout the pipeline
- **Format Family Identification**: Special handling for format families (e.g., ogg_vorbis)

## Architecture

```
File System → Scanner → Hash Calculator → Message Publisher → RabbitMQ
```

## Project Structure

```
file_watcher/
├── src/
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # Service entry point
│   ├── file_scanner.py         # File discovery and scanning
│   └── message_publisher.py    # RabbitMQ message publishing
├── Dockerfile                   # Container definition
├── pyproject.toml              # Python dependencies
└── README.md                   # This file
```

## Configuration

The service is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RABBITMQ_HOST` | RabbitMQ server hostname | `localhost` |
| `RABBITMQ_PORT` | RabbitMQ server port | `5672` |
| `RABBITMQ_USER` | RabbitMQ username | `guest` |
| `RABBITMQ_PASSWORD` | RabbitMQ password | `guest` |
| `RABBITMQ_VHOST` | RabbitMQ virtual host | `/` |
| `EXCHANGE_NAME` | RabbitMQ exchange name | `tracktion_exchange` |
| `ROUTING_KEY` | Message routing key | `file.discovered` |
| `WATCH_DIRECTORIES` | Comma-separated directories to watch | Required |
| `SCAN_INTERVAL` | Seconds between directory scans | `60` |
| `ENABLE_HASHING` | Enable file hash calculation | `true` |

## Supported File Formats

The service monitors for the following audio file extensions:

### Lossy Formats
- `.mp3` - MPEG Layer 3
- `.m4a` - MPEG-4 Audio
- `.mp4` - MPEG-4 (audio tracks)
- `.m4b` - MPEG-4 Audiobook
- `.m4p` - MPEG-4 Protected
- `.m4v` - MPEG-4 Video (audio tracks)
- `.m4r` - MPEG-4 Ringtone
- `.ogg` - OGG Vorbis
- `.oga` - OGG Audio

### Lossless Formats
- `.flac` - Free Lossless Audio Codec
- `.wav` - Waveform Audio
- `.wave` - Waveform Audio (alternative extension)

## Message Format

The service publishes messages in the following JSON format:

```json
{
  "file_info": {
    "path": "/path/to/file.ogg",
    "name": "file.ogg",
    "extension": ".ogg",
    "size": 5242880,
    "modified": "2024-01-01T12:00:00Z",
    "hash": "sha256_hash_value",
    "xxh128": "xxhash128_value"
  },
  "format_family": "ogg_vorbis",
  "correlation_id": "uuid-string",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Format Family Identification

Special format families are identified for proper downstream processing:
- OGG files (`.ogg`, `.oga`) → `format_family: "ogg_vorbis"`

## Running Locally

1. Install dependencies:
```bash
cd services/file_watcher
uv pip install -e .
```

2. Set environment variables:
```bash
export WATCH_DIRECTORIES="/path/to/music,/another/path"
export RABBITMQ_HOST="localhost"
```

3. Run the service:
```bash
uv run python -m src.main
```

## Running with Docker

The service is included in the main docker-compose configuration:

```bash
docker-compose up file-watcher
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit/file_watcher/
```

## File Discovery Process

1. **Directory Scanning**: Recursively scans configured directories
2. **Format Filtering**: Filters files by supported extensions
3. **Hash Calculation**: Computes SHA-256 and xxHash128 for duplicate detection
4. **State Tracking**: Maintains internal state of discovered files
5. **Change Detection**: Identifies new or modified files
6. **Message Publishing**: Publishes discovery events to RabbitMQ

## Error Handling

The service implements robust error handling:

1. **File Access Errors**: Logged and skipped
2. **Hash Calculation Errors**: Falls back to file size/modification time
3. **Connection Errors**: Automatic reconnection with exponential backoff
4. **Invalid Files**: Logged with structured error information

## Performance Considerations

- **Parallel Hashing**: Uses multiprocessing for hash calculation
- **Incremental Scanning**: Only processes new/changed files
- **Memory Efficiency**: Streams large files for hashing
- **Connection Pooling**: Reuses RabbitMQ connections

## Troubleshooting

### Service won't start
- Check WATCH_DIRECTORIES environment variable is set
- Verify RabbitMQ connection parameters
- Ensure watched directories exist and are accessible

### Files not being discovered
- Check file extensions match supported formats
- Verify file permissions allow reading
- Check service logs for scanning errors

### High CPU usage
- Adjust SCAN_INTERVAL to reduce scanning frequency
- Consider disabling hashing for large files
- Check for symbolic link loops in directories

## Future Enhancements

- Real-time file system monitoring (inotify/FSEvents)
- Support for additional formats (AAC, OPUS, WebM)
- Configurable file size limits
- Parallel directory scanning
- REST API for manual file submission
