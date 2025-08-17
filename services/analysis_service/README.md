# Analysis Service

The Analysis Service is responsible for extracting metadata from audio files and storing it in both PostgreSQL and Neo4j databases.

## Features

- **Audio Format Support**: MP3, FLAC, WAV, M4A
- **Metadata Extraction**: Title, artist, album, duration, bitrate, sample rate, format
- **Dual Storage**: PostgreSQL for relational data, Neo4j for graph relationships
- **Message Queue Integration**: RabbitMQ for asynchronous processing
- **Error Handling**: Retry logic with exponential backoff
- **Health Checks**: Service health monitoring

## Architecture

The service follows a microservices architecture:

```
RabbitMQ Message → Consumer → Metadata Extractor → Storage Handler → Databases
```

## Project Structure

```
analysis_service/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── main.py                # Service entry point
│   ├── message_consumer.py   # RabbitMQ message consumer
│   ├── metadata_extractor.py # Audio metadata extraction
│   ├── storage_handler.py    # Database storage logic
│   └── exceptions.py         # Custom exceptions
├── Dockerfile                 # Container definition
├── pyproject.toml            # Python dependencies
└── README.md                 # This file
```

## Configuration

The service is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `NEO4J_URI` | Neo4j Bolt connection URI | Required |
| `NEO4J_USER` | Neo4j username | Required |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `RABBITMQ_URL` | RabbitMQ connection URL | Required |
| `ANALYSIS_QUEUE` | Queue name for messages | `analysis_queue` |
| `EXCHANGE_NAME` | RabbitMQ exchange name | `tracktion_exchange` |
| `ANALYSIS_ROUTING_KEY` | Routing key for messages | `file.analyze` |
| `MAX_RETRIES` | Maximum retry attempts | `3` |
| `RETRY_DELAY` | Base retry delay in seconds | `5.0` |

## Message Format

The service expects messages in the following JSON format:

```json
{
  "recording_id": "uuid-string",
  "file_path": "/path/to/audio/file.mp3",
  "timestamp": 1234567890.123
}
```

## Running Locally

1. Install dependencies:
```bash
cd services/analysis_service
uv pip install -e .
```

2. Set environment variables:
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/tracktion"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
```

3. Run the service:
```bash
uv run python -m src.main
```

## Running with Docker

The service is included in the main docker-compose configuration:

```bash
docker-compose up analysis-service
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit/services/analysis/
```

Run integration tests:
```bash
uv run pytest tests/integration/test_analysis_service.py
```

## Extracted Metadata

The service extracts the following metadata fields:

### Basic Metadata
- `title` - Track title
- `artist` - Track artist
- `album` - Album name
- `date` - Release date
- `genre` - Music genre
- `track` - Track number

### Technical Metadata
- `duration` - Track duration in seconds
- `bitrate` - Audio bitrate in bps
- `sample_rate` - Sample rate in Hz
- `channels` - Number of audio channels
- `format` - File format

### Format-Specific Metadata
- MP3: `version`, `layer`
- FLAC: `bits_per_sample`, `albumartist`
- MP4/M4A: `codec`, `codec_description`

## Database Schema

### PostgreSQL (Metadata table)
- `id` - UUID primary key
- `recording_id` - Foreign key to Recording
- `key` - Metadata field name
- `value` - Metadata value

### Neo4j Graph Structure
- Nodes: `Recording`, `Metadata`, `Artist`, `Album`, `Genre`
- Relationships: `HAS_METADATA`, `PERFORMED_BY`, `PART_OF`, `HAS_GENRE`

## Error Handling

The service implements comprehensive error handling:

1. **Invalid Audio Files**: Marked as "invalid" status
2. **Extraction Errors**: Retried with exponential backoff
3. **Storage Errors**: Retried and logged
4. **Connection Errors**: Automatic reconnection attempts

## Troubleshooting

### Service won't start
- Check all environment variables are set
- Verify database connections
- Ensure RabbitMQ is running

### Files not processing
- Check RabbitMQ queue for messages
- Verify file paths are accessible
- Check service logs for errors

### Metadata not extracted
- Ensure file format is supported
- Check file isn't corrupted
- Verify mutagen library is installed

## Future Enhancements

- Support for additional audio formats (OGG, AAC, OPUS)
- Advanced audio analysis (BPM, key detection, mood)
- Batch processing optimization
- Caching layer for repeated files
- Web API for direct queries