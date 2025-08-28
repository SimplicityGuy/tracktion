# Cataloging Service

The Cataloging Service is responsible for managing the database catalog of music files in the Tracktion system. It handles file lifecycle events (created, modified, deleted, moved, renamed) and maintains the recordings database.

## Features

- **File Lifecycle Management**: Handles creation, modification, deletion, movement, and renaming of files
- **Soft Delete Support**: Files are soft-deleted by default, allowing for recovery
- **Automatic Cleanup**: Periodic cleanup of old soft-deleted records
- **Database Integrity**: Maintains referential integrity across the system
- **Event-Driven Architecture**: Consumes file events from RabbitMQ

## Architecture

The service follows an event-driven architecture:

1. File watcher service publishes file events to RabbitMQ
2. Cataloging service consumes these events
3. Database is updated based on event type
4. Soft-deleted records are periodically cleaned up

## Configuration

The service is configured through environment variables:

### Database Configuration
- `DB_HOST`: PostgreSQL host (default: localhost)
- `DB_PORT`: PostgreSQL port (default: 5432)
- `DB_NAME`: Database name (default: tracktion)
- `DB_USER`: Database user (default: tracktion)
- `DB_PASSWORD`: Database password (default: tracktion)

### RabbitMQ Configuration
- `RABBITMQ_HOST`: RabbitMQ host (default: localhost)
- `RABBITMQ_PORT`: RabbitMQ port (default: 5672)
- `RABBITMQ_USERNAME`: RabbitMQ username (default: guest)
- `RABBITMQ_PASSWORD`: RabbitMQ password (default: guest)
- `RABBITMQ_EXCHANGE`: Exchange name (default: file_events)
- `RABBITMQ_QUEUE`: Queue name (default: cataloging.file.events)

### Service Configuration
- `SOFT_DELETE_ENABLED`: Enable soft delete (default: true)
- `CLEANUP_INTERVAL_DAYS`: Days to keep soft-deleted records (default: 30)
- `LOG_LEVEL`: Logging level (default: INFO)

## Running the Service

### Local Development

```bash
# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# Run the service
python -m cataloging_service.main
```

### Docker

```bash
# Build the image
docker build -t tracktion-cataloging-service .

# Run the container
docker run --rm \
  -e DB_HOST=postgres \
  -e RABBITMQ_HOST=rabbitmq \
  --network tracktion_network \
  tracktion-cataloging-service
```

### Docker Compose

The service is included in the main docker-compose.yaml:

```bash
docker-compose up cataloging-service
```

## Event Types

The service handles the following event types:

### Created
- Creates a new recording in the database
- Stores file path, hashes, and metadata

### Modified
- Updates existing recording with new hashes
- Marks for reprocessing

### Deleted
- Soft-deletes the recording (by default)
- Sets deleted_at timestamp
- Can be configured for hard delete

### Moved
- Updates file path to new location
- Preserves all metadata and relationships

### Renamed
- Updates file name and path
- Preserves all metadata and relationships

## Database Schema

The service manages the `recordings` table with the following key fields:
- `id`: UUID primary key
- `file_path`: Full path to the file
- `file_name`: Name of the file
- `sha256_hash`: SHA256 hash of file content
- `xxh128_hash`: XXH128 hash for fast comparison
- `file_size`: Size in bytes
- `processing_status`: Current processing status
- `deleted_at`: Soft delete timestamp
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

## Testing

```bash
# Run unit tests
uv run pytest tests/unit/

# Run integration tests
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=cataloging_service tests/
```

## Monitoring

The service logs all operations and can be monitored through:
- Application logs (stdout)
- RabbitMQ management interface
- Database query monitoring
- Health check endpoint (when running in Docker)

## Error Handling

- Failed messages are sent to dead letter queue
- Database errors trigger rollback
- Connection failures trigger automatic reconnection
- Soft-deleted records can be recovered
