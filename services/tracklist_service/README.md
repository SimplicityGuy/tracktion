# Tracklist Service

## Overview

The Tracklist Service is a comprehensive microservice within the Tracktion ecosystem that provides functionality to search, retrieve, and process tracklist data from 1001tracklists.com. It serves as the primary interface for DJ set data acquisition, featuring advanced web scraping capabilities, intelligent caching, message queue processing, and CUE file generation.

## Features

- **Advanced Web Scraping**: Intelligent scraping of 1001tracklists.com with respect for rate limits and robots.txt
- **Search Functionality**: Comprehensive search for DJs, events, and individual tracklists
- **Caching System**: Redis-based caching with configurable TTL for improved performance
- **Message Queue Processing**: Asynchronous processing via RabbitMQ for scalable operations
- **CUE File Generation**: Generate industry-standard CUE files from tracklist data
- **Audio Validation**: Validate audio files against tracklist metadata
- **Draft Management**: Support for draft tracklists and version control
- **Batch Operations**: Bulk processing and synchronization capabilities
- **Rate Limiting**: Intelligent rate limiting to prevent service abuse
- **Authentication**: JWT-based authentication for secure API access
- **Monitoring & Logging**: Structured logging and health monitoring

## Architecture

The Tracklist Service follows a modular microservices architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tracklist Service                        │
├─────────────────┬───────────────────┬───────────────────────────┤
│   REST API      │   Message Queue   │      Background Tasks     │
│   (FastAPI)     │   (RabbitMQ)      │      (AsyncIO)           │
│                 │                   │                          │
│ • Search        │ • Async Jobs      │ • Periodic Sync          │
│ • Tracklist     │ • Batch Ops       │ • Cache Cleanup          │
│ • CUE Gen       │ • File Events     │ • Health Checks          │
│ • Admin         │                   │                          │
└─────────────────┴───────────────────┴───────────────────────────┘
        │                   │                     │
        ▼                   ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│                 │ │                 │ │                 │
│   Redis Cache   │ │   PostgreSQL    │ │  File Storage   │
│                 │ │   Database      │ │    (Local)      │
│ • Search Cache  │ │ • Tracklists    │ │ • CUE Files     │
│ • Rate Limits   │ │ • Recordings    │ │ • Audio Files   │
│ • Session Data  │ │ • Operations    │ │ • Drafts        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                     │
        └───────────────────┼─────────────────────┘
                            │
                            ▼
                ┌─────────────────┐
                │                 │
                │ 1001tracklists  │
                │    .com API     │
                │                 │
                └─────────────────┘
```

### Core Components

- **API Layer**: FastAPI-based REST endpoints with OpenAPI documentation
- **Scraping Engine**: BeautifulSoup-based web scraping with retry logic
- **Cache Layer**: Redis-based caching for performance optimization
- **Message Queue**: RabbitMQ integration for asynchronous processing
- **Database Layer**: SQLAlchemy ORM with PostgreSQL backend
- **File Management**: Local filesystem storage for CUE files
- **Authentication**: JWT-based security with role-based access
- **Monitoring**: Structured logging with health check endpoints

## Installation

### Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Redis 6+
- RabbitMQ 3.8+
- UV package manager

### Setup

```bash
# Clone the repository (if not already done)
cd services/tracklist_service

# Install dependencies using UV
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Set up the database
uv run alembic upgrade head

# Run the service
uv run python -m src.main
```

## Configuration

The service uses environment variables for configuration. All variables are prefixed with `TRACKLIST_`:

### API Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_API_HOST` | API bind address | `0.0.0.0` |
| `TRACKLIST_API_PORT` | API port | `8000` |
| `TRACKLIST_API_WORKERS` | Number of API workers | `1` |
| `TRACKLIST_API_RELOAD` | Enable auto-reload for development | `false` |
| `TRACKLIST_API_LOG_LEVEL` | API log level | `info` |
| `TRACKLIST_API_PREFIX` | API path prefix | `/api/v1` |
| `TRACKLIST_API_DOCS_ENABLED` | Enable API documentation | `true` |

### Scraping Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_SCRAPING_BASE_URL` | Base URL for scraping | `https://1001tracklists.com` |
| `TRACKLIST_SCRAPING_REQUEST_TIMEOUT` | HTTP request timeout (seconds) | `30` |
| `TRACKLIST_SCRAPING_MAX_RETRIES` | Maximum retry attempts | `3` |
| `TRACKLIST_SCRAPING_RETRY_DELAY_BASE` | Base retry delay (seconds) | `1.0` |
| `TRACKLIST_SCRAPING_RATE_LIMIT_DELAY` | Delay between requests (seconds) | `2.0` |
| `TRACKLIST_SCRAPING_SESSION_TIMEOUT` | Session timeout (seconds) | `3600` |
| `TRACKLIST_SCRAPING_RESPECT_ROBOTS_TXT` | Respect robots.txt | `true` |

### Cache Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_CACHE_ENABLED` | Enable Redis caching | `true` |
| `TRACKLIST_CACHE_REDIS_HOST` | Redis host | `localhost` |
| `TRACKLIST_CACHE_REDIS_PORT` | Redis port | `6379` |
| `TRACKLIST_CACHE_REDIS_DB` | Redis database number | `1` |
| `TRACKLIST_CACHE_REDIS_PASSWORD` | Redis password | `null` |
| `TRACKLIST_CACHE_SEARCH_TTL_HOURS` | Search cache TTL (hours) | `24` |
| `TRACKLIST_CACHE_FAILED_SEARCH_TTL_MINUTES` | Failed search cache TTL (minutes) | `30` |
| `TRACKLIST_CACHE_KEY_PREFIX` | Cache key prefix | `tracklist:` |

### Message Queue Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_MQ_RABBITMQ_URL` | RabbitMQ connection URL | `amqp://guest:guest@localhost:5672/` |
| `TRACKLIST_MQ_EXCHANGE_NAME` | Exchange name | `tracktion_exchange` |
| `TRACKLIST_MQ_SEARCH_QUEUE` | Search queue name | `tracklist_search_queue` |
| `TRACKLIST_MQ_SEARCH_ROUTING_KEY` | Search routing key | `tracklist.search` |
| `TRACKLIST_MQ_RESULT_ROUTING_KEY` | Result routing key | `tracklist.result` |
| `TRACKLIST_MQ_MAX_RETRIES` | Maximum message retries | `3` |
| `TRACKLIST_MQ_BASE_DELAY_SECONDS` | Base delay for retries | `2.0` |
| `TRACKLIST_MQ_PREFETCH_COUNT` | Message prefetch count | `1` |

### Database Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_DATABASE_HOST` | PostgreSQL host | `localhost` |
| `TRACKLIST_DATABASE_PORT` | PostgreSQL port | `5432` |
| `TRACKLIST_DATABASE_NAME` | Database name | `tracktion_tracklist` |
| `TRACKLIST_DATABASE_USER` | Database user | `tracklist_user` |
| `TRACKLIST_DATABASE_PASSWORD` | Database password | `tracklist_password` |

### Service Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `TRACKLIST_LOG_LEVEL` | Service log level | `INFO` |
| `TRACKLIST_METRICS_ENABLED` | Enable metrics collection | `true` |
| `TRACKLIST_HEALTH_CHECK_ENABLED` | Enable health checks | `true` |
| `TRACKLIST_DEBUG_MODE` | Enable debug mode | `false` |

## Usage

### Starting the Service

```bash
# Development mode with auto-reload
uv run python -m src.main

# Production mode with multiple workers
TRACKLIST_API_WORKERS=4 uv run python -m src.main
```

### Python Client Examples

#### Search for DJs
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://localhost:8000/api/v1/search/",
        params={
            "query": "Carl Cox",
            "search_type": "dj",
            "limit": 10
        }
    )
    results = response.json()

    for result in results["results"]:
        print(f"{result['dj_name']} - {result['event_name']}")
```

#### Retrieve a Tracklist
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/tracklist",
        json={
            "url": "https://www.1001tracklists.com/tracklist/...",
            "include_transitions": True
        }
    )
    tracklist_data = response.json()

    for track in tracklist_data["tracklist"]["tracks"]:
        timestamp = track["timestamp"]["formatted_time"] if track["timestamp"] else "N/A"
        print(f"{timestamp} - {track['artist']} - {track['title']}")
```

#### Generate CUE File
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/cue/generate",
        json={
            "tracklist_url": "https://www.1001tracklists.com/tracklist/...",
            "format": "standard",
            "include_pregap": True
        }
    )
    cue_data = response.json()

    with open("set.cue", "w") as f:
        f.write(cue_data["cue_content"])
```

## API Reference

### Search Endpoints

#### `GET /api/v1/search/`
Search for DJs, events, or tracklists.

**Parameters:**
- `query` (string, required): Search query
- `search_type` (string): Type of search (`dj`, `event`, `track`)
- `page` (integer): Page number for pagination
- `limit` (integer): Results per page (max 100)
- `start_date` (string): Start date filter (YYYY-MM-DD)
- `end_date` (string): End date filter (YYYY-MM-DD)

#### `GET /api/v1/search/dj/{dj_slug}`
Get tracklists for a specific DJ.

#### `GET /api/v1/search/event/{event_slug}`
Get tracklists for a specific event.

### Tracklist Endpoints

#### `POST /api/v1/tracklist`
Retrieve a tracklist by URL.

**Request Body:**
```json
{
    "url": "string",
    "force_refresh": false,
    "include_transitions": true,
    "correlation_id": "uuid"
}
```

#### `GET /api/v1/tracklist/status/{correlation_id}`
Check the status of an async tracklist job.

#### `DELETE /api/v1/tracklist/cache`
Clear cached tracklist data.

### CUE Generation Endpoints

#### `POST /api/v1/cue/generate`
Generate a CUE file from tracklist data.

**Request Body:**
```json
{
    "tracklist_url": "string",
    "audio_file_path": "string",
    "format": "standard",
    "include_pregap": true
}
```

#### `GET /api/v1/cue/{cue_id}`
Retrieve a generated CUE file.

### Admin Endpoints

#### `GET /api/v1/admin/stats`
Get service statistics and metrics.

#### `POST /api/v1/admin/cache/clear`
Clear all cache data.

#### `GET /api/v1/admin/health`
Detailed health check with component status.

### Health Endpoint

#### `GET /health`
Basic health check endpoint.

## Data Models

### Tracklist Model
```python
{
    "id": "uuid",
    "url": "https://www.1001tracklists.com/...",
    "dj_name": "Carl Cox",
    "event_name": "Tomorrowland 2024",
    "venue": "Boom, Belgium",
    "date": "2024-07-20",
    "tracks": [...],
    "transitions": [...],
    "metadata": {...},
    "scraped_at": "2024-08-21T10:00:00Z",
    "source_html_hash": "abc123"
}
```

### Track Model
```python
{
    "number": 1,
    "timestamp": {
        "track_number": 1,
        "timestamp_ms": 0,
        "formatted_time": "00:00"
    },
    "artist": "Carl Cox",
    "title": "The Revolution Continues",
    "remix": "Original Mix",
    "label": "Intec",
    "is_id": false,
    "bpm": 128.0,
    "key": "Am",
    "genre": "Techno",
    "notes": "Opening track"
}
```

### Transition Model
```python
{
    "from_track": 1,
    "to_track": 2,
    "transition_type": "blend",
    "timestamp_ms": 180000,
    "duration_ms": 8000,
    "notes": "Smooth transition"
}
```

### CUE File Model
```python
{
    "id": "uuid",
    "tracklist_id": "uuid",
    "filename": "set.cue",
    "cue_content": "...",
    "format": "standard",
    "created_at": "2024-08-21T10:00:00Z",
    "file_size": 1024,
    "checksum": "sha256:..."
}
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests only
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
uv run pytest tests/unit/api/test_search.py -v

# Run tests with specific markers
uv run pytest -m "not slow"
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and database operations
- **API Tests**: Test REST endpoints and response formats
- **Scraping Tests**: Test web scraping functionality (mocked)

## Deployment

### Docker Deployment

```bash
# Build the image
docker build -t tracklist-service .

# Run with docker-compose
docker-compose up -d tracklist-service
```

### Environment Setup

1. **Database Setup**:
   ```bash
   # Create database and user
   createdb tracktion_tracklist
   uv run alembic upgrade head
   ```

2. **Redis Setup**:
   ```bash
   # Start Redis server
   redis-server
   ```

3. **RabbitMQ Setup**:
   ```bash
   # Start RabbitMQ server
   rabbitmq-server

   # Create exchange and queues
   uv run python -m src.messaging.setup_queues
   ```

### Production Considerations

- Use a reverse proxy (nginx) for SSL termination
- Set up log aggregation (ELK stack or similar)
- Configure monitoring (Prometheus/Grafana)
- Set up backup procedures for database and files
- Use environment-specific configuration files
- Implement graceful shutdown handling

## Monitoring

### Health Checks

The service provides multiple levels of health checks:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with component status
curl http://localhost:8000/api/v1/admin/health
```

### Metrics

Key metrics tracked by the service:

- **Request Metrics**: Response times, request counts, error rates
- **Cache Metrics**: Hit rates, miss rates, eviction counts
- **Scraping Metrics**: Success rates, failure counts, rate limit hits
- **Queue Metrics**: Message counts, processing times, retry counts
- **Database Metrics**: Connection pool status, query performance

### Logging

The service uses structured logging (JSON format) with the following log levels:

- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning conditions that don't require immediate action
- `ERROR`: Error conditions that need attention
- `CRITICAL`: Critical errors that may cause service failure

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms**: Service exits immediately or fails to bind to port

**Solutions**:
```bash
# Check if port is already in use
lsof -i :8000

# Check configuration
uv run python -c "from src.config import get_config; print(get_config().validate())"

# Check dependencies
uv run python -c "import redis, aio_pika, sqlalchemy; print('Dependencies OK')"
```

#### 2. Scraping Failures

**Symptoms**: 403 errors, timeouts, or parsing failures

**Solutions**:
```bash
# Check rate limiting settings
export TRACKLIST_SCRAPING_RATE_LIMIT_DELAY=5.0

# Update user agents
# Edit user_agents list in ScrapingConfig

# Check site structure changes
uv run python -c "from src.scraper.search_scraper import SearchScraper; SearchScraper().test_connection()"
```

#### 3. Cache Issues

**Symptoms**: High memory usage, cache misses, connection errors

**Solutions**:
```bash
# Check Redis connection
redis-cli ping

# Clear cache
curl -X POST http://localhost:8000/api/v1/admin/cache/clear

# Adjust TTL settings
export TRACKLIST_CACHE_SEARCH_TTL_HOURS=12
```

#### 4. Database Connection Issues

**Symptoms**: Connection timeouts, pool exhaustion

**Solutions**:
```bash
# Check database connectivity
uv run python -c "from src.database import get_db_session; next(get_db_session())"

# Check connection pool settings
export TRACKLIST_DATABASE_POOL_SIZE=20

# Run database migrations
uv run alembic upgrade head
```

#### 5. Message Queue Problems

**Symptoms**: Messages not processing, queue buildup

**Solutions**:
```bash
# Check RabbitMQ status
rabbitmqctl status

# Check queue lengths
rabbitmqctl list_queues name messages

# Restart message consumer
# Kill and restart the service
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
export TRACKLIST_DEBUG_MODE=true
export TRACKLIST_LOG_LEVEL=DEBUG
uv run python -m src.main
```

### Performance Tuning

#### Scraping Performance
- Adjust `TRACKLIST_SCRAPING_RATE_LIMIT_DELAY` based on rate limits
- Increase `TRACKLIST_SCRAPING_MAX_RETRIES` for unstable connections
- Use connection pooling for better performance

#### Cache Optimization
- Monitor cache hit rates and adjust TTL values
- Use cache warming for frequently accessed data
- Configure Redis memory limits appropriately

#### Database Performance
- Monitor slow queries and add indexes as needed
- Adjust connection pool size based on load
- Use read replicas for heavy read workloads

## Integration with Other Services

### Analysis Service
The tracklist service publishes events when tracklists are processed:

```json
{
    "event_type": "tracklist.processed",
    "tracklist_id": "uuid",
    "url": "https://...",
    "track_count": 42,
    "processing_time_ms": 1500
}
```

### File Watcher Service
Monitors CUE file changes and triggers reprocessing:

```json
{
    "event_type": "file.changed",
    "file_path": "/path/to/set.cue",
    "file_type": "cue",
    "change_type": "modified"
}
```

### Cataloging Service
Receives tracklist metadata for cataloging:

```json
{
    "event_type": "tracklist.cataloged",
    "tracklist_id": "uuid",
    "metadata": {...},
    "tags": ["techno", "live-set"]
}
```

## Security Considerations

### Authentication
- JWT tokens with configurable expiration
- Role-based access control for admin endpoints
- API key authentication for service-to-service communication

### Rate Limiting
- Per-IP rate limiting for public endpoints
- Authenticated user rate limiting with higher limits
- Scraping rate limiting to respect target site policies

### Data Protection
- Sensitive configuration via environment variables
- Encrypted storage of authentication tokens
- Audit logging for administrative actions

## Contributing

### Development Setup

1. **Fork and Clone**: Fork the repository and clone locally
2. **Environment Setup**: Follow installation instructions
3. **Pre-commit Hooks**: Install and configure pre-commit hooks
4. **Code Quality**: Ensure all tests pass and code follows style guidelines

### Code Style

The project uses:
- **Ruff**: For linting and code formatting
- **MyPy**: For static type checking
- **Pytest**: For testing

### Pull Request Process

1. Create a feature branch from `main`
2. Make changes and add tests
3. Ensure all tests pass: `uv run pytest`
4. Run code quality checks: `pre-commit run --all-files`
5. Update documentation if needed
6. Submit pull request with clear description

### Testing Guidelines

- Write unit tests for all public functions
- Add integration tests for database operations
- Mock external dependencies (web scraping, external APIs)
- Maintain test coverage above 80%

## License

Proprietary - See LICENSE file for details

---

For more information, see the [main project documentation](../../README.md) or contact the development team.
