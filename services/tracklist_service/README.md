# Tracklist Service

Service for retrieving and parsing tracklist data from 1001tracklists.com.

## Overview

The Tracklist Service provides functionality to:
- Search for DJs, events, and tracklists
- Retrieve complete tracklist data including tracks, timestamps, and transitions
- Cache results for improved performance
- Process requests asynchronously via message queue
- Handle errors gracefully with retry and circuit breaker patterns

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│                 │     │              │     │             │
│   REST API      │────▶│   Scraper    │────▶│  1001tracks │
│   (FastAPI)     │     │   (BeautifulSoup)  │  lists.com  │
│                 │     │              │     │             │
└─────────────────┘     └──────────────┘     └─────────────┘
        │                       │
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌──────────────┐
│                 │     │              │
│   Redis Cache   │     │  Message     │
│                 │     │  Queue       │
│                 │     │  (RabbitMQ)  │
└─────────────────┘     └──────────────┘
```

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The service uses environment variables for configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Scraping Configuration
SCRAPING_USER_AGENT="Mozilla/5.0..."
SCRAPING_REQUEST_TIMEOUT=30
SCRAPING_MAX_RETRIES=3
SCRAPING_DELAY_MIN=1.0
SCRAPING_DELAY_MAX=3.0

# Cache Configuration
CACHE_ENABLED=true
CACHE_REDIS_HOST=localhost
CACHE_REDIS_PORT=6379
CACHE_TTL_HOURS=168  # 7 days

# Message Queue Configuration
MQ_ENABLED=true
MQ_RABBITMQ_URL=amqp://guest:guest@localhost:5672/
MQ_EXCHANGE_NAME=tracklist_exchange
MQ_PREFETCH_COUNT=10
```

## API Endpoints

### Search Endpoints

#### Search for DJs, Events, or Tracklists
```http
POST /api/v1/search
Content-Type: application/json

{
  "search_type": "dj",
  "query": "Carl Cox",
  "limit": 20
}
```

### Tracklist Endpoints

#### Retrieve Tracklist by URL
```http
POST /api/v1/tracklist
Content-Type: application/json

{
  "url": "https://www.1001tracklists.com/tracklist/...",
  "force_refresh": false,
  "include_transitions": true
}
```

#### Get Async Job Status
```http
GET /api/v1/tracklist/status/{correlation_id}
```

#### Clear Cache
```http
DELETE /api/v1/tracklist/cache?url=https://...
```

#### Health Check
```http
GET /api/v1/health
```

## Data Models

### Tracklist
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
  "metadata": {...}
}
```

### Track
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
  "key": "Am"
}
```

### Transition
```python
{
  "from_track": 1,
  "to_track": 2,
  "transition_type": "blend",
  "timestamp_ms": 180000,
  "duration_ms": 8000
}
```

## Usage Examples

### Python Client

```python
import requests

# Search for a DJ
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "search_type": "dj",
        "query": "Amelie Lens",
        "limit": 10
    }
)
results = response.json()

# Retrieve a tracklist
response = requests.post(
    "http://localhost:8000/api/v1/tracklist",
    json={
        "url": results["results"][0]["url"],
        "include_transitions": True
    }
)
tracklist = response.json()

# Print tracks
for track in tracklist["tracklist"]["tracks"]:
    print(f"{track['timestamp']['formatted_time']} - {track['artist']} - {track['title']}")
```

### Async Processing

```python
# Submit for async processing
response = requests.post(
    "http://localhost:8000/api/v1/tracklist?async_processing=true",
    json={"url": "https://www.1001tracklists.com/..."}
)
job_data = response.json()
correlation_id = job_data["correlation_id"]

# Check job status
import time
while True:
    response = requests.get(
        f"http://localhost:8000/api/v1/tracklist/status/{correlation_id}"
    )
    status = response.json()

    if status["status"] == "completed":
        tracklist = status["tracklist"]
        break
    elif status["status"] == "failed":
        print(f"Job failed: {status['error']}")
        break

    time.sleep(2)
```

## Error Handling

The service implements several resilience patterns:

### Retry Logic
- Automatic retry with exponential backoff for transient failures
- Configurable max attempts and delays

### Circuit Breaker
- Prevents cascading failures by stopping requests to failing services
- Automatic recovery attempts after timeout period

### Custom Exceptions
- `TracklistNotFoundError`: When a tracklist page doesn't exist
- `ParseError`: When HTML parsing fails
- `RateLimitError`: When rate limits are exceeded
- `ScrapingError`: General scraping failures

## Testing

```bash
# Run unit tests
uv run pytest tests/unit/tracklist_service/

# Run integration tests
uv run pytest tests/integration/tracklist_service/

# Run with coverage
uv run pytest tests/ --cov=services.tracklist_service --cov-report=html

# Run specific test file
uv run pytest tests/unit/tracklist_service/test_tracklist_models.py -v
```

## Development

### Project Structure
```
services/tracklist_service/
├── src/
│   ├── api/               # REST API endpoints
│   ├── cache/             # Redis caching
│   ├── messaging/         # RabbitMQ handlers
│   ├── models/            # Pydantic models
│   ├── resilience/        # Error handling & retry logic
│   └── scraper/           # Web scraping logic
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
└── README.md
```

### Adding New Features

1. **New Scrapers**: Extend `ScraperBase` class
2. **New Models**: Add to `models/` directory
3. **New Endpoints**: Add to `api/` directory
4. **New Message Handlers**: Add to `messaging/` directory

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy services/tracklist_service/
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "service": "tracklist_api",
  "status": "healthy",
  "timestamp": "2024-08-21T10:00:00Z",
  "components": {
    "cache": "healthy",
    "message_queue": "healthy",
    "scraper": "healthy"
  }
}
```

### Metrics
- Cache hit rate
- Average response time
- Scraping success rate
- Queue processing time

## Troubleshooting

### Common Issues

#### 1. Connection Refused
**Problem**: Cannot connect to Redis or RabbitMQ
**Solution**: Ensure services are running:
```bash
docker-compose up -d redis rabbitmq
```

#### 2. Rate Limiting
**Problem**: Getting rate limited by 1001tracklists.com
**Solution**: Increase delay between requests in configuration:
```bash
SCRAPING_DELAY_MIN=3.0
SCRAPING_DELAY_MAX=5.0
```

#### 3. Parsing Errors
**Problem**: HTML structure has changed
**Solution**: Check for site updates and update selectors in `scraper/tracklist_scraper.py`

#### 4. Cache Not Working
**Problem**: Cache misses even for recent requests
**Solution**: Check Redis connection and TTL settings

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Other Services

### Cataloging Service
The tracklist service publishes events when tracklists are retrieved:
```json
{
  "event_type": "tracklist.completed",
  "tracklist_id": "uuid",
  "url": "https://...",
  "track_count": 42
}
```

### Analysis Service
Retrieved tracklists can be analyzed for:
- BPM progression
- Key compatibility
- Energy flow
- Genre classification

### CUE Generation
Tracklist data can be used to generate CUE files for DJ sets.

## Contributing

1. Create a feature branch
2. Make changes and add tests
3. Ensure all tests pass
4. Run pre-commit hooks
5. Submit pull request

## License

Proprietary - See LICENSE file for details
