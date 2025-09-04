# Tracktion Configuration Guide

## Overview

This guide provides comprehensive documentation for all configuration options across the Tracktion microservices. Each service supports environment variable configuration with sensible defaults and validation.

## Configuration Hierarchy

Configuration sources are applied in the following priority order:
1. Environment variables (highest priority)
2. Configuration files (if supported)
3. Default values (lowest priority)

## Common Configuration Patterns

### Database Configuration
All services that use databases follow this pattern:
```bash
# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tracktion
DB_USER=tracktion_user
DB_PASSWORD=secure_password
DATABASE_URL=postgresql://user:pass@host:port/db  # Optional: overrides individual settings
```

### Message Queue Configuration
Services using RabbitMQ:
```bash
# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=rabbitmq_user
RABBITMQ_PASSWORD=secure_password
RABBITMQ_VHOST=/
RABBITMQ_URL=amqp://user:pass@host:port/vhost  # Optional: overrides individual settings
```

### Redis Configuration
Services using Redis for caching:
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password  # Optional
REDIS_DB=0
REDIS_URL=redis://[:password]@host:port/db  # Optional: overrides individual settings
```

## Service-Specific Configuration

### Analysis Service

The Analysis Service performs audio analysis including BPM detection, key detection, and metadata extraction.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=analysis_service
SERVICE_VERSION=1.0.0
INSTANCE_ID=analysis_01  # For multi-instance deployments

# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false  # Set to true for development only
```

#### Audio Processing Configuration
```bash
# Audio Processing Settings
MAX_CONCURRENT_ANALYSES=4        # Number of concurrent audio analyses
AUDIO_CHUNK_SIZE=8192           # Buffer size for audio processing (bytes)
AUDIO_SAMPLE_RATE=44100         # Target sample rate for analysis
SUPPORTED_FORMATS=mp3,flac,wav,ogg,m4a,aac  # Comma-separated list

# Analysis Timeouts
AUDIO_LOAD_TIMEOUT=30           # Seconds to load audio file
BPM_DETECTION_TIMEOUT=60        # Seconds for BPM analysis
KEY_DETECTION_TIMEOUT=45        # Seconds for key detection
METADATA_EXTRACTION_TIMEOUT=15   # Seconds for metadata extraction
```

#### Algorithm Configuration
```bash
# BPM Detection Settings
BPM_CONFIDENCE_THRESHOLD=0.7     # Minimum confidence for reliable BPM detection
BPM_AGREEMENT_TOLERANCE=5.0      # BPM difference tolerance for algorithm agreement
BPM_FALLBACK_ENABLED=true       # Enable Percival fallback algorithm

# Key Detection Settings
KEY_CONFIDENCE_THRESHOLD=0.7     # Minimum confidence for reliable key detection
KEY_AGREEMENT_BOOST=1.2          # Confidence multiplier when algorithms agree
KEY_DISAGREEMENT_PENALTY=0.8     # Confidence multiplier when algorithms disagree
KEY_NEEDS_REVIEW_THRESHOLD=0.7   # Threshold below which manual review is suggested
```

#### Performance Settings
```bash
# Threading and Concurrency
THREAD_POOL_SIZE=4              # Number of worker threads for CPU-bound tasks
ASYNC_POOL_SIZE=10              # Size of async task pool
MAX_MEMORY_PER_ANALYSIS=512     # Maximum memory per analysis (MB)

# Caching Configuration
ENABLE_RESULT_CACHING=true      # Cache analysis results
CACHE_TTL_SECONDS=86400        # Cache time-to-live (24 hours)
MAX_CACHE_SIZE_MB=1024         # Maximum cache size in memory
```

### Tracklist Service

The Tracklist Service handles web scraping, search, and matching with 1001tracklists.com.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=tracklist_service
SERVICE_VERSION=1.0.0
INSTANCE_ID=tracklist_01

# API Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=false
```

#### Web Scraping Configuration
```bash
# Rate Limiting
SCRAPING_RATE_LIMIT=2.0         # Requests per second (respects site robots.txt)
REQUEST_DELAY_MIN=0.5           # Minimum delay between requests (seconds)
REQUEST_DELAY_MAX=2.0           # Maximum delay between requests (seconds)
MAX_RETRIES=3                   # Maximum retry attempts for failed requests

# User Agent Rotation
ROTATE_USER_AGENTS=true         # Enable user agent rotation for anti-detection
USER_AGENT_LIST=Mozilla/5.0...  # Comma-separated list of user agents

# Timeouts
HTTP_CONNECT_TIMEOUT=10         # Connection timeout (seconds)
HTTP_READ_TIMEOUT=30           # Read timeout (seconds)
HTTP_TOTAL_TIMEOUT=60          # Total request timeout (seconds)
```

#### Search and Caching Configuration
```bash
# Search Configuration
DEFAULT_SEARCH_LIMIT=10         # Default number of search results
MAX_SEARCH_LIMIT=100           # Maximum allowed search results
SEARCH_TIMEOUT=30              # Search operation timeout (seconds)

# Redis Caching
CACHE_SEARCH_RESULTS=true      # Cache search results
SEARCH_CACHE_TTL=3600          # Search result cache TTL (1 hour)
FAILED_SEARCH_CACHE_TTL=300    # Failed search cache TTL (5 minutes)
MAX_CACHE_ENTRIES=10000        # Maximum cached entries
```

#### Matching Algorithm Configuration
```bash
# Confidence Weights (must sum to 1.0)
MATCH_WEIGHT_TITLE=0.3         # Title matching importance
MATCH_WEIGHT_ARTIST=0.25       # Artist matching importance
MATCH_WEIGHT_DURATION=0.25     # Duration matching importance
MATCH_WEIGHT_DATE=0.1          # Date matching importance
MATCH_WEIGHT_EVENT=0.1         # Event matching importance

# Fuzzy Matching Settings
FUZZY_MATCH_THRESHOLD=0.7      # Minimum similarity for fuzzy matching
SUBSTRING_MATCH_SCORE=0.7      # Score for substring matches
```

### File Watcher Service

The File Watcher Service monitors file system changes and triggers processing.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=file_watcher
SERVICE_VERSION=1.0.0
INSTANCE_ID=file_watcher_01

# File System Monitoring
DATA_DIR=/path/to/music/files    # Directory to monitor (overrides FILE_WATCHER_SCAN_PATH)
FILE_WATCHER_SCAN_PATH=/default/path  # Fallback scan path
SUPPORTED_EXTENSIONS=mp3,flac,wav,ogg,m4a,aac  # Monitored file extensions
```

#### Performance Configuration
```bash
# Concurrency Settings
MAX_CONCURRENT_FILES=100        # Maximum concurrent file processing
BATCH_SIZE=100                 # Files processed per batch
SCAN_BATCH_SIZE=1000           # Files scanned per batch during initial scan

# File Processing
HASH_CHUNK_SIZE=8192           # Chunk size for file hashing (bytes)
ENABLE_DUAL_HASHING=true       # Calculate both SHA256 and XXH128 hashes
METADATA_EXTRACTION_TIMEOUT=10  # Timeout for metadata extraction (seconds)
```

#### Monitoring Configuration
```bash
# Watchdog Settings
OBSERVER_HEALTH_CHECK_INTERVAL=30  # Observer health check interval (seconds)
AUTO_RESTART_OBSERVER=true         # Automatically restart failed observers
MAX_RESTART_ATTEMPTS=5             # Maximum observer restart attempts

# Event Processing
EVENT_PROCESSING_DELAY=1.0      # Delay before processing events (seconds)
DEDUPLICATE_EVENTS=true         # Remove duplicate file system events
EVENT_BATCH_TIMEOUT=5          # Batch processing timeout (seconds)
```

### Cataloging Service

The Cataloging Service manages recording metadata and provides search functionality.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=cataloging_service
SERVICE_VERSION=1.0.0
INSTANCE_ID=cataloging_01

# API Configuration
HOST=0.0.0.0
PORT=8002
DEBUG=false
```

#### Search Configuration
```bash
# Search Settings
DEFAULT_SEARCH_LIMIT=20        # Default number of search results
MAX_SEARCH_LIMIT=1000         # Maximum allowed search results
SEARCH_TIMEOUT=10             # Search operation timeout (seconds)
ENABLE_FUZZY_SEARCH=true      # Enable fuzzy text search
FUZZY_SEARCH_THRESHOLD=0.3    # Minimum similarity for fuzzy search

# Pagination Settings
DEFAULT_PAGE_SIZE=50          # Default pagination size
MAX_PAGE_SIZE=500            # Maximum pagination size
```

#### Recording Management
```bash
# Recording Processing
AUTO_GENERATE_THUMBNAILS=true    # Generate thumbnails for audio files
THUMBNAIL_SIZE=300x300          # Thumbnail dimensions
MAX_RECORDING_SIZE_MB=500       # Maximum recording file size

# Metadata Processing
EXTRACT_WAVEFORM=true           # Extract waveform data
WAVEFORM_RESOLUTION=1000        # Waveform data points
AUTO_TAG_RECORDINGS=true        # Automatically tag recordings
```

### Notification Service

The Notification Service handles event notifications and messaging.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=notification_service
SERVICE_VERSION=1.0.0
INSTANCE_ID=notification_01

# API Configuration
HOST=0.0.0.0
PORT=8003
DEBUG=false
```

#### Notification Configuration
```bash
# Message Processing
MAX_CONCURRENT_NOTIFICATIONS=10   # Concurrent notification processing
NOTIFICATION_RETRY_ATTEMPTS=3     # Retry attempts for failed notifications
NOTIFICATION_TIMEOUT=30           # Notification delivery timeout (seconds)

# Queue Configuration
NOTIFICATION_QUEUE_SIZE=1000      # Maximum queue size
BATCH_NOTIFICATION_SIZE=50        # Notifications processed per batch
QUEUE_PROCESSING_INTERVAL=5       # Queue processing interval (seconds)
```

### File Rename Service

The File Rename Service provides ML-based file renaming suggestions.

#### Core Settings
```bash
# Service Identity
SERVICE_NAME=file_rename_service
SERVICE_VERSION=1.0.0
INSTANCE_ID=file_rename_01

# API Configuration
HOST=0.0.0.0
PORT=8004
DEBUG=false
```

#### ML Configuration
```bash
# Machine Learning Settings
MODEL_PATH=/app/models             # Path to ML models
ENABLE_MODEL_TRAINING=true         # Enable model training endpoints
TRAINING_DATA_RETENTION_DAYS=90    # Keep training data for N days
MIN_TRAINING_SAMPLES=100           # Minimum samples required for training

# Prediction Settings
PREDICTION_CONFIDENCE_THRESHOLD=0.7  # Minimum confidence for suggestions
MAX_SUGGESTIONS_PER_REQUEST=10       # Maximum rename suggestions
SUGGESTION_TIMEOUT=15                # Suggestion generation timeout (seconds)
```

## Environment-Specific Configuration

### Development Environment
```bash
# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Relaxed Timeouts for Development
HTTP_TIMEOUT=300
DB_QUERY_TIMEOUT=60
ANALYSIS_TIMEOUT=300

# Development Database (use local instances)
DB_HOST=localhost
REDIS_HOST=localhost
RABBITMQ_HOST=localhost
```

### Production Environment
```bash
# Production Settings
DEBUG=false
LOG_LEVEL=INFO
ENABLE_CORS=false

# Optimized Performance Settings
MAX_CONCURRENT_ANALYSES=8        # Scale based on CPU cores
THREAD_POOL_SIZE=8              # Scale based on CPU cores
MAX_MEMORY_PER_ANALYSIS=256     # Conservative memory usage

# Production Security
DB_SSL_MODE=require             # Require SSL for database connections
REDIS_TLS=true                  # Use TLS for Redis connections
API_KEY_REQUIRED=true           # Require API keys for endpoints
```

### Staging Environment
```bash
# Staging mirrors production but with relaxed settings
DEBUG=false
LOG_LEVEL=DEBUG                 # More verbose logging for testing
ENABLE_PERFORMANCE_MONITORING=true

# Reduced resource limits for cost optimization
MAX_CONCURRENT_ANALYSES=2
CACHE_TTL_SECONDS=1800         # Shorter cache TTL for testing
```

## Configuration Validation

All services validate configuration on startup and will fail to start with clear error messages if:

- Required environment variables are missing
- Values are outside valid ranges
- Database/external service connections fail
- File paths don't exist or aren't accessible
- Resource limits are incompatible with system resources

## Performance Tuning Guidelines

### CPU-Intensive Services (Analysis Service)
- Set `MAX_CONCURRENT_ANALYSES` to 1-2x CPU cores
- Increase `THREAD_POOL_SIZE` for I/O-heavy workloads
- Monitor memory usage and adjust `MAX_MEMORY_PER_ANALYSIS`

### I/O-Intensive Services (File Watcher)
- Increase `MAX_CONCURRENT_FILES` for high-throughput scenarios
- Adjust `BATCH_SIZE` based on storage performance
- Consider SSD storage for hash calculation performance

### Network-Intensive Services (Tracklist Service)
- Respect rate limits with appropriate `SCRAPING_RATE_LIMIT`
- Tune cache settings based on usage patterns
- Monitor external service response times

## Security Configuration

### Database Security
```bash
DB_SSL_MODE=require            # Always use SSL in production
DB_CERT_PATH=/path/to/cert     # Client certificate path
DB_KEY_PATH=/path/to/key       # Client key path
```

### API Security
```bash
API_KEY_HEADER=X-API-Key       # API key header name
RATE_LIMIT_ENABLED=true        # Enable API rate limiting
RATE_LIMIT_REQUESTS=100        # Requests per minute per IP
RATE_LIMIT_WINDOW=60          # Rate limit window (seconds)
```

### Service Communication
```bash
INTERNAL_API_TOKEN=secret      # Token for inter-service communication
TLS_ENABLED=true              # Enable TLS for service communication
CERT_PATH=/path/to/cert       # TLS certificate path
KEY_PATH=/path/to/key         # TLS key path
```

## Monitoring and Observability

### Logging Configuration
```bash
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json              # json or text
LOG_FILE=/var/log/service.log # Log file path (optional)
STRUCTURED_LOGGING=true      # Enable structured logging
```

### Metrics Configuration
```bash
ENABLE_METRICS=true          # Enable Prometheus metrics
METRICS_PORT=9090           # Metrics endpoint port
METRICS_PATH=/metrics       # Metrics endpoint path
```

### Health Checks
```bash
HEALTH_CHECK_ENABLED=true    # Enable health check endpoint
HEALTH_CHECK_PATH=/health    # Health check endpoint path
HEALTH_CHECK_TIMEOUT=5      # Health check timeout (seconds)
```

## Troubleshooting Configuration Issues

### Common Issues
1. **Service won't start**: Check required environment variables
2. **Database connection failed**: Verify connection settings and network access
3. **High memory usage**: Reduce concurrent processing limits
4. **Slow performance**: Tune cache settings and resource limits
5. **Rate limiting errors**: Adjust external API rate limits

### Debug Mode
Enable debug mode for detailed configuration logging:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Configuration Validation
Most services provide a configuration validation command:
```bash
python -m src.config --validate
```

This will check all configuration values and report any issues before starting the service.
