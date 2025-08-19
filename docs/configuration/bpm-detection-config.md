# BPM Detection Configuration Guide

## Overview

This guide covers all configuration options for the BPM detection system, including environment variables, configuration files, and deployment-specific settings.

## Configuration Hierarchy

Configuration values are loaded in the following order (later values override earlier ones):

1. **Default Values** - Hard-coded defaults in configuration classes
2. **Configuration Files** - YAML/JSON configuration files
3. **Environment Variables** - Runtime environment settings
4. **CLI Arguments** - Command-line overrides
5. **API Parameters** - Request-specific overrides

## Environment Variables

All environment variables use the `TRACKTION_` prefix to avoid conflicts.

### BPM Detection Settings

```bash
# Algorithm Configuration
TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.7          # Minimum confidence for primary algorithm (0.0-1.0)
TRACKTION_BPM_FALLBACK_THRESHOLD=0.5            # Threshold for using fallback algorithm (0.0-1.0)
TRACKTION_BPM_AGREEMENT_TOLERANCE=5.0           # BPM difference tolerance for algorithm agreement
TRACKTION_BPM_MAX_FILE_SIZE_MB=500              # Maximum file size for processing (MB)

# Supported Formats (comma-separated)
TRACKTION_BPM_SUPPORTED_FORMATS=".mp3,.wav,.flac,.m4a,.ogg,.wma,.aac"
```

### Temporal Analysis Settings

```bash
# Window Configuration
TRACKTION_TEMPORAL_WINDOW_SIZE_SECONDS=10.0     # Analysis window size in seconds
TRACKTION_TEMPORAL_START_WINDOW_SECONDS=30.0    # Start region analysis duration
TRACKTION_TEMPORAL_END_WINDOW_SECONDS=30.0      # End region analysis duration
TRACKTION_TEMPORAL_MIN_WINDOWS_FOR_ANALYSIS=3   # Minimum windows required

# Stability Analysis
TRACKTION_TEMPORAL_STABILITY_THRESHOLD=0.8      # Threshold for stable tempo classification
TRACKTION_TEMPORAL_ENABLE_STORAGE=true          # Store temporal data in database
```

### Caching Configuration

```bash
# Redis Connection
TRACKTION_CACHE_ENABLED=true                    # Enable/disable caching
TRACKTION_CACHE_REDIS_HOST=localhost            # Redis server hostname
TRACKTION_CACHE_REDIS_PORT=6379                 # Redis server port
TRACKTION_CACHE_REDIS_DB=0                      # Redis database number
TRACKTION_CACHE_REDIS_PASSWORD=                 # Redis password (optional)

# TTL Settings
TRACKTION_CACHE_DEFAULT_TTL_DAYS=30             # Default cache TTL in days
TRACKTION_CACHE_FAILED_TTL_HOURS=1              # TTL for failed results in hours
TRACKTION_CACHE_LOW_CONFIDENCE_TTL_DAYS=7       # TTL for low confidence results

# Hash Configuration
TRACKTION_CACHE_ALGORITHM_VERSION=1.0           # Algorithm version for cache keys
TRACKTION_CACHE_USE_XXH128=true                 # Use XXHash128 for performance
```

### Message Queue Configuration

```bash
# RabbitMQ Connection
TRACKTION_QUEUE_RABBITMQ_URL="amqp://guest:guest@localhost:5672/"
TRACKTION_QUEUE_QUEUE_NAME=analysis_queue       # Queue name for audio analysis
TRACKTION_QUEUE_EXCHANGE_NAME=tracktion_exchange # Exchange name
TRACKTION_QUEUE_ROUTING_KEY=file.analyze        # Routing key for messages

# Processing Configuration
TRACKTION_QUEUE_MAX_RETRIES=5                   # Maximum retry attempts
TRACKTION_QUEUE_BASE_DELAY_SECONDS=2.0          # Base delay for exponential backoff
TRACKTION_QUEUE_PREFETCH_COUNT=1                # Message prefetch count
```

### Storage Configuration

```bash
# Database Connections
TRACKTION_STORAGE_POSTGRES_URL="postgresql://user:pass@localhost:5432/tracktion"
TRACKTION_STORAGE_NEO4J_URI="bolt://localhost:7687"
TRACKTION_STORAGE_NEO4J_USER=neo4j
TRACKTION_STORAGE_NEO4J_PASSWORD=password

# Storage Options
TRACKTION_STORAGE_STORE_TEMPORAL_ARRAY=false    # Store full temporal array data
TRACKTION_STORAGE_BATCH_SIZE=100                # Batch size for bulk operations
```

### Performance Configuration

```bash
# Parallel Processing
TRACKTION_PERFORMANCE_PARALLEL_WORKERS=4        # Number of worker threads
TRACKTION_PERFORMANCE_PROCESSING_TIMEOUT_SECONDS=300  # Processing timeout

# Memory Management
TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB=1000      # Memory limit per worker (MB)
TRACKTION_PERFORMANCE_ENABLE_STREAMING=true     # Enable streaming for large files
TRACKTION_PERFORMANCE_STREAMING_THRESHOLD_MB=50 # File size threshold for streaming
TRACKTION_PERFORMANCE_CHUNK_SIZE_BYTES=8192     # Chunk size for streaming

# Optimization
TRACKTION_PERFORMANCE_ENABLE_GPU=false          # Enable GPU acceleration (if available)
TRACKTION_PERFORMANCE_CPU_THREADS=0             # CPU threads (0 = auto)
```

### Service Configuration

```bash
# General Service Settings
TRACKTION_SERVICE_ENABLE_TEMPORAL_ANALYSIS=true # Enable temporal analysis globally
TRACKTION_SERVICE_LOG_LEVEL=INFO                # Logging level (DEBUG, INFO, WARNING, ERROR)
TRACKTION_SERVICE_METRICS_ENABLED=true          # Enable metrics collection
TRACKTION_SERVICE_HEALTH_CHECK_PORT=8080        # Health check endpoint port

# Environment
TRACKTION_ENV=development                       # Environment (development, staging, production)
```

## Configuration Files

### YAML Configuration

Create a `config.yaml` file for structured configuration:

```yaml
# config.yaml
bpm:
  confidence_threshold: 0.7
  fallback_threshold: 0.5
  agreement_tolerance: 5.0
  supported_formats:
    - ".mp3"
    - ".wav"
    - ".flac"
    - ".m4a"

temporal:
  window_size_seconds: 10.0
  stability_threshold: 0.8
  enable_storage: true

cache:
  enabled: true
  redis:
    host: "localhost"
    port: 6379
    db: 0
  ttl:
    default_days: 30
    failed_hours: 1
    low_confidence_days: 7

performance:
  parallel_workers: 4
  memory_limit_mb: 1000
  enable_streaming: true
  streaming_threshold_mb: 50

logging:
  level: "INFO"
  format: "json"
  output: "stdout"
```

### JSON Configuration

Alternative JSON format for configuration:

```json
{
  "bpm": {
    "confidence_threshold": 0.7,
    "fallback_threshold": 0.5,
    "agreement_tolerance": 5.0,
    "max_file_size_mb": 500
  },
  "temporal": {
    "window_size_seconds": 10.0,
    "stability_threshold": 0.8,
    "enable_storage": true
  },
  "cache": {
    "enabled": true,
    "redis": {
      "host": "localhost",
      "port": 6379
    },
    "ttl": {
      "default_days": 30
    }
  }
}
```

## Environment-Specific Configurations

### Development Environment

```bash
# Development settings
TRACKTION_ENV=development
TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.6          # Lower threshold for testing
TRACKTION_CACHE_DEFAULT_TTL_DAYS=1              # Shorter cache TTL
TRACKTION_PERFORMANCE_PARALLEL_WORKERS=2        # Fewer workers for development
TRACKTION_SERVICE_LOG_LEVEL=DEBUG               # Verbose logging
TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB=500       # Lower memory limit
```

### Staging Environment

```bash
# Staging settings
TRACKTION_ENV=staging
TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.7          # Production-like settings
TRACKTION_CACHE_DEFAULT_TTL_DAYS=7              # Moderate cache TTL
TRACKTION_PERFORMANCE_PARALLEL_WORKERS=4        # Production-like workers
TRACKTION_SERVICE_LOG_LEVEL=INFO                # Standard logging
TRACKTION_SERVICE_METRICS_ENABLED=true          # Enable metrics collection
```

### Production Environment

```bash
# Production settings
TRACKTION_ENV=production
TRACKTION_BPM_CONFIDENCE_THRESHOLD=0.8          # High confidence threshold
TRACKTION_CACHE_DEFAULT_TTL_DAYS=90             # Long cache TTL
TRACKTION_PERFORMANCE_PARALLEL_WORKERS=8        # Maximum workers
TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB=2000      # High memory limit
TRACKTION_SERVICE_LOG_LEVEL=WARNING             # Minimal logging
TRACKTION_PERFORMANCE_ENABLE_STREAMING=true     # Enable all optimizations
```

## Docker Configuration

### Environment File

Create a `.env` file for Docker Compose:

```bash
# .env file for Docker
TRACKTION_CACHE_REDIS_HOST=redis
TRACKTION_STORAGE_POSTGRES_URL=postgresql://postgres:password@db:5432/tracktion
TRACKTION_STORAGE_NEO4J_URI=bolt://neo4j:7687
TRACKTION_QUEUE_RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
TRACKTION_PERFORMANCE_PARALLEL_WORKERS=4
TRACKTION_SERVICE_LOG_LEVEL=INFO
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  analysis-service:
    build: .
    environment:
      - TRACKTION_CACHE_REDIS_HOST=redis
      - TRACKTION_STORAGE_POSTGRES_URL=postgresql://postgres:password@db:5432/tracktion
      - TRACKTION_PERFORMANCE_PARALLEL_WORKERS=4
    depends_on:
      - redis
      - db
      - neo4j
      - rabbitmq
    volumes:
      - ./audio:/app/audio:ro

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: tracktion
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

## Kubernetes Configuration

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bpm-detection-config
data:
  TRACKTION_BPM_CONFIDENCE_THRESHOLD: "0.8"
  TRACKTION_PERFORMANCE_PARALLEL_WORKERS: "8"
  TRACKTION_CACHE_DEFAULT_TTL_DAYS: "90"
  TRACKTION_SERVICE_LOG_LEVEL: "INFO"
  TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB: "2000"
```

### Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: bpm-detection-secrets
type: Opaque
stringData:
  TRACKTION_STORAGE_POSTGRES_URL: "postgresql://user:pass@postgres:5432/tracktion"
  TRACKTION_STORAGE_NEO4J_PASSWORD: "neo4j_password"
  TRACKTION_CACHE_REDIS_PASSWORD: "redis_password"
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bpm-detection-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bpm-detection
  template:
    metadata:
      labels:
        app: bpm-detection
    spec:
      containers:
      - name: analysis-service
        image: tracktion/bpm-detection:latest
        envFrom:
        - configMapRef:
            name: bpm-detection-config
        - secretRef:
            name: bpm-detection-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## Configuration Validation

### Runtime Validation

The service performs configuration validation on startup:

```python
# Configuration validation examples
def validate_config(config: ServiceConfig) -> List[str]:
    errors = []

    # BPM configuration validation
    if not 0.0 <= config.bpm.confidence_threshold <= 1.0:
        errors.append("BPM confidence_threshold must be between 0.0 and 1.0")

    # Performance validation
    if config.performance.parallel_workers < 1:
        errors.append("parallel_workers must be at least 1")

    if config.performance.memory_limit_mb < 100:
        errors.append("memory_limit_mb should be at least 100MB")

    return errors
```

### Configuration Test

Test your configuration with the validation script:

```bash
# Test configuration
python -m services.analysis_service.config --validate

# Example output:
# ✅ Configuration valid
# 📊 BPM confidence threshold: 0.7
# 🚀 Parallel workers: 4
# 💾 Cache enabled: True
# 🔗 Redis connection: OK
```

## Monitoring Configuration

### Metrics Configuration

```bash
# Metrics collection
TRACKTION_METRICS_ENABLED=true
TRACKTION_METRICS_PORT=9090
TRACKTION_METRICS_PATH=/metrics

# Prometheus integration
TRACKTION_METRICS_PROMETHEUS_ENABLED=true
TRACKTION_METRICS_EXPORT_INTERVAL_SECONDS=30
```

### Health Check Configuration

```bash
# Health check settings
TRACKTION_HEALTH_CHECK_PORT=8080
TRACKTION_HEALTH_CHECK_PATH=/health
TRACKTION_HEALTH_CHECK_INTERVAL_SECONDS=30

# Dependency checks
TRACKTION_HEALTH_CHECK_REDIS=true
TRACKTION_HEALTH_CHECK_DATABASE=true
TRACKTION_HEALTH_CHECK_ESSENTIA=true
```

## Troubleshooting Configuration

### Common Issues

#### Redis Connection Issues
```bash
# Check Redis configuration
echo "PING" | redis-cli -h $TRACKTION_CACHE_REDIS_HOST -p $TRACKTION_CACHE_REDIS_PORT

# Test with authentication
redis-cli -h $TRACKTION_CACHE_REDIS_HOST -p $TRACKTION_CACHE_REDIS_PORT -a $TRACKTION_CACHE_REDIS_PASSWORD ping
```

#### Memory Issues
```bash
# Check memory configuration
echo "Current memory limit: $TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB MB"
echo "Parallel workers: $TRACKTION_PERFORMANCE_PARALLEL_WORKERS"
echo "Total estimated memory: $(($TRACKTION_PERFORMANCE_MEMORY_LIMIT_MB * $TRACKTION_PERFORMANCE_PARALLEL_WORKERS)) MB"
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
psql $TRACKTION_STORAGE_POSTGRES_URL -c "SELECT 1;"

# Test Neo4j connection
cypher-shell -a $TRACKTION_STORAGE_NEO4J_URI -u $TRACKTION_STORAGE_NEO4J_USER -p $TRACKTION_STORAGE_NEO4J_PASSWORD "RETURN 1;"
```

### Debug Configuration

Enable debug mode for detailed configuration logging:

```bash
# Debug configuration loading
TRACKTION_DEBUG_CONFIG=true
TRACKTION_SERVICE_LOG_LEVEL=DEBUG

# This will output:
# 🔧 Loading configuration from: config.yaml
# 🌍 Environment overrides: 12 variables
# ⚙️ Final configuration: {...}
```

This configuration guide provides comprehensive coverage of all settings and deployment scenarios for the BPM detection system.
