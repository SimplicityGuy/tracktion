# Sample Configurations

This guide provides comprehensive configuration examples for different Tracktion deployment scenarios. Each configuration includes explanations, best practices, and environment-specific settings.

## Table of Contents

- [Development Environment](#development-environment)
- [Production Environment](#production-environment)
- [Docker Configurations](#docker-configurations)
- [Performance Tuning](#performance-tuning)
- [Security Configurations](#security-configurations)
- [Service-Specific Configurations](#service-specific-configurations)

---

## Development Environment

### Basic Development Setup

```bash
# .env.development
# Basic development configuration for local testing

# Database Configuration
DATABASE_URL=postgresql://localhost:5432/tracktion_dev
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_ECHO=false

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=10

# RabbitMQ Configuration
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_QUEUE_PREFIX=tracktion_dev

# Analysis Service Configuration
ANALYSIS_BPM_CONFIDENCE_THRESHOLD=0.7
ANALYSIS_KEY_CONFIDENCE_THRESHOLD=0.6
ANALYSIS_ENABLE_MOOD_ANALYSIS=true
ANALYSIS_ENABLE_GENRE_CLASSIFICATION=true
ANALYSIS_CONCURRENT_WORKERS=2

# File Processing Configuration
FILE_MAX_SIZE_MB=100
FILE_SUPPORTED_FORMATS=mp3,flac,wav,aiff,m4a
FILE_TEMP_DIRECTORY=/tmp/tracktion
FILE_AUTO_CLEANUP=true
FILE_BACKUP_ENABLED=false

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/tracktion_dev.log
LOG_ROTATION_SIZE=10MB
LOG_RETENTION_DAYS=7

# Development Features
DEV_AUTO_RELOAD=true
DEV_DEBUG_TOOLBAR=true
DEV_MOCK_EXTERNAL_APIS=true
DEV_SAMPLE_DATA=true

# Performance Settings (Relaxed for Development)
WORKER_TIMEOUT=300
REQUEST_TIMEOUT=60
BATCH_PROCESSING_SIZE=5
ENABLE_CACHING=false

# Security (Relaxed for Development)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
CORS_ALLOW_CREDENTIALS=true

# External APIs (Use test endpoints)
SPOTIFY_CLIENT_ID=your_dev_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_dev_spotify_client_secret
MUSICBRAINZ_USER_AGENT=TracktionDev/1.0
```

### Advanced Development Configuration

```bash
# .env.development.advanced
# Advanced development setup with additional features

# Extend basic development configuration
include=.env.development

# Advanced Analysis Features
ANALYSIS_ALGORITHM_BPM=multiband
ANALYSIS_ALGORITHM_KEY=hpcp_advanced
ANALYSIS_ENABLE_SPECTRAL_FEATURES=true
ANALYSIS_ENABLE_RHYTHM_PATTERNS=true
ANALYSIS_ENABLE_HARMONIC_ANALYSIS=true

# Machine Learning Configuration
ML_MODEL_PATH=models/dev
ML_ENABLE_TRAINING_MODE=true
ML_BATCH_SIZE=16
ML_LEARNING_RATE=0.001

# Advanced File Processing
FILE_ENABLE_PREVIEW_GENERATION=true
FILE_PREVIEW_DURATION=30
FILE_ENABLE_WAVEFORM_GENERATION=true
FILE_WAVEFORM_RESOLUTION=1024

# Development Database Seeding
DB_SEED_SAMPLE_TRACKS=100
DB_SEED_SAMPLE_PLAYLISTS=10
DB_SEED_SAMPLE_USERS=5

# Monitoring and Profiling
ENABLE_PROFILING=true
PROFILING_OUTPUT_DIR=profiling/
ENABLE_METRICS_COLLECTION=true
METRICS_EXPORT_INTERVAL=30

# Testing Configuration
TEST_DATABASE_URL=postgresql://localhost:5432/tracktion_test
TEST_ENABLE_COVERAGE=true
TEST_COVERAGE_THRESHOLD=80
TEST_PARALLEL_WORKERS=4

# Development API Keys (Use test/development keys only)
LASTFM_API_KEY=dev_lastfm_key
DISCOGS_API_KEY=dev_discogs_key
ECHONEST_API_KEY=dev_echonest_key
```

---

## Production Environment

### Standard Production Configuration

```bash
# .env.production
# Production configuration with security and performance optimizations

# Database Configuration (Production)
DATABASE_URL=postgresql://tracktion_user:${DB_PASSWORD}@db-cluster.internal:5432/tracktion_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_PRE_PING=true
DATABASE_POOL_RECYCLE=3600
DATABASE_ECHO=false
DATABASE_SSL_MODE=require

# Redis Configuration (Production)
REDIS_URL=redis://:${REDIS_PASSWORD}@redis-cluster.internal:6379/0
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_KEEPALIVE=true
REDIS_SOCKET_KEEPALIVE_OPTIONS=1,3,5
REDIS_SSL=true

# RabbitMQ Configuration (Production)
RABBITMQ_URL=amqps://tracktion_user:${RABBITMQ_PASSWORD}@rabbitmq-cluster.internal:5671/tracktion_prod
RABBITMQ_QUEUE_PREFIX=tracktion_prod
RABBITMQ_EXCHANGE_TYPE=topic
RABBITMQ_CONNECTION_POOL_SIZE=10
RABBITMQ_CONFIRM_DELIVERY=true

# Analysis Service Configuration (Production)
ANALYSIS_BPM_CONFIDENCE_THRESHOLD=0.8
ANALYSIS_KEY_CONFIDENCE_THRESHOLD=0.7
ANALYSIS_ENABLE_MOOD_ANALYSIS=true
ANALYSIS_ENABLE_GENRE_CLASSIFICATION=true
ANALYSIS_CONCURRENT_WORKERS=8
ANALYSIS_BATCH_SIZE=50
ANALYSIS_TIMEOUT=300

# File Processing Configuration (Production)
FILE_MAX_SIZE_MB=200
FILE_SUPPORTED_FORMATS=mp3,flac,wav,aiff,m4a,ogg,wma
FILE_TEMP_DIRECTORY=/var/lib/tracktion/temp
FILE_AUTO_CLEANUP=true
FILE_BACKUP_ENABLED=true
FILE_BACKUP_LOCATION=s3://tracktion-backups/files
FILE_COMPRESSION_ENABLED=true

# Logging Configuration (Production)
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s
LOG_FILE=/var/log/tracktion/tracktion.log
LOG_ROTATION_SIZE=100MB
LOG_RETENTION_DAYS=30
LOG_STRUCTURED=true
LOG_EXPORT_TO_ELASTICSEARCH=true

# Performance Settings (Production)
WORKER_TIMEOUT=120
REQUEST_TIMEOUT=30
BATCH_PROCESSING_SIZE=20
ENABLE_CACHING=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Security Configuration (Production)
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ACCESS_TOKEN_EXPIRES=3600
JWT_REFRESH_TOKEN_EXPIRES=86400
CORS_ORIGINS=https://app.tracktion.com,https://admin.tracktion.com
CORS_ALLOW_CREDENTIALS=false
RATE_LIMIT_REQUESTS_PER_MINUTE=100
ENABLE_HTTPS_ONLY=true

# Monitoring and Health Checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
TRACING_SAMPLE_RATE=0.1

# External Service Configuration
EXTERNAL_API_TIMEOUT=10
EXTERNAL_API_RETRY_ATTEMPTS=3
EXTERNAL_API_RETRY_BACKOFF=exponential

# Production External APIs
SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID}
SPOTIFY_CLIENT_SECRET=${SPOTIFY_CLIENT_SECRET}
MUSICBRAINZ_USER_AGENT=Tracktion/1.0 (https://tracktion.com)
LASTFM_API_KEY=${LASTFM_API_KEY}

# Deployment Configuration
DEPLOYMENT_ENVIRONMENT=production
DEPLOYMENT_VERSION=${CI_COMMIT_SHA}
DEPLOYMENT_TIMESTAMP=${CI_PIPELINE_CREATED_AT}
```

### High-Availability Production Configuration

```bash
# .env.production.ha
# High-availability production configuration

# Extend standard production configuration
include=.env.production

# Database Configuration (HA)
DATABASE_URL=postgresql://tracktion_user:${DB_PASSWORD}@db-primary.internal:5432/tracktion_prod
DATABASE_REPLICA_URLS=postgresql://tracktion_user:${DB_PASSWORD}@db-replica-1.internal:5432/tracktion_prod,postgresql://tracktion_user:${DB_PASSWORD}@db-replica-2.internal:5432/tracktion_prod
DATABASE_READ_REPLICA_ENABLED=true
DATABASE_AUTOMATIC_FAILOVER=true
DATABASE_CONNECTION_MAX_AGE=300

# Redis Configuration (HA with Sentinel)
REDIS_SENTINEL_SERVICE_NAME=tracktion-redis
REDIS_SENTINEL_HOSTS=redis-sentinel-1:26379,redis-sentinel-2:26379,redis-sentinel-3:26379
REDIS_SENTINEL_SOCKET_TIMEOUT=0.5

# RabbitMQ Configuration (Cluster)
RABBITMQ_CLUSTER_NODES=rabbitmq-1.internal,rabbitmq-2.internal,rabbitmq-3.internal
RABBITMQ_HA_POLICY=all
RABBITMQ_QUEUE_MIRRORING=true

# Load Balancing and Scaling
LOAD_BALANCER_HEALTH_CHECK=/health
AUTO_SCALING_ENABLED=true
AUTO_SCALING_MIN_INSTANCES=2
AUTO_SCALING_MAX_INSTANCES=10
AUTO_SCALING_TARGET_CPU=70
AUTO_SCALING_TARGET_MEMORY=80

# Advanced Performance Settings
WORKER_PROCESSES=auto
WORKER_CONNECTIONS=1000
KEEPALIVE_TIMEOUT=65
CLIENT_MAX_BODY_SIZE=200M
GZIP_COMPRESSION=true
STATIC_FILE_CACHING=true

# Disaster Recovery
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=90
BACKUP_ENCRYPTION=true
BACKUP_DESTINATION=s3://tracktion-dr-backups
DISASTER_RECOVERY_RTO=4h
DISASTER_RECOVERY_RPO=15m

# Advanced Security
WAF_ENABLED=true
DDoS_PROTECTION=true
IP_WHITELIST_ENABLED=false
GEO_BLOCKING_ENABLED=false
SECURITY_HEADERS_ENABLED=true
CONTENT_SECURITY_POLICY=strict

# Compliance and Auditing
AUDIT_LOGGING_ENABLED=true
GDPR_COMPLIANCE_MODE=true
DATA_RETENTION_POLICY=2y
PII_ANONYMIZATION=true
COMPLIANCE_REPORTING=true
```

---

## Docker Configurations

### Development Docker Compose

```yaml
# docker-compose.dev.yml
# Development environment with all services

version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15-alpine
    container_name: tracktion_postgres_dev
    environment:
      POSTGRES_DB: tracktion_dev
      POSTGRES_USER: tracktion_user
      POSTGRES_PASSWORD: dev_password
    volumes:
      - postgres_data_dev:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    networks:
      - tracktion_dev

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: tracktion_redis_dev
    command: redis-server --appendonly yes
    volumes:
      - redis_data_dev:/data
    ports:
      - "6379:6379"
    networks:
      - tracktion_dev

  # RabbitMQ Message Queue
  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: tracktion_rabbitmq_dev
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    volumes:
      - rabbitmq_data_dev:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    networks:
      - tracktion_dev

  # Analysis Service
  analysis_service:
    build:
      context: ./services/analysis_service
      dockerfile: Dockerfile.dev
    container_name: tracktion_analysis_dev
    environment:
      - DATABASE_URL=postgresql://tracktion_user:dev_password@postgres:5432/tracktion_dev
      - REDIS_URL=redis://redis:6379/0
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - LOG_LEVEL=DEBUG
    volumes:
      - ./services/analysis_service:/app
      - ./data/music:/data/music:ro
    depends_on:
      - postgres
      - redis
      - rabbitmq
    ports:
      - "8001:8001"
    networks:
      - tracktion_dev
    restart: unless-stopped

  # Tracklist Service
  tracklist_service:
    build:
      context: ./services/tracklist_service
      dockerfile: Dockerfile.dev
    container_name: tracktion_tracklist_dev
    environment:
      - DATABASE_URL=postgresql://tracktion_user:dev_password@postgres:5432/tracktion_dev
      - REDIS_URL=redis://redis:6379/0
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - LOG_LEVEL=DEBUG
    volumes:
      - ./services/tracklist_service:/app
    depends_on:
      - postgres
      - redis
      - rabbitmq
    ports:
      - "8002:8002"
    networks:
      - tracktion_dev
    restart: unless-stopped

  # File Watcher Service
  file_watcher:
    build:
      context: ./services/file_watcher
      dockerfile: Dockerfile.dev
    container_name: tracktion_file_watcher_dev
    environment:
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
      - WATCH_DIRECTORY=/data/music
      - LOG_LEVEL=DEBUG
    volumes:
      - ./services/file_watcher:/app
      - ./data/music:/data/music
    depends_on:
      - rabbitmq
    networks:
      - tracktion_dev
    restart: unless-stopped

  # Development Database Admin (Optional)
  pgadmin:
    image: dpage/pgadmin4
    container_name: tracktion_pgadmin_dev
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@tracktion.dev
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "8080:80"
    depends_on:
      - postgres
    networks:
      - tracktion_dev
    profiles:
      - admin

volumes:
  postgres_data_dev:
  redis_data_dev:
  rabbitmq_data_dev:

networks:
  tracktion_dev:
    driver: bridge
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml
# Production environment with optimizations

version: '3.8'

services:
  # Database (Production)
  postgres:
    image: postgres:15
    container_name: tracktion_postgres_prod
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data_prod:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    networks:
      - tracktion_backend
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Redis (Production with Persistence)
  redis:
    image: redis:7
    container_name: tracktion_redis_prod
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - redis_data_prod:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    networks:
      - tracktion_backend
    restart: unless-stopped

  # RabbitMQ (Production)
  rabbitmq:
    image: rabbitmq:3-management
    container_name: tracktion_rabbitmq_prod
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
      RABBITMQ_VM_MEMORY_HIGH_WATERMARK: 0.8
    volumes:
      - rabbitmq_data_prod:/var/lib/rabbitmq
      - ./config/rabbitmq.conf:/etc/rabbitmq/rabbitmq.conf
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    networks:
      - tracktion_backend
    restart: unless-stopped

  # Analysis Service (Production)
  analysis_service:
    image: tracktion/analysis_service:${VERSION:-latest}
    container_name: tracktion_analysis_prod
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis:6379/0
      - RABBITMQ_URL=amqp://${RABBITMQ_USER}:${RABBITMQ_PASSWORD}@rabbitmq:5672/
      - LOG_LEVEL=INFO
      - WORKER_PROCESSES=4
    volumes:
      - analysis_temp:/tmp/tracktion
      - ${MUSIC_STORAGE_PATH}:/data/music:ro
    depends_on:
      - postgres
      - redis
      - rabbitmq
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    networks:
      - tracktion_backend
      - tracktion_frontend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Load Balancer
  nginx:
    image: nginx:alpine
    container_name: tracktion_nginx_prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - analysis_service
      - tracklist_service
    networks:
      - tracktion_frontend
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: tracktion_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - tracktion_monitoring
    restart: unless-stopped
    profiles:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: tracktion_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - tracktion_monitoring
    restart: unless-stopped
    profiles:
      - monitoring

volumes:
  postgres_data_prod:
  redis_data_prod:
  rabbitmq_data_prod:
  analysis_temp:
  prometheus_data:
  grafana_data:

networks:
  tracktion_backend:
    driver: bridge
    internal: true
  tracktion_frontend:
    driver: bridge
  tracktion_monitoring:
    driver: bridge

secrets:
  postgres_password:
    external: true
  rabbitmq_password:
    external: true
```

---

## Performance Tuning

### High-Performance Analysis Configuration

```bash
# .env.performance
# Configuration optimized for high-throughput analysis

# Analysis Performance Settings
ANALYSIS_CONCURRENT_WORKERS=16
ANALYSIS_BATCH_SIZE=100
ANALYSIS_QUEUE_SIZE=1000
ANALYSIS_PREFETCH_COUNT=10
ANALYSIS_TIMEOUT=60

# Memory Management
ANALYSIS_MAX_MEMORY_USAGE=8GB
ANALYSIS_MEMORY_CLEANUP_INTERVAL=300
ANALYSIS_SWAP_USAGE_LIMIT=20

# CPU Optimization
ANALYSIS_CPU_AFFINITY=auto
ANALYSIS_THREAD_POOL_SIZE=32
ANALYSIS_ENABLE_MULTIPROCESSING=true
ANALYSIS_PROCESS_POOL_SIZE=8

# I/O Optimization
ANALYSIS_IO_BUFFER_SIZE=65536
ANALYSIS_TEMP_FILE_LOCATION=/dev/shm/tracktion
ANALYSIS_ASYNC_IO=true
ANALYSIS_FILE_CACHE_SIZE=1GB

# Algorithm Optimization
BPM_ALGORITHM=onset_strength_optimized
KEY_ALGORITHM=chroma_cqt_optimized
MOOD_ALGORITHM=spectral_features_fast
ENABLE_ALGORITHM_CACHING=true

# Database Performance
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_POOL_PRE_PING=false
DATABASE_ECHO=false
DATABASE_ISOLATION_LEVEL=READ_COMMITTED

# Connection Pooling
DATABASE_CONNECTION_TIMEOUT=30
DATABASE_CONNECTION_MAX_AGE=3600
DATABASE_PREPARED_STATEMENTS=true

# Query Optimization
DATABASE_QUERY_CACHE_SIZE=1000
DATABASE_ENABLE_QUERY_LOGGING=false
DATABASE_SLOW_QUERY_THRESHOLD=1.0

# Redis Performance
REDIS_CONNECTION_POOL_SIZE=100
REDIS_MAX_CONNECTIONS=200
REDIS_SOCKET_KEEPALIVE=true
REDIS_SOCKET_TIMEOUT=5.0
REDIS_RESPONSE_TIMEOUT=5.0

# Cache Configuration
CACHE_DEFAULT_TTL=3600
CACHE_MAX_SIZE=10000
CACHE_EVICTION_POLICY=lru
CACHE_COMPRESSION=true

# Message Queue Performance
RABBITMQ_CONNECTION_POOL_SIZE=20
RABBITMQ_CHANNEL_POOL_SIZE=100
RABBITMQ_PREFETCH_COUNT=50
RABBITMQ_CONFIRM_DELIVERY=false
RABBITMQ_PERSISTENT_MESSAGES=false

# Worker Configuration
WORKER_CONCURRENCY=16
WORKER_PREFETCH_MULTIPLIER=4
WORKER_MAX_TASKS_PER_CHILD=1000
WORKER_TIME_LIMIT=300

# System Resource Limits
MAX_OPEN_FILES=65536
MAX_PROCESSES=32768
MEMORY_OVERCOMMIT=2
SWAPPINESS=10
```

### PostgreSQL Performance Configuration

```ini
# postgresql.conf
# PostgreSQL optimizations for Tracktion

# Connection Settings
max_connections = 200
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = 2GB                # 25% of RAM
effective_cache_size = 6GB          # 75% of RAM
work_mem = 16MB
maintenance_work_mem = 512MB
dynamic_shared_memory_type = posix

# Checkpoint Settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_timeout = 15min
max_wal_size = 4GB
min_wal_size = 1GB

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200
default_statistics_target = 100

# Logging
log_destination = 'csvlog'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_truncate_on_rotation = on
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Vacuum and Autovacuum
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 15s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02
autovacuum_vacuum_cost_delay = 2ms
autovacuum_vacuum_cost_limit = 400

# Background Writer
bgwriter_delay = 100ms
bgwriter_lru_maxpages = 100
bgwriter_lru_multiplier = 2.0
```

---

## Security Configurations

### Production Security Configuration

```bash
# .env.security
# Comprehensive security configuration

# Authentication & Authorization
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRES=900  # 15 minutes
JWT_REFRESH_TOKEN_EXPIRES=604800  # 7 days
JWT_ISSUER=tracktion.com
JWT_AUDIENCE=tracktion-api

# Password Security
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SYMBOLS=true
PASSWORD_HASH_ALGORITHM=bcrypt
PASSWORD_HASH_ROUNDS=12

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20
RATE_LIMIT_STRATEGY=sliding_window

# API Security
API_KEY_HEADER=X-API-Key
API_RATE_LIMIT_PER_KEY=1000
API_REQUIRE_HTTPS=true
API_VALIDATE_CONTENT_TYPE=true

# CORS Configuration
CORS_ALLOWED_ORIGINS=https://app.tracktion.com,https://admin.tracktion.com
CORS_ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOWED_HEADERS=Content-Type,Authorization,X-Requested-With
CORS_EXPOSE_HEADERS=X-Total-Count,X-Rate-Limit-Remaining
CORS_ALLOW_CREDENTIALS=false
CORS_MAX_AGE=3600

# Security Headers
SECURITY_HEADERS_ENABLED=true
HSTS_MAX_AGE=31536000
HSTS_INCLUDE_SUBDOMAINS=true
CONTENT_SECURITY_POLICY=default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:
X_FRAME_OPTIONS=DENY
X_CONTENT_TYPE_OPTIONS=nosniff
REFERRER_POLICY=strict-origin-when-cross-origin

# SSL/TLS Configuration
SSL_CERT_PATH=/etc/ssl/certs/tracktion.crt
SSL_KEY_PATH=/etc/ssl/private/tracktion.key
SSL_PROTOCOLS=TLSv1.2,TLSv1.3
SSL_CIPHERS=ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS
SSL_VERIFY_CLIENT=optional
SSL_SESSION_TIMEOUT=1d

# Database Security
DATABASE_SSL_MODE=require
DATABASE_SSL_CERT=/etc/ssl/certs/client.crt
DATABASE_SSL_KEY=/etc/ssl/private/client.key
DATABASE_SSL_ROOT_CERT=/etc/ssl/certs/ca.crt

# Encryption
FIELD_ENCRYPTION_KEY=${FIELD_ENCRYPTION_KEY}
FIELD_ENCRYPTION_ALGORITHM=AES-256-GCM
FILE_ENCRYPTION_ENABLED=false
BACKUP_ENCRYPTION_ENABLED=true

# Audit Logging
AUDIT_LOG_ENABLED=true
AUDIT_LOG_LEVEL=INFO
AUDIT_LOG_FILE=/var/log/tracktion/audit.log
AUDIT_LOG_ROTATE_SIZE=100MB
AUDIT_LOG_RETENTION_DAYS=365
AUDIT_LOG_INCLUDE_REQUEST_BODY=false
AUDIT_LOG_INCLUDE_RESPONSE_BODY=false

# Security Monitoring
SECURITY_MONITORING_ENABLED=true
FAILED_LOGIN_THRESHOLD=5
FAILED_LOGIN_WINDOW=300
FAILED_LOGIN_LOCKOUT=900
SUSPICIOUS_ACTIVITY_DETECTION=true
GEO_IP_VALIDATION=false

# Data Privacy
GDPR_MODE=true
DATA_RETENTION_DAYS=2555  # 7 years
PII_ANONYMIZATION=true
DATA_EXPORT_ENABLED=true
DATA_DELETION_ENABLED=true
COOKIE_SECURE=true
COOKIE_HTTP_ONLY=true
COOKIE_SAME_SITE=Strict
```

---

## Service-Specific Configurations

### Analysis Service Configuration

```yaml
# config/analysis_service.yml
# Comprehensive Analysis Service configuration

analysis:
  # BPM Detection
  bpm:
    algorithm: "multiband"  # multiband, onset_strength, beat_tracking
    confidence_threshold: 0.8
    sample_rate: 44100
    hop_length: 512
    frame_length: 2048

    # Advanced BPM settings
    tempo_range: [60, 200]
    enable_subdivision_detection: true
    smoothing_window: 5

    # Performance settings
    batch_size: 32
    enable_caching: true
    cache_ttl: 3600

  # Key Detection
  key:
    algorithm: "hpcp"  # hpcp, chroma_cqt, chroma_stft
    confidence_threshold: 0.7
    chromagram_resolution: 12
    enable_mode_detection: true

    # HPCP-specific settings
    hpcp_size: 12
    hpcp_reference_frequency: 440.0
    hpcp_harmonics: 4
    hpcp_band_preset: true

    # Performance settings
    batch_size: 16
    enable_caching: true
    cache_ttl: 3600

  # Mood Analysis
  mood:
    model_path: "models/mood_classifier_v2.pkl"
    enable_training_mode: false
    confidence_threshold: 0.6

    # Feature extraction
    enable_spectral_features: true
    enable_rhythm_features: true
    enable_harmonic_features: true
    enable_timbral_features: true

    # Categories
    mood_categories:
      - "happy"
      - "sad"
      - "energetic"
      - "calm"
      - "aggressive"
      - "romantic"
      - "melancholic"
      - "uplifting"

    # Performance settings
    batch_size: 8
    enable_gpu: false
    enable_caching: true

  # Genre Classification
  genre:
    enabled: true
    model_path: "models/genre_classifier_v1.pkl"
    confidence_threshold: 0.5

    genre_categories:
      - "rock"
      - "pop"
      - "electronic"
      - "hip-hop"
      - "jazz"
      - "classical"
      - "country"
      - "reggae"
      - "blues"
      - "folk"

# Processing Configuration
processing:
  # Worker settings
  workers: 4
  worker_timeout: 300
  max_retries: 3
  retry_delay: 5

  # Queue settings
  queue_name: "analysis_queue"
  prefetch_count: 1
  queue_durable: true
  queue_auto_delete: false

  # File processing
  temp_directory: "/tmp/tracktion_analysis"
  cleanup_temp_files: true
  max_file_size_mb: 200
  supported_formats:
    - "mp3"
    - "flac"
    - "wav"
    - "aiff"
    - "m4a"
    - "ogg"

# Monitoring
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_port: 8001
  health_check_path: "/health"

  # Performance metrics
  track_processing_time: true
  track_queue_size: true
  track_error_rate: true
  track_memory_usage: true
```

### Tracklist Service Configuration

```yaml
# config/tracklist_service.yml
# Tracklist Service configuration

tracklist:
  # Playlist Generation
  playlist_generation:
    default_max_tracks: 50
    default_duration_minutes: 60

    # Matching algorithms
    similarity_algorithms:
      - "euclidean"
      - "cosine"
      - "manhattan"

    default_similarity_algorithm: "cosine"
    similarity_threshold: 0.7

    # Diversification
    enable_diversification: true
    diversification_factors:
      - "artist"
      - "genre"
      - "year"
      - "energy"
      - "key"

    # Energy progression
    energy_progression_types:
      - "linear_increase"
      - "linear_decrease"
      - "build_and_sustain"
      - "wave"
      - "random"

    default_energy_progression: "build_and_sustain"

  # Track Matching
  matching:
    # Feature weights for similarity calculation
    feature_weights:
      bpm: 0.3
      key: 0.25
      energy: 0.2
      danceability: 0.15
      valence: 0.1

    # BPM matching
    bpm_tolerance: 10  # ±10 BPM
    key_compatibility_mode: "camelot"  # camelot, circle_of_fifths

    # Mood matching
    mood_similarity_threshold: 0.6
    enable_mood_progression: true

  # Database queries
  database:
    # Query optimization
    enable_query_caching: true
    query_cache_ttl: 300
    max_query_results: 1000

    # Connection settings
    pool_size: 10
    max_overflow: 20

    # Indexes for performance
    ensure_indexes: true
    analyze_tables: true

# API Configuration
api:
  # Rate limiting
  rate_limit_per_minute: 100
  rate_limit_burst: 20

  # Pagination
  default_page_size: 25
  max_page_size: 100

  # Response format
  include_metadata: true
  include_analysis_confidence: true

  # CORS
  cors_origins:
    - "http://localhost:3000"
    - "https://app.tracktion.com"

# Caching
caching:
  # Redis configuration
  redis_ttl: 3600
  cache_prefixes:
    playlists: "playlist:"
    tracks: "track:"
    similarity: "sim:"

  # Cache strategies
  enable_playlist_caching: true
  enable_similarity_caching: true
  cache_invalidation_strategy: "ttl"  # ttl, manual, hybrid
```

### File Watcher Configuration

```yaml
# config/file_watcher.yml
# File Watcher Service configuration

file_watcher:
  # Watch settings
  watch_paths:
    - path: "/data/music"
      recursive: true
      enabled: true
    - path: "/data/imports"
      recursive: false
      enabled: true

  # File patterns
  include_patterns:
    - "*.mp3"
    - "*.flac"
    - "*.wav"
    - "*.aiff"
    - "*.m4a"
    - "*.ogg"

  exclude_patterns:
    - ".*"  # Hidden files
    - "*.tmp"
    - "*.part"
    - "Thumbs.db"
    - ".DS_Store"

  # Processing behavior
  auto_analyze: true
  auto_organize: false
  duplicate_detection: true

  # Timing settings
  debounce_seconds: 5
  batch_processing: true
  batch_size: 10
  batch_timeout: 30

  # File monitoring
  poll_interval: 1  # seconds
  use_inotify: true  # Linux
  use_kqueue: false  # BSD/macOS

  # Error handling
  max_retries: 3
  retry_delay: 10
  skip_corrupted_files: true
  quarantine_path: "/data/quarantine"

# Integration
integration:
  # Message queue
  queue_name: "file_events"
  exchange_name: "tracktion_files"
  routing_key: "file.detected"

  # Event types
  event_types:
    - "file_added"
    - "file_modified"
    - "file_deleted"
    - "file_moved"

  # Metadata extraction
  extract_metadata: true
  metadata_cache_ttl: 86400

# Performance
performance:
  # Worker threads
  worker_threads: 2
  io_threads: 4

  # Memory management
  max_memory_mb: 512
  enable_memory_monitoring: true

  # File system
  fs_buffer_size: 65536
  max_open_files: 1000
```

These comprehensive configuration examples provide a solid foundation for deploying Tracktion in various environments. Each configuration can be customized based on specific requirements, infrastructure, and performance needs.

---

**Next Steps:**
- Choose the appropriate configuration for your environment
- Customize values based on your infrastructure
- Test configurations in a development environment first
- Monitor performance and adjust settings as needed

For more specific configuration options, refer to the individual service documentation and deployment guides.
