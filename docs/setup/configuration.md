# Configuration Guide

Complete configuration reference for all Tracktion services and infrastructure components.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Service Configuration](#service-configuration)
4. [Database Configuration](#database-configuration)
5. [Infrastructure Configuration](#infrastructure-configuration)
6. [Security Configuration](#security-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Development vs Production](#development-vs-production)

## Configuration Overview

Tracktion uses a hierarchical configuration system:

1. **Environment Variables**: Primary configuration method
2. **Configuration Files**: Service-specific YAML/JSON configs
3. **Docker Compose**: Infrastructure and service orchestration
4. **Default Values**: Built-in fallbacks for all settings

### Configuration Priority

1. Environment Variables (highest priority)
2. Configuration files (`.env`, `config.yaml`)
3. Command line arguments
4. Default values (lowest priority)

## Environment Variables

### Core Infrastructure Variables

#### PostgreSQL Database

```bash
# Connection
DATABASE_URL=postgresql://tracktion:password@localhost:5432/tracktion
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=tracktion
DATABASE_USER=tracktion
DATABASE_PASSWORD=secure_password

# Connection Pool
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
DATABASE_ECHO=false

# SSL Configuration
DATABASE_SSL_MODE=prefer
DATABASE_SSL_CERT=/path/to/client-cert.pem
DATABASE_SSL_KEY=/path/to/client-key.pem
DATABASE_SSL_ROOT_CERT=/path/to/ca-cert.pem
```

#### Neo4j Graph Database

```bash
# Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password
NEO4J_DATABASE=neo4j

# Connection Pool
NEO4J_MAX_CONNECTION_LIFETIME=300
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_CONNECTION_TIMEOUT=5
NEO4J_MAX_RETRY_TIME=30

# Security
NEO4J_ENCRYPTED=false
NEO4J_TRUST=TRUST_ALL_CERTIFICATES
```

#### Redis Cache

```bash
# Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
REDIS_DB=0
REDIS_USERNAME=default

# Connection Pool
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=5
REDIS_RETRY_ON_TIMEOUT=true
REDIS_HEALTH_CHECK_INTERVAL=30

# SSL
REDIS_SSL=false
REDIS_SSL_CERT_REQS=required
REDIS_SSL_CA_CERTS=/path/to/ca-cert.pem
REDIS_SSL_CERTFILE=/path/to/client-cert.pem
REDIS_SSL_KEYFILE=/path/to/client-key.pem
```

#### RabbitMQ Message Queue

```bash
# Connection
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_VIRTUAL_HOST=/
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest

# SSL Configuration
RABBITMQ_SSL=false
RABBITMQ_SSL_CACERT=/path/to/cacert.pem
RABBITMQ_SSL_CERT=/path/to/cert.pem
RABBITMQ_SSL_KEY=/path/to/key.pem

# Connection Pool
RABBITMQ_HEARTBEAT=600
RABBITMQ_CONNECTION_TIMEOUT=10
RABBITMQ_RETRY_DELAY=1.0
RABBITMQ_MAX_RETRIES=5

# Exchange Configuration
EXCHANGE_NAME=tracktion_exchange
EXCHANGE_TYPE=topic
EXCHANGE_DURABLE=true
EXCHANGE_AUTO_DELETE=false
```

### Service-Specific Configuration

#### Analysis Service

```bash
# Service Configuration
ANALYSIS_SERVICE_HOST=0.0.0.0
ANALYSIS_SERVICE_PORT=8001
ANALYSIS_SERVICE_WORKERS=4
ANALYSIS_SERVICE_DEBUG=false

# Queue Configuration
ANALYSIS_QUEUE=analysis_queue
ANALYSIS_ROUTING_KEY=file.analyze
ANALYSIS_DEAD_LETTER_QUEUE=analysis_dlq
ANALYSIS_QUEUE_DURABLE=true
ANALYSIS_QUEUE_EXCLUSIVE=false
ANALYSIS_QUEUE_AUTO_DELETE=false

# Processing Configuration
MAX_RETRIES=3
RETRY_DELAY=5.0
BATCH_SIZE=10
PROCESSING_TIMEOUT=300

# Audio Analysis Features
ENABLE_AUDIO_ANALYSIS=true
ENABLE_BPM_DETECTION=true
ENABLE_KEY_DETECTION=true
ENABLE_MOOD_ANALYSIS=true
ENABLE_GENRE_CLASSIFICATION=true
ENABLE_TEMPORAL_ANALYSIS=true

# BPM Detection
BPM_CONFIDENCE_THRESHOLD=0.7
BPM_ALGORITHM=multifeature
BPM_TEMPORAL_WINDOW=10.0
BPM_MIN_BPM=60
BPM_MAX_BPM=200

# Key Detection
KEY_CONFIDENCE_THRESHOLD=0.6
KEY_ALGORITHM=edma
KEY_USE_THREE_CHORD_MODEL=true
KEY_SCALE_VALIDATION=true

# Mood Analysis
MOOD_MODEL_PATH=./models/mood_model.pb
MOOD_CONFIDENCE_THRESHOLD=0.5
MOOD_DIMENSIONS=happy,sad,aggressive,relaxed,acoustic,electronic

# Genre Classification
GENRE_MODEL_PATH=./models/discogs_effnet.pb
GENRE_CONFIDENCE_THRESHOLD=0.3
GENRE_TOP_N=3

# Performance
PARALLEL_WORKERS=4
MEMORY_LIMIT_MB=1024
CACHE_REDIS_TTL=3600
ENABLE_GPU_ACCELERATION=false

# Models
MODELS_DIR=./models
AUTO_DOWNLOAD_MODELS=true
MODEL_CACHE_SIZE=5
DISCOGS_EFFNET_MODEL_URL=https://essentia.upf.edu/models/classifiers/discogs_effnet-bs64-1.pb
```

#### Tracklist Service

```bash
# Service Configuration
TRACKLIST_SERVICE_HOST=0.0.0.0
TRACKLIST_SERVICE_PORT=8002
TRACKLIST_SERVICE_WORKERS=2
TRACKLIST_SERVICE_DEBUG=false

# API Configuration
API_RATE_LIMIT_REQUESTS_PER_SECOND=10
API_RATE_LIMIT_REQUESTS_PER_MINUTE=100
API_RATE_LIMIT_REQUESTS_PER_HOUR=1000
API_RATE_LIMIT_BURST_SIZE=20
API_MAX_CONCURRENT_CONNECTIONS=1000
API_MAX_CONNECTIONS_PER_IP=10

# Authentication
API_KEY_REQUIRED=false
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Web Scraping
ENABLE_WEB_SCRAPING=true
SCRAPING_DELAY=1.0
SCRAPING_TIMEOUT=30
MAX_SCRAPING_RETRIES=3
SCRAPING_USER_AGENT=TracktionBot/1.0 (+https://tracktion.com/bot)
RESPECT_ROBOTS_TXT=true

# Supported Sources
ENABLE_1001TRACKLISTS=true
ENABLE_MIXESDB=true
ENABLE_SOUNDCLOUD=false
ENABLE_MIXCLOUD=false

# Cache Settings
ENABLE_REDIS_CACHE=true
TRACKLIST_CACHE_TTL=7200
SEARCH_CACHE_TTL=900
CACHE_SEARCH_RESULTS=true
CACHE_TRACKLIST_DATA=true

# Search Configuration
SEARCH_DEFAULT_LIMIT=20
SEARCH_MAX_LIMIT=100
SEARCH_FUZZY_THRESHOLD=0.8
SEARCH_INDEX_BATCH_SIZE=1000

# CUE Generation
CUE_OUTPUT_DIR=./cue_files
CUE_DEFAULT_FORMAT=standard
CUE_SUPPORTED_FORMATS=standard,cdj,traktor,serato,rekordbox
ENABLE_CUE_VALIDATION=true
CUE_TIMING_PRECISION=2
```

#### File Watcher Service

```bash
# Service Configuration
FILE_WATCHER_HOST=0.0.0.0
FILE_WATCHER_PORT=8003
FILE_WATCHER_DEBUG=false

# Watch Configuration
WATCH_DIRECTORIES=/path/to/music,/path/to/uploads
WATCH_DIRECTORIES_RECURSIVE=true
WATCH_PATTERNS=*.mp3,*.flac,*.wav,*.m4a,*.ogg,*.wma,*.aac
IGNORE_PATTERNS=*.tmp,*.part,.*,Thumbs.db

# Processing
ENABLE_HASH_GENERATION=true
HASH_ALGORITHM=sha256
SECONDARY_HASH_ALGORITHM=xxhash128
ENABLE_METADATA_EXTRACTION=true
METADATA_TIMEOUT=30

# Batch Processing
BATCH_SIZE=10
BATCH_TIMEOUT=60
ENABLE_PARALLEL_PROCESSING=true
MAX_WORKERS=4

# Message Publishing
FILE_WATCHER_EXCHANGE=file_events
FILE_CREATED_ROUTING_KEY=file.created
FILE_MODIFIED_ROUTING_KEY=file.modified
FILE_DELETED_ROUTING_KEY=file.deleted
FILE_MOVED_ROUTING_KEY=file.moved

# Queue Configuration
PUBLISH_TIMEOUT=30
PUBLISH_RETRIES=3
ENABLE_MESSAGE_PERSISTENCE=true
MESSAGE_TTL=86400

# Performance
MEMORY_LIMIT_MB=512
IO_TIMEOUT=30
ENABLE_ASYNC_PROCESSING=true
```

#### Notification Service

```bash
# Service Configuration
NOTIFICATION_SERVICE_HOST=0.0.0.0
NOTIFICATION_SERVICE_PORT=8004
NOTIFICATION_SERVICE_DEBUG=false

# Discord Integration
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-url
DISCORD_ENABLE=true
DISCORD_RATE_LIMIT=5
DISCORD_TIMEOUT=30
DISCORD_RETRY_COUNT=3

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_TLS=true
SMTP_SSL=false
ENABLE_EMAIL=false

# Slack Integration (future)
SLACK_WEBHOOK_URL=
SLACK_ENABLE=false
SLACK_CHANNEL=#tracktion

# Alert Configuration
ALERT_LEVELS=info,warning,error,critical
ENABLE_ERROR_NOTIFICATIONS=true
ENABLE_ANALYSIS_NOTIFICATIONS=false
ENABLE_SYSTEM_NOTIFICATIONS=true
ENABLE_SECURITY_NOTIFICATIONS=true

# Rate Limiting
NOTIFICATION_RATE_LIMIT=10
NOTIFICATION_RATE_WINDOW=60
NOTIFICATION_BURST_LIMIT=20

# Message Formatting
MESSAGE_MAX_LENGTH=2000
ENABLE_MARKDOWN=true
ENABLE_EMBEDS=true
INCLUDE_TIMESTAMPS=true
INCLUDE_CORRELATION_IDS=false
```

## Service Configuration

### Docker Compose Configuration

#### Base Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

x-common-variables: &common-variables
  DATABASE_URL: ${DATABASE_URL}
  NEO4J_URI: ${NEO4J_URI}
  NEO4J_USER: ${NEO4J_USER}
  NEO4J_PASSWORD: ${NEO4J_PASSWORD}
  REDIS_HOST: redis
  REDIS_PORT: 6379
  RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672/

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: tracktion
      POSTGRES_USER: tracktion
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tracktion"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  neo4j:
    image: neo4j:5.12
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_apoc_export_file_enabled: 'true'
      NEO4J_apoc_import_file_enabled: 'true'
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  postgres_data:
  neo4j_data:
  neo4j_logs:
  redis_data:
  rabbitmq_data:
```

#### Development Override (`docker-compose.dev.yml`)

```yaml
version: '3.8'

services:
  postgres:
    environment:
      POSTGRES_DB: tracktion_dev
      POSTGRES_PASSWORD: dev_password
    command: postgres -c log_statement=all

  neo4j:
    environment:
      NEO4J_AUTH: neo4j/dev_password

  redis:
    command: redis-server --appendonly yes

  rabbitmq:
    ports:
      - "15672:15672"  # Management UI

  analysis-service:
    build:
      context: ./services/analysis_service
      dockerfile: Dockerfile.dev
    environment:
      <<: *common-variables
      LOG_LEVEL: DEBUG
      ENABLE_AUTO_RELOAD: 'true'
    volumes:
      - ./services/analysis_service:/app
      - ./shared:/app/shared
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - neo4j
      - redis
      - rabbitmq

  tracklist-service:
    build:
      context: ./services/tracklist_service
      dockerfile: Dockerfile.dev
    environment:
      <<: *common-variables
      LOG_LEVEL: DEBUG
    volumes:
      - ./services/tracklist_service:/app
    ports:
      - "8002:8002"
    depends_on:
      - postgres
      - redis
      - rabbitmq
```

#### Production Override (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  postgres:
    restart: unless-stopped
    environment:
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    volumes:
      - /data/postgres:/var/lib/postgresql/data

  neo4j:
    restart: unless-stopped
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_dbms_memory_heap_initial__size: 1G
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G
    volumes:
      - /data/neo4j:/data

  redis:
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - /data/redis:/data

  rabbitmq:
    restart: unless-stopped
    environment:
      RABBITMQ_VM_MEMORY_HIGH_WATERMARK: 0.6
    volumes:
      - /data/rabbitmq:/var/lib/rabbitmq

  analysis-service:
    restart: unless-stopped
    environment:
      <<: *common-variables
      LOG_LEVEL: INFO
      WORKERS: 4
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  tracklist-service:
    restart: unless-stopped
    environment:
      <<: *common-variables
      LOG_LEVEL: INFO
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

## Database Configuration

### PostgreSQL Advanced Configuration

#### Connection Pool Settings

```python
# In services/shared/core_types/src/database.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False
)
```

#### Alembic Configuration

```ini
# alembic.ini
[alembic]
script_location = alembic
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s
sqlalchemy.url = postgresql://tracktion:password@localhost:5432/tracktion

[loggers]
keys = root,sqlalchemy,alembic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

### Neo4j Configuration

#### Cypher Query Configuration

```python
# In services/shared/core_types/src/neo4j_manager.py
driver = GraphDatabase.driver(
    neo4j_uri,
    auth=(neo4j_user, neo4j_password),
    max_connection_lifetime=300,
    max_connection_pool_size=100,
    connection_timeout=5,
    max_retry_time=30
)
```

#### Index and Constraint Setup

```cypher
-- Performance indexes
CREATE INDEX recording_file_path IF NOT EXISTS FOR (r:Recording) ON (r.file_path);
CREATE INDEX recording_hash IF NOT EXISTS FOR (r:Recording) ON (r.file_hash);
CREATE INDEX metadata_key_value IF NOT EXISTS FOR (m:Metadata) ON (m.key, m.value);
CREATE INDEX artist_name_search IF NOT EXISTS FOR (a:Artist) ON (a.name);
CREATE INDEX album_title_search IF NOT EXISTS FOR (al:Album) ON (al.title);
CREATE INDEX genre_name IF NOT EXISTS FOR (g:Genre) ON (g.name);

-- Uniqueness constraints
CREATE CONSTRAINT recording_id IF NOT EXISTS FOR (r:Recording) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT artist_name_unique IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT album_composite IF NOT EXISTS FOR (al:Album) REQUIRE (al.title, al.artist) IS UNIQUE;
CREATE CONSTRAINT genre_name_unique IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;
```

### Redis Configuration

#### Memory and Persistence

```bash
# redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

#### Cache Key Patterns

```python
# Cache key naming convention
CACHE_KEYS = {
    'analysis_result': 'analysis:{recording_id}',
    'bpm_result': 'bpm:{file_hash}',
    'key_result': 'key:{file_hash}',
    'search_result': 'search:{query_hash}:{page}',
    'tracklist': 'tracklist:{tracklist_id}',
    'user_session': 'session:{session_id}',
}

# TTL settings
CACHE_TTL = {
    'analysis_result': 86400,  # 24 hours
    'bpm_result': 604800,      # 7 days
    'key_result': 604800,      # 7 days
    'search_result': 900,      # 15 minutes
    'tracklist': 7200,         # 2 hours
    'user_session': 3600,      # 1 hour
}
```

## Infrastructure Configuration

### NGINX Load Balancer

```nginx
# /etc/nginx/sites-available/tracktion
upstream analysis_service {
    server analysis-service-1:8001 weight=1;
    server analysis-service-2:8001 weight=1;
}

upstream tracklist_service {
    server tracklist-service-1:8002 weight=1;
    server tracklist-service-2:8002 weight=1;
}

server {
    listen 80;
    server_name api.tracktion.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Analysis Service
    location /v1/analysis/ {
        proxy_pass http://analysis_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_timeout 300s;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Tracklist Service
    location /v1/tracklist/ {
        proxy_pass http://tracklist_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://analysis_service;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Kubernetes Configuration

```yaml
# k8s/analysis-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-service
  namespace: tracktion
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analysis-service
  template:
    metadata:
      labels:
        app: analysis-service
    spec:
      containers:
      - name: analysis-service
        image: tracktion/analysis-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tracktion-secrets
              key: database-url
        - name: REDIS_HOST
          value: "redis-service"
        - name: RABBITMQ_URL
          valueFrom:
            secretKeyRef:
              name: tracktion-secrets
              key: rabbitmq-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /v1/health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/health/ready
            port: 8001
          initialDelaySeconds: 15
          periodSeconds: 5
```

## Security Configuration

### API Authentication

```python
# JWT Configuration
JWT_SETTINGS = {
    'SECRET_KEY': os.getenv('JWT_SECRET_KEY'),
    'ALGORITHM': 'HS256',
    'ACCESS_TOKEN_EXPIRE_MINUTES': 30,
    'REFRESH_TOKEN_EXPIRE_DAYS': 7,
    'ISSUER': 'tracktion-api',
    'AUDIENCE': 'tracktion-users'
}

# API Key Configuration
API_KEY_SETTINGS = {
    'HEADER_NAME': 'X-API-Key',
    'MIN_LENGTH': 32,
    'EXPIRY_DAYS': 365,
    'RATE_LIMIT_MULTIPLIER': 1.0
}
```

### TLS/SSL Configuration

```yaml
# Docker Compose SSL
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    environment:
      - SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
      - SSL_KEY_PATH=/etc/nginx/ssl/key.pem
```

### Database Security

```bash
# PostgreSQL SSL
DATABASE_SSL_MODE=require
DATABASE_SSL_CERT=/path/to/client-cert.pem
DATABASE_SSL_KEY=/path/to/client-key.pem
DATABASE_SSL_ROOT_CERT=/path/to/ca-cert.pem

# Neo4j Encryption
NEO4J_ENCRYPTED=true
NEO4J_TRUST=TRUST_SYSTEM_CA_SIGNED_CERTIFICATES

# Redis SSL
REDIS_SSL=true
REDIS_SSL_CERT_REQS=required
REDIS_SSL_CA_CERTS=/path/to/ca-cert.pem
```

## Performance Tuning

### Analysis Service Optimization

```bash
# CPU Optimization
PARALLEL_WORKERS=8              # Number of CPU cores
ENABLE_GPU_ACCELERATION=true    # If NVIDIA GPU available
MEMORY_LIMIT_MB=4096           # Maximum memory per worker

# Audio Analysis Optimization
BPM_BUFFER_SIZE=8192           # Audio buffer size
ENABLE_STREAMING_ANALYSIS=true  # For large files
ANALYSIS_CHUNK_SIZE=30         # Seconds per chunk

# Cache Optimization
REDIS_CONNECTION_POOL_SIZE=20   # Connection pool
CACHE_BATCH_SIZE=100           # Batch cache operations
ENABLE_CACHE_COMPRESSION=true   # Compress cached data
```

### Database Performance

```sql
-- PostgreSQL optimization
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

```cypher
// Neo4j optimization
// neo4j.conf
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
dbms.tx_log.rotation.retention_policy=1G size
```

### Resource Monitoring

```bash
# Service resource limits
services:
  analysis-service:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
```

This comprehensive configuration guide covers all aspects of Tracktion configuration, from basic environment variables to advanced performance tuning and security settings.
