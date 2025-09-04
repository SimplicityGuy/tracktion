# Tracktion Installation Guide

Complete installation and setup guide for the Tracktion music analysis and management system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (Docker)](#quick-start-docker)
3. [Development Setup](#development-setup)
4. [Database Configuration](#database-configuration)
5. [Environment Variables](#environment-variables)
6. [Service Configuration](#service-configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, Windows 10+ with WSL2
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 10GB free space (50GB+ for large music libraries)
- **CPU**: 2 cores (4+ cores recommended for parallel processing)

### Software Dependencies

- **Docker**: 20.10+ with Docker Compose v2
- **Python**: 3.11+ (for local development)
- **uv**: Latest version for Python package management
- **Git**: For source code management

### Audio Format Support

The system supports the following audio formats:
- **MP3**: Full metadata extraction including ID3v1/ID3v2
- **FLAC**: Complete Vorbis comment extraction
- **WAV**: Basic metadata and audio analysis
- **M4A/MP4**: iTunes metadata and AAC audio analysis
- **OGG Vorbis**: Comprehensive Vorbis comment support
- **WMA**: Basic Windows Media metadata
- **AAC**: Advanced Audio Coding analysis

## Quick Start (Docker)

The fastest way to get Tracktion running is using Docker Compose.

### 1. Clone Repository

```bash
git clone https://github.com/your-org/tracktion.git
cd tracktion
```

### 2. Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables (see Configuration section)
nano .env
```

### 3. Start Services

```bash
# Start all services in background
docker-compose up -d

# View service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Verify Installation

```bash
# Check service health
curl http://localhost:8001/v1/health  # Analysis Service
curl http://localhost:8002/v1/health  # Tracklist Service

# Access web interfaces
open http://localhost:8001/v1/docs     # Analysis API docs
open http://localhost:8002/v1/docs     # Tracklist API docs
open http://localhost:7474             # Neo4j Browser
```

### 5. Process Your First File

```bash
# Copy an audio file to the watched directory
cp your-audio-file.mp3 ./watched_directory/

# Monitor processing logs
docker-compose logs -f analysis-service
```

## Development Setup

For local development and contribution to the project.

### 1. Prerequisites Installation

#### Install uv (Python Package Manager)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify installation:**
```bash
uv --version
```

#### Install Docker

**macOS:**
```bash
# Install Docker Desktop
brew install --cask docker
```

**Ubuntu/Debian:**
```bash
# Add Docker repository
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Project Setup

```bash
# Clone repository
git clone https://github.com/your-org/tracktion.git
cd tracktion

# Install all development dependencies
uv pip install -e . --all-extras

# Install pre-commit hooks (MANDATORY)
uv run pre-commit install

# Verify pre-commit setup
uv run pre-commit run --all-files
```

### 3. Database Setup

#### PostgreSQL Setup

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
sleep 10

# Run database migrations
uv run alembic upgrade head

# Verify database connection
uv run python -c "from shared.core_types.src.database import DatabaseManager; db = DatabaseManager(); print('Database connected:', db.test_connection())"
```

#### Neo4j Setup

```bash
# Start Neo4j container
docker-compose up -d neo4j

# Wait for Neo4j to be ready
sleep 30

# Initialize Neo4j constraints and indexes
uv run python scripts/initialize_neo4j.py

# Verify Neo4j connection
uv run python -c "from shared.core_types.src.neo4j_manager import Neo4jManager; neo = Neo4jManager(); print('Neo4j connected:', neo.test_connection())"
```

#### Redis Setup

```bash
# Start Redis container
docker-compose up -d redis

# Verify Redis connection
docker-compose exec redis redis-cli ping
```

### 4. RabbitMQ Setup

```bash
# Start RabbitMQ container
docker-compose up -d rabbitmq

# Wait for RabbitMQ to be ready
sleep 15

# Access RabbitMQ Management UI
open http://localhost:15672
# Username: guest, Password: guest

# Verify RabbitMQ connection
uv run python -c "import pika; conn = pika.BlockingConnection(pika.URLParameters('amqp://guest:guest@localhost:5672/')); print('RabbitMQ connected'); conn.close()"
```

### 5. Service-Specific Setup

#### Analysis Service

```bash
cd services/analysis_service

# Install service dependencies
uv pip install -e .

# Download ML models (if enabled)
uv run python src/model_manager.py --download-all

# Test the service
uv run pytest tests/unit/ -v
```

#### Tracklist Service

```bash
cd services/tracklist_service

# Install service dependencies
uv pip install -e .

# Test the service
uv run pytest tests/unit/ -v
```

#### File Watcher Service

```bash
cd services/file_watcher

# Install service dependencies
uv pip install -e .

# Test the service
uv run pytest tests/unit/ -v
```

## Database Configuration

### PostgreSQL Configuration

**Connection String Format:**
```
postgresql://username:password@host:port/database
```

**Production Settings:**
```bash
# In .env file
DATABASE_URL=postgresql://tracktion_user:secure_password@postgres.example.com:5432/tracktion_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

**Migration Management:**

```bash
# Create new migration
uv run alembic revision --autogenerate -m "Add new table"

# Upgrade to latest
uv run alembic upgrade head

# Downgrade one revision
uv run alembic downgrade -1

# Show migration history
uv run alembic history

# Show current revision
uv run alembic current
```

### Neo4j Configuration

**Connection Settings:**
```bash
# In .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=neo4j
NEO4J_MAX_CONNECTION_LIFETIME=300
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_CONNECTION_TIMEOUT=5
```

**Initial Setup Commands:**

```cypher
// Create constraints
CREATE CONSTRAINT recording_id IF NOT EXISTS FOR (r:Recording) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT artist_name IF NOT EXISTS FOR (a:Artist) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT album_title IF NOT EXISTS FOR (al:Album) REQUIRE (al.title, al.artist) IS UNIQUE;
CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;

// Create indexes
CREATE INDEX recording_file_path IF NOT EXISTS FOR (r:Recording) ON (r.file_path);
CREATE INDEX metadata_key IF NOT EXISTS FOR (m:Metadata) ON (m.key);
CREATE INDEX artist_search IF NOT EXISTS FOR (a:Artist) ON (a.name);
```

### Redis Configuration

**Connection Settings:**
```bash
# In .env file
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=5
REDIS_RETRY_ON_TIMEOUT=true
REDIS_HEALTH_CHECK_INTERVAL=30
```

**Cache Configuration:**
```bash
# Cache TTL settings (seconds)
CACHE_DEFAULT_TTL=3600          # 1 hour
CACHE_ANALYSIS_TTL=86400        # 24 hours
CACHE_TRACKLIST_TTL=7200        # 2 hours
CACHE_SEARCH_TTL=900            # 15 minutes

# Cache size limits
CACHE_MAX_MEMORY=512mb
CACHE_MAX_MEMORY_POLICY=allkeys-lru
```

## Environment Variables

### Core Infrastructure

```bash
# Database URLs
DATABASE_URL=postgresql://tracktion:tracktion@localhost:5432/tracktion
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=tracktion_neo4j
REDIS_HOST=localhost
REDIS_PORT=6379

# Message Queue
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
EXCHANGE_NAME=tracktion_exchange

# Service Ports
ANALYSIS_SERVICE_PORT=8001
TRACKLIST_SERVICE_PORT=8002
FILE_WATCHER_PORT=8003
NOTIFICATION_SERVICE_PORT=8004
```

### Analysis Service

```bash
# Queue Configuration
ANALYSIS_QUEUE=analysis_queue
ANALYSIS_ROUTING_KEY=file.analyze
MAX_RETRIES=3
RETRY_DELAY=5.0

# Audio Analysis
ENABLE_AUDIO_ANALYSIS=true
ENABLE_BPM_DETECTION=true
ENABLE_KEY_DETECTION=true
ENABLE_MOOD_ANALYSIS=true
ENABLE_TEMPORAL_ANALYSIS=true

# Performance
BPM_CONFIDENCE_THRESHOLD=0.7
PARALLEL_WORKERS=4
MEMORY_LIMIT_MB=1024
CACHE_REDIS_TTL=3600

# Models
MODELS_DIR=./models
AUTO_DOWNLOAD_MODELS=true
DISCOGS_EFFNET_MODEL_URL=https://essentia.upf.edu/models/classifiers/discogs_effnet-bs64-1.pb
```

### Tracklist Service

```bash
# API Configuration
TRACKLIST_API_PORT=8002
API_RATE_LIMIT=100
API_RATE_WINDOW=60

# Scraping
ENABLE_WEB_SCRAPING=true
SCRAPING_DELAY=1.0
MAX_SCRAPING_RETRIES=3
USER_AGENT=TracktionBot/1.0

# Cache Settings
ENABLE_REDIS_CACHE=true
TRACKLIST_CACHE_TTL=7200
SEARCH_CACHE_TTL=900

# CUE Generation
CUE_OUTPUT_DIR=./cue_files
CUE_DEFAULT_FORMAT=standard
ENABLE_CUE_VALIDATION=true
```

### File Watcher Service

```bash
# Watch Configuration
WATCH_DIRECTORIES=/path/to/music,/path/to/uploads
RECURSIVE_WATCH=true
WATCH_PATTERNS=*.mp3,*.flac,*.wav,*.m4a,*.ogg

# Processing
ENABLE_HASH_GENERATION=true
HASH_ALGORITHM=sha256
ENABLE_METADATA_EXTRACTION=true
BATCH_SIZE=10
PARALLEL_PROCESSING=true

# Message Publishing
FILE_WATCHER_EXCHANGE=file_events
FILE_CREATED_ROUTING_KEY=file.created
FILE_MODIFIED_ROUTING_KEY=file.modified
FILE_DELETED_ROUTING_KEY=file.deleted
```

### Notification Service

```bash
# Discord Integration
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-url
DISCORD_ENABLE=true
DISCORD_RATE_LIMIT=5

# Email Configuration (future)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ENABLE_EMAIL=false

# Alert Settings
ALERT_LEVELS=info,warning,error,critical
ENABLE_ERROR_NOTIFICATIONS=true
ENABLE_ANALYSIS_NOTIFICATIONS=false
ENABLE_SYSTEM_NOTIFICATIONS=true
```

## Service Configuration

### Docker Compose Configuration

**Development Profile:**
```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

**Production Profile:**
```bash
# Start production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Service Dependencies:**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tracktion"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.12
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "$NEO4J_PASSWORD", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  rabbitmq:
    image: rabbitmq:3-management
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Service Startup Order

The services should start in this order:

1. **Infrastructure**: PostgreSQL, Neo4j, Redis, RabbitMQ
2. **File Watcher**: Monitors directories and publishes events
3. **Analysis Service**: Processes audio files and extracts metadata
4. **Cataloging Service**: Organizes and catalogs processed files
5. **Tracklist Service**: Provides search and tracklist functionality
6. **Notification Service**: Handles alerts and notifications

## Verification

### Health Checks

**Service Health:**
```bash
# Check all services
curl http://localhost:8001/v1/health    # Analysis Service
curl http://localhost:8002/v1/health    # Tracklist Service
curl http://localhost:8003/health       # File Watcher Service
curl http://localhost:8004/health       # Notification Service
```

**Database Health:**
```bash
# PostgreSQL
docker-compose exec postgres pg_isready -U tracktion

# Neo4j
docker-compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 'Neo4j is running' as status"

# Redis
docker-compose exec redis redis-cli ping

# RabbitMQ
docker-compose exec rabbitmq rabbitmq-diagnostics ping
```

### Functional Testing

**End-to-End Test:**
```bash
# Copy test file to watched directory
cp tests/fixtures/test_audio.mp3 ./watched_directory/

# Monitor processing
docker-compose logs -f file-watcher analysis-service cataloging-service

# Check results
curl "http://localhost:8001/v1/recordings" | jq .

# Search for the file
curl -X POST "http://localhost:8002/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test_audio", "limit": 10}' | jq .
```

**Unit Tests:**
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest --cov=services --cov-report=html
```

### Performance Testing

**Load Testing:**
```bash
# Install load testing tools
uv pip install locust

# Run load tests
uv run locust -f tests/load_tests/api_load_test.py --host=http://localhost:8001

# Monitor resource usage
docker-compose top
docker stats
```

## Troubleshooting

### Common Issues

#### 1. Services Won't Start

**Symptoms:**
- Docker containers exit immediately
- Connection refused errors
- Port binding failures

**Solutions:**
```bash
# Check port conflicts
netstat -tulpn | grep :8001
lsof -i :8001

# Check Docker logs
docker-compose logs service-name

# Restart with fresh containers
docker-compose down -v
docker-compose up --build
```

#### 2. Database Connection Issues

**Symptoms:**
- Connection timeout errors
- Authentication failures
- Migration failures

**Solutions:**
```bash
# Check database container status
docker-compose ps postgres neo4j redis

# Check connection strings
echo $DATABASE_URL
echo $NEO4J_URI

# Reset database
docker-compose down postgres
docker volume rm tracktion_postgres_data
docker-compose up -d postgres
sleep 10
uv run alembic upgrade head
```

#### 3. File Processing Issues

**Symptoms:**
- Files not being processed
- Analysis stuck in pending state
- Missing metadata

**Solutions:**
```bash
# Check file permissions
ls -la watched_directory/
chmod 755 watched_directory/
chmod 644 watched_directory/*.mp3

# Check RabbitMQ queues
open http://localhost:15672  # Management UI

# Restart file watcher
docker-compose restart file-watcher

# Check analysis service logs
docker-compose logs -f analysis-service
```

#### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow processing
- Container restarts

**Solutions:**
```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Advanced > Memory

# Reduce parallel workers
echo "PARALLEL_WORKERS=2" >> .env
docker-compose restart
```

#### 5. Network Connectivity Issues

**Symptoms:**
- Service discovery failures
- Timeout errors between services
- External API failures

**Solutions:**
```bash
# Check Docker network
docker network ls
docker network inspect tracktion_default

# Test internal connectivity
docker-compose exec analysis-service ping postgres
docker-compose exec analysis-service nslookup redis

# Check external connectivity
docker-compose exec tracklist-service curl -I https://www.1001tracklists.com
```

### Log Analysis

**Centralized Logging:**
```bash
# View all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f analysis-service

# Filter by time
docker-compose logs --since="2024-01-01T00:00:00" analysis-service

# Search logs
docker-compose logs analysis-service 2>&1 | grep ERROR
```

**Log Levels:**
```bash
# Set log level for debugging
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose restart

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Performance Monitoring

**Resource Monitoring:**
```bash
# Docker stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# System monitoring
htop
iotop
nethogs
```

**Application Metrics:**
```bash
# Service-specific metrics
curl http://localhost:8001/v1/metrics    # Analysis Service metrics
curl http://localhost:8002/v1/metrics    # Tracklist Service metrics

# Database metrics
docker-compose exec postgres psql -U tracktion -c "SELECT * FROM pg_stat_activity;"
```

### Getting Help

#### Community Support

- **GitHub Issues**: https://github.com/your-org/tracktion/issues
- **Discord Server**: https://discord.gg/tracktion
- **Documentation**: https://docs.tracktion.com

#### Support Channels

- **Bug Reports**: Use GitHub Issues with bug report template
- **Feature Requests**: Use GitHub Issues with feature request template
- **Security Issues**: Email security@tracktion.com
- **General Questions**: Use GitHub Discussions

#### Providing Debug Information

When reporting issues, include:

```bash
# System information
uname -a
docker --version
docker-compose --version
uv --version

# Service versions
docker-compose exec analysis-service python --version
docker-compose exec analysis-service pip list

# Configuration (sanitized)
cat .env | sed 's/PASSWORD=.*/PASSWORD=***HIDDEN***/'

# Recent logs
docker-compose logs --tail=100 service-name

# Resource usage
docker stats --no-stream
```

This installation guide provides comprehensive setup instructions for both production deployment and development environments. Follow the steps appropriate for your use case, and refer to the troubleshooting section if you encounter any issues.
