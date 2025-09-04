# Docker Setup Guide

Complete guide for running Tracktion with Docker and Docker Compose, including development and production configurations.

## Table of Contents

1. [Docker Overview](#docker-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Development Setup](#development-setup)
5. [Production Deployment](#production-deployment)
6. [Container Management](#container-management)
7. [Networking](#networking)
8. [Storage and Volumes](#storage-and-volumes)
9. [Monitoring and Logs](#monitoring-and-logs)
10. [Troubleshooting](#troubleshooting)

## Docker Overview

Tracktion uses a multi-container Docker architecture with the following components:

### Infrastructure Containers
- **PostgreSQL**: Primary relational database
- **Neo4j**: Graph database for relationships
- **Redis**: Caching layer and session storage
- **RabbitMQ**: Message queue for service communication

### Application Containers
- **Analysis Service**: Audio analysis and metadata extraction
- **Tracklist Service**: Tracklist search and CUE generation
- **File Watcher Service**: File monitoring and event publishing
- **Cataloging Service**: Music catalog management
- **Notification Service**: Alert and notification handling

## Prerequisites

### System Requirements

- **Docker Engine**: 20.10+
- **Docker Compose**: v2.0+ (plugin version preferred)
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 10GB+ free space
- **OS**: Linux, macOS, or Windows with WSL2

### Installation

#### Docker Desktop (Recommended)

**macOS:**
```bash
brew install --cask docker
```

**Windows:**
Download from https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe

#### Docker Engine (Linux Server)

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

**Verify Installation:**
```bash
docker --version
docker compose version
```

## Quick Start

Get Tracktion running in under 5 minutes:

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/tracktion.git
cd tracktion

# Copy environment template
cp .env.example .env
```

### 2. Basic Configuration

Edit `.env` with minimal required settings:
```bash
# Database passwords
DATABASE_PASSWORD=secure_db_password
NEO4J_PASSWORD=secure_neo4j_password
REDIS_PASSWORD=secure_redis_password

# Watch directories (adjust paths as needed)
WATCH_DIRECTORIES=/path/to/your/music

# Optional: Discord notifications
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

### 3. Start Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### 4. Verify Installation

```bash
# Health checks
curl http://localhost:8001/v1/health    # Analysis Service
curl http://localhost:8002/v1/health    # Tracklist Service

# Access web interfaces
open http://localhost:8001/v1/docs      # Analysis API docs
open http://localhost:7474              # Neo4j Browser
open http://localhost:15672             # RabbitMQ Management
```

## Development Setup

For local development with hot-reload and debugging capabilities.

### Development Configuration

Create `docker-compose.dev.yml`:
```yaml
version: '3.8'

services:
  analysis-service:
    build:
      context: ./services/analysis_service
      dockerfile: Dockerfile.dev
    volumes:
      - ./services/analysis_service:/app
      - ./shared:/app/shared
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - RELOAD=true
    ports:
      - "8001:8001"
      - "5678:5678"  # Debug port

  tracklist-service:
    build:
      context: ./services/tracklist_service
      dockerfile: Dockerfile.dev
    volumes:
      - ./services/tracklist_service:/app
      - ./shared:/app/shared
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    ports:
      - "8002:8002"

  file-watcher:
    build:
      context: ./services/file_watcher
      dockerfile: Dockerfile.dev
    volumes:
      - ./services/file_watcher:/app
      - ./shared:/app/shared
      - ${WATCH_DIRECTORIES}:/watched:ro
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
```

### Development Dockerfile

Example `Dockerfile.dev`:
```dockerfile
# services/analysis_service/Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -e .

# Install development dependencies
RUN uv pip install --system debugpy pytest pytest-cov

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Development command with hot reload
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
```

### Start Development Environment

```bash
# Start with development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Or use the development profile
docker compose --profile development up --build
```

### Development Workflow

```bash
# Rebuild specific service after code changes
docker compose build analysis-service
docker compose up -d analysis-service

# View logs for specific service
docker compose logs -f analysis-service

# Execute commands in running container
docker compose exec analysis-service bash
docker compose exec analysis-service python -m pytest

# Debug with remote debugger
# Connect your IDE to localhost:5678
```

## Production Deployment

Production-ready configuration with security, monitoring, and scalability.

### Production Configuration

Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"

services:
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: tracktion
      POSTGRES_USER: tracktion
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backup:/backup
    command: postgres -c shared_preload_libraries=pg_stat_statements
    logging: *default-logging
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tracktion"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  neo4j:
    image: neo4j:5.12-enterprise
    restart: unless-stopped
    environment:
      NEO4J_AUTH_FILE: /run/secrets/neo4j_auth
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_dbms_memory_heap_initial__size: 2G
      NEO4J_dbms_memory_heap_max__size: 4G
      NEO4J_dbms_memory_pagecache_size: 2G
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    secrets:
      - neo4j_auth
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./backup:/backup
    logging: *default-logging

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --requirepass-file /run/secrets/redis_password
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    logging: *default-logging

  rabbitmq:
    image: rabbitmq:3.12-management
    restart: unless-stopped
    environment:
      RABBITMQ_DEFAULT_USER_FILE: /run/secrets/rabbitmq_user
      RABBITMQ_DEFAULT_PASS_FILE: /run/secrets/rabbitmq_password
      RABBITMQ_VM_MEMORY_HIGH_WATERMARK: 0.6
    secrets:
      - rabbitmq_user
      - rabbitmq_password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    logging: *default-logging

  analysis-service:
    image: tracktion/analysis-service:${VERSION:-latest}
    restart: unless-stopped
    environment:
      - LOG_LEVEL=INFO
      - WORKERS=4
      - DATABASE_URL_FILE=/run/secrets/database_url
    secrets:
      - database_url
      - redis_password
      - rabbitmq_credentials
    volumes:
      - audio_models:/app/models:ro
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging: *default-logging
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      rabbitmq:
        condition: service_started

secrets:
  db_password:
    file: ./secrets/db_password.txt
  neo4j_auth:
    file: ./secrets/neo4j_auth.txt
  redis_password:
    file: ./secrets/redis_password.txt
  rabbitmq_user:
    file: ./secrets/rabbitmq_user.txt
  rabbitmq_password:
    file: ./secrets/rabbitmq_password.txt
  database_url:
    file: ./secrets/database_url.txt
  rabbitmq_credentials:
    file: ./secrets/rabbitmq_credentials.txt

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/tracktion/postgres
  neo4j_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/tracktion/neo4j
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/tracktion/redis
  rabbitmq_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/tracktion/rabbitmq
  audio_models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/tracktion/models
```

### Production Secrets

Create secret files:
```bash
mkdir -p secrets

# Database password
echo "super_secure_db_password" > secrets/db_password.txt

# Neo4j authentication
echo "neo4j/super_secure_neo4j_password" > secrets/neo4j_auth.txt

# Redis password
echo "super_secure_redis_password" > secrets/redis_password.txt

# RabbitMQ credentials
echo "tracktion_user" > secrets/rabbitmq_user.txt
echo "super_secure_rabbitmq_password" > secrets/rabbitmq_password.txt

# Database URL
echo "postgresql://tracktion:super_secure_db_password@postgres:5432/tracktion" > secrets/database_url.txt

# RabbitMQ connection
echo "amqp://tracktion_user:super_secure_rabbitmq_password@rabbitmq:5672/" > secrets/rabbitmq_credentials.txt

# Set appropriate permissions
chmod 600 secrets/*
```

### Start Production Environment

```bash
# Create data directories
sudo mkdir -p /data/tracktion/{postgres,neo4j,redis,rabbitmq,models,backup}
sudo chown -R $USER:$USER /data/tracktion

# Start production services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check service health
docker compose ps
docker compose logs --tail=50 -f
```

## Container Management

### Service Operations

```bash
# Start all services
docker compose up -d

# Start specific services
docker compose up -d postgres redis rabbitmq

# Stop all services
docker compose down

# Stop and remove everything including volumes (DESTRUCTIVE)
docker compose down -v

# Restart specific service
docker compose restart analysis-service

# Scale services
docker compose up -d --scale analysis-service=3

# Update and restart services
docker compose pull
docker compose up -d
```

### Container Inspection

```bash
# View running containers
docker compose ps

# View detailed container info
docker compose ps --format table

# Check container resource usage
docker stats

# Execute commands in containers
docker compose exec analysis-service bash
docker compose exec postgres psql -U tracktion -d tracktion

# Copy files to/from containers
docker compose cp analysis-service:/app/logs/app.log ./logs/
docker compose cp ./config.yaml analysis-service:/app/config.yaml
```

### Image Management

```bash
# Build all images
docker compose build

# Build specific service
docker compose build analysis-service

# Build without cache
docker compose build --no-cache

# Pull latest images
docker compose pull

# Remove unused images
docker system prune -f
docker image prune -f
```

## Networking

### Default Network

Docker Compose automatically creates a network for the stack:
```bash
# View networks
docker network ls

# Inspect tracktion network
docker network inspect tracktion_default
```

### Custom Networks

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

services:
  nginx:
    networks:
      - frontend

  analysis-service:
    networks:
      - frontend
      - backend

  postgres:
    networks:
      - backend
```

### Service Discovery

Services can communicate using service names:
```python
# From analysis-service to postgres
DATABASE_URL = "postgresql://user:pass@postgres:5432/tracktion"

# From analysis-service to redis
REDIS_HOST = "redis"

# From any service to rabbitmq
RABBITMQ_URL = "amqp://user:pass@rabbitmq:5672/"
```

### Port Mapping

```yaml
services:
  analysis-service:
    ports:
      - "8001:8001"        # host:container
      - "127.0.0.1:8001:8001"  # bind to localhost only
      - "8001"             # random host port
```

## Storage and Volumes

### Volume Types

```yaml
services:
  postgres:
    volumes:
      # Named volume (managed by Docker)
      - postgres_data:/var/lib/postgresql/data

      # Bind mount (host path)
      - /host/path:/container/path

      # Anonymous volume
      - /tmp

      # Read-only bind mount
      - ./config:/app/config:ro

volumes:
  postgres_data:
    driver: local
```

### Backup Strategies

#### Database Backups

**PostgreSQL:**
```bash
# Create backup script
cat << 'EOF' > scripts/backup_postgres.sh
#!/bin/bash
BACKUP_DIR="/backup/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

docker compose exec -T postgres pg_dump -U tracktion -d tracktion | \
  gzip > $BACKUP_DIR/tracktion_${TIMESTAMP}.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
EOF

chmod +x scripts/backup_postgres.sh

# Schedule with cron
echo "0 2 * * * /path/to/tracktion/scripts/backup_postgres.sh" | crontab -
```

**Neo4j:**
```bash
# Neo4j backup
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/backup
```

#### Volume Backups

```bash
# Backup volumes using rsync
rsync -av /data/tracktion/ /backup/tracktion-$(date +%Y%m%d)/

# Backup using tar
docker run --rm -v tracktion_postgres_data:/data -v $(pwd):/backup alpine \
  tar -czf /backup/postgres-backup-$(date +%Y%m%d).tar.gz -C /data .
```

### Data Migration

```bash
# Migrate data between environments
docker compose down
docker run --rm -v tracktion_postgres_data:/from -v /new/location:/to alpine \
  sh -c "cd /from && cp -av . /to"
```

## Monitoring and Logs

### Log Management

```bash
# View logs from all services
docker compose logs -f

# View logs from specific service
docker compose logs -f analysis-service

# View logs with timestamps
docker compose logs -f -t

# Follow last 100 lines
docker compose logs -f --tail=100

# View logs from specific time
docker compose logs --since="2024-01-01T00:00:00"

# Save logs to file
docker compose logs analysis-service > analysis-service.log
```

### Log Configuration

```yaml
services:
  analysis-service:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=analysis"
```

### Health Monitoring

```bash
# Check container health
docker compose ps
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Monitor resource usage
docker stats

# Custom health check
docker compose exec analysis-service curl -f http://localhost:8001/v1/health
```

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro

volumes:
  prometheus_data:
  grafana_data:
```

## Troubleshooting

### Common Issues

#### 1. Containers Won't Start

**Check logs:**
```bash
docker compose logs service-name
docker compose events
```

**Common causes:**
- Port conflicts: `netstat -tulpn | grep PORT`
- Insufficient memory: Increase Docker memory limit
- Missing environment variables: Check `.env` file
- Volume permission issues: `sudo chown -R $USER:$USER /data/path`

#### 2. Service Discovery Issues

**Test connectivity:**
```bash
docker compose exec analysis-service ping postgres
docker compose exec analysis-service nslookup redis
```

**Check networks:**
```bash
docker network ls
docker network inspect tracktion_default
```

#### 3. Database Connection Issues

**PostgreSQL:**
```bash
# Check if PostgreSQL is ready
docker compose exec postgres pg_isready -U tracktion

# Connect to database
docker compose exec postgres psql -U tracktion -d tracktion

# Check connections
docker compose exec postgres psql -U tracktion -c "SELECT * FROM pg_stat_activity;"
```

**Neo4j:**
```bash
# Check Neo4j status
docker compose exec neo4j cypher-shell -u neo4j -p password "RETURN 'Hello World'"
```

#### 4. Performance Issues

**Check resource usage:**
```bash
# Container resources
docker stats

# System resources
htop
iostat -x 1
```

**Database performance:**
```bash
# PostgreSQL slow queries
docker compose exec postgres psql -U tracktion -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Redis memory usage
docker compose exec redis redis-cli info memory
```

#### 5. Volume Issues

**Check volume mounts:**
```bash
docker compose exec service-name df -h
docker compose exec service-name ls -la /mount/point
```

**Fix permissions:**
```bash
docker compose exec service-name chown -R user:group /path
```

### Debug Mode

Enable debug mode for troubleshooting:

```yaml
services:
  analysis-service:
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    ports:
      - "5678:5678"  # Debugger port
    command: python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn src.api.app:app --host 0.0.0.0 --reload
```

### Recovery Procedures

#### Complete Reset

```bash
# Stop all services
docker compose down -v

# Remove all containers, networks, and volumes
docker system prune -a -f --volumes

# Rebuild and restart
docker compose build --no-cache
docker compose up -d
```

#### Partial Recovery

```bash
# Reset specific service
docker compose stop analysis-service
docker compose rm -f analysis-service
docker compose up -d analysis-service

# Reset database only
docker compose down postgres
docker volume rm tracktion_postgres_data
docker compose up -d postgres
```

### Getting Help

When seeking help, provide:
```bash
# System information
docker --version
docker compose version
uname -a

# Container status
docker compose ps

# Service logs
docker compose logs --tail=50 service-name

# Resource usage
docker stats --no-stream

# Network information
docker network ls
docker network inspect tracktion_default
```

This comprehensive Docker guide provides everything needed to run Tracktion in any environment, from development to production, with proper monitoring, backup, and troubleshooting procedures.
