# Multi-Instance File Watcher Deployment Guide

## Overview

The file_watcher service supports running multiple instances simultaneously to monitor different directories. Each instance operates independently with unique identification and can publish events to the same RabbitMQ message queue without conflicts.

## Instance Identification

Each file_watcher instance has:
- **Unique Instance ID**: Either set via `INSTANCE_ID` environment variable or auto-generated
- **Watched Directory**: The specific directory being monitored by that instance
- **Connection Name**: Unique RabbitMQ connection identifier for management visibility

## Configuration

### Environment Variables

Each instance requires the following configuration:

```bash
# Instance Identification
INSTANCE_ID=watcher-music        # Unique identifier for this instance
DATA_DIR=/data/music             # Directory to monitor

# RabbitMQ Connection
RABBITMQ_HOST=rabbitmq           # RabbitMQ hostname
RABBITMQ_PORT=5672               # RabbitMQ port
RABBITMQ_USER=tracktion          # RabbitMQ username
RABBITMQ_PASS=changeme           # RabbitMQ password

# Logging
LOG_LEVEL=INFO                   # Log level (DEBUG, INFO, WARNING, ERROR)
```

## Deployment Options

### 1. Docker Compose (Recommended)

Use the provided `docker-compose.multi-instance.yaml` for easy multi-instance deployment:

```bash
# Start 3 instances (music, downloads, imports)
docker-compose -f docker-compose.multi-instance.yaml up

# Start 5 instances (includes podcasts and audiobooks)
docker-compose -f docker-compose.multi-instance.yaml --profile extended up

# View logs for specific instance
docker-compose -f docker-compose.multi-instance.yaml logs file_watcher_music

# Scale instances dynamically
docker-compose -f docker-compose.multi-instance.yaml up --scale file_watcher_music=2
```

### 2. Kubernetes Deployment

Example Kubernetes deployment with multiple instances:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: file-watcher-music
spec:
  replicas: 1
  selector:
    matchLabels:
      app: file-watcher
      instance: music
  template:
    metadata:
      labels:
        app: file-watcher
        instance: music
    spec:
      containers:
      - name: file-watcher
        image: tracktion/file-watcher:latest
        env:
        - name: INSTANCE_ID
          value: "watcher-music"
        - name: DATA_DIR
          value: "/data/music"
        - name: RABBITMQ_HOST
          value: "rabbitmq-service"
        volumeMounts:
        - name: music-data
          mountPath: /data/music
          readOnly: true
      volumes:
      - name: music-data
        hostPath:
          path: /mnt/music
```

### 3. Docker Run Commands

For manual deployment of individual instances:

```bash
# Instance 1: Music Directory
docker run -d \
  --name file-watcher-music \
  -e INSTANCE_ID=watcher-music \
  -e DATA_DIR=/data/music \
  -e RABBITMQ_HOST=rabbitmq \
  -v /host/music:/data/music:ro \
  tracktion/file-watcher:latest

# Instance 2: Downloads Directory
docker run -d \
  --name file-watcher-downloads \
  -e INSTANCE_ID=watcher-downloads \
  -e DATA_DIR=/data/downloads \
  -e RABBITMQ_HOST=rabbitmq \
  -v /host/downloads:/data/downloads:ro \
  tracktion/file-watcher:latest
```

## Message Structure

Each instance includes identification metadata in published messages:

```json
{
  "correlation_id": "uuid-v4",
  "timestamp": "2025-08-28T10:00:00Z",
  "event_type": "file_discovered",
  "instance_id": "watcher-music",
  "watched_directory": "/data/music",
  "file_info": {
    "path": "/data/music/song.mp3",
    "name": "song.mp3",
    "extension": ".mp3",
    "size_bytes": "5242880",
    "sha256_hash": "...",
    "xxh128_hash": "..."
  }
}
```

## Monitoring

### RabbitMQ Management Console

View all connected instances in RabbitMQ Management (http://localhost:15672):
- Connections tab shows each instance with unique connection name
- Each connection displays: `file_watcher_{instance_id}`

### Log Aggregation

All instances include instance metadata in logs:

```json
{
  "timestamp": "2025-08-28T10:00:00Z",
  "instance_id": "watcher-music",
  "level": "INFO",
  "message": "File discovered",
  "path": "/data/music/song.mp3"
}
```

Use log aggregation tools (ELK, Grafana Loki) to:
- Filter logs by instance_id
- Track events per directory
- Monitor instance health

### Metrics

Recommended metrics to monitor:
- Files discovered per instance
- Processing time per instance
- Connection status per instance
- Error rate per instance

## Troubleshooting

### Common Issues

#### 1. Instance Identification Conflicts
**Problem**: Multiple instances with same ID
**Solution**: Ensure each instance has unique INSTANCE_ID environment variable

#### 2. Directory Access Issues
**Problem**: Instance can't read directory
**Solution**: Check volume mounts and permissions

#### 3. RabbitMQ Connection Failures
**Problem**: Instance can't connect to RabbitMQ
**Solution**: Verify network connectivity and credentials

#### 4. High Resource Usage
**Problem**: Multiple instances consuming too much CPU/memory
**Solution**:
- Adjust scan intervals
- Limit number of concurrent instances
- Use resource limits in Docker/Kubernetes

### Debugging Commands

```bash
# View instance logs
docker logs file-watcher-music

# Check instance environment
docker exec file-watcher-music env | grep INSTANCE

# Monitor RabbitMQ connections
rabbitmqctl list_connections name client_properties

# Check instance resource usage
docker stats file-watcher-music file-watcher-downloads file-watcher-imports
```

## Best Practices

1. **Instance Naming**: Use descriptive instance IDs (e.g., `watcher-music`, `watcher-podcasts`)
2. **Resource Limits**: Set CPU and memory limits for each instance
3. **Monitoring**: Implement health checks and alerting
4. **Log Rotation**: Configure log rotation to prevent disk space issues
5. **Graceful Shutdown**: Ensure instances handle SIGTERM properly
6. **Network Isolation**: Use separate networks for production and testing
7. **Backup Strategy**: Regular backups of tracked file state

## Performance Considerations

- Each instance maintains its own file tracking state
- Message queue handles concurrent publishers efficiently
- Database operations are handled by downstream services
- Use XXH128 hash for quick duplicate detection
- SHA256 provides cryptographic integrity verification

## Security

- Run instances with read-only volume mounts
- Use non-root user in containers
- Implement network policies in Kubernetes
- Rotate RabbitMQ credentials regularly
- Monitor for unusual file access patterns
