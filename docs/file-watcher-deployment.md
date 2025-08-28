# File Watcher Deployment Guide

## Overview

The File Watcher service monitors directories for new audio files and sends notifications to the message queue for processing by other services. This guide explains how to configure and deploy the service with custom directory paths.

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DATA_DIR` | Directory to monitor inside container | `/data/music` | `/app/media` |
| `HOST_MUSIC_DIR` | Host directory to mount (docker-compose) | `./data/music` | `/home/user/Music` |
| `RABBITMQ_HOST` | RabbitMQ hostname | `localhost` | `rabbitmq` |
| `RABBITMQ_PORT` | RabbitMQ port | `5672` | `5672` |
| `RABBITMQ_USER` | RabbitMQ username | `guest` | `tracktion` |
| `RABBITMQ_PASS` | RabbitMQ password | `guest` | `changeme` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |

### Volume Mounts

The service requires read-only access to the directories you want to monitor. Mount your host directories to the container's `DATA_DIR` path.

## Deployment Methods

### 1. Docker Run

#### Basic Usage
```bash
# Monitor default directory
docker run -d \
  --name file-watcher \
  -v /path/to/music:/data/music:ro \
  tracktion/file-watcher
```

#### Custom Directory with Environment Variable
```bash
# Monitor custom directory path
docker run -d \
  --name file-watcher \
  -e DATA_DIR=/app/media \
  -v /home/user/Music:/app/media:ro \
  tracktion/file-watcher
```

#### Full Configuration
```bash
docker run -d \
  --name file-watcher \
  --network tracktion-network \
  -e DATA_DIR=/data/music \
  -e RABBITMQ_HOST=rabbitmq \
  -e RABBITMQ_PORT=5672 \
  -e RABBITMQ_USER=tracktion \
  -e RABBITMQ_PASS=changeme \
  -e LOG_LEVEL=DEBUG \
  -v /home/user/Music:/data/music:ro \
  tracktion/file-watcher
```

### 2. Docker Compose

#### Basic Configuration

Add to your `.env` file:
```bash
# Host directory to monitor
HOST_MUSIC_DIR=/home/user/Music
# Optional: Custom container path
DATA_DIR=/data/music
```

Run with docker-compose:
```bash
docker-compose up -d file_watcher
```

#### Multiple Directories

To monitor multiple directories, use the multi-directory example:

```bash
# Copy and customize the multi-directory configuration
cp infrastructure/docker-compose.multi-dirs.yaml docker-compose.override.yaml

# Edit docker-compose.override.yaml to set your directories

# Start all file watchers
docker-compose up -d
```

#### Custom Volume Mount Examples

```yaml
# docker-compose.yaml
services:
  file_watcher:
    volumes:
      # Monitor user's Music directory
      - /home/user/Music:/data/music:ro

      # Monitor multiple directories (create multiple services)
      # See docker-compose.multi-dirs.yaml for full example

      # Monitor network share
      - /mnt/nas/music:/data/music:ro

      # Monitor with environment variable
      - ${MUSIC_LIBRARY}:/data/music:ro
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: file-watcher
spec:
  replicas: 1
  selector:
    matchLabels:
      app: file-watcher
  template:
    metadata:
      labels:
        app: file-watcher
    spec:
      containers:
      - name: file-watcher
        image: tracktion/file-watcher:latest
        env:
        - name: DATA_DIR
          value: "/data/music"
        - name: RABBITMQ_HOST
          value: "rabbitmq-service"
        volumeMounts:
        - name: music-volume
          mountPath: /data/music
          readOnly: true
      volumes:
      - name: music-volume
        hostPath:
          path: /mnt/music
          type: Directory
```

## Troubleshooting

### Common Issues

#### 1. Directory Not Found Error
```
ERROR: Data directory /data/music does not exist
```
**Solution**: Ensure the volume mount is correct and the host directory exists:
```bash
# Check host directory exists
ls -la /path/to/your/music

# Verify volume mount in docker inspect
docker inspect file-watcher | grep -A5 Mounts
```

#### 2. Permission Denied Error
```
ERROR: No read permission for /data/music
```
**Solution**: Ensure the directory has read permissions:
```bash
# Check permissions
ls -la /path/to/your/music

# Fix permissions if needed (be careful with permissions)
chmod +r /path/to/your/music
```

#### 3. Container Fails to Start
**Check logs**:
```bash
docker logs file-watcher
```

**Common causes**:
- Missing volume mount
- Incorrect environment variables
- RabbitMQ not accessible

#### 4. No Files Being Detected
**Verify the service is monitoring the correct directory**:
```bash
# Check container logs for monitoring message
docker logs file-watcher | grep "Monitoring directory"
```

**Check supported file extensions**:
```bash
# The service monitors: .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma, .opus, .aiff, .ape
```

### Debugging

#### Enable Debug Logging
```bash
docker run -d \
  --name file-watcher \
  -e LOG_LEVEL=DEBUG \
  -v /path/to/music:/data/music:ro \
  tracktion/file-watcher
```

#### Test Directory Access
```bash
# Test if container can access the directory
docker run --rm \
  -v /path/to/music:/data/music:ro \
  tracktion/file-watcher \
  ls -la /data/music
```

#### Verify Environment Variables
```bash
# Check environment variables in running container
docker exec file-watcher env | grep -E "(DATA_DIR|RABBITMQ)"
```

## Best Practices

1. **Use Read-Only Mounts**: Always mount directories as read-only (`:ro`) to prevent accidental modifications
2. **Absolute Paths**: Use absolute paths for volume mounts to avoid confusion
3. **Environment Files**: Store configuration in `.env` files for easy management
4. **Health Checks**: Monitor container health and logs regularly
5. **Resource Limits**: Set appropriate resource limits for production deployments

## Security Considerations

1. **Read-Only Access**: The service only needs read access to monitored directories
2. **Non-Root User**: The container runs as a non-root user (`appuser`) for security
3. **Network Isolation**: Use Docker networks to isolate services
4. **Secrets Management**: Use Docker secrets or environment files for sensitive data
5. **Volume Permissions**: Ensure proper file permissions on host directories

## Migration from Legacy Configuration

If migrating from the old `FILE_WATCHER_SCAN_PATH` variable:

```bash
# Old configuration
FILE_WATCHER_SCAN_PATH=/data/music

# New configuration (both work for backward compatibility)
DATA_DIR=/data/music
```

The service supports both variables during the transition period, with `DATA_DIR` taking precedence.
