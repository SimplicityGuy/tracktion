# Common Errors and Solutions

## Overview

This document catalogs frequently encountered errors in the Tracktion system along with their proven solutions. It's organized by error category and includes error codes, log patterns, root causes, and step-by-step resolution procedures.

## Audio Processing Errors

### Error: BPM Detection Failed

**Error Patterns:**
```
ERROR [analysis_service] BPM detection failed: RuntimeError: Audio loading failed
ERROR [analysis_service] essentia.standard.MonoLoader: could not load audio file
ERROR [analysis_service] BPM detection returned NaN or invalid value
```

**Root Causes:**
1. **Corrupted audio file**: File is damaged or incomplete
2. **Unsupported format**: File format not supported by Essentia
3. **File permissions**: Service cannot read audio file
4. **Memory exhaustion**: Insufficient memory for large audio files
5. **Missing dependencies**: Audio codec libraries not installed

**Solutions:**

#### Solution 1: Verify Audio File
```bash
# Check file integrity and format
docker-compose exec analysis_service file /path/to/audio/file.mp3
docker-compose exec analysis_service ffprobe /path/to/audio/file.mp3

# Test with known good file
docker-compose exec analysis_service python -c "
import essentia.standard as es
loader = es.MonoLoader(filename='/path/to/test.mp3')
audio = loader()
print(f'Loaded {len(audio)} samples')
"
```

#### Solution 2: Fix File Permissions
```bash
# Check and fix file permissions
ls -la /path/to/audio/files/
sudo chown -R tracktion:tracktion /path/to/audio/files/
sudo chmod -R 644 /path/to/audio/files/
```

#### Solution 3: Memory and Resource Issues
```bash
# Increase container memory limits
# Edit docker-compose.yml:
services:
  analysis_service:
    mem_limit: 4g
    mem_reservation: 2g

# Restart with new limits
docker-compose up -d analysis_service

# Monitor memory usage
docker stats analysis_service --no-stream
```

#### Solution 4: Reinstall Audio Dependencies
```bash
# Rebuild container with fresh dependencies
docker-compose build --no-cache analysis_service
docker-compose up -d analysis_service
```

### Error: Key Detection Algorithm Failure

**Error Patterns:**
```
ERROR [analysis_service] Key detection failed: ValueError: Invalid HPCP size
ERROR [analysis_service] KeyExtractor returned invalid key signature
WARNING [analysis_service] Key detection confidence below threshold: 0.23
```

**Root Causes:**
1. **Audio too short**: Less than 30 seconds of audio
2. **Silent audio**: No musical content detected
3. **Atonal music**: Music without clear key center
4. **Algorithm misconfiguration**: Wrong parameters for HPCP analysis

**Solutions:**

#### Solution 1: Validate Audio Content
```bash
# Check audio duration and content
docker-compose exec analysis_service python -c "
import essentia.standard as es
import numpy as np

loader = es.MonoLoader(filename='/path/to/audio/file.mp3')
audio = loader()

print(f'Duration: {len(audio) / 44100:.2f} seconds')
print(f'RMS energy: {np.sqrt(np.mean(audio**2)):.6f}')
print(f'Max amplitude: {np.max(np.abs(audio)):.6f}')

if len(audio) < 44100 * 30:
    print('WARNING: Audio too short for reliable key detection')
if np.sqrt(np.mean(audio**2)) < 0.001:
    print('WARNING: Very quiet audio, may affect key detection')
"
```

#### Solution 2: Adjust Algorithm Parameters
```bash
# Test with different algorithm settings
docker-compose exec analysis_service python -c "
from services.analysis_service.src.key_detector import KeyDetector

# Try with relaxed confidence threshold
detector = KeyDetector(confidence_threshold=0.5)
result = detector.detect_key('/path/to/audio/file.mp3')
print(f'Key: {result.key}, Confidence: {result.confidence}')
"

# Update configuration if needed
echo 'KEY_DETECTION_CONFIDENCE_THRESHOLD=0.5' >> .env
docker-compose restart analysis_service
```

## Database Errors

### Error: Connection Pool Exhausted

**Error Patterns:**
```
ERROR [tracklist_service] sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 10 reached
ERROR [tracklist_service] Connection could not be returned to pool
CRITICAL [tracklist_service] Database connection pool exhausted
```

**Root Causes:**
1. **Connection leaks**: Connections not properly closed
2. **Long-running queries**: Blocking connection pool
3. **High concurrent load**: More connections needed than configured
4. **Database deadlocks**: Connections waiting indefinitely

**Solutions:**

#### Solution 1: Identify Connection Leaks
```bash
# Check active database connections
docker-compose exec postgres psql -U tracktion -c "
SELECT
    state,
    count(*) as connections,
    usename,
    application_name
FROM pg_stat_activity
WHERE datname='tracktion_dev'
GROUP BY state, usename, application_name
ORDER BY connections DESC;
"

# Look for idle connections
docker-compose exec postgres psql -U tracktion -c "
SELECT
    pid,
    state,
    state_change,
    query_start,
    query
FROM pg_stat_activity
WHERE datname='tracktion_dev'
AND state = 'idle in transaction'
AND state_change < now() - interval '5 minutes';
"
```

#### Solution 2: Kill Problematic Connections
```bash
# Kill long-idle connections
docker-compose exec postgres psql -U tracktion -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname='tracktion_dev'
AND state = 'idle in transaction'
AND state_change < now() - interval '10 minutes';
"

# Restart service to reset connection pool
docker-compose restart tracklist_service
```

#### Solution 3: Increase Pool Size
```bash
# Update connection pool configuration
echo 'SQLALCHEMY_POOL_SIZE=30' >> .env
echo 'SQLALCHEMY_MAX_OVERFLOW=20' >> .env
echo 'SQLALCHEMY_POOL_TIMEOUT=30' >> .env

docker-compose restart tracklist_service

# Monitor pool usage
docker-compose logs tracklist_service | grep -i "pool"
```

#### Solution 4: Optimize Query Performance
```bash
# Find slow queries
docker-compose exec postgres psql -U tracktion -c "
SELECT
    query,
    calls,
    total_time / calls as avg_time_ms,
    mean_time,
    stddev_time
FROM pg_stat_statements
WHERE calls > 10
ORDER BY avg_time_ms DESC
LIMIT 10;
"

# Add missing indexes
docker-compose exec postgres psql -U tracktion -c "
-- Example: Add index for frequently searched columns
CREATE INDEX CONCURRENTLY idx_tracks_artist_title
ON tracks(artist_id, title);
"
```

### Error: Database Migration Failures

**Error Patterns:**
```
ERROR [alembic] Can't locate revision identified by 'abc123'
ERROR [alembic] Target database is not up to date
sqlalchemy.exc.ProgrammingError: relation "new_table" does not exist
```

**Root Causes:**
1. **Migration conflicts**: Multiple branches creating conflicting migrations
2. **Manual schema changes**: Database modified outside of Alembic
3. **Corrupted migration state**: Alembic version table corrupted
4. **Incomplete migration**: Previous migration failed partially

**Solutions:**

#### Solution 1: Check Migration Status
```bash
# Check current migration state
docker-compose exec analysis_service uv run alembic current
docker-compose exec analysis_service uv run alembic history --verbose

# Show pending migrations
docker-compose exec analysis_service uv run alembic show current
```

#### Solution 2: Resolve Migration Conflicts
```bash
# Create merge migration for conflicting branches
docker-compose exec analysis_service uv run alembic merge -m "merge conflicting migrations" head1 head2

# Apply merge migration
docker-compose exec analysis_service uv run alembic upgrade head
```

#### Solution 3: Fix Corrupted Migration State
```bash
# Manual migration state repair (use with caution)
docker-compose exec postgres psql -U tracktion -c "
-- Check alembic version table
SELECT * FROM alembic_version;

-- If corrupted, set to known good state
-- DELETE FROM alembic_version;
-- INSERT INTO alembic_version (version_num) VALUES ('known_good_revision');
"

# Stamp database with current state
docker-compose exec analysis_service uv run alembic stamp head
```

## Message Queue Errors

### Error: RabbitMQ Connection Failed

**Error Patterns:**
```
ERROR [analysis_service] pika.exceptions.AMQPConnectionError: Connection to localhost:5672 failed
ERROR [file_watcher] ConnectionResetError: [Errno 104] Connection reset by peer
WARNING [notification_service] Message queue connection lost, retrying...
```

**Root Causes:**
1. **RabbitMQ service down**: Container stopped or crashed
2. **Network connectivity**: DNS or routing issues
3. **Authentication failure**: Wrong credentials
4. **Resource exhaustion**: RabbitMQ out of memory/disk space

**Solutions:**

#### Solution 1: Check RabbitMQ Status
```bash
# Check RabbitMQ container
docker-compose ps rabbitmq
docker-compose logs rabbitmq --tail=50

# Check RabbitMQ service status
docker-compose exec rabbitmq rabbitmqctl status
docker-compose exec rabbitmq rabbitmqctl cluster_status
```

#### Solution 2: Verify Connection Parameters
```bash
# Test connection manually
docker-compose exec analysis_service python -c "
import pika
import os

try:
    connection = pika.BlockingConnection(
        pika.URLParameters(os.getenv('RABBITMQ_URL'))
    )
    print('RabbitMQ connection successful')
    connection.close()
except Exception as e:
    print(f'RabbitMQ connection failed: {e}')
"

# Check environment variables
docker-compose exec analysis_service env | grep RABBITMQ
```

#### Solution 3: Restart RabbitMQ
```bash
# Restart RabbitMQ container
docker-compose restart rabbitmq

# Wait for RabbitMQ to be ready
sleep 30

# Restart dependent services
docker-compose restart analysis_service file_watcher notification_service
```

#### Solution 4: Check Resource Usage
```bash
# Check RabbitMQ memory usage
docker-compose exec rabbitmq rabbitmqctl status | grep -A5 -B5 memory

# Check disk space
docker-compose exec rabbitmq df -h

# If needed, clean up old messages
docker-compose exec rabbitmq rabbitmqctl list_queues name messages
docker-compose exec rabbitmq rabbitmqctl purge_queue queue_name
```

### Error: Message Processing Stuck

**Error Patterns:**
```
WARNING [analysis_service] Message queue backlog detected: 1500 messages
ERROR [analysis_service] Message processing timeout after 300 seconds
ERROR [analysis_service] Consumer stopped due to repeated failures
```

**Root Causes:**
1. **Consumer crashed**: Service stopped processing messages
2. **Processing bottleneck**: Messages taking too long to process
3. **Error loop**: Failed messages being requeued indefinitely
4. **Resource constraints**: Insufficient CPU/memory for processing

**Solutions:**

#### Solution 1: Check Queue Status
```bash
# Examine queue backlogs
docker-compose exec rabbitmq rabbitmqctl list_queues name messages messages_ready messages_unacknowledged consumers

# Check consumer details
docker-compose exec rabbitmq rabbitmqctl list_consumers

# Look for problematic messages
docker-compose exec rabbitmq rabbitmqctl list_queue_bindings
```

#### Solution 2: Restart Consumers
```bash
# Restart message consumers
docker-compose restart analysis_service file_watcher

# Check if consumers reconnected
sleep 10
docker-compose exec rabbitmq rabbitmqctl list_consumers
```

#### Solution 3: Clear Stuck Messages
```bash
# For development/testing only - this loses messages!
docker-compose exec rabbitmq rabbitmqctl purge_queue audio.analysis.requests
docker-compose exec rabbitmq rabbitmqctl purge_queue file.operations.requests

# For production - inspect messages first
docker-compose exec rabbitmq rabbitmqctl get_messages audio.analysis.requests 5
```

## File System and Storage Errors

### Error: Disk Space Exhausted

**Error Patterns:**
```
ERROR [file_watcher] OSError: [Errno 28] No space left on device
ERROR [analysis_service] Cannot write temporary file: disk full
CRITICAL [system] Disk usage critical: 98% full
```

**Root Causes:**
1. **Log files growing**: Application logs consuming space
2. **Temporary files**: Audio processing temporary files not cleaned
3. **Docker images**: Accumulated Docker layers and images
4. **Database growth**: Database files growing beyond expectations

**Solutions:**

#### Solution 1: Immediate Cleanup
```bash
# Check disk usage
df -h
du -h /var/lib/docker/ | tail -10

# Clean up Docker resources
docker system prune -a --volumes
docker image prune -a

# Clean up application logs
find logs/ -name "*.log" -size +100M -exec truncate -s 50M {} \;
find /tmp -name "tracktion_*" -mtime +1 -delete

# Clean up audio processing temp files
find /tmp -name "audio_*" -mtime +0 -delete
```

#### Solution 2: Configure Log Rotation
```bash
# Set up logrotate for application logs
sudo tee /etc/logrotate.d/tracktion << 'EOF'
/path/to/tracktion/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 tracktion tracktion
    postrotate
        docker-compose restart analysis_service tracklist_service file_watcher
    endscript
}
EOF

# Test log rotation
sudo logrotate -d /etc/logrotate.d/tracktion
```

#### Solution 3: Implement Disk Monitoring
```bash
# Create disk monitoring script
cat > scripts/disk-monitor.sh << 'EOF'
#!/bin/bash
THRESHOLD=85
USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

if [ $USAGE -gt $THRESHOLD ]; then
    echo "WARNING: Disk usage is ${USAGE}% (threshold: ${THRESHOLD}%)"

    # Automated cleanup
    docker system prune -f
    find /tmp -name "tracktion_*" -mtime +0 -delete

    echo "Cleanup completed. New usage: $(df / | tail -1 | awk '{print $5}')"
fi
EOF

chmod +x scripts/disk-monitor.sh

# Add to crontab
echo "*/15 * * * * /path/to/tracktion/scripts/disk-monitor.sh" | sudo crontab -
```

### Error: File Permission Denied

**Error Patterns:**
```
ERROR [file_watcher] PermissionError: [Errno 13] Permission denied: '/audio/files/'
ERROR [analysis_service] Cannot read audio file: permission denied
ERROR [file_rename_service] Failed to move file: operation not permitted
```

**Root Causes:**
1. **Wrong file ownership**: Files owned by different user
2. **Incorrect permissions**: Files not readable by service user
3. **SELinux/AppArmor**: Security policies blocking access
4. **Container user mismatch**: UID/GID mismatch between host and container

**Solutions:**

#### Solution 1: Fix File Ownership
```bash
# Check current ownership
ls -la /path/to/audio/files/

# Fix ownership for Docker containers
sudo chown -R 1000:1000 /path/to/audio/files/
sudo chmod -R 755 /path/to/audio/files/

# For specific service user
sudo chown -R tracktion:tracktion /path/to/audio/files/
```

#### Solution 2: Configure Docker User Mapping
```bash
# Update docker-compose.yml to use host user
services:
  analysis_service:
    user: "${UID}:${GID}"

  file_watcher:
    user: "${UID}:${GID}"

# Set environment variables
echo "UID=$(id -u)" >> .env
echo "GID=$(id -g)" >> .env

# Restart containers
docker-compose down
docker-compose up -d
```

#### Solution 3: Check SELinux Context
```bash
# Check SELinux status (RedHat/CentOS)
getenforce
ls -Z /path/to/audio/files/

# Fix SELinux context if needed
sudo setsebool -P container_manage_cgroup on
sudo chcon -R -t container_file_t /path/to/audio/files/
```

## Network and Connectivity Errors

### Error: Service Communication Timeout

**Error Patterns:**
```
ERROR [tracklist_service] httpx.ReadTimeout: Read timeout on request
ERROR [analysis_service] ConnectionError: Failed to reach tracklist_service
WARNING [file_watcher] Service discovery failed: DNS resolution error
```

**Root Causes:**
1. **Network latency**: Slow network between services
2. **Service overloaded**: Target service not responding in time
3. **DNS issues**: Service names not resolving correctly
4. **Firewall blocking**: Network policies blocking connections

**Solutions:**

#### Solution 1: Test Network Connectivity
```bash
# Test service-to-service connectivity
docker-compose exec analysis_service ping tracklist_service
docker-compose exec analysis_service curl -v http://tracklist_service:8002/health

# Check DNS resolution
docker-compose exec analysis_service nslookup tracklist_service
docker-compose exec analysis_service cat /etc/resolv.conf
```

#### Solution 2: Increase Timeouts
```bash
# Update service configuration with higher timeouts
echo 'HTTP_TIMEOUT=30' >> .env
echo 'DATABASE_TIMEOUT=60' >> .env

# Restart services
docker-compose restart analysis_service tracklist_service
```

#### Solution 3: Check Docker Network
```bash
# Inspect Docker network
docker network ls
docker network inspect tracktion_default

# Recreate network if needed
docker-compose down
docker network prune
docker-compose up -d
```

## Application-Specific Errors

### Error: Track Matching Failed

**Error Patterns:**
```
ERROR [tracklist_service] Track matching timeout: fuzzy search took too long
WARNING [tracklist_service] No matches found for track: "Unknown Title"
ERROR [tracklist_service] Matching service returned malformed response
```

**Root Causes:**
1. **Database performance**: Slow fuzzy matching queries
2. **Large catalog**: Too many tracks to search efficiently
3. **Poor data quality**: Inconsistent track metadata
4. **Algorithm tuning**: Matching thresholds too strict/loose

**Solutions:**

#### Solution 1: Optimize Database Queries
```bash
# Add indexes for matching
docker-compose exec postgres psql -U tracktion -c "
CREATE INDEX CONCURRENTLY idx_tracks_title_trgm
ON tracks USING gin(title gin_trgm_ops);

CREATE INDEX CONCURRENTLY idx_tracks_artist_trgm
ON tracks USING gin(artist gin_trgm_ops);

-- Enable trigram extension if needed
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"

# Analyze query performance
docker-compose exec postgres psql -U tracktion -c "
EXPLAIN ANALYZE
SELECT * FROM tracks
WHERE title % 'search term'
ORDER BY similarity(title, 'search term') DESC
LIMIT 10;
"
```

#### Solution 2: Tune Matching Parameters
```bash
# Adjust matching sensitivity
echo 'FUZZY_MATCH_THRESHOLD=0.6' >> .env
echo 'MATCH_TIMEOUT_SECONDS=30' >> .env

# Test with different parameters
docker-compose exec tracklist_service python -c "
from services.tracklist_service.src.services.matching_service import MatchingService

matcher = MatchingService()
results = matcher.find_matches('test track', 'test artist', threshold=0.5)
print(f'Found {len(results)} matches')
"
```

### Error: Audio Analysis Inconsistent Results

**Error Patterns:**
```
WARNING [analysis_service] BPM detection variance high: 120 vs 240
ERROR [analysis_service] Key detection disagreement: C major vs F# minor
WARNING [analysis_service] Mood analysis confidence too low: 0.23
```

**Root Causes:**
1. **Algorithm sensitivity**: Different algorithms giving different results
2. **Audio quality**: Poor quality affecting analysis accuracy
3. **Musical content**: Complex music difficult to analyze
4. **Model training**: ML models not optimized for music type

**Solutions:**

#### Solution 1: Validate with Reference Files
```bash
# Test with known reference tracks
mkdir -p tests/fixtures/reference_audio/

# Test BPM detection accuracy
docker-compose exec analysis_service python -c "
from services.analysis_service.src.bpm_detector import BPMDetector

detector = BPMDetector()
test_files = [
    ('/reference/120bpm.mp3', 120),
    ('/reference/128bpm.mp3', 128),
    ('/reference/140bpm.mp3', 140)
]

for file_path, expected_bpm in test_files:
    result = detector.detect_bpm(file_path)
    error = abs(result['bpm'] - expected_bpm)
    print(f'{file_path}: detected {result[\"bpm\"]} (error: {error})')

    if error > 5:
        print(f'WARNING: High BPM error for {file_path}')
"
```

#### Solution 2: Implement Consensus Algorithm
```bash
# Update analysis to use multiple algorithms and consensus
echo 'USE_CONSENSUS_ANALYSIS=true' >> .env
echo 'MIN_CONSENSUS_ALGORITHMS=2' >> .env
echo 'CONSENSUS_AGREEMENT_THRESHOLD=0.1' >> .env

docker-compose restart analysis_service
```

## Prevention and Monitoring

### Automated Error Detection

#### Log Monitoring Script
```bash
cat > scripts/error-monitor.sh << 'EOF'
#!/bin/bash

# Monitor for critical errors
LOGDIR="/path/to/tracktion/logs"
ERROR_COUNT=$(find $LOGDIR -name "*.log" -mtime -1 -exec grep -c "ERROR\|CRITICAL" {} \; | paste -sd+ | bc)

if [ $ERROR_COUNT -gt 50 ]; then
    echo "High error rate detected: $ERROR_COUNT errors in last 24h"

    # Show recent errors
    find $LOGDIR -name "*.log" -mtime -1 -exec grep -h "ERROR\|CRITICAL" {} \; | tail -10

    # Alert mechanism (email, Slack, etc.)
    # curl -X POST -H 'Content-type: application/json' \
    #   --data "{\"text\":\"Tracktion error rate high: $ERROR_COUNT errors\"}" \
    #   $SLACK_WEBHOOK_URL
fi
EOF

chmod +x scripts/error-monitor.sh

# Schedule regular monitoring
echo "0 */6 * * * /path/to/tracktion/scripts/error-monitor.sh" | crontab -
```

#### Health Check Automation
```bash
cat > scripts/health-check-cron.sh << 'EOF'
#!/bin/bash

if ! ./scripts/health-check.sh > /dev/null 2>&1; then
    echo "Health check failed at $(date)"

    # Attempt automatic recovery
    echo "Attempting service restart..."
    docker-compose restart

    sleep 60

    # Verify recovery
    if ./scripts/health-check.sh > /dev/null 2>&1; then
        echo "Recovery successful"
    else
        echo "Recovery failed - manual intervention required"
        # Send alert
    fi
fi
EOF

chmod +x scripts/health-check-cron.sh
echo "*/10 * * * * /path/to/tracktion/scripts/health-check-cron.sh" | crontab -
```

This document should be updated regularly as new error patterns are discovered and resolved. Each solution should be tested in a development environment before applying to production systems.
