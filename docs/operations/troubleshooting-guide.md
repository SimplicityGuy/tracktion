# Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps operators and developers diagnose and resolve common issues in the Tracktion system. It covers service-specific problems, infrastructure issues, and systematic debugging approaches.

## Quick Diagnostic Commands

### System Health Check
```bash
# Check all services status
docker-compose ps

# Check service health endpoints
curl -f http://localhost:8001/health  # Analysis Service
curl -f http://localhost:8002/health  # Tracklist Service
curl -f http://localhost:8003/health  # File Watcher

# Check external dependencies
docker-compose exec postgres pg_isready -U tracktion
docker-compose exec redis redis-cli ping
docker-compose exec rabbitmq rabbitmqctl status

# Check resource usage
docker stats --no-stream
df -h  # Disk space
free -h  # Memory usage
```

### Log Analysis
```bash
# View recent service logs
docker-compose logs --tail=100 analysis_service
docker-compose logs --tail=100 tracklist_service
docker-compose logs --tail=100 file_watcher

# Search for errors across all services
docker-compose logs | grep -i error | tail -20
docker-compose logs | grep -i exception | tail -20
docker-compose logs | grep -i failed | tail -20

# Monitor logs in real-time
docker-compose logs -f
```

## Service-Specific Troubleshooting

### Analysis Service Issues

#### Audio Processing Failures

**Symptoms:**
- BPM detection returns null or fails
- Audio files not being processed
- High CPU usage without results
- Memory consumption growing continuously

**Diagnostic Commands:**
```bash
# Check audio processing queue
docker-compose exec rabbitmq rabbitmqctl list_queues | grep audio

# Test audio file loading
docker-compose exec analysis_service python -c "
import essentia.standard as es
try:
    loader = es.MonoLoader(filename='/path/to/test.mp3')
    audio = loader()
    print(f'Audio loaded: {len(audio)} samples, duration: {len(audio)/44100:.2f}s')
except Exception as e:
    print(f'Audio loading failed: {e}')
"

# Check available disk space for temp files
docker-compose exec analysis_service df -h /tmp

# Monitor memory usage during processing
docker stats analysis_service --no-stream
```

**Common Solutions:**
```bash
# 1. Restart analysis service
docker-compose restart analysis_service

# 2. Clear temp directories
docker-compose exec analysis_service rm -rf /tmp/tracktion_*

# 3. Check audio file permissions and format
docker-compose exec analysis_service ls -la /audio/files/
docker-compose exec analysis_service file /audio/files/problem_file.mp3

# 4. Increase memory limits
# Edit docker-compose.yml:
# services:
#   analysis_service:
#     mem_limit: 4g

# 5. Check Essentia library installation
docker-compose exec analysis_service python -c "import essentia; print('Essentia OK')"
```

#### Algorithm Confidence Issues

**Symptoms:**
- BPM detection confidence consistently low
- Key detection returning incorrect results
- Mood analysis failing validation

**Diagnostic Approach:**
```bash
# Test with known reference files
docker-compose exec analysis_service python -c "
from services.analysis_service.src.bpm_detector import BPMDetector
detector = BPMDetector()

# Test known 120 BPM file
result = detector.detect_bpm('/path/to/120bpm_test.mp3')
print(f'BPM: {result[\"bpm\"]}, Confidence: {result[\"confidence\"]}')

# Should be close to 120 with high confidence
if abs(result['bpm'] - 120) > 5:
    print('WARNING: BPM detection may be miscalibrated')
if result['confidence'] < 0.8:
    print('WARNING: Low confidence suggests algorithm issues')
"

# Check algorithm parameters
docker-compose exec analysis_service python -c "
from services.analysis_service.src.config import get_config
config = get_config()
print(f'BPM confidence threshold: {config.bpm_confidence_threshold}')
print(f'Sample rate: {config.audio_sample_rate}')
"
```

### Tracklist Service Issues

#### Database Connection Problems

**Symptoms:**
- API returning 500 errors
- "Connection pool exhausted" messages
- Slow query responses
- Database timeout errors

**Diagnostic Commands:**
```bash
# Check database connection
docker-compose exec tracklist_service python -c "
from shared.database import get_engine
try:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute('SELECT 1')
        print('Database connection OK')
except Exception as e:
    print(f'Database connection failed: {e}')
"

# Check connection pool status
docker-compose logs tracklist_service | grep -i "pool"

# Monitor active connections
docker-compose exec postgres psql -U tracktion -c "
SELECT state, count(*)
FROM pg_stat_activity
WHERE datname='tracktion_dev'
GROUP BY state;
"

# Check for long-running queries
docker-compose exec postgres psql -U tracktion -c "
SELECT pid, now() - query_start as duration, query
FROM pg_stat_activity
WHERE state = 'active'
AND now() - query_start > interval '5 seconds'
ORDER BY duration DESC;
"
```

**Solutions:**
```bash
# 1. Restart service and database
docker-compose restart tracklist_service postgres

# 2. Kill long-running queries
docker-compose exec postgres psql -U tracktion -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active'
AND now() - query_start > interval '30 seconds';
"

# 3. Increase connection pool size
# Edit service configuration:
# SQLALCHEMY_POOL_SIZE=20
# SQLALCHEMY_MAX_OVERFLOW=30

# 4. Optimize slow queries
docker-compose exec postgres psql -U tracktion -c "
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"
```

#### Track Matching Performance

**Symptoms:**
- Slow track matching API responses
- High CPU usage in matching service
- Memory growth during batch operations

**Performance Analysis:**
```bash
# Profile matching service
docker-compose exec tracklist_service python -c "
import cProfile
from services.tracklist_service.src.services.matching_service import MatchingService

matching_service = MatchingService()

# Profile a typical matching operation
pr = cProfile.Profile()
pr.enable()

# Simulate matching operation
result = matching_service.find_matches('test track', 'test artist')

pr.disable()
pr.print_stats(10)  # Top 10 functions
"

# Check database indexes
docker-compose exec postgres psql -U tracktion -c "
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE tablename IN ('tracks', 'artists', 'albums')
ORDER BY n_distinct DESC;
"
```

### File Watcher Issues

#### File System Monitoring Problems

**Symptoms:**
- New files not being detected
- Duplicate processing events
- High inode usage
- File watcher service crashes

**Diagnostic Commands:**
```bash
# Check inode usage
df -i

# Check file watcher limits
docker-compose exec file_watcher cat /proc/sys/fs/inotify/max_user_watches
docker-compose exec file_watcher cat /proc/sys/fs/inotify/max_user_instances

# Test file events manually
docker-compose exec file_watcher python -c "
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TestHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        print(f'{time.strftime(\"%H:%M:%S\")} {event.event_type}: {event.src_path}')

observer = Observer()
observer.schedule(TestHandler(), '/watch/directory', recursive=True)
observer.start()

print('Watching for file events... (Ctrl+C to stop)')
try:
    time.sleep(60)
except KeyboardInterrupt:
    observer.stop()
observer.join()
"

# Check watched directories
docker-compose logs file_watcher | grep -i "watching"
```

**Solutions:**
```bash
# 1. Increase inotify limits
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 2. Restart file watcher service
docker-compose restart file_watcher

# 3. Clear duplicate events
docker-compose exec rabbitmq rabbitmqctl purge_queue file.events

# 4. Optimize directory watching
# Configure selective watching in file_watcher config:
# WATCH_EXTENSIONS=.mp3,.flac,.wav
# IGNORE_PATTERNS=.tmp,.part,.*
```

## Infrastructure Issues

### Docker and Container Problems

#### Container Won't Start

**Symptoms:**
- Service exits immediately after start
- Port binding failures
- Volume mount issues
- Resource constraints

**Diagnostic Steps:**
```bash
# Check container status and exit codes
docker-compose ps
docker-compose logs service_name

# Check for port conflicts
sudo lsof -i :8001  # Analysis service port
sudo lsof -i :5432  # PostgreSQL port
sudo lsof -i :6379  # Redis port

# Check disk space
df -h
docker system df

# Check memory usage
free -h
docker stats --no-stream

# Inspect container configuration
docker-compose config
docker inspect tracktion_analysis_service_1
```

**Solutions:**
```bash
# 1. Clean up Docker resources
docker system prune -f
docker volume prune -f

# 2. Free up disk space
docker image prune -a
docker container prune

# 3. Restart Docker daemon (if needed)
sudo systemctl restart docker

# 4. Check for permission issues
ls -la /var/lib/docker/
sudo chown -R $(whoami):$(whoami) ./data/

# 5. Recreate containers
docker-compose down
docker-compose up --force-recreate
```

### Database Issues

#### PostgreSQL Performance Problems

**Symptoms:**
- Slow query execution
- High CPU usage
- Connection pool exhaustion
- Lock waits

**Performance Analysis:**
```bash
# Check database size and statistics
docker-compose exec postgres psql -U tracktion -c "
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"

# Analyze query performance
docker-compose exec postgres psql -U tracktion -c "
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
WHERE calls > 100
ORDER BY mean_time DESC
LIMIT 10;
"

# Check for blocking queries
docker-compose exec postgres psql -U tracktion -c "
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
"
```

**Optimization Steps:**
```bash
# 1. Update database statistics
docker-compose exec postgres psql -U tracktion -c "ANALYZE;"

# 2. Reindex heavily used tables
docker-compose exec postgres psql -U tracktion -c "REINDEX TABLE tracks;"

# 3. Check and create missing indexes
docker-compose exec postgres psql -U tracktion -c "
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE n_distinct > 100
AND correlation < 0.1
ORDER BY n_distinct DESC;
"

# 4. Vacuum and analyze
docker-compose exec postgres psql -U tracktion -c "VACUUM ANALYZE;"
```

#### Redis Issues

**Symptoms:**
- Cache misses increasing
- Memory usage growing
- Connection timeouts
- Slow response times

**Diagnostic Commands:**
```bash
# Check Redis status
docker-compose exec redis redis-cli info server
docker-compose exec redis redis-cli info memory
docker-compose exec redis redis-cli info stats

# Check key distribution
docker-compose exec redis redis-cli --scan --pattern "*" | head -20

# Monitor Redis operations
docker-compose exec redis redis-cli monitor

# Check memory usage by key type
docker-compose exec redis redis-cli info memory | grep used_memory
docker-compose exec redis redis-cli memory usage "key_name"
```

**Solutions:**
```bash
# 1. Clear expired keys
docker-compose exec redis redis-cli --scan --pattern "*" | xargs docker-compose exec redis redis-cli del

# 2. Optimize memory usage
docker-compose exec redis redis-cli config set maxmemory-policy allkeys-lru
docker-compose exec redis redis-cli config set maxmemory 1gb

# 3. Restart Redis (data loss warning)
docker-compose restart redis
```

### Message Queue Issues

#### RabbitMQ Problems

**Symptoms:**
- Messages accumulating in queues
- Consumer connection failures
- High memory usage
- Message processing delays

**Queue Analysis:**
```bash
# Check queue status
docker-compose exec rabbitmq rabbitmqctl list_queues name messages consumers

# Check exchange bindings
docker-compose exec rabbitmq rabbitmqctl list_bindings

# Monitor message rates
docker-compose exec rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged message_stats

# Check consumer details
docker-compose exec rabbitmq rabbitmqctl list_consumers
```

**Solutions:**
```bash
# 1. Purge stuck queues (development only)
docker-compose exec rabbitmq rabbitmqctl purge_queue audio.analysis.requests

# 2. Restart consumers
docker-compose restart analysis_service

# 3. Check disk space (RabbitMQ is sensitive to disk space)
docker-compose exec rabbitmq df -h

# 4. Increase memory limits
# Edit rabbitmq configuration:
# vm_memory_high_watermark.relative = 0.6
```

## Network and Connectivity Issues

### Service Communication Problems

**Symptoms:**
- Services can't reach each other
- API timeouts
- DNS resolution failures
- Load balancer issues

**Network Diagnostics:**
```bash
# Test service connectivity
docker-compose exec analysis_service ping tracklist_service
docker-compose exec analysis_service curl -f http://tracklist_service:8002/health

# Check Docker network
docker network ls
docker network inspect tracktion_default

# Test external connectivity
docker-compose exec analysis_service ping 8.8.8.8
docker-compose exec analysis_service curl -f https://api.external-service.com/health

# Check DNS resolution
docker-compose exec analysis_service nslookup postgres
docker-compose exec analysis_service cat /etc/resolv.conf
```

**Solutions:**
```bash
# 1. Restart Docker networking
docker-compose down
docker network prune
docker-compose up

# 2. Check firewall rules
sudo iptables -L
sudo ufw status

# 3. Verify Docker daemon configuration
cat /etc/docker/daemon.json
sudo systemctl restart docker
```

## Performance Issues

### System Resource Problems

#### High CPU Usage

**Investigation:**
```bash
# Identify CPU-intensive processes
htop
top -p $(pgrep -d, -f tracktion)

# Check container resource usage
docker stats

# Profile service performance
docker-compose exec analysis_service py-spy top --pid 1

# Check for CPU throttling
docker-compose logs analysis_service | grep -i throttl
```

#### Memory Leaks

**Detection:**
```bash
# Monitor memory usage over time
while true; do
    docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"
    sleep 60
done

# Check for memory growth patterns
docker-compose exec analysis_service python -c "
import psutil
import time

process = psutil.Process()
for i in range(10):
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f'Memory usage: {memory_mb:.1f} MB')
    time.sleep(30)
"

# Use memory profiler for detailed analysis
docker-compose exec analysis_service python -m memory_profiler services/analysis_service/src/main.py
```

#### Disk Space Issues

**Monitoring:**
```bash
# Check disk usage
df -h
du -h /var/lib/docker/ | tail -20

# Find large files
find /tmp -size +100M -ls
find /var/log -name "*.log" -size +50M -ls

# Check Docker volume usage
docker system df -v
```

**Cleanup:**
```bash
# Clean up Docker resources
docker system prune -a --volumes

# Clean up application logs
find logs/ -name "*.log" -mtime +7 -delete

# Clean up temporary files
rm -rf /tmp/tracktion_*
rm -rf /tmp/audio_processing_*
```

## Emergency Procedures

### Service Recovery

#### Complete System Restart
```bash
# 1. Graceful shutdown
docker-compose stop

# 2. Check for stuck processes
docker ps -a
docker kill $(docker ps -aq) 2>/dev/null || true

# 3. Clean restart
docker-compose down --volumes
docker system prune -f
docker-compose up -d

# 4. Verify services
./scripts/health-check.sh
```

#### Database Recovery
```bash
# 1. Stop all services
docker-compose stop

# 2. Backup current state (if possible)
docker-compose exec postgres pg_dump -U tracktion tracktion_dev > emergency_backup.sql

# 3. Restart database
docker-compose restart postgres

# 4. Verify database integrity
docker-compose exec postgres psql -U tracktion -c "SELECT count(*) FROM tracks;"

# 5. Restart dependent services
docker-compose start analysis_service tracklist_service
```

### Data Recovery

#### Backup Restoration
```bash
# 1. Stop services
docker-compose stop

# 2. Restore database
docker-compose exec -T postgres psql -U tracktion tracktion_dev < backup.sql

# 3. Verify data integrity
docker-compose exec postgres psql -U tracktion -c "
SELECT
    table_name,
    (SELECT count(*) FROM information_schema.columns WHERE table_name = t.table_name) as columns,
    (SELECT count(*) FROM pg_stat_user_tables WHERE relname = t.table_name) as rows
FROM information_schema.tables t
WHERE table_schema = 'public'
ORDER BY table_name;
"

# 4. Restart services
docker-compose up -d
```

## Monitoring and Alerting Setup

### Key Metrics to Monitor

#### Application Metrics
```bash
# Queue depths
rabbitmqctl list_queues name messages

# Processing rates
grep "completed" logs/analysis_service.log | tail -100 | wc -l

# Error rates
grep "ERROR" logs/*.log | wc -l

# Response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8001/health
```

#### System Metrics
```bash
# CPU and Memory
free -m
vmstat 1 5

# Disk I/O
iostat -x 1 5

# Network
ss -tuln | grep LISTEN
netstat -i
```

### Automated Health Checks

#### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

set -e

echo "=== Tracktion Health Check ==="
echo "Timestamp: $(date)"
echo

# Check services
services=("analysis_service:8001" "tracklist_service:8002" "file_watcher:8003")
for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}

    if curl -f -s "http://localhost:${port}/health" > /dev/null; then
        echo "✅ ${name}: healthy"
    else
        echo "❌ ${name}: unhealthy"
        exit 1
    fi
done

# Check databases
if docker-compose exec -T postgres pg_isready -U tracktion > /dev/null; then
    echo "✅ PostgreSQL: healthy"
else
    echo "❌ PostgreSQL: unhealthy"
    exit 1
fi

if docker-compose exec redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis: healthy"
else
    echo "❌ Redis: unhealthy"
    exit 1
fi

# Check RabbitMQ
if docker-compose exec rabbitmq rabbitmqctl status > /dev/null 2>&1; then
    echo "✅ RabbitMQ: healthy"
else
    echo "❌ RabbitMQ: unhealthy"
    exit 1
fi

echo
echo "🎉 All systems healthy!"
```

This troubleshooting guide provides systematic approaches to diagnosing and resolving common issues in the Tracktion system. Use it as a reference during incidents and for routine maintenance tasks.
