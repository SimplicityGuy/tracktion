# Analysis Pipeline Tuning Guide

## Overview

This guide provides comprehensive tuning recommendations for the Tracktion analysis pipeline to achieve optimal performance for different scales of operation. The pipeline has been designed to process 1000+ music files per hour with proper configuration.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Configuration Parameters](#configuration-parameters)
- [Performance Tuning](#performance-tuning)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Hardware Requirements

### Minimum Configuration (100-500 files/hour)
Suitable for personal music collections and small-scale operations.

- **CPU**: 2 cores (Intel i3/AMD Ryzen 3 or equivalent)
- **RAM**: 4 GB
- **Storage**:
  - Type: SSD recommended (HDD acceptable)
  - Space: 100 GB minimum
  - IOPS: 500+ read, 200+ write
- **Network**: 10 Mbps (if using cloud storage)
- **Docker**: 2 GB allocated memory

**Recommended Settings**:
```bash
# Environment variables
ANALYSIS_MAX_WORKERS=2
ANALYSIS_BATCH_SIZE=5
QUEUE_MAX_SIZE=1000
ANALYSIS_TIMEOUT=60
MAX_MEMORY_MB=2048
```

### Standard Configuration (1,000-5,000 files/hour)
Recommended for music studios and medium-scale operations.

- **CPU**: 4 cores (Intel i5/AMD Ryzen 5 or equivalent)
- **RAM**: 8 GB
- **Storage**:
  - Type: NVMe SSD
  - Space: 500 GB
  - IOPS: 3000+ read, 1000+ write
- **Network**: 100 Mbps
- **Docker**: 4 GB allocated memory

**Recommended Settings**:
```bash
# Environment variables
ANALYSIS_MAX_WORKERS=4
ANALYSIS_BATCH_SIZE=10
QUEUE_MAX_SIZE=5000
ANALYSIS_TIMEOUT=30
MAX_MEMORY_MB=4096
CACHE_SIZE_MB=512
```

### High Performance (5,000-10,000 files/hour)
For professional music libraries and production environments.

- **CPU**: 8 cores (Intel i7/AMD Ryzen 7 or equivalent)
- **RAM**: 16 GB
- **Storage**:
  - Type: NVMe SSD (PCIe Gen 4)
  - Space: 1 TB
  - IOPS: 10000+ read, 5000+ write
  - Consider RAID 0 for maximum performance
- **Network**: 1 Gbps
- **Docker**: 8 GB allocated memory

**Recommended Settings**:
```bash
# Environment variables
ANALYSIS_MAX_WORKERS=8
ANALYSIS_BATCH_SIZE=20
QUEUE_MAX_SIZE=10000
ANALYSIS_TIMEOUT=20
MAX_MEMORY_MB=8192
CACHE_SIZE_MB=1024
PARALLEL_ANALYSIS=true
```

### Enterprise Configuration (10,000+ files/hour)
For large-scale deployments and commercial operations.

- **CPU**: 16+ cores (Intel Xeon/AMD EPYC or equivalent)
- **RAM**: 32+ GB
- **Storage**:
  - Type: NVMe SSD RAID 10
  - Space: 2+ TB
  - IOPS: 50000+ read, 20000+ write
  - Consider dedicated storage array
- **Network**: 10 Gbps
- **Docker**: 16+ GB allocated memory

**Recommended Settings**:
```bash
# Environment variables
ANALYSIS_MAX_WORKERS=16
ANALYSIS_BATCH_SIZE=50
QUEUE_MAX_SIZE=50000
ANALYSIS_TIMEOUT=15
MAX_MEMORY_MB=16384
CACHE_SIZE_MB=4096
PARALLEL_ANALYSIS=true
DISTRIBUTED_MODE=true
```

## Configuration Parameters

### Core Parameters

#### Worker Configuration
```bash
# Number of concurrent analysis workers
ANALYSIS_MAX_WORKERS=4  # Should not exceed CPU cores

# Batch size for processing
ANALYSIS_BATCH_SIZE=10  # Larger = better throughput, more memory

# Queue configuration
QUEUE_MAX_SIZE=10000  # Maximum items in queue
QUEUE_PRIORITY_LEVELS=3  # Number of priority levels
```

#### Timeout Configuration
```bash
# Analysis timeout per file (seconds)
ANALYSIS_TIMEOUT=30  # Increase for large files

# Health check timeout
HEALTH_CHECK_TIMEOUT=5

# Database connection timeout
DB_CONNECT_TIMEOUT=10

# Message queue timeout
RABBITMQ_TIMEOUT=30
```

#### Memory Configuration
```bash
# Maximum memory usage (MB)
MAX_MEMORY_MB=8192

# Cache size for results
CACHE_SIZE_MB=512

# Redis memory limit
REDIS_MAXMEMORY=1gb

# PostgreSQL shared buffers
POSTGRES_SHARED_BUFFERS=2GB
```

### Database Tuning

#### PostgreSQL Optimization
```sql
-- postgresql.conf adjustments
shared_buffers = 2GB  -- 25% of RAM
effective_cache_size = 6GB  -- 75% of RAM
maintenance_work_mem = 512MB
work_mem = 50MB
max_connections = 200
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1  -- For SSD
```

#### Neo4j Optimization
```properties
# neo4j.conf adjustments
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
dbms.memory.pagecache.size=2g
dbms.connector.bolt.thread_pool_max_size=400
dbms.checkpoint.interval.time=15m
```

#### Redis Optimization
```conf
# redis.conf adjustments
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for performance
tcp-keepalive 60
tcp-backlog 511
```

### Message Queue Tuning

#### RabbitMQ Configuration
```bash
# Environment variables
RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.6
RABBITMQ_DISK_FREE_LIMIT=5GB

# rabbitmq.conf
vm_memory_high_watermark.relative = 0.6
disk_free_limit.absolute = 5GB
channel_max = 2048
heartbeat = 60
consumer_timeout = 3600000  # 1 hour

# Enable lazy queues for large volumes
queue_mode = lazy
```

## Performance Tuning

### CPU Optimization

#### Thread Pool Sizing
```python
# Optimal worker count calculation
import os

cpu_count = os.cpu_count()
optimal_workers = min(cpu_count * 2, 32)  # 2x CPU cores, max 32

# For I/O bound operations
io_workers = cpu_count * 4

# For CPU bound operations
cpu_workers = cpu_count
```

#### CPU Affinity
```bash
# Pin workers to specific CPU cores
taskset -c 0-3 python analysis_worker.py  # Use cores 0-3
```

### Memory Optimization

#### Garbage Collection Tuning
```python
# Python GC optimization
import gc

# Reduce GC frequency for batch processing
gc.set_threshold(700, 10, 10)

# Disable GC during critical sections
gc.disable()
# ... critical processing ...
gc.enable()
```

#### Memory Profiling
```bash
# Monitor memory usage
python -m memory_profiler analysis_service.py

# Track memory leaks
python -m tracemalloc analysis_service.py
```

### I/O Optimization

#### File System Tuning
```bash
# Linux kernel parameters (/etc/sysctl.conf)
vm.swappiness=10  # Reduce swap usage
vm.dirty_ratio=15
vm.dirty_background_ratio=5
fs.file-max=2097152

# Mount options for data partition
mount -o noatime,nodiratime,nobarrier /dev/nvme0n1 /data
```

#### Storage Configuration
```bash
# RAID configuration for performance
mdadm --create /dev/md0 --level=0 --raid-devices=2 /dev/nvme0n1 /dev/nvme1n1

# File system optimization
mkfs.ext4 -E stride=128,stripe-width=256 /dev/md0
```

### Network Optimization

#### TCP Tuning
```bash
# /etc/sysctl.conf
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
net.ipv4.tcp_congestion_control=bbr
net.core.netdev_max_backlog=5000
```

## Monitoring Setup

### Prometheus Configuration

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'analysis_service'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'

  - job_name: 'rabbitmq'
    static_configs:
      - targets: ['localhost:15692']

  - job_name: 'postgresql'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### Key Metrics to Monitor

#### Application Metrics
```
# Throughput
rate(analysis_files_processed_total[5m])  # Files per second

# Latency
histogram_quantile(0.95, analysis_processing_duration_seconds)  # P95 latency

# Queue depth
analysis_queue_depth

# Error rate
rate(analysis_failures_total[5m]) / rate(analysis_files_processed_total[5m])
```

#### System Metrics
```
# CPU usage
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes

# Disk I/O
rate(node_disk_read_bytes_total[5m])
rate(node_disk_written_bytes_total[5m])

# Network I/O
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])
```

### Grafana Dashboard

Import the dashboard from `infrastructure/monitoring/grafana/analysis-pipeline-dashboard.json` for comprehensive visualization of:
- Real-time throughput graphs
- Queue depth visualization
- Error rate tracking
- Resource utilization
- Performance trends

### Alert Rules

#### prometheus-alerts.yml
```yaml
groups:
  - name: analysis_pipeline
    rules:
      - alert: HighErrorRate
        expr: rate(analysis_failures_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"

      - alert: QueueBacklog
        expr: analysis_queue_depth > 10000
        for: 10m
        annotations:
          summary: "Large queue backlog"
          description: "Queue depth exceeds 10,000 items"

      - alert: SlowProcessing
        expr: histogram_quantile(0.95, analysis_processing_duration_seconds) > 60
        for: 5m
        annotations:
          summary: "Slow file processing"
          description: "P95 processing time exceeds 60 seconds"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 8192
        for: 5m
        annotations:
          summary: "High memory usage"
          description: "Process using more than 8GB of memory"
```

## Troubleshooting

### Common Performance Issues

#### 1. Low Throughput (<500 files/hour)

**Symptoms**:
- Processing rate below expectations
- Queue not being consumed quickly

**Diagnosis**:
```bash
# Check worker status
curl http://localhost:8080/metrics | grep analysis_workers

# Check database performance
psql -c "SELECT * FROM pg_stat_activity WHERE state != 'idle';"

# Check I/O wait
iostat -x 1
```

**Solutions**:
- Increase `ANALYSIS_MAX_WORKERS`
- Optimize database queries (add indexes)
- Upgrade to SSD storage
- Check network latency to external services

#### 2. High Memory Usage

**Symptoms**:
- Out of memory errors
- System swapping
- Slow response times

**Diagnosis**:
```bash
# Check memory usage by component
docker stats

# Find memory leaks
python -m tracemalloc --traceback=10 analysis_service.py

# Check Redis memory
redis-cli INFO memory
```

**Solutions**:
- Reduce `ANALYSIS_BATCH_SIZE`
- Enable result streaming
- Implement periodic garbage collection
- Increase swap space as temporary measure
- Configure Redis maxmemory policy

#### 3. Queue Backlog

**Symptoms**:
- Queue depth continuously increasing
- Processing lag increasing
- Memory pressure from queue

**Diagnosis**:
```bash
# Check queue depth
curl http://localhost:15672/api/queues/%2F/analysis_queue | jq .messages

# Check processing rate
curl http://localhost:8080/metrics | grep analysis_files_processed_total
```

**Solutions**:
- Scale workers horizontally
- Implement priority queue processing
- Add circuit breakers for slow operations
- Enable batch acknowledgments
- Consider distributed processing

#### 4. Database Bottlenecks

**Symptoms**:
- High database CPU usage
- Slow queries in logs
- Connection pool exhaustion

**Diagnosis**:
```sql
-- PostgreSQL slow queries
SELECT * FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC;

-- Check connections
SELECT count(*) FROM pg_stat_activity;

-- Neo4j query analysis
CALL dbms.listQueries() YIELD query, elapsedTimeMillis
WHERE elapsedTimeMillis > 1000;
```

**Solutions**:
- Add missing indexes
- Optimize query patterns
- Increase connection pool size
- Implement query caching
- Consider read replicas

#### 5. Network Issues

**Symptoms**:
- Intermittent failures
- High latency
- Connection timeouts

**Diagnosis**:
```bash
# Check network latency
ping -c 100 database-host

# Monitor network errors
netstat -s | grep -i error

# Check bandwidth usage
iftop -i eth0
```

**Solutions**:
- Implement retry logic with exponential backoff
- Use connection pooling
- Enable TCP keepalive
- Consider local caching
- Optimize payload sizes

### Performance Debugging Tools

#### Profiling Commands
```bash
# CPU profiling
py-spy record -o profile.svg -- python analysis_service.py

# Memory profiling
memray run analysis_service.py
memray flamegraph output.bin

# I/O profiling
strace -c python analysis_service.py

# Network profiling
tcpdump -i any -w capture.pcap port 5432
```

#### Benchmarking Tools
```bash
# Load testing
locust -f tests/performance/locustfile.py --host=http://localhost:8080

# Database benchmarking
pgbench -i -s 50 tracktion_db
pgbench -c 10 -j 2 -t 1000 tracktion_db

# Queue benchmarking
rabbitmq-perf-test -x 1 -y 2 -u analysis_queue -a
```

## Best Practices

### 1. Capacity Planning

- **Monitor trends**: Track throughput over time
- **Plan for peaks**: Size for 2x average load
- **Test limits**: Regular load testing
- **Set alerts**: Proactive monitoring
- **Document changes**: Track configuration evolution

### 2. Optimization Strategy

- **Measure first**: Profile before optimizing
- **Fix bottlenecks**: Address slowest components
- **Batch operations**: Reduce overhead
- **Cache strategically**: Cache expensive operations
- **Parallelize**: Use all available resources

### 3. Operational Excellence

- **Automate scaling**: Use auto-scaling groups
- **Regular maintenance**: Schedule optimization windows
- **Backup configuration**: Version control settings
- **Test changes**: Use staging environment
- **Document everything**: Maintain runbooks

### 4. Security Considerations

- **Limit resources**: Set memory/CPU limits
- **Network isolation**: Use private networks
- **Encrypt data**: TLS for all connections
- **Audit logging**: Track all operations
- **Regular updates**: Keep dependencies current

### 5. Cost Optimization

- **Right-sizing**: Match resources to load
- **Spot instances**: Use for batch processing
- **Reserved capacity**: For predictable loads
- **Compression**: Reduce storage/network costs
- **Lifecycle policies**: Archive old data

## Scaling Strategies

### Vertical Scaling
- Increase CPU/RAM on existing servers
- Upgrade storage to faster SSDs
- Best for: Simple deployments, quick wins

### Horizontal Scaling
- Add more worker nodes
- Implement load balancing
- Best for: High availability, large scale

### Distributed Processing
- Use message queue for distribution
- Implement map-reduce patterns
- Best for: Massive scale, parallel workloads

### Cloud-Native Scaling
- Kubernetes autoscaling
- Serverless functions for peaks
- Best for: Variable loads, cost optimization

## Conclusion

The Tracktion analysis pipeline is designed for scalability and can be tuned to handle workloads from personal collections to enterprise-scale music libraries. Key success factors:

1. **Match hardware to workload**: Use the configuration guidelines
2. **Monitor continuously**: Track metrics and trends
3. **Optimize iteratively**: Start simple, improve based on data
4. **Plan for growth**: Design for 2x current capacity
5. **Automate operations**: Reduce manual intervention

For additional support, consult the [Performance Test Documentation](../tests/performance/README.md) and the [Architecture Documentation](./architecture/).

## Appendix

### Quick Reference Card

| Scale | Files/Hour | Workers | Batch Size | RAM | CPU |
|-------|------------|---------|------------|-----|-----|
| Small | 100-500 | 2 | 5 | 4GB | 2 cores |
| Medium | 1,000-5,000 | 4 | 10 | 8GB | 4 cores |
| Large | 5,000-10,000 | 8 | 20 | 16GB | 8 cores |
| Enterprise | 10,000+ | 16+ | 50 | 32GB+ | 16+ cores |

### Environment Template

Save as `.env` and adjust values:

```bash
# Analysis Service Configuration
ANALYSIS_MAX_WORKERS=4
ANALYSIS_BATCH_SIZE=10
ANALYSIS_TIMEOUT=30
QUEUE_MAX_SIZE=10000
QUEUE_PRIORITY_LEVELS=3
MAX_MEMORY_MB=8192
CACHE_SIZE_MB=512

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/tracktion
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379

# Message Queue Configuration
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_PREFETCH_COUNT=10

# Monitoring
PROMETHEUS_PORT=8080
HEALTH_CHECK_PORT=8081
LOG_LEVEL=INFO
```
