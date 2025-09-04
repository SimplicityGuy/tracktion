# Monitoring Setup Guide

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Architecture](#monitoring-architecture)
3. [Application Metrics](#application-metrics)
4. [Infrastructure Monitoring](#infrastructure-monitoring)
5. [Alerting Configuration](#alerting-configuration)
6. [Log Aggregation](#log-aggregation)
7. [Monitoring Tools Setup](#monitoring-tools-setup)
8. [Dashboards](#dashboards)
9. [Performance Monitoring](#performance-monitoring)
10. [Health Checks](#health-checks)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

## Overview

This guide provides comprehensive instructions for setting up monitoring, alerting, and observability for the Tracktion system. The monitoring stack is designed to provide full visibility into application performance, infrastructure health, and business metrics.

### Monitoring Objectives

- **Reliability**: Detect and prevent system failures
- **Performance**: Monitor response times and resource usage
- **Security**: Track security events and anomalies
- **Business Metrics**: Monitor key performance indicators
- **Capacity Planning**: Track resource utilization trends

### Key Metrics Categories

1. **Golden Signals**: Latency, Traffic, Errors, Saturation
2. **Application Metrics**: Business logic, performance, errors
3. **Infrastructure Metrics**: CPU, memory, disk, network
4. **Security Metrics**: Authentication, authorization, threats
5. **Business Metrics**: User activity, data processing, revenue

## Monitoring Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │────│   Prometheus    │────│    Grafana      │
│                 │    │   (Metrics)     │    │  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐               │
         │              │   AlertManager  │               │
         │              │   (Alerting)    │               │
         │              └─────────────────┘               │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ELK Stack     │    │   PagerDuty     │    │   Slack/Teams   │
│   (Logging)     │    │  (Incidents)    │    │ (Notifications) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Overview

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notification
- **Elasticsearch**: Log storage and search
- **Logstash**: Log processing pipeline
- **Kibana**: Log visualization and analysis
- **Jaeger**: Distributed tracing
- **Node Exporter**: System metrics collection

## Application Metrics

### Core Application Metrics

#### HTTP Request Metrics
```python
# Example: FastAPI metrics implementation
from prometheus_client import Counter, Histogram, Gauge
import time

# Request counter
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# Request duration
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Active connections
ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Active HTTP connections'
)
```

#### Business Logic Metrics
```python
# Analysis service metrics
AUDIO_FILES_PROCESSED = Counter(
    'audio_files_processed_total',
    'Total audio files processed',
    ['status', 'format']
)

ANALYSIS_DURATION = Histogram(
    'audio_analysis_duration_seconds',
    'Audio analysis processing time',
    ['analysis_type']
)

QUEUE_SIZE = Gauge(
    'processing_queue_size',
    'Current processing queue size'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections'
)
```

### Service-Specific Metrics

#### Analysis Service
- `analysis_requests_total`: Total analysis requests
- `analysis_duration_seconds`: Analysis processing time
- `analysis_errors_total`: Analysis failures
- `audio_files_queued`: Files waiting for processing
- `feature_extraction_duration`: Time for feature extraction

#### File Watcher Service
- `files_watched_total`: Total files being monitored
- `file_events_total`: File system events detected
- `processing_lag_seconds`: Delay in file processing
- `directory_scan_duration`: Time to scan directories

#### Tracklist Service
- `tracklist_matches_total`: Successful track matches
- `tracklist_search_duration`: Search operation time
- `tracklist_cache_hits`: Cache hit rate
- `database_query_duration`: Database query performance

#### Notification Service
- `notifications_sent_total`: Total notifications sent
- `notification_delivery_duration`: Notification delivery time
- `notification_failures_total`: Failed notifications
- `webhook_response_time`: Webhook endpoint response time

### Metric Collection Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'tracktion-analysis'
    static_configs:
      - targets: ['analysis-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'tracktion-file-watcher'
    static_configs:
      - targets: ['file-watcher:8001']

  - job_name: 'tracktion-tracklist'
    static_configs:
      - targets: ['tracklist-service:8002']

  - job_name: 'tracktion-notifications'
    static_configs:
      - targets: ['notification-service:8003']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## Infrastructure Monitoring

### System Metrics

#### Server Metrics
- **CPU Usage**: `cpu_usage_percent`
- **Memory Usage**: `memory_usage_percent`
- **Disk Usage**: `disk_usage_percent`
- **Network I/O**: `network_bytes_total`
- **Load Average**: `system_load_average`
- **Open Files**: `process_open_fds`

#### Docker Metrics
```yaml
# docker-compose monitoring addition
services:
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
```

#### Database Metrics (PostgreSQL)
```sql
-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Key metrics to monitor:
-- 1. Connection count
SELECT count(*) FROM pg_stat_activity;

-- 2. Query performance
SELECT query, total_time, calls, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC LIMIT 10;

-- 3. Database size
SELECT pg_size_pretty(pg_database_size('tracktion'));

-- 4. Table sizes
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(tablename::regclass) DESC;
```

### Infrastructure Alerting Rules

```yaml
# alert_rules.yml
groups:
  - name: infrastructure
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: DiskSpaceLow
        expr: disk_usage_percent > 90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Disk space critically low"
          description: "Disk usage is above 90%"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
```

## Alerting Configuration

### Alert Categories

#### Critical Alerts (Immediate Response)
- Service down
- Database connection failure
- Disk space critically low (>95%)
- High error rate (>5%)
- Security breach detected

#### Warning Alerts (Action Required)
- High resource usage (CPU >80%, Memory >85%)
- Slow response times (>2s average)
- Queue buildup
- Certificate expiring (30 days)

#### Info Alerts (Monitoring)
- Deployment completed
- Configuration changes
- Scheduled maintenance
- Performance improvements

### AlertManager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@tracktion.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#critical-alerts'
        title: 'Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#warnings'
        title: 'Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'web.hook'
    webhook_configs:
      - url: 'http://127.0.0.1:5001/'
```

### Application-Specific Alerts

```yaml
# Application alert rules
groups:
  - name: tracktion-application
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for {{ $labels.service }}"

      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow response time"
          description: "95th percentile response time is above 2 seconds"

      - alert: QueueBacklog
        expr: processing_queue_size > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Processing queue backlog"
          description: "Queue size is {{ $value }} items"

      - alert: DatabaseConnectionLoss
        expr: database_connections_active == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection lost"
          description: "No active database connections detected"
```

## Log Aggregation

### ELK Stack Setup

#### Elasticsearch Configuration
```yaml
# docker-compose.yml - Elasticsearch
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
  container_name: elasticsearch
  environment:
    - discovery.type=single-node
    - ES_JAVA_OPTS=-Xms1g -Xmx1g
    - xpack.security.enabled=false
  volumes:
    - elasticsearch_data:/usr/share/elasticsearch/data
  ports:
    - "9200:9200"
  networks:
    - monitoring
```

#### Logstash Configuration
```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "analysis" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{DATA:logger} %{GREEDYDATA:message}" }
    }
  }

  if [fields][service] == "tracklist" {
    json {
      source => "message"
    }
  }

  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "tracktion-logs-%{+YYYY.MM.dd}"
  }
}
```

#### Filebeat Configuration
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/analysis-service/*.log
  fields:
    service: analysis
    environment: production

- type: log
  enabled: true
  paths:
    - /app/logs/file-watcher/*.log
  fields:
    service: file-watcher
    environment: production

output.logstash:
  hosts: ["logstash:5044"]

processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded
```

### Structured Logging Implementation

#### Python Logging Configuration
```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration

        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/application.log')
    ]
)

# Add JSON formatter
for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())
```

## Monitoring Tools Setup

### Docker Compose Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

volumes:
  grafana_data:
  elasticsearch_data:

networks:
  monitoring:
    driver: bridge
```

### Setup Scripts

#### Monitoring Setup Script
```bash
#!/bin/bash
# setup-monitoring.sh

set -e

echo "Setting up Tracktion monitoring stack..."

# Create monitoring directories
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources},alertmanager}

# Create Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'tracktion-services'
    static_configs:
      - targets: ['analysis-service:8000', 'file-watcher:8001', 'tracklist-service:8002', 'notification-service:8003']
    metrics_path: '/metrics'
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

# Create Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

echo "Monitoring stack setup complete!"
echo "Grafana: http://localhost:3000 (admin/admin123)"
echo "Prometheus: http://localhost:9090"
echo "AlertManager: http://localhost:9093"
```

## Dashboards

### Grafana Dashboard Templates

#### System Overview Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "Tracktion System Overview",
    "tags": ["tracktion"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}} - {{method}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

#### Application Performance Dashboard
```json
{
  "dashboard": {
    "title": "Tracktion Application Performance",
    "panels": [
      {
        "title": "Audio Files Processed",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(audio_files_processed_total[1h])",
            "legendFormat": "Files/hour"
          }
        ]
      },
      {
        "title": "Processing Queue Size",
        "type": "graph",
        "targets": [
          {
            "expr": "processing_queue_size",
            "legendFormat": "Queue Size"
          }
        ]
      },
      {
        "title": "Analysis Duration",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(audio_analysis_duration_seconds_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ]
  }
}
```

### Key Performance Indicators (KPIs)

#### Business Metrics
- Files processed per hour
- Average processing time
- User satisfaction score
- System availability (uptime)

#### Technical Metrics
- Request rate (RPS)
- Response time (95th percentile)
- Error rate (%)
- Resource utilization

#### Alert Coverage
- Mean time to detection (MTTD)
- Mean time to resolution (MTTR)
- Alert fatigue ratio
- False positive rate

## Performance Monitoring

### Application Performance Monitoring (APM)

#### Distributed Tracing Setup
```python
# tracing_config.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage example
@tracer.start_as_current_span("audio_analysis")
def analyze_audio_file(file_path: str):
    with tracer.start_as_current_span("load_file") as span:
        span.set_attribute("file.path", file_path)
        # Load file logic

    with tracer.start_as_current_span("extract_features"):
        # Feature extraction logic
        pass

    with tracer.start_as_current_span("classify_audio"):
        # Classification logic
        pass
```

#### Performance Metrics Collection
```python
# performance_metrics.py
from prometheus_client import Histogram, Counter
import time
import functools

# Performance decorators
REQUEST_TIME = Histogram(
    'request_processing_seconds',
    'Time spent processing request',
    ['endpoint', 'method']
)

def track_performance(endpoint: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_TIME.labels(endpoint=endpoint, method='success').observe(time.time() - start_time)
                return result
            except Exception as e:
                REQUEST_TIME.labels(endpoint=endpoint, method='error').observe(time.time() - start_time)
                raise
        return wrapper
    return decorator

# Usage
@track_performance('analyze_audio')
def analyze_audio(file_path: str):
    # Analysis logic
    pass
```

### Database Performance Monitoring

```sql
-- Database monitoring queries

-- Slow queries
SELECT query, total_time, calls, mean_time, stddev_time
FROM pg_stat_statements
WHERE mean_time > 1000  -- queries taking more than 1 second
ORDER BY total_time DESC;

-- Index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE tablename IN ('audio_files', 'tracklists', 'analysis_results');

-- Connection monitoring
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

-- Table sizes and bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(tablename::regclass)) as total_size,
       pg_size_pretty(pg_relation_size(tablename::regclass)) as table_size,
       pg_size_pretty(pg_total_relation_size(tablename::regclass) - pg_relation_size(tablename::regclass)) as index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(tablename::regclass) DESC;
```

## Health Checks

### Service Health Endpoints

```python
# health_checks.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
import redis
import aiohttp
import asyncio

app = FastAPI()

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'external_api': self.check_external_api,
            'disk_space': self.check_disk_space,
            'memory': self.check_memory
        }

    async def check_database(self):
        """Check database connectivity"""
        try:
            async with get_db_session() as session:
                result = await session.execute(text("SELECT 1"))
                return {"status": "healthy", "response_time": "< 100ms"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_redis(self):
        """Check Redis connectivity"""
        try:
            redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
            redis_client.ping()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_external_api(self):
        """Check external API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.example.com/health', timeout=5) as response:
                    if response.status == 200:
                        return {"status": "healthy", "response_code": 200}
                    else:
                        return {"status": "degraded", "response_code": response.status}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_disk_space(self):
        """Check available disk space"""
        import shutil
        total, used, free = shutil.disk_usage('/')
        free_percent = (free / total) * 100

        if free_percent > 20:
            return {"status": "healthy", "free_space_percent": free_percent}
        elif free_percent > 10:
            return {"status": "warning", "free_space_percent": free_percent}
        else:
            return {"status": "critical", "free_space_percent": free_percent}

    async def check_memory(self):
        """Check memory usage"""
        import psutil
        memory = psutil.virtual_memory()

        if memory.percent < 80:
            return {"status": "healthy", "memory_usage_percent": memory.percent}
        elif memory.percent < 90:
            return {"status": "warning", "memory_usage_percent": memory.percent}
        else:
            return {"status": "critical", "memory_usage_percent": memory.percent}

health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    results = {}
    overall_status = "healthy"

    for check_name, check_func in health_checker.checks.items():
        try:
            result = await check_func()
            results[check_name] = result

            if result["status"] in ["unhealthy", "critical"]:
                overall_status = "unhealthy"
            elif result["status"] in ["warning", "degraded"] and overall_status == "healthy":
                overall_status = "degraded"

        except Exception as e:
            results[check_name] = {"status": "error", "error": str(e)}
            overall_status = "unhealthy"

    if overall_status != "healthy":
        raise HTTPException(status_code=503, detail={
            "status": overall_status,
            "checks": results
        })

    return {
        "status": overall_status,
        "checks": results,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    # Check if service can handle requests
    critical_checks = ['database', 'redis']

    for check_name in critical_checks:
        result = await health_checker.checks[check_name]()
        if result["status"] != "healthy":
            raise HTTPException(status_code=503, detail=f"{check_name} not ready")

    return {"status": "ready"}

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    # Simple check that service is responsive
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
```

### Kubernetes Health Check Configuration

```yaml
# k8s-health-checks.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: analysis-service
        image: tracktion/analysis-service:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Troubleshooting

### Common Monitoring Issues

#### High Memory Usage in Prometheus
**Symptoms**: Prometheus consuming excessive memory
**Solutions**:
```yaml
# Reduce retention period
prometheus:
  command:
    - '--storage.tsdb.retention.time=7d'  # Reduce from default 15d
    - '--storage.tsdb.retention.size=10GB'

# Optimize scrape intervals
scrape_configs:
  - job_name: 'non-critical'
    scrape_interval: 60s  # Increase interval for non-critical metrics
```

#### Missing Metrics
**Symptoms**: Metrics not appearing in Grafana
**Debugging Steps**:
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://service:8000/metrics

# Check Prometheus logs
docker logs prometheus

# Test metric query
curl -G http://localhost:9090/api/v1/query --data-urlencode 'query=up'
```

#### Alert Fatigue
**Symptoms**: Too many alerts, team ignoring notifications
**Solutions**:
- Adjust alert thresholds based on historical data
- Implement alert grouping and routing
- Use different severity levels
- Add alert dependencies
- Regular alert review and cleanup

### Performance Issues

#### Slow Grafana Dashboards
```yaml
# Optimize queries
# Bad: rate(http_requests_total[1h])
# Good: rate(http_requests_total[5m])

# Use recording rules for expensive queries
groups:
  - name: recording_rules
    interval: 30s
    rules:
      - record: tracktion:request_rate_5m
        expr: rate(http_requests_total[5m])

      - record: tracktion:error_rate_5m
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

#### Log Volume Issues
```yaml
# Logstash pipeline optimization
pipeline:
  batch.size: 1000
  batch.delay: 50
  workers: 4

# Elasticsearch optimization
indices:
  template:
    settings:
      number_of_shards: 1
      number_of_replicas: 0  # For single-node development
      refresh_interval: 30s
```

### Monitoring Best Practices

#### Metric Naming Convention
```python
# Good naming patterns
http_requests_total          # Counter with _total suffix
http_request_duration_seconds # Histogram with base unit
current_users               # Gauge without suffix
processing_queue_size       # Gauge for current state

# Service-specific prefixes
analysis_files_processed_total
file_watcher_events_total
tracklist_search_duration_seconds
```

#### Alert Design Principles

1. **Actionable**: Every alert should require specific action
2. **Contextual**: Include relevant information for debugging
3. **Prioritized**: Use severity levels appropriately
4. **Documented**: Link to runbooks or troubleshooting guides

```yaml
# Good alert example
- alert: DatabaseConnectionLoss
  expr: database_connections_active == 0
  for: 1m
  labels:
    severity: critical
    service: analysis
  annotations:
    summary: "Database connection lost for Analysis Service"
    description: "The Analysis Service has lost all database connections. Check database server health and network connectivity."
    runbook_url: "https://wiki.company.com/runbooks/database-connection-loss"
    dashboard_url: "https://grafana.company.com/d/database-overview"
```

#### Dashboard Design Guidelines

1. **User-Focused**: Design for the audience (developers, ops, business)
2. **Hierarchical**: Overview → Detailed → Diagnostic
3. **Consistent**: Use consistent colors, units, and layouts
4. **Annotated**: Add descriptions and context
5. **Actionable**: Include links to relevant tools and runbooks

## Best Practices

### Monitoring Strategy

1. **Start with the Golden Signals**: Latency, Traffic, Errors, Saturation
2. **Monitor User Experience**: Focus on what users actually experience
3. **Automate Everything**: Alerts, dashboards, remediation where possible
4. **Document Everything**: Runbooks, escalation procedures, system architecture
5. **Regular Reviews**: Weekly metric reviews, monthly alert tuning

### Operational Excellence

1. **Monitor the Monitors**: Ensure monitoring system health
2. **Test Alert Paths**: Regular alert testing and drill exercises
3. **Capacity Planning**: Proactive scaling based on trends
4. **Security Monitoring**: Include security metrics and alerts
5. **Cost Optimization**: Monitor and optimize monitoring costs

### Team Practices

1. **On-Call Rotation**: Shared responsibility for system health
2. **Post-Incident Reviews**: Learn from every incident
3. **Metrics-Driven Decisions**: Use data for all technical decisions
4. **Continuous Improvement**: Regular retrospectives and improvements
5. **Knowledge Sharing**: Document and share monitoring knowledge

This monitoring setup provides comprehensive visibility into the Tracktion system's health, performance, and business metrics. Regular review and optimization of monitoring configurations ensure the system continues to meet operational requirements as it scales.
