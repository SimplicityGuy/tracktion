# Monitoring Dashboard Setup Guide

Complete setup and usage guide for the Resilient Scraping System monitoring dashboard.

## Overview

The monitoring dashboard provides real-time visibility into the resilient scraping system with:
- **System Health Monitoring**: Overall health scores and status
- **Parser Performance**: Success rates and error tracking per page type
- **Cache Analytics**: Hit rates, fallback usage, and performance metrics
- **Alert Management**: Real-time alerts with severity-based routing
- **Issue Detection**: Automated issue identification with recommendations
- **Performance Trends**: Historical analysis and trend visualization

## Components

### 1. Core Dashboard (`dashboard.py`)
- Metrics collection and aggregation
- Health status calculation
- Issue detection and prioritization
- Performance trend analysis
- Metrics export capabilities

### 2. Web Interface (`web_dashboard.py`)
- HTML dashboard with real-time updates
- REST API for programmatic access
- Auto-refresh and manual refresh capabilities
- Mobile-responsive interface

## Quick Start

### Prerequisites
```bash
# Install required dependencies
pip install aiohttp aiohttp-cors redis
```

### Basic Setup
```python
from services.tracklist_service.src.monitoring.dashboard import MonitoringDashboard
from services.tracklist_service.src.monitoring.alert_manager import AlertManager
from services.tracklist_service.src.monitoring.structure_monitor import StructureMonitor
from services.tracklist_service.src.cache.fallback_cache import FallbackCache
from redis.asyncio import Redis

# Initialize Redis connection
redis_client = Redis(host="localhost", port=6379, decode_responses=True)

# Create core components
alert_manager = AlertManager(
    redis_client=redis_client,
    discord_webhook_url="https://discord.com/api/webhooks/..."
)

structure_monitor = StructureMonitor()
fallback_cache = FallbackCache(redis_client=redis_client)

# Create dashboard
dashboard = MonitoringDashboard(
    alert_manager=alert_manager,
    structure_monitor=structure_monitor,
    fallback_cache=fallback_cache
)
```

### Web Dashboard
```python
from services.tracklist_service.src.monitoring.web_dashboard import WebMonitoringDashboard

# Create web dashboard
web_dashboard = WebMonitoringDashboard(dashboard)

# Start server
await web_dashboard.start_server(host="0.0.0.0", port=8080)
```

### Command Line Usage
```bash
# Start web dashboard
python -m services.tracklist_service.src.monitoring.web_dashboard

# Access dashboard
open http://localhost:8080
```

## Dashboard Features

### System Health Overview
- **Overall Health Score**: Calculated based on parser health, cache performance, and alert volume
- **Status Indicators**: Healthy (green), Warning (yellow), Degraded (orange), Critical (red)
- **Uptime Metrics**: Parser availability, cache hit rates, and service availability

### Parser Health Monitoring
```python
# Check parser health for specific page type
health = await dashboard.get_system_metrics(["tracklist", "artist"])

for page_type, metrics in health.parser_health.items():
    print(f"{page_type}: {metrics['success_rate']:.1%} success rate")
    if not metrics['healthy']:
        print(f"  Issues: {metrics.get('anomaly_count', 0)} anomalies")
```

### Cache Performance Analytics
- **Hit Rate**: Percentage of cache hits vs total requests
- **Fallback Usage**: Frequency of fallback cache utilization
- **Performance Trends**: Historical cache performance data
- **Memory Usage**: Cache size and memory consumption

### Alert Management
```python
# Get recent alerts
recent_alerts = await dashboard.alert_manager.get_recent_alerts(limit=50)

# Get active issues requiring attention
active_issues = await dashboard.get_active_issues()

for issue in active_issues:
    print(f"[{issue['priority'].upper()}] {issue['title']}")
    print(f"  Recommendations: {', '.join(issue['recommendations'][:2])}")
```

### Performance Trends
```python
# Get 24-hour performance trends
trends = await dashboard.get_performance_trends(hours=24)

print(f"Data points: {trends['data_points']}")
print(f"Average cache hit rate: {sum(trends['cache_hit_rates']) / len(trends['cache_hit_rates']):.1%}")

# Export trends data
csv_data = await dashboard.export_metrics("csv")
with open("performance_metrics.csv", "w") as f:
    f.write(csv_data)
```

## API Endpoints

### Health Status
```bash
GET /api/health
```
Returns overall system health with status, score, and issues.

Example response:
```json
{
  "status": "healthy",
  "health_score": 0.95,
  "timestamp": "2025-08-22T10:30:00Z",
  "issues": [],
  "uptime_metrics": {
    "parsers_healthy": 4,
    "total_parsers": 4,
    "cache_hit_rate": 0.82,
    "alerts_24h": 3
  }
}
```

### System Metrics
```bash
GET /api/metrics?page_types=tracklist,artist
```
Returns detailed system metrics including parser health, cache stats, and alerts.

### Active Issues
```bash
GET /api/issues
```
Returns list of active issues requiring attention.

Example response:
```json
[
  {
    "type": "parser_health",
    "priority": "medium",
    "title": "Parser health degraded: tracklist",
    "description": "Success rate: 78.5%",
    "recommendations": [
      "Check for recent site changes",
      "Review extraction selectors"
    ],
    "page_type": "tracklist"
  }
]
```

### Performance Trends
```bash
GET /api/trends?hours=24
```
Returns performance trend data over specified time period.

### Metrics Export
```bash
GET /api/export?format=json
GET /api/export?format=csv
```
Export metrics in JSON or CSV format.

## Configuration

### Environment Variables
```bash
# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_AUTO_REFRESH=30

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password

# Alert Configuration
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Monitoring Thresholds
HEALTH_CHECK_INTERVAL=300
CACHE_HIT_RATE_THRESHOLD=0.7
ALERT_VOLUME_THRESHOLD=20
PARSER_SUCCESS_RATE_THRESHOLD=0.8
```

### Dashboard Configuration
```python
dashboard_config = {
    "metrics_history_size": 1000,  # Number of metric snapshots to keep
    "auto_refresh_interval": 30,   # Seconds between auto-refresh
    "health_score_weights": {
        "parser_health": 0.4,
        "cache_performance": 0.3,
        "alert_volume": 0.2,
        "structure_changes": 0.1
    },
    "alert_thresholds": {
        "critical_health_score": 0.5,
        "warning_health_score": 0.7,
        "cache_hit_rate_warning": 0.7,
        "max_alerts_24h": 20
    }
}
```

## Customization

### Custom Metrics
```python
class CustomDashboard(MonitoringDashboard):
    async def get_custom_metrics(self):
        """Add custom metrics collection."""
        custom_data = {
            "database_connections": await self.check_db_connections(),
            "external_api_health": await self.check_external_apis(),
            "queue_sizes": await self.check_queue_sizes()
        }
        return custom_data

    async def check_db_connections(self):
        # Custom database health check
        return {"active": 10, "max": 100, "status": "healthy"}
```

### Custom Alerts
```python
# Register custom anomaly detector
async def database_connection_detector(page_type):
    connections = await check_database_connections()
    if connections["active"] / connections["max"] > 0.9:
        return f"High database connection usage: {connections['active']}/{connections['max']}"
    return None

alert_manager.register_anomaly_detector("database", database_connection_detector)
```

### Custom Web Interface
```python
class CustomWebDashboard(WebMonitoringDashboard):
    async def custom_endpoint(self, request):
        """Add custom API endpoint."""
        custom_data = await self.dashboard.get_custom_metrics()
        return web.json_response(custom_data)

    def _create_app(self):
        app = super()._create_app()
        app.router.add_get("/api/custom", self.custom_endpoint)
        return app
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "services.tracklist_service.src.monitoring.web_dashboard"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  dashboard:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379
      - DASHBOARD_HOST=0.0.0.0
      - DASHBOARD_PORT=8080
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: monitoring-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: monitoring-dashboard
  template:
    metadata:
      labels:
        app: monitoring-dashboard
    spec:
      containers:
      - name: dashboard
        image: tracktion/monitoring-dashboard:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DASHBOARD_HOST
          value: "0.0.0.0"
---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  selector:
    app: monitoring-dashboard
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Monitoring & Alerting

### Health Checks
```bash
# Health check endpoint for load balancers
curl http://localhost:8080/api/health

# Detailed status check
curl "http://localhost:8080/api/metrics?page_types=tracklist"
```

### Prometheus Integration
```python
from prometheus_client import start_http_server, Gauge, Counter

# Create Prometheus metrics
health_score_gauge = Gauge('system_health_score', 'Overall system health score')
cache_hit_rate_gauge = Gauge('cache_hit_rate', 'Cache hit rate percentage')
alerts_counter = Counter('alerts_total', 'Total alerts', ['severity'])

# Update metrics in dashboard
async def update_prometheus_metrics():
    health = await dashboard.get_health_status()
    metrics = await dashboard.get_system_metrics()

    health_score_gauge.set(health['health_score'])
    cache_hit_rate_gauge.set(metrics.cache_stats['hit_rate'])

    for severity, count in metrics.alert_summary.items():
        alerts_counter.labels(severity=severity).inc(count)

# Start Prometheus metrics server
start_http_server(9090)
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Resilient Scraping System",
    "panels": [
      {
        "title": "System Health Score",
        "type": "stat",
        "targets": [
          {
            "expr": "system_health_score",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "cache_hit_rate",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check server status
curl -I http://localhost:8080

# Check logs
python -m services.tracklist_service.src.monitoring.web_dashboard

# Verify Redis connection
redis-cli ping
```

#### High Memory Usage
```python
# Reduce metrics history size
dashboard._max_history = 100

# Clear old metrics
dashboard._metrics_history = dashboard._metrics_history[-100:]
```

#### Slow API Responses
```python
# Enable caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
async def cached_health_check(page_type, timestamp_minute):
    return await alert_manager.check_parser_health(page_type)
```

### Performance Optimization
- Use Redis for metric storage and caching
- Implement request rate limiting
- Enable gzip compression for API responses
- Use connection pooling for database connections
- Implement metric sampling for high-volume data

### Security Considerations
- Enable HTTPS in production
- Implement authentication and authorization
- Use environment variables for sensitive configuration
- Regularly update dependencies for security patches
- Monitor dashboard access logs

## Support

For issues or questions:
1. Check the dashboard logs for error details
2. Verify Redis connectivity and performance
3. Review configuration settings
4. Test API endpoints individually
5. Check system resource usage (CPU, memory, network)
