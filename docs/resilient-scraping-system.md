# Resilient Scraping System

A comprehensive infrastructure for handling site updates gracefully through intelligent monitoring, adaptive caching, and automated recovery mechanisms.

## Overview

The Resilient Scraping System provides a robust foundation for web scraping operations that can automatically detect and adapt to website changes, maintain service availability through intelligent caching strategies, and provide comprehensive monitoring and alerting.

## Core Components

### 1. Structure Monitor
**File**: `services/tracklist_service/src/monitoring/structure_monitor.py`

Continuously monitors website structure changes using content fingerprinting and similarity analysis.

#### Key Features
- **Content Fingerprinting**: Creates unique signatures for page structures
- **Change Detection**: Identifies structural modifications with severity assessment
- **Automated Validation**: Validates scraping selectors against new page structures
- **Performance Monitoring**: Tracks extraction success rates and response times

#### Usage
```python
from services.tracklist_service.src.monitoring.structure_monitor import StructureMonitor

monitor = StructureMonitor()
change_report = await monitor.detect_changes("tracklist", html_content, url)

if change_report.has_breaking_changes:
    print(f"Breaking changes detected: {change_report.changes}")
```

#### Configuration
- **similarity_threshold**: Minimum similarity for detecting changes (default: 0.8)
- **check_interval**: Frequency of structure validation (default: 3600 seconds)
- **max_fingerprint_age**: Maximum age for cached fingerprints (default: 7 days)

### 2. Alert Manager
**File**: `services/tracklist_service/src/monitoring/alert_manager.py`

Provides comprehensive alerting and health monitoring capabilities with Discord notifications.

#### Key Features
- **Discord Alerts**: Log, Discord, Dashboard notifications
- **Health Monitoring**: Parser success rate tracking and anomaly detection
- **Severity-Based Routing**: Automatic channel selection based on alert severity
- **Custom Anomaly Detectors**: Extensible framework for custom monitoring logic

#### Alert Severities & Channels
- **INFO/WARNING**: Log + Dashboard
- **ERROR**: Log + Dashboard + Discord
- **CRITICAL**: Log + Dashboard + Discord

#### Usage
```python
from services.tracklist_service.src.monitoring.alert_manager import AlertManager

alert_manager = AlertManager(
    redis_client=redis_client,
    discord_webhook_url="https://discord.com/api/webhooks/..."
)

# Monitor parser health
health = await alert_manager.check_parser_health("tracklist")
if health.requires_alert:
    await alert_manager.send_health_alert(health, "tracklist")

# Send custom alerts
await alert_manager.send_alert("error", "Extraction failed", ["discord", "log"])
```

#### Custom Anomaly Detection
```python
async def custom_detector(page_type):
    # Custom logic here
    if anomaly_detected:
        return "Description of anomaly"
    return None

alert_manager.register_anomaly_detector("custom", custom_detector)
```

### 3. Fallback Cache
**File**: `services/tracklist_service/src/cache/fallback_cache.py`

Multi-layered caching system with intelligent fallback strategies for maintaining service availability.

#### Cache Strategies
- **STRICT**: Only fresh, valid data
- **FLEXIBLE**: Allows stale data with quality scoring
- **FALLBACK**: Uses any available data when primary fails
- **PROGRESSIVE**: Adaptive strategy based on availability

#### Key Features
- **Multi-Layer Storage**: Memory + Redis with automatic tier management
- **Quality Scoring**: Data validity assessment based on age and source
- **Intelligent Expiration**: Dynamic TTL based on data quality
- **Statistics Tracking**: Comprehensive cache performance metrics

#### Usage
```python
from services.tracklist_service.src.cache.fallback_cache import FallbackCache, CacheStrategy

cache = FallbackCache(redis_client=redis_client)

# Store data with quality scoring
await cache.set_with_quality("key", data, quality_score=0.9)

# Retrieve with fallback strategy
result = await cache.get_with_fallback(
    "key",
    strategy=CacheStrategy.FLEXIBLE,
    max_age=3600
)

# Cache warming
async def fetch_fresh_data(key):
    return {"fresh": "data"}

results = await cache.warm_cache(["key1", "key2"], fetch_fresh_data)
```

#### Cache Statistics
```python
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.2%}")
print(f"Fallback usage: {stats['fallback_hits']}")
```

### 4. Adaptive Parser
**File**: `services/tracklist_service/src/scrapers/adaptive_parser.py`

Self-improving parser framework that learns from extraction patterns and adapts to site changes.

#### Key Features
- **Pattern Learning**: Automatically discovers new extraction patterns
- **A/B Testing**: Compares strategy effectiveness in real-time
- **Version Management**: Rollback capabilities for parser configurations
- **Hot Reload**: Dynamic configuration updates without restart

#### Usage
```python
from services.tracklist_service.src.scrapers.adaptive_parser import AdaptiveParser

parser = AdaptiveParser(config_path="parser_config.json")

# Parse with adaptation
result = await parser.parse_with_adaptation(
    "tracklist",
    html_content,
    learn_patterns=True
)

# Version management
await parser.create_version("v1.1", "Added new selectors")
await parser.rollback_to_version("v1.0")

# A/B testing
await parser.start_ab_test(strategy_a, strategy_b, traffic_split=0.5)
```

#### Pattern Learning
```python
# Enable pattern learning
await parser.learn_patterns("tracklist", successful_extraction, html_content)

# Promote successful patterns
await parser.promote_pattern("tracklist", "title", pattern_selector)
```

### 5. Resilient Extractor
**File**: `services/tracklist_service/src/scrapers/resilient_extractor.py`

High-reliability data extraction with multiple fallback strategies and automatic recovery.

#### Extraction Strategies
- **CSS**: CSS selector-based extraction
- **XPath**: XPath expression evaluation
- **Text**: Text pattern matching
- **Regex**: Regular expression parsing

#### Key Features
- **Multi-Strategy Fallback**: Automatic strategy switching on failure
- **Confidence Scoring**: Quality assessment of extracted data
- **Error Recovery**: Graceful handling of parsing failures
- **Performance Monitoring**: Extraction timing and success metrics

#### Usage
```python
from services.tracklist_service.src.scrapers.resilient_extractor import (
    ResilientExtractor, CSSStrategy, XPathStrategy
)

extractor = ResilientExtractor([
    CSSStrategy("h1.title"),
    XPathStrategy("//h1[@class='title']/text()"),
])

result = await extractor.extract(html_content)
if result.success:
    print(f"Extracted: {result.data} (confidence: {result.confidence})")
```

## System Integration

### Complete Workflow
```python
async def resilient_scraping_workflow(url, page_type):
    # 1. Monitor for changes
    monitor = StructureMonitor()
    change_report = await monitor.detect_changes(page_type, html_content, url)

    # 2. Check cache first
    cache = FallbackCache(redis_client=redis_client)
    cached_data = await cache.get_with_fallback(
        f"parsed:{page_type}:{url_hash}",
        strategy=CacheStrategy.FLEXIBLE
    )

    if cached_data and not change_report.has_breaking_changes:
        return cached_data

    # 3. Parse with adaptation
    parser = AdaptiveParser()
    result = await parser.parse_with_adaptation(
        page_type,
        html_content,
        learn_patterns=True
    )

    # 4. Store with quality scoring
    quality_score = 0.9 if result.confidence > 0.8 else 0.6
    await cache.set_with_quality(
        f"parsed:{page_type}:{url_hash}",
        result.data,
        quality_score=quality_score
    )

    # 5. Alert on issues
    alert_manager = AlertManager()
    if change_report.has_breaking_changes:
        await alert_manager.send_change_alert(change_report)

    health = await alert_manager.check_parser_health(page_type)
    if health.requires_alert:
        await alert_manager.send_health_alert(health, page_type)

    return result.data
```

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password

# Alert Configuration
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Monitoring Configuration
STRUCTURE_CHECK_INTERVAL=3600
HEALTH_CHECK_INTERVAL=300
MAX_FALLBACK_AGE=604800
```

### Parser Configuration (JSON)
```json
{
  "tracklist": {
    "version": "1.0",
    "selectors": {
      "title": [
        {"type": "CSS", "selector": "h1.title"},
        {"type": "XPath", "selector": "//h1[@class='title']/text()"}
      ],
      "tracks": [
        {"type": "CSS", "selector": ".track-item"},
        {"type": "XPath", "selector": "//div[@class='track-list']//div[@class='track']"}
      ]
    },
    "learning_enabled": true,
    "ab_testing_enabled": true,
    "confidence_threshold": 0.8
  }
}
```

## Monitoring & Metrics

### Key Metrics
- **Parser Success Rate**: Percentage of successful extractions
- **Cache Hit Rate**: Cache effectiveness and performance
- **Structure Change Frequency**: Rate of website modifications
- **Alert Volume**: Number and severity of alerts generated
- **Recovery Time**: Time to adapt to breaking changes

### Dashboard Endpoints
```python
# Health status
GET /api/health/parser/{page_type}

# Cache statistics
GET /api/cache/stats

# Recent alerts
GET /api/alerts?severity=error&limit=10

# Version history
GET /api/parser/{page_type}/versions

# Performance metrics
GET /api/metrics/{page_type}?period=24h
```

## Best Practices

### Development
1. **Always Use Fallback Strategies**: Implement multiple extraction approaches
2. **Monitor Continuously**: Set up comprehensive alerting for all components
3. **Cache Intelligently**: Use appropriate cache strategies for different data types
4. **Learn from Failures**: Enable pattern learning to improve over time
5. **Version Control**: Maintain parser version history for rollback capability

### Operations
1. **Monitor Alert Channels**: Ensure Discord notifications are working
2. **Review Cache Performance**: Monitor hit rates and adjust TTL settings
3. **Update Patterns Regularly**: Review learned patterns and promote successful ones
4. **Capacity Planning**: Monitor Redis usage and scale as needed
5. **Security**: Regularly update dependencies and monitor for vulnerabilities

### Troubleshooting
1. **High Alert Volume**: Check for new site changes or bot detection
2. **Low Cache Hit Rate**: Review TTL settings and data quality scoring
3. **Poor Extraction Success**: Enable pattern learning and check selectors
4. **Performance Issues**: Monitor Redis latency and consider clustering
5. **False Positives**: Adjust similarity thresholds and anomaly detection

## Testing

The system includes comprehensive test coverage:
- **62 tests** covering all core components
- **81% coverage** for AlertManager
- **88% coverage** for FallbackCache
- **Integration tests** for complete workflows
- **Performance benchmarks** for cache and parsing operations

Run tests with:
```bash
uv run pytest tests/unit/tracklist_service/ -v
```

## Support

For issues or questions:
1. Check the logs for detailed error information
2. Review the monitoring dashboard for system health
3. Verify configuration settings and environment variables
4. Consult the test files for usage examples
5. Check Redis connectivity and performance
