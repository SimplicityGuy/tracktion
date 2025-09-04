# Notification Service

Unified Discord notification service for the Tracktion project, providing comprehensive alert management with rate limiting, retry logic, and message queuing across multiple Discord channels.

## Features

### Notification Capabilities
- **Multi-Channel Support**: Dedicated Discord webhooks for different alert types (general, error, critical, tracklist, monitoring, security)
- **Rich Discord Embeds**: Formatted messages with custom colors, fields, timestamps, and alert-specific templates
- **Message Templates**: Pre-built templates for common notification types (error, critical, tracklist, monitoring, security)
- **Custom Message Builder**: Flexible embed builder with support for thumbnails, images, author info, and custom fields

### Infrastructure Features
- **Rate Limiting**: Token bucket and sliding window rate limiters with per-channel configuration
- **Retry Logic**: Configurable retry policies with exponential backoff and jitter
- **Circuit Breaker**: Prevents cascading failures with automatic recovery
- **Message Queuing**: Async message queues for rate-limited scenarios
- **History Logging**: Redis-backed notification history with statistics tracking
- **Health Checks**: Service health monitoring and webhook validation

## Architecture

The service follows a plugin-based notification architecture:

```
Message → Rate Limiter → Discord Channel → Webhook → Discord Server
     ↓          ↓              ↓              ↓
History Logger  Queue     Circuit Breaker   Retry Manager
```

## Project Structure

```
notification_service/
├── src/
│   ├── __init__.py                  # Package initialization and exports
│   ├── channels/
│   │   ├── __init__.py              # Channel exports
│   │   └── discord.py               # Discord webhook implementation
│   ├── core/
│   │   ├── __init__.py              # Core module exports
│   │   ├── base.py                  # Base abstractions and types
│   │   ├── history.py               # Notification history tracking
│   │   ├── rate_limiter.py          # Rate limiting implementations
│   │   └── retry.py                 # Retry logic and circuit breaker
│   └── templates/
│       ├── __init__.py              # Template exports
│       └── discord_templates.py     # Discord embed templates
├── pyproject.toml                   # Python dependencies
└── README.md                        # This file
```

## Configuration

The service is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| **Discord Webhooks** |
| `DISCORD_WEBHOOK_GENERAL` | General notifications webhook URL | Required |
| `DISCORD_WEBHOOK_ERRORS` | Error notifications webhook URL | Required |
| `DISCORD_WEBHOOK_CRITICAL` | Critical alerts webhook URL | Required |
| `DISCORD_WEBHOOK_TRACKLIST` | Tracklist updates webhook URL | Required |
| `DISCORD_WEBHOOK_MONITORING` | Monitoring alerts webhook URL | Required |
| `DISCORD_WEBHOOK_SECURITY` | Security alerts webhook URL | Required |
| **Rate Limiting** |
| `DISCORD_RATE_LIMIT` | Requests per minute limit | `30` |
| `DISCORD_QUEUE_SIZE` | Max queued messages per channel | `100` |
| **Retry Configuration** |
| `DISCORD_RETRY_ATTEMPTS` | Max retry attempts for failed sends | `3` |
| `DISCORD_TIMEOUT_SECONDS` | Request timeout in seconds | `10` |
| **Redis Configuration** |
| `REDIS_HOST` | Redis server hostname | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_DB` | Redis database number | `0` |

## Alert Types

The service supports six distinct alert types, each with dedicated Discord channels:

| Alert Type | Purpose | Color | Typical Use Cases |
|-----------|---------|--------|------------------|
| `GENERAL` | General information | Blue | Service starts, configuration changes |
| `ERROR` | Non-critical errors | Orange | Failed processing, recoverable errors |
| `CRITICAL` | Critical system failures | Red | Service crashes, data corruption |
| `TRACKLIST` | Tracklist operations | Green | New tracklists, updates, deletions |
| `MONITORING` | Performance metrics | Purple | Performance alerts, threshold breaches |
| `SECURITY` | Security events | Yellow | Authentication failures, security breaches |

## Usage Examples

### Basic Notification

```python
from services.notification_service import (
    DiscordNotificationService,
    NotificationMessage,
    AlertType
)

# Initialize service
service = DiscordNotificationService()

# Create notification
message = NotificationMessage(
    alert_type=AlertType.GENERAL,
    title="Service Started",
    message="Analysis service has started successfully",
    fields=[
        {
            "name": "Version",
            "value": "1.0.0",
            "inline": True
        }
    ]
)

# Send notification
result = await service.send(message)
print(f"Success: {result.success}, Status: {result.status}")
```

### Error Notification with Template

```python
from services.notification_service.src.templates.discord_templates import DiscordEmbedBuilder

builder = DiscordEmbedBuilder()

# Build error embed
error_embed = builder.build_error_embed(
    error_message="Failed to process audio file",
    error_details={
        "file_path": "/music/track.mp3",
        "error_code": "DECODE_ERROR",
        "service": "analysis_service"
    },
    traceback="Traceback (most recent call last):\n  File..."
)

# Send via service
service = DiscordNotificationService()
message = NotificationMessage(
    alert_type=AlertType.ERROR,
    title=error_embed["embeds"][0]["title"],
    message=error_embed["embeds"][0]["description"]
)
await service.send(message)
```

### Critical Alert with Mentions

```python
# Critical alerts automatically include @here mentions
critical_embed = builder.build_critical_embed(
    title="Database Connection Lost",
    message="PostgreSQL connection has been lost",
    impact="All audio analysis operations are halted",
    action_required="Restart database service immediately"
)

# This will ping @here in Discord
```

### Monitoring Alert

```python
monitoring_embed = builder.build_monitoring_embed(
    metric_name="CPU Usage",
    current_value="95%",
    threshold="80%",
    status="error"
)
```

## Running Locally

1. Install dependencies:
```bash
cd services/notification_service
uv pip install -e .
```

2. Set environment variables:
```bash
export DISCORD_WEBHOOK_GENERAL="https://discord.com/api/webhooks/..."
export DISCORD_WEBHOOK_ERRORS="https://discord.com/api/webhooks/..."
export DISCORD_WEBHOOK_CRITICAL="https://discord.com/api/webhooks/..."
# ... set other webhook URLs
```

3. Use in your application:
```python
from services.notification_service import DiscordNotificationService
```

## Running with Docker

The service is designed to be imported as a library. Include it in your service's dependencies:

```dockerfile
COPY services/notification_service /app/services/notification_service
RUN pip install -e /app/services/notification_service
```

## Testing

Run unit tests:
```bash
uv run pytest tests/unit/services/notification/
```

Run integration tests (requires Discord webhooks):
```bash
uv run pytest tests/integration/test_notification_service.py
```

## Rate Limiting

The service implements sophisticated rate limiting to respect Discord's API limits:

### Token Bucket Rate Limiter
- **Default**: 30 requests per 60 seconds per webhook
- **Burst Support**: Allows temporary bursts up to the limit
- **Per-Channel**: Independent rate limits for each Discord webhook

### Sliding Window Rate Limiter
- More accurate rate limiting for strict compliance
- Tracks actual request timestamps within the time window
- Prevents thundering herd scenarios

### Message Queuing
- Automatically queues messages when rate limited
- Background processors handle queued messages
- Configurable queue size (default: 100 messages per channel)

## Retry Logic and Circuit Breaker

### Retry Manager
- **Exponential Backoff**: 2^attempt with jitter to prevent synchronization
- **Configurable Policies**: Max attempts, backoff base, retry conditions
- **Smart Exception Handling**: Only retries specific exception types

### Circuit Breaker
- **Failure Threshold**: Opens circuit after 5 consecutive failures
- **Recovery Timeout**: 60 seconds before attempting recovery
- **States**: Closed (normal) → Open (failing) → Half-Open (testing)

## History and Analytics

### Notification History
- **Redis Persistence**: All notifications stored with metadata
- **Memory Cache**: Recent notifications cached for fast access
- **Retention Policy**: Configurable retention (default: 7 days)

### Statistics and Metrics
- **Success Rates**: Track delivery success/failure rates
- **Response Times**: Monitor Discord API response times
- **Alert Type Breakdown**: Statistics by alert category
- **Retry Analysis**: Track retry patterns and effectiveness

## Error Handling

The service implements comprehensive error handling:

1. **Webhook Validation**: Validates webhook URLs on startup
2. **Network Errors**: Automatic retry with exponential backoff
3. **Rate Limiting**: Graceful queuing when limits exceeded
4. **Invalid Payloads**: Validation and sanitization of Discord payloads
5. **Service Degradation**: Circuit breaker prevents cascading failures

## Discord Integration Details

### Webhook Configuration
Each alert type requires a dedicated Discord webhook URL. To create webhooks:
1. Go to Discord Server Settings → Integrations → Webhooks
2. Create webhooks for each channel you want to receive alerts
3. Copy webhook URLs to environment variables

### Message Formatting
- **Embed Limits**: Respects Discord's 6000 character limit per embed
- **Field Limits**: Maximum 25 fields per embed, 1024 characters per field
- **Color Coding**: Automatic color assignment based on alert type
- **Timestamps**: All messages include UTC timestamps

### Rich Content Support
- **Custom Fields**: Add structured data with name/value pairs
- **Images**: Support for thumbnail and main images
- **Links**: Clickable URLs in titles and descriptions
- **Code Blocks**: Syntax highlighting for error tracebacks

## Troubleshooting

### Service Health Check
```python
# Check if service is healthy
healthy = await service.health_check()
print(f"Service healthy: {healthy}")

# Validate webhook configuration
valid = await service.validate_configuration()
print(f"Configuration valid: {valid}")
```

### Rate Limit Status
```python
# Check current rate limit status
status = await service.get_rate_limit_status()
for alert_type, info in status.items():
    print(f"{alert_type}: {info['remaining']}/{info['limit']} remaining")
```

### Common Issues

**Webhooks not working**:
- Verify webhook URLs are correct and active
- Check Discord server permissions
- Test with simple curl command

**Messages not sending**:
- Check rate limiting status
- Verify network connectivity to Discord
- Review service logs for errors

**High latency**:
- Monitor Discord API response times
- Check network connectivity
- Consider adjusting timeout settings

## Future Enhancements

- Support for additional notification channels (Slack, email, SMS)
- Message templates for custom alert types
- Advanced analytics dashboard
- Webhook endpoint for receiving Discord events
- Integration with monitoring systems (Prometheus, Grafana)
- Notification routing based on alert severity and time of day
