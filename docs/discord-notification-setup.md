# Discord Notification Setup Guide

## Overview

The Tracktion system uses Discord as its exclusive notification channel. Different types of alerts are sent to specific Discord channels for better organization and monitoring.

## Discord Webhook Setup

### Step 1: Create Discord Channels

Create the following channels in your Discord server:

1. `#tracktion-general` - General notifications and status updates
2. `#tracktion-errors` - Error alerts and warnings
3. `#tracktion-critical` - Critical system failures and urgent issues
4. `#tracktion-tracklists` - Tracklist generation notifications
5. `#tracktion-monitoring` - System health and performance alerts
6. `#tracktion-security` - Security alerts and authentication issues

### Step 2: Create Webhooks

For each channel:

1. Right-click the channel and select "Edit Channel"
2. Go to "Integrations" → "Webhooks"
3. Click "New Webhook"
4. Give it a descriptive name (e.g., "Tracktion Alerts")
5. Copy the webhook URL
6. Save the webhook

### Step 3: Configure Environment Variables

Add the webhook URLs to your `.env` file:

```bash
# Discord Webhook Configuration
DISCORD_WEBHOOK_GENERAL=https://discord.com/api/webhooks/123456789/abcdefg
DISCORD_WEBHOOK_ERRORS=https://discord.com/api/webhooks/234567890/bcdefgh
DISCORD_WEBHOOK_CRITICAL=https://discord.com/api/webhooks/345678901/cdefghi
DISCORD_WEBHOOK_TRACKLIST=https://discord.com/api/webhooks/456789012/defghij
DISCORD_WEBHOOK_MONITORING=https://discord.com/api/webhooks/567890123/efghijk
DISCORD_WEBHOOK_SECURITY=https://discord.com/api/webhooks/678901234/fghijkl

# Optional settings
DISCORD_RATE_LIMIT=10        # Messages per minute per channel
DISCORD_RETRY_ATTEMPTS=3      # Retry failed sends
DISCORD_TIMEOUT_SECONDS=10    # Request timeout
```

## Alert Types and Routing

| Alert Type | Discord Channel | Use Case |
|------------|----------------|----------|
| `GENERAL` | #tracktion-general | Info messages, status updates, non-critical notifications |
| `ERROR` | #tracktion-errors | Non-critical errors, warnings, issues that need attention |
| `CRITICAL` | #tracktion-critical | System failures, data loss risks, urgent issues |
| `TRACKLIST` | #tracktion-tracklists | Successful tracklist generation, parsing updates |
| `MONITORING` | #tracktion-monitoring | Health checks, performance metrics, resource usage |
| `SECURITY` | #tracktion-security | Auth failures, suspicious activity, security events |

## Message Format

Discord notifications include:

- **Title**: Clear description of the event
- **Color**: Visual indicator of severity (green/yellow/red)
- **Timestamp**: When the event occurred
- **Details**: Relevant information about the event
- **Source**: Which service generated the alert

### Example Message

```json
{
  "embeds": [{
    "title": "🎵 Tracklist Generated",
    "description": "Successfully generated tracklist for recording",
    "color": 3066993,  // Green
    "fields": [
      {
        "name": "Recording",
        "value": "Artist - Mix Title",
        "inline": true
      },
      {
        "name": "Tracks",
        "value": "15",
        "inline": true
      }
    ],
    "timestamp": "2024-01-01T12:00:00Z",
    "footer": {
      "text": "Tracklist Service"
    }
  }]
}
```

## Testing Your Configuration

1. Run the notification test command:
```bash
uv run python -m services.tracklist_service.test_notifications
```

2. Check each Discord channel to verify messages are received

3. Monitor the logs for any connection errors:
```bash
docker-compose logs -f tracklist_service | grep -i discord
```

## Troubleshooting

### Messages Not Arriving

1. Verify webhook URLs are correct and not expired
2. Check network connectivity to Discord
3. Ensure environment variables are loaded correctly
4. Check logs for rate limiting errors

### Rate Limiting

Discord webhooks have rate limits:
- 30 requests per minute per webhook
- Configure `DISCORD_RATE_LIMIT` to stay below this threshold

### Connection Timeouts

If you experience timeouts:
- Increase `DISCORD_TIMEOUT_SECONDS`
- Check your network connection
- Verify Discord's status at https://discordstatus.com

## Security Best Practices

1. **Never commit webhook URLs** to version control
2. **Rotate webhooks** periodically
3. **Use channel permissions** to restrict who can view alerts
4. **Monitor webhook usage** for suspicious activity
5. **Test in development** before production deployment

## Migration from Slack/Email

If migrating from previous notification systems:

1. All Slack webhook references should be replaced with Discord webhooks
2. Email notification code has been removed - no SMTP configuration needed
3. Update any monitoring dashboards to point to Discord channels
4. Notify team members of the new Discord channels for alerts

## Support

For issues with Discord notifications:
1. Check this documentation
2. Review logs in `logs/notifications.log`
3. Verify webhook configuration in Discord server settings
4. Contact the development team if issues persist
