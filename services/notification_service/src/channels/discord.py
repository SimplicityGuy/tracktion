"""Discord notification channel implementation."""

import asyncio
import logging
import os
from typing import Any

import aiohttp

from services.notification_service.src.core.base import (
    AlertType,
    NotificationChannel,
    NotificationMessage,
    NotificationResult,
    NotificationStatus,
)
from services.notification_service.src.core.history import NotificationHistoryLogger
from services.notification_service.src.core.rate_limiter import PerChannelRateLimiter, RateLimitConfig
from services.notification_service.src.core.retry import CircuitBreaker, RetryManager, RetryPolicy
from services.notification_service.src.templates.discord_templates import DiscordEmbedBuilder

logger = logging.getLogger(__name__)


class DiscordWebhookClient:
    """Client for sending messages to Discord webhooks."""

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 10.0,
        retry_policy: RetryPolicy | None = None,
    ):
        """Initialize Discord webhook client.

        Args:
            webhook_url: Discord webhook URL
            timeout: Request timeout in seconds
            retry_policy: Retry policy for failed requests
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.retry_manager = RetryManager(retry_policy or RetryPolicy())
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=aiohttp.ClientError,
        )

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send payload to Discord webhook.

        Args:
            payload: Discord webhook payload

        Returns:
            Response data from Discord

        Raises:
            aiohttp.ClientError: If request fails
        """

        async def _send() -> dict[str, Any]:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
                if response.status == 429:
                    # Rate limited
                    retry_after = response.headers.get("X-RateLimit-Reset-After", "60")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message=f"Rate limited. Retry after {retry_after}s",
                    )
                if response.status >= 400:
                    text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Discord webhook error: {text}",
                    )

                # Parse rate limit headers
                rate_limit_info = {
                    "limit": response.headers.get("X-RateLimit-Limit"),
                    "remaining": response.headers.get("X-RateLimit-Remaining"),
                    "reset": response.headers.get("X-RateLimit-Reset"),
                    "reset_after": response.headers.get("X-RateLimit-Reset-After"),
                }

                return {
                    "status": response.status,
                    "rate_limit": rate_limit_info,
                }

        # Use circuit breaker and retry manager
        result: dict[str, Any] = await self.circuit_breaker.call(lambda: self.retry_manager.execute(_send))
        return result

    async def validate(self) -> bool:
        """Validate webhook URL is working.

        Returns:
            True if webhook is valid, False otherwise
        """
        try:
            # Send a simple test message
            test_payload = {
                "content": "🔧 Webhook validation test",
            }
            await self.send(test_payload)
            return True
        except Exception as e:
            logger.error(f"Webhook validation failed: {e}")
            return False


class DiscordNotificationService(NotificationChannel):
    """Discord notification service implementation."""

    def __init__(
        self,
        redis_client: Any | None = None,
        history_logger: NotificationHistoryLogger | None = None,
    ):
        """Initialize Discord notification service.

        Args:
            redis_client: Optional Redis client for persistence
            history_logger: Optional history logger
        """
        self.webhooks: dict[AlertType, str | None] = {
            AlertType.GENERAL: os.getenv("DISCORD_WEBHOOK_GENERAL"),
            AlertType.ERROR: os.getenv("DISCORD_WEBHOOK_ERRORS"),
            AlertType.CRITICAL: os.getenv("DISCORD_WEBHOOK_CRITICAL"),
            AlertType.TRACKLIST: os.getenv("DISCORD_WEBHOOK_TRACKLIST"),
            AlertType.MONITORING: os.getenv("DISCORD_WEBHOOK_MONITORING"),
            AlertType.SECURITY: os.getenv("DISCORD_WEBHOOK_SECURITY"),
        }

        # Initialize rate limiter with Discord's limits
        rate_limit = int(os.getenv("DISCORD_RATE_LIMIT", "30"))
        self.rate_limiter = PerChannelRateLimiter(RateLimitConfig(limit=rate_limit, window=60.0))

        # Initialize retry manager
        retry_attempts = int(os.getenv("DISCORD_RETRY_ATTEMPTS", "3"))
        self.retry_policy = RetryPolicy(
            max_attempts=retry_attempts,
            backoff_base=2.0,
            retry_on=(aiohttp.ClientError,),
        )

        # Initialize clients
        self.clients: dict[AlertType, DiscordWebhookClient | None] = {}
        for alert_type, webhook_url in self.webhooks.items():
            if webhook_url:
                self.clients[alert_type] = DiscordWebhookClient(
                    webhook_url=webhook_url,
                    timeout=float(os.getenv("DISCORD_TIMEOUT_SECONDS", "10")),
                    retry_policy=self.retry_policy,
                )
            else:
                self.clients[alert_type] = None

        # Initialize history logger
        self.history_logger = history_logger or NotificationHistoryLogger(redis_client)

        # Initialize template builder
        self.embed_builder = DiscordEmbedBuilder()

        # Queue for rate limited messages
        self.message_queue: dict[AlertType, asyncio.Queue] = {
            alert_type: asyncio.Queue(maxsize=int(os.getenv("DISCORD_QUEUE_SIZE", "100"))) for alert_type in AlertType
        }

        # Start queue processors
        self.queue_tasks: list[asyncio.Task] = []
        self._start_queue_processors()

    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send notification to Discord.

        Args:
            message: Notification message to send

        Returns:
            Result of the send operation
        """
        alert_type = message.alert_type
        client = self.clients.get(alert_type)

        if not client:
            result = NotificationResult(
                success=False,
                status=NotificationStatus.FAILED,
                error=f"No webhook configured for {alert_type.value}",
            )
            await self.history_logger.log(message, result)
            return result

        # Check rate limits
        webhook_url = self.webhooks[alert_type]
        if webhook_url and not await self.rate_limiter.allow(webhook_url):
            # Queue the message
            try:
                self.message_queue[alert_type].put_nowait(message)
                result = NotificationResult(
                    success=True,
                    status=NotificationStatus.QUEUED,
                    metadata={"queue_size": self.message_queue[alert_type].qsize()},
                )
            except asyncio.QueueFull:
                result = NotificationResult(
                    success=False,
                    status=NotificationStatus.RATE_LIMITED,
                    error="Message queue full",
                )
            await self.history_logger.log(message, result)
            return result

        # Build Discord embed
        embed_payload = self.embed_builder.build_embed(
            alert_type=alert_type,
            title=message.title,
            description=message.message,
            color=message.color,
            fields=message.fields,
        )

        # Send to Discord
        try:
            response = await client.send(embed_payload)
            result = NotificationResult(
                success=True,
                status=NotificationStatus.SENT,
                message_id=str(response.get("status")),
                metadata=response.get("rate_limit", {}),
            )
        except aiohttp.ClientError as e:
            result = NotificationResult(
                success=False,
                status=NotificationStatus.FAILED,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error sending to Discord: {e}")
            result = NotificationResult(
                success=False,
                status=NotificationStatus.FAILED,
                error=f"Unexpected error: {e}",
            )

        # Log to history
        await self.history_logger.log(message, result)
        return result

    async def validate_configuration(self) -> bool:
        """Validate Discord configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        valid = True
        for alert_type, client in self.clients.items():
            if client:
                if not await client.validate():
                    logger.error(f"Webhook validation failed for {alert_type.value}")
                    valid = False
            else:
                logger.warning(f"No webhook configured for {alert_type.value}")

        return valid

    async def get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Dictionary containing rate limit information
        """
        status = {}
        for alert_type, webhook_url in self.webhooks.items():
            if webhook_url:
                status[alert_type.value] = self.rate_limiter.get_status(webhook_url)

        # Add queue sizes
        for alert_type, queue in self.message_queue.items():
            if alert_type.value in status:
                status[alert_type.value]["queue_size"] = queue.qsize()

        return status

    async def health_check(self) -> bool:
        """Check if Discord service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        # Check if at least one webhook is configured and working
        for client in self.clients.values():
            if client:
                try:
                    if await client.validate():
                        return True
                except Exception:
                    continue

        return False

    def _start_queue_processors(self) -> None:
        """Start background tasks to process queued messages."""
        for alert_type in AlertType:
            if self.clients.get(alert_type):
                task = asyncio.create_task(self._process_queue(alert_type))
                self.queue_tasks.append(task)

    async def _process_queue(self, alert_type: AlertType) -> None:
        """Process queued messages for an alert type.

        Args:
            alert_type: Alert type to process queue for
        """
        queue = self.message_queue[alert_type]
        webhook_url = self.webhooks[alert_type]

        while True:
            try:
                # Wait for a message
                message = await queue.get()

                # Wait for rate limit to clear
                if webhook_url:
                    await self.rate_limiter.wait_for_token(webhook_url)

                # Send the message
                result = await self.send(message)

                if not result.success:
                    logger.error(f"Failed to send queued message for {alert_type.value}: {result.error}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error for {alert_type.value}: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def shutdown(self) -> None:
        """Shutdown the notification service."""
        # Cancel queue processors
        for task in self.queue_tasks:
            task.cancel()

        await asyncio.gather(*self.queue_tasks, return_exceptions=True)
        self.queue_tasks.clear()

        # Process remaining queued messages
        for alert_type, queue in self.message_queue.items():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    logger.warning(f"Dropping queued message for {alert_type.value} during shutdown")
                except asyncio.QueueEmpty:
                    break
