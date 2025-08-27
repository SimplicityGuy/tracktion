"""
Message handler for tracklist import operations.

Handles RabbitMQ message publishing and consuming for async import processing
with proper error handling, retry logic, and message acknowledgments.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List

import aio_pika
from aio_pika import Message, IncomingMessage, ExchangeType
from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractExchange

from ..config import get_config
from ..exceptions import MessageQueueError
from ..models.tracklist import ImportTracklistRequest

logger = logging.getLogger(__name__)


@dataclass
class ImportJobMessage:
    """Message schema for import job requests."""

    correlation_id: str
    request: ImportTracklistRequest
    created_at: str
    retry_count: int = 0
    priority: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "correlation_id": self.correlation_id,
            "request": self.request.model_dump(),
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImportJobMessage":
        """Create from dictionary."""
        return cls(
            correlation_id=data["correlation_id"],
            request=ImportTracklistRequest(**data["request"]),
            created_at=data["created_at"],
            retry_count=data.get("retry_count", 0),
            priority=data.get("priority", 5),
        )


@dataclass
class ImportResultMessage:
    """Message schema for import job results."""

    correlation_id: str
    success: bool
    tracklist_id: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "correlation_id": self.correlation_id,
            "success": self.success,
            "tracklist_id": self.tracklist_id,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "completed_at": self.completed_at or datetime.now(timezone.utc).isoformat(),
        }


class ImportMessageHandler:
    """RabbitMQ message handler for import operations."""

    def __init__(self, connection_url: Optional[str] = None):
        """
        Initialize the import message handler.

        Args:
            connection_url: RabbitMQ connection URL
        """
        self.config = get_config()
        self.connection_url = connection_url or self.config.message_queue.rabbitmq_url

        # Connection objects
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.exchange: Optional[AbstractExchange] = None

        # Queue names
        self.import_queue = "tracklist_import_queue"
        self.import_result_queue = "tracklist_import_result_queue"
        self.import_retry_queue = "tracklist_import_retry_queue"
        self.import_dlq = "tracklist_import_dlq"

        # Exchange name
        self.exchange_name = self.config.message_queue.exchange_name

        # Message handlers
        self.import_handlers: List[Callable] = []

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.connection_url}")

            self.connection = await aio_pika.connect_robust(
                self.connection_url, client_properties={"service": "tracklist_import_handler"}
            )

            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)  # Process one message at a time

            # Declare exchange
            self.exchange = await self.channel.declare_exchange(self.exchange_name, ExchangeType.DIRECT, durable=True)

            # Declare queues
            await self._declare_queues()

            logger.info("Successfully connected to RabbitMQ")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise MessageQueueError(f"Failed to connect to message queue: {str(e)}", queue_name="connection")

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def _declare_queues(self) -> None:
        """Declare all required queues."""
        if not self.channel:
            raise MessageQueueError("Channel not initialized")

        # Import queue (main processing queue)
        import_queue = await self.channel.declare_queue(
            self.import_queue,
            durable=True,
            arguments={
                "x-message-ttl": 3600000,  # 1 hour TTL
                "x-max-length": 1000,  # Max 1000 messages
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": self.import_dlq,
            },
        )

        # Bind to exchange
        await import_queue.bind(self.exchange, routing_key="tracklist.import")

        # Result queue (for publishing results)
        result_queue = await self.channel.declare_queue(
            self.import_result_queue,
            durable=True,
            arguments={
                "x-message-ttl": 86400000,  # 24 hour TTL
                "x-max-length": 5000,  # Max 5000 results
            },
        )
        await result_queue.bind(self.exchange, routing_key="tracklist.import.result")

        # Retry queue (for failed imports)
        retry_queue = await self.channel.declare_queue(
            self.import_retry_queue,
            durable=True,
            arguments={
                "x-message-ttl": 300000,  # 5 minute TTL
                "x-max-length": 500,  # Max 500 retries
            },
        )
        await retry_queue.bind(self.exchange, routing_key="tracklist.import.retry")

        # Dead letter queue (for permanently failed imports)
        await self.channel.declare_queue(self.import_dlq, durable=True)

        logger.info("All queues declared successfully")

    async def publish_import_job(self, job_message: ImportJobMessage) -> bool:
        """
        Publish an import job to the queue.

        Args:
            job_message: Import job message

        Returns:
            True if published successfully
        """
        try:
            if not self.exchange:
                raise MessageQueueError("Not connected to message queue")

            message_body = json.dumps(job_message.to_dict()).encode()

            message = Message(
                message_body,
                priority=job_message.priority,
                correlation_id=job_message.correlation_id,
                timestamp=datetime.now(timezone.utc),
                headers={"retry_count": job_message.retry_count, "created_at": job_message.created_at},
            )

            await self.exchange.publish(message, routing_key="tracklist.import")

            logger.info(
                "Published import job",
                extra={
                    "correlation_id": job_message.correlation_id,
                    "url": job_message.request.url,
                    "retry_count": job_message.retry_count,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish import job: {e}", extra={"correlation_id": job_message.correlation_id})
            raise MessageQueueError(
                f"Failed to publish import job: {str(e)}",
                queue_name=self.import_queue,
                correlation_id=job_message.correlation_id,
            )

    async def publish_import_result(self, result_message: ImportResultMessage) -> bool:
        """
        Publish an import result to the result queue.

        Args:
            result_message: Import result message

        Returns:
            True if published successfully
        """
        try:
            if not self.exchange:
                raise MessageQueueError("Not connected to message queue")

            message_body = json.dumps(result_message.to_dict()).encode()

            message = Message(
                message_body,
                correlation_id=result_message.correlation_id,
                timestamp=datetime.now(timezone.utc),
                headers={"success": result_message.success, "completed_at": result_message.completed_at},
            )

            await self.exchange.publish(message, routing_key="tracklist.import.result")

            logger.info(
                "Published import result",
                extra={
                    "correlation_id": result_message.correlation_id,
                    "success": result_message.success,
                    "tracklist_id": result_message.tracklist_id,
                },
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to publish import result: {e}", extra={"correlation_id": result_message.correlation_id}
            )
            return False

    async def start_consuming(self) -> None:
        """Start consuming import job messages."""
        if not self.channel:
            raise MessageQueueError("Channel not initialized")

        import_queue = await self.channel.get_queue(self.import_queue)

        async def process_import_message(message: IncomingMessage) -> None:
            """Process a single import message."""
            async with message.process():
                try:
                    # Parse message
                    job_data = json.loads(message.body.decode())
                    job_message = ImportJobMessage.from_dict(job_data)

                    logger.info(
                        "Processing import job",
                        extra={
                            "correlation_id": job_message.correlation_id,
                            "url": job_message.request.url,
                            "retry_count": job_message.retry_count,
                        },
                    )

                    # Process through registered handlers
                    for handler in self.import_handlers:
                        await handler(job_message)

                    # Message will be automatically acknowledged on successful completion

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse import message: {e}")
                    # Message will be rejected and moved to DLQ

                except Exception as e:
                    logger.error(f"Error processing import message: {e}")
                    # Message will be rejected and potentially retried
                    raise

        await import_queue.consume(process_import_message)
        logger.info("Started consuming import messages")

    def register_import_handler(self, handler: Callable[[ImportJobMessage], None]) -> None:
        """
        Register a handler for import messages.

        Args:
            handler: Async function to handle import messages
        """
        self.import_handlers.append(handler)
        logger.info(f"Registered import handler: {handler.__name__}")

    async def ping(self) -> bool:
        """
        Check if message queue is accessible.

        Returns:
            True if accessible, False otherwise
        """
        try:
            if self.connection and not self.connection.is_closed:
                return True
            else:
                # Try to establish connection
                await self.connect()
                return True
        except Exception:
            return False

    async def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all queues.

        Returns:
            Dictionary with queue statistics
        """
        if not self.channel:
            raise MessageQueueError("Channel not initialized")

        stats = {}

        for queue_name in [self.import_queue, self.import_result_queue, self.import_retry_queue, self.import_dlq]:
            try:
                queue = await self.channel.get_queue(queue_name)
                stats[queue_name] = {
                    "message_count": queue.declaration_result.message_count,
                    "consumer_count": queue.declaration_result.consumer_count,
                }
            except Exception as e:
                logger.warning(f"Could not get stats for queue {queue_name}: {e}")
                stats[queue_name] = {"error": str(e)}

        return stats


# Global instance for easy access
import_message_handler = ImportMessageHandler()


async def setup_import_message_handler() -> ImportMessageHandler:
    """
    Setup and connect the import message handler.

    Returns:
        Connected message handler instance
    """
    if not import_message_handler.connection or import_message_handler.connection.is_closed:
        await import_message_handler.connect()

    return import_message_handler


async def cleanup_import_message_handler() -> None:
    """Cleanup the import message handler connection."""
    await import_message_handler.disconnect()
