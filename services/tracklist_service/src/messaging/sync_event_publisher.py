"""RabbitMQ event publishers for synchronization events."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import aio_pika
from aio_pika import ExchangeType, Message

from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQClient, RabbitMQConfig
from services.tracklist_service.src.messaging.sync_message_schemas import (
    BatchSyncMessage,
    ConflictDetectedMessage,
    CueRegenerationTriggeredMessage,
    SyncCompletedMessage,
    SyncEventMessage,
    SyncEventType,
    SyncFailedMessage,
    SyncStatusUpdateMessage,
    VersionCreatedMessage,
)

if TYPE_CHECKING:
    from aio_pika.abc import AbstractChannel, AbstractExchange

logger = logging.getLogger(__name__)


class SyncEventPublisher:
    """Publisher for synchronization-related events."""

    def __init__(self, rabbitmq_client: RabbitMQClient | None = None):
        """Initialize sync event publisher.

        Args:
            rabbitmq_client: RabbitMQ client instance
        """

        self.rabbitmq_client = rabbitmq_client or RabbitMQClient(RabbitMQConfig())
        self.exchange_name = "tracklist.sync.events"
        self.exchange_type = ExchangeType.TOPIC
        self.exchange: AbstractExchange | None = None
        self.channel: AbstractChannel | None = None

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup exchange."""
        try:
            await self.rabbitmq_client.connect()
            if not self.rabbitmq_client.connection:
                raise ConnectionError("Failed to establish RabbitMQ connection")

            self.channel = await self.rabbitmq_client.connection.channel()

            # Declare the sync events exchange
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                self.exchange_type,
                durable=True,
            )

            # Declare dead letter exchange for failed messages
            await self.channel.declare_exchange(
                f"{self.exchange_name}.dlx",
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare queues for different event types
            await self._setup_queues()

            logger.info(f"Connected to RabbitMQ exchange: {self.exchange_name}")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def _setup_queues(self) -> None:
        """Setup queues for different event types."""
        if not self.channel:
            raise ConnectionError("Channel not established")

        queues = [
            ("sync.events.conflicts", "sync.conflict.*"),
            ("sync.events.versions", "sync.version.*"),
            ("sync.events.cue", "sync.cue.*"),
            ("sync.events.status", "sync.status.*"),
            ("sync.events.batch", "sync.batch.*"),
            ("sync.events.all", "sync.#"),  # Catch-all queue
        ]

        for queue_name, routing_pattern in queues:
            # Declare queue with dead letter exchange
            queue = await self.channel.declare_queue(
                queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{self.exchange_name}.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                    "x-max-retries": 3,
                },
            )

            # Bind queue to exchange
            if not self.exchange:
                raise ConnectionError("Exchange not established")
            await queue.bind(self.exchange, routing_pattern)

            # Declare dead letter queue
            dlq = await self.channel.declare_queue(
                f"{queue_name}.dlq",
                durable=True,
                arguments={
                    "x-message-ttl": 604800000,  # 7 days
                },
            )

            # Bind dead letter queue
            await dlq.bind(
                f"{self.exchange_name}.dlx",
                routing_pattern,
            )

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        await self.rabbitmq_client.disconnect()

    async def publish_sync_started(
        self,
        tracklist_id: UUID,
        source: str,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish sync started event.

        Args:
            tracklist_id: ID of the tracklist
            source: Sync source
            actor: Who triggered the sync
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = SyncEventMessage(
                message_id=uuid4(),
                event_type=SyncEventType.SYNC_STARTED,
                tracklist_id=tracklist_id,
                source=source,
                actor=actor,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key="sync.status.started",
                priority=7,
            )

            logger.info(f"Published sync started event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish sync started event: {e}")
            return False

    async def publish_sync_completed(
        self,
        tracklist_id: UUID,
        source: str,
        changes_applied: int,
        confidence: float,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish sync completed event.

        Args:
            tracklist_id: ID of the tracklist
            source: Sync source
            changes_applied: Number of changes applied
            confidence: Confidence score
            actor: Who triggered the sync
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = SyncCompletedMessage(
                message_id=uuid4(),
                event_type=SyncEventType.SYNC_COMPLETED,
                tracklist_id=tracklist_id,
                source=source,
                actor=actor,
                changes_applied=changes_applied,
                confidence=confidence,
                duration_seconds=metadata.get("duration_seconds", 0) if metadata else 0,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key="sync.status.completed",
                priority=5,
            )

            logger.info(f"Published sync completed event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish sync completed event: {e}")
            return False

    async def publish_sync_failed(
        self,
        tracklist_id: UUID,
        source: str,
        error_message: str,
        retry_count: int = 0,
        will_retry: bool = False,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish sync failed event.

        Args:
            tracklist_id: ID of the tracklist
            source: Sync source
            error_message: Error message
            retry_count: Number of retries attempted
            will_retry: Whether retry is scheduled
            actor: Who triggered the sync
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = SyncFailedMessage(
                message_id=uuid4(),
                event_type=SyncEventType.SYNC_FAILED,
                tracklist_id=tracklist_id,
                source=source,
                actor=actor,
                error_message=error_message,
                retry_count=retry_count,
                will_retry=will_retry,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key="sync.status.failed",
                priority=8,  # Higher priority for failures
            )

            logger.info(f"Published sync failed event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish sync failed event: {e}")
            return False

    async def publish_conflict_detected(
        self,
        tracklist_id: UUID,
        conflicts: list[dict[str, Any]],
        source: str,
        auto_resolvable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish conflict detected event.

        Args:
            tracklist_id: ID of the tracklist
            conflicts: List of detected conflicts
            source: Source of the conflict
            auto_resolvable: Whether conflicts can be auto-resolved
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = ConflictDetectedMessage(
                message_id=uuid4(),
                event_type=SyncEventType.CONFLICT_DETECTED,
                tracklist_id=tracklist_id,
                source=source,
                conflicts=conflicts,
                conflict_count=len(conflicts),
                auto_resolvable=auto_resolvable,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key="sync.conflict.detected",
                priority=9,  # High priority for conflicts
            )

            logger.info(f"Published conflict detected event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish conflict detected event: {e}")
            return False

    async def publish_conflict_resolved(
        self,
        tracklist_id: UUID,
        resolution_count: int,
        resolution_strategy: str,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish conflict resolved event.

        Args:
            tracklist_id: ID of the tracklist
            resolution_count: Number of conflicts resolved
            resolution_strategy: Strategy used for resolution
            actor: Who resolved the conflicts
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = SyncEventMessage(
                message_id=uuid4(),
                event_type=SyncEventType.CONFLICT_RESOLVED,
                tracklist_id=tracklist_id,
                source="conflict_resolution",
                actor=actor,
                correlation_id=uuid4(),
                metadata={
                    "resolution_count": resolution_count,
                    "resolution_strategy": resolution_strategy,
                    **(metadata or {}),
                },
            )

            await self._publish_message(
                message,
                routing_key="sync.conflict.resolved",
                priority=6,
            )

            logger.info(f"Published conflict resolved event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish conflict resolved event: {e}")
            return False

    async def publish_version_created(
        self,
        tracklist_id: UUID,
        version_id: UUID,
        version_number: int,
        change_type: str,
        change_summary: str,
        created_by: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish version created event.

        Args:
            tracklist_id: ID of the tracklist
            version_id: ID of the new version
            version_number: Version number
            change_type: Type of change
            change_summary: Summary of changes
            created_by: Who created the version
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = VersionCreatedMessage(
                message_id=uuid4(),
                event_type=SyncEventType.VERSION_CREATED,
                tracklist_id=tracklist_id,
                version_id=version_id,
                version_number=version_number,
                change_type=change_type,
                change_summary=change_summary,
                created_by=created_by,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key="sync.version.created",
                priority=5,
            )

            logger.info(f"Published version created event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish version created event: {e}")
            return False

    async def publish_cue_regeneration_triggered(
        self,
        tracklist_id: UUID,
        trigger: str,
        priority: str,
        cue_formats: list[str],
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish CUE regeneration triggered event.

        Args:
            tracklist_id: ID of the tracklist
            trigger: What triggered regeneration
            priority: Regeneration priority
            cue_formats: CUE formats to regenerate
            actor: Who triggered regeneration
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = CueRegenerationTriggeredMessage(
                message_id=uuid4(),
                event_type=SyncEventType.CUE_REGENERATION_TRIGGERED,
                tracklist_id=tracklist_id,
                trigger=trigger,
                priority=priority,
                cue_formats=cue_formats,
                job_count=len(cue_formats),
                actor=actor,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            # Determine priority based on regeneration priority
            message_priority = {
                "critical": 10,
                "high": 8,
                "normal": 5,
                "low": 3,
                "batch": 1,
            }.get(priority, 5)

            await self._publish_message(
                message,
                routing_key="sync.cue.triggered",
                priority=message_priority,
            )

            logger.info(f"Published CUE regeneration triggered event for tracklist {tracklist_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish CUE regeneration event: {e}")
            return False

    async def publish_batch_sync(
        self,
        tracklist_ids: list[UUID],
        source: str,
        operation: str,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish batch sync event.

        Args:
            tracklist_ids: List of tracklist IDs
            source: Sync source
            operation: Batch operation type
            actor: Who triggered the batch sync
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            message = BatchSyncMessage(
                message_id=uuid4(),
                event_type=SyncEventType.BATCH_SYNC_STARTED,
                tracklist_ids=tracklist_ids,
                tracklist_count=len(tracklist_ids),
                source=source,
                operation=operation,
                actor=actor,
                correlation_id=uuid4(),
                progress=0,
                metadata=metadata or {},
            )

            await self._publish_message(
                message,
                routing_key=f"sync.batch.{operation}",
                priority=4,  # Lower priority for batch operations
            )

            logger.info(f"Published batch sync event for {len(tracklist_ids)} tracklists")
            return True

        except Exception as e:
            logger.error(f"Failed to publish batch sync event: {e}")
            return False

    async def publish_sync_status_update(
        self,
        tracklist_id: UUID,
        status: str,
        progress: int | None = None,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Publish sync status update.

        Args:
            tracklist_id: ID of the tracklist
            status: Current status
            progress: Progress percentage (0-100)
            message: Status message
            metadata: Additional metadata

        Returns:
            True if published successfully
        """
        try:
            update_message = SyncStatusUpdateMessage(
                message_id=uuid4(),
                event_type=SyncEventType.SYNC_STATUS_UPDATE,
                tracklist_id=tracklist_id,
                status=status,
                progress=progress,
                message=message,
                correlation_id=uuid4(),
                metadata=metadata or {},
            )

            await self._publish_message(
                update_message,
                routing_key="sync.status.update",
                priority=3,  # Low priority for status updates
            )

            logger.debug(f"Published status update for tracklist {tracklist_id}: {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish status update: {e}")
            return False

    async def _publish_message(
        self,
        message: Any,
        routing_key: str,
        priority: int = 5,
    ) -> None:
        """Publish a message to RabbitMQ.

        Args:
            message: Message to publish
            routing_key: Routing key for the message
            priority: Message priority (1-10)
        """

        if not self.exchange:
            await self.connect()

        if not self.exchange:
            raise ConnectionError("Exchange not established after connection attempt")

        # Create AMQP message
        amqp_message = Message(
            body=message.to_json().encode(),
            content_type="application/json",
            priority=priority,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            timestamp=datetime.now(UTC),
            message_id=str(message.message_id),
            headers={
                "event_type": (str(message.event_type.value) if hasattr(message, "event_type") else ""),
                "retry_count": 0,
                "max_retries": 3,
            },
        )

        # Publish message
        await self.exchange.publish(
            amqp_message,
            routing_key=routing_key,
        )

        logger.debug(f"Published message {message.message_id} with routing key {routing_key}")
