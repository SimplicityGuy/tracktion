"""RabbitMQ event consumer for handling synchronization events."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aio_pika
from aio_pika import IncomingMessage, ExchangeType
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQClient
from services.tracklist_service.src.messaging.sync_message_schemas import (
    SyncTriggerRequest,
    ConflictResolutionRequest,
    VersionRollbackRequest,
    BatchSyncRequest,
    CueRegenerationTriggeredMessage,
)
from services.tracklist_service.src.services.sync_service import (
    SynchronizationService,
    SyncSource,
)
from services.tracklist_service.src.services.conflict_resolution_service import ConflictResolutionService
from services.tracklist_service.src.services.version_service import VersionService
from services.tracklist_service.src.services.cue_regeneration_service import CueRegenerationService

logger = logging.getLogger(__name__)


class SyncEventConsumer:
    """Consumer for synchronization-related events."""

    def __init__(self, rabbitmq_client: Optional[RabbitMQClient] = None):
        """Initialize sync event consumer.

        Args:
            rabbitmq_client: RabbitMQ client instance
        """
        from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQConfig
        self.rabbitmq_client = rabbitmq_client or RabbitMQClient(RabbitMQConfig())
        self.channel: Any = None
        self.queue = None
        self.consumer_tag = None

        # Database setup
        config = get_config()
        # Build database URL from config
        db_url = f"postgresql+asyncpg://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
        self.engine = create_async_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Services will be initialized per session
        self.sync_service = None
        self.conflict_service = None
        self.version_service = None
        self.cue_service = None

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            await self.rabbitmq_client.connect()
            if self.rabbitmq_client.connection:
                self.channel = await self.rabbitmq_client.connection.channel()
            else:
                raise RuntimeError("RabbitMQ connection not established")

            # Set prefetch count for load balancing
            await self.channel.set_qos(prefetch_count=10)

            # Declare the sync events exchange
            exchange = await self.channel.declare_exchange(
                "tracklist.sync.events",
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare consumer queue
            self.queue = await self.channel.declare_queue(
                "sync.events.consumer",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "tracklist.sync.events.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                },
            )

            # Bind queue to relevant routing keys
            routing_keys = [
                "sync.trigger.*",
                "sync.conflict.resolve",
                "sync.version.rollback",
                "sync.batch.*",
                "sync.cue.process",
            ]

            for routing_key in routing_keys:
                await self.queue.bind(exchange, routing_key)

            logger.info("Connected to RabbitMQ for sync event consumption")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        if not self.queue:
            await self.connect()

        # Start consuming
        self.consumer_tag = await self.queue.consume(self.process_message)
        logger.info("Started consuming sync events")

        # Keep the consumer running
        await asyncio.Future()  # Run forever

    async def stop_consuming(self) -> None:
        """Stop consuming messages."""
        if self.consumer_tag and self.queue:
            await self.queue.cancel(self.consumer_tag)
            self.consumer_tag = None
            logger.info("Stopped consuming sync events")

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        await self.stop_consuming()
        await self.rabbitmq_client.disconnect()
        await self.engine.dispose()

    async def process_message(self, message: IncomingMessage) -> None:
        """Process an incoming message.

        Args:
            message: Incoming RabbitMQ message
        """
        async with message.process():
            try:
                # Parse message
                body = json.loads(message.body.decode())
                routing_key = message.routing_key

                logger.info(f"Processing message with routing key: {routing_key}")

                # Route message based on routing key
                if routing_key and routing_key.startswith("sync.trigger"):
                    await self._handle_sync_trigger(body)
                elif routing_key == "sync.conflict.resolve":
                    await self._handle_conflict_resolution(body)
                elif routing_key == "sync.version.rollback":
                    await self._handle_version_rollback(body)
                elif routing_key and routing_key.startswith("sync.batch"):
                    await self._handle_batch_sync(body)
                elif routing_key == "sync.cue.process":
                    await self._handle_cue_regeneration(body)
                else:
                    logger.warning(f"Unknown routing key: {routing_key}")

            except Exception as e:
                logger.error(f"Failed to process message: {e}")

                # Check retry count
                headers = message.headers or {}
                retry_count = headers.get("retry_count", 0)
                max_retries = headers.get("max_retries", 3)

                if int(retry_count) < int(max_retries):
                    # Requeue with increased retry count
                    await self._requeue_message(message, retry_count + 1)
                else:
                    # Max retries reached, send to DLQ
                    logger.error(f"Max retries reached for message {message.message_id}")
                    # Message will go to DLQ automatically

    async def _handle_sync_trigger(self, body: Dict[str, Any]) -> None:
        """Handle sync trigger request.

        Args:
            body: Message body
        """
        try:
            request = SyncTriggerRequest(**body)

            async with self.SessionLocal() as session:
                sync_service = SynchronizationService(session)

                # Parse source
                source = SyncSource(request.source)

                # Trigger sync
                result = await sync_service.trigger_manual_sync(
                    tracklist_id=request.tracklist_id,
                    source=source,
                    force=request.force,
                    actor=request.actor,
                )

                logger.info(f"Sync triggered for tracklist {request.tracklist_id}: {result['status']}")

        except Exception as e:
            logger.error(f"Failed to handle sync trigger: {e}")
            raise

    async def _handle_conflict_resolution(self, body: Dict[str, Any]) -> None:
        """Handle conflict resolution request.

        Args:
            body: Message body
        """
        try:
            request = ConflictResolutionRequest(**body)

            async with self.SessionLocal() as session:
                conflict_service = ConflictResolutionService(session)

                if request.auto_resolve:
                    # Auto-resolve conflicts
                    # This would need to fetch conflicts and auto-resolve them
                    logger.info(f"Auto-resolving conflicts for {request.tracklist_id}")
                else:
                    # Apply provided resolutions
                    success, error = await conflict_service.resolve_conflicts(
                        tracklist_id=request.tracklist_id,
                        resolutions=request.conflict_resolutions,
                        actor=request.actor,
                    )

                    if success:
                        logger.info(f"Resolved conflicts for tracklist {request.tracklist_id}")
                    else:
                        logger.error(f"Failed to resolve conflicts: {error}")

        except Exception as e:
            logger.error(f"Failed to handle conflict resolution: {e}")
            raise

    async def _handle_version_rollback(self, body: Dict[str, Any]) -> None:
        """Handle version rollback request.

        Args:
            body: Message body
        """
        try:
            request = VersionRollbackRequest(**body)

            async with self.SessionLocal() as session:
                version_service = VersionService(session)

                # Perform rollback
                result = await version_service.rollback_to_version(
                    version_id=request.version_id,
                    create_backup=request.create_backup,
                )

                if result:
                    logger.info(f"Rolled back tracklist {request.tracklist_id} to version {request.version_id}")
                else:
                    logger.error(f"Failed to rollback tracklist {request.tracklist_id}")

        except Exception as e:
            logger.error(f"Failed to handle version rollback: {e}")
            raise

    async def _handle_batch_sync(self, body: Dict[str, Any]) -> None:
        """Handle batch sync request.

        Args:
            body: Message body
        """
        try:
            request = BatchSyncRequest(**body)

            async with self.SessionLocal() as session:
                sync_service = SynchronizationService(session)

                # Process batch sync
                results = []
                source = SyncSource(request.source)

                if request.parallel:
                    # Process in parallel with semaphore
                    semaphore = asyncio.Semaphore(request.max_parallel)

                    async def sync_with_semaphore(tracklist_id):
                        async with semaphore:
                            return await sync_service.trigger_manual_sync(
                                tracklist_id=tracklist_id,
                                source=source,
                                force=False,
                                actor=request.actor,
                            )

                    tasks = [sync_with_semaphore(tid) for tid in request.tracklist_ids]

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # Process sequentially
                    for tracklist_id in request.tracklist_ids:
                        try:
                            result = await sync_service.trigger_manual_sync(
                                tracklist_id=tracklist_id,
                                source=source,
                                force=False,
                                actor=request.actor,
                            )
                            results.append(result)
                        except Exception as e:
                            if request.continue_on_error:
                                results.append({"status": "error", "error": str(e)})
                            else:
                                raise

                # Log results
                successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
                logger.info(f"Batch sync completed: {successful}/{len(request.tracklist_ids)} successful")

        except Exception as e:
            logger.error(f"Failed to handle batch sync: {e}")
            raise

    async def _handle_cue_regeneration(self, body: Dict[str, Any]) -> None:
        """Handle CUE regeneration request.

        Args:
            body: Message body
        """
        try:
            CueRegenerationTriggeredMessage(**body)  # Validate message format

            async with self.SessionLocal() as session:
                cue_service = CueRegenerationService(session)

                # Process regeneration queue
                processed = await cue_service.process_regeneration_queue(
                    max_jobs=10,
                    priority_filter=None,
                )

                logger.info(f"Processed {len(processed)} CUE regeneration jobs")

        except Exception as e:
            logger.error(f"Failed to handle CUE regeneration: {e}")
            raise

    async def _requeue_message(
        self,
        message: IncomingMessage,
        retry_count: int,
    ) -> None:
        """Requeue a message with increased retry count.

        Args:
            message: Original message
            retry_count: New retry count
        """
        try:
            # Update headers
            headers = dict(message.headers or {})
            headers["retry_count"] = retry_count

            # Create new message
            new_message = aio_pika.Message(
                body=message.body,
                content_type=message.content_type,
                priority=message.priority,
                delivery_mode=message.delivery_mode,
                headers=headers,
            )

            # Publish back to exchange with delay
            exchange = await self.channel.get_exchange("tracklist.sync.events")
            await exchange.publish(
                new_message,
                routing_key=message.routing_key,
            )

            logger.info(f"Requeued message with retry count {retry_count}")

        except Exception as e:
            logger.error(f"Failed to requeue message: {e}")
