"""RabbitMQ message consumer for tracklist events."""

import json
import logging
from typing import TYPE_CHECKING

import aio_pika
from aio_pika import ExchangeType

if TYPE_CHECKING:
    from aio_pika.abc import AbstractChannel, AbstractIncomingMessage, AbstractQueue, AbstractRobustConnection

from services.cataloging_service.src.config import get_config
from services.cataloging_service.src.database import get_db_manager
from services.cataloging_service.src.repositories import MetadataRepository, RecordingRepository, TracklistRepository

logger = logging.getLogger(__name__)


class TracklistMessageConsumer:
    """Consumer for tracklist events."""

    def __init__(self) -> None:
        """Initialize the tracklist message consumer."""
        self.config = get_config()
        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.queue: AbstractQueue | None = None
        self.db_manager = get_db_manager()

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            # Connect to RabbitMQ
            self.connection = await aio_pika.connect_robust(
                self.config.rabbitmq.url,
                client_properties={
                    "connection_name": "cataloging-tracklist-consumer",
                },
            )

            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)

            # Declare the tracklist events exchange
            exchange = await self.channel.declare_exchange(
                "tracklist_events",
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare consumer queue
            self.queue = await self.channel.declare_queue(
                "cataloging.tracklist.events",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "tracklist_events.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                },
            )

            # Bind queue to routing keys
            routing_keys = [
                "tracklist.generated",
                "metadata.extracted",
            ]

            for routing_key in routing_keys:
                await self.queue.bind(exchange, routing_key)

            logger.info("Connected to RabbitMQ for tracklist events")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ for tracklist events: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            logger.info("Disconnected from RabbitMQ (tracklist consumer)")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def process_message(self, message: "AbstractIncomingMessage") -> None:
        """Process a tracklist event message.

        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Parse message body
            body = json.loads(message.body.decode())
            event_type = body.get("event_type")
            file_path = body.get("file_path")
            correlation_id = body.get("correlation_id", "unknown")

            logger.info(
                f"Processing {event_type} event for {file_path}",
                extra={"correlation_id": correlation_id},
            )

            # Handle event based on type
            async with self.db_manager.get_session() as session:
                recording_repo = RecordingRepository(session)
                tracklist_repo = TracklistRepository(session)

                if event_type == "tracklist.generated":
                    # Find the recording
                    recording = await recording_repo.get_by_file_path(file_path)
                    if recording:
                        # Extract tracklist data
                        source = body.get("source", "generated")
                        tracks = body.get("tracks", [])
                        cue_file_path = body.get("cue_file_path")

                        # Upsert the tracklist
                        if recording.id is None:
                            raise ValueError("Recording ID cannot be None")
                        await tracklist_repo.upsert(
                            recording_id=recording.id,
                            source=source,
                            tracks=tracks,
                            cue_file_path=cue_file_path,
                        )

                        logger.info(
                            f"Stored tracklist for {file_path} with {len(tracks)} tracks",
                            extra={"correlation_id": correlation_id},
                        )
                    else:
                        logger.warning(
                            f"Recording not found for tracklist: {file_path}",
                            extra={"correlation_id": correlation_id},
                        )

                elif event_type == "metadata.extracted":
                    # Find the recording
                    recording = await recording_repo.get_by_file_path(file_path)
                    if recording:
                        # Extract metadata
                        metadata = body.get("metadata", {})

                        # Store metadata
                        metadata_repo = MetadataRepository(session)

                        if recording.id is None:
                            raise ValueError("Recording ID cannot be None")
                        for key, value in metadata.items():
                            await metadata_repo.upsert(recording.id, key, str(value))

                        logger.info(
                            f"Stored {len(metadata)} metadata entries for {file_path}",
                            extra={"correlation_id": correlation_id},
                        )
                    else:
                        logger.warning(
                            f"Recording not found for metadata: {file_path}",
                            extra={"correlation_id": correlation_id},
                        )

                else:
                    logger.warning(f"Unknown event type: {event_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing tracklist message: {e}")
            raise
