"""Async RabbitMQ message consumer for analysis service."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from uuid import UUID

import aio_pika
from aio_pika import ExchangeType, IncomingMessage

from .async_storage_handler import AsyncStorageHandler
from .async_config import get_config

logger = logging.getLogger(__name__)


class AsyncAnalysisProcessor:
    """Async processor for audio analysis tasks."""

    def __init__(self, storage_handler: AsyncStorageHandler) -> None:
        """Initialize async analysis processor.

        Args:
            storage_handler: Async storage handler instance
        """
        self.storage = storage_handler
        self._processing_semaphore = asyncio.Semaphore(10)  # Limit concurrent processing

    async def process_audio_analysis(
        self, recording_id: UUID, file_path: str, analysis_type: str = "basic"
    ) -> Dict[str, Any]:
        """Process audio analysis asynchronously.

        Args:
            recording_id: UUID of the recording
            file_path: Path to the audio file
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results
        """
        async with self._processing_semaphore:
            # Check cache first
            cached = await self.storage.get_cached_analysis(recording_id=recording_id, analysis_type=analysis_type)
            if cached:
                logger.info(f"Using cached analysis for {recording_id}")
                return cached

            # Simulate async audio analysis (would be actual analysis in production)
            await asyncio.sleep(0.1)  # Simulate processing time

            results = {
                "analysis_type": analysis_type,
                "metadata": {
                    "duration": "5:32",
                    "bitrate": "320kbps",
                    "sample_rate": "44100Hz",
                    "channels": 2,
                    "format": "mp3",
                },
                "features": {"tempo": 128, "key": "A minor", "energy": 0.85, "danceability": 0.72},
            }

            # Store results
            await self.storage.store_analysis_results(
                recording_id=recording_id, analysis_type=analysis_type, results=results
            )

            return results

    async def process_similarity_analysis(self, recording_id: UUID, limit: int = 10) -> Dict[str, Any]:
        """Find similar recordings asynchronously.

        Args:
            recording_id: UUID of the recording
            limit: Maximum number of similar recordings

        Returns:
            Similarity analysis results
        """
        async with self._processing_semaphore:
            similar = await self.storage.find_similar_recordings(recording_id=recording_id, limit=limit)

            return {"recording_id": str(recording_id), "similar_recordings": similar, "count": len(similar)}

    async def process_tracklist_extraction(
        self, recording_id: UUID, file_path: str, source: str = "automatic"
    ) -> Dict[str, Any]:
        """Extract and store tracklist information.

        Args:
            recording_id: UUID of the recording
            file_path: Path to the audio file
            source: Source of the tracklist

        Returns:
            Extraction results
        """
        async with self._processing_semaphore:
            # Simulate async tracklist extraction
            await asyncio.sleep(0.2)

            # Mock tracklist data
            tracks = [
                {"position": 1, "name": "Track 1", "artist": "Artist A", "start_time": "00:00", "duration": "5:32"},
                {"position": 2, "name": "Track 2", "artist": "Artist B", "start_time": "05:32", "duration": "6:15"},
            ]

            # Store tracklist
            success = await self.storage.store_tracklist(recording_id=recording_id, source=source, tracks=tracks)

            return {"success": success, "tracks_count": len(tracks), "source": source}


class AsyncAnalysisMessageConsumer:
    """Async consumer for analysis service messages."""

    def __init__(self) -> None:
        """Initialize the async message consumer."""
        self.config = get_config()
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.queue: Optional[aio_pika.Queue] = None

        # Initialize storage handler
        self.storage = AsyncStorageHandler(
            postgres_url=self.config.database.url,
            neo4j_uri=self.config.neo4j.uri,
            neo4j_auth=(self.config.neo4j.user, self.config.neo4j.password),
            redis_url=self.config.redis.url,
        )

        # Initialize processor
        self.processor = AsyncAnalysisProcessor(self.storage)

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            # Connect to RabbitMQ
            self.connection = await aio_pika.connect_robust(
                self.config.rabbitmq.url,
                client_properties={
                    "connection_name": "analysis-service",
                },
            )

            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)

            # Declare the analysis exchange
            exchange = await self.channel.declare_exchange(
                "analysis.events",
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare consumer queue
            self.queue = await self.channel.declare_queue(
                "analysis.tasks",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "analysis.events.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                },
            )

            # Bind queue to routing keys
            routing_keys = ["analysis.audio", "analysis.similarity", "analysis.tracklist", "analysis.metadata"]

            for routing_key in routing_keys:
                await self.queue.bind(exchange, routing_key)

            logger.info("Connected to RabbitMQ for analysis service")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ and close storage connections."""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            await self.storage.close()
            logger.info("Disconnected from RabbitMQ and storage")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    async def start_consuming(self) -> None:
        """Start consuming messages asynchronously."""
        if not self.queue:
            await self.connect()

        try:
            # Start consuming messages
            assert self.queue is not None
            async with self.queue.iterator() as queue_iter:
                logger.info("Started consuming analysis tasks")

                async for message in queue_iter:
                    async with message.process():
                        await self.process_message(message)

        except asyncio.CancelledError:
            logger.info("Analysis consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in analysis consumer: {e}")
            raise

    async def process_message(self, message: IncomingMessage) -> None:
        """Process an analysis task message.

        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Parse message body
            body = json.loads(message.body.decode())
            task_type = body.get("task_type")
            recording_id = UUID(body.get("recording_id"))
            file_path = body.get("file_path")
            correlation_id = body.get("correlation_id", "unknown")

            logger.info(f"Processing {task_type} task for {recording_id}", extra={"correlation_id": correlation_id})

            # Route to appropriate processor
            if task_type == "audio_analysis":
                result = await self.processor.process_audio_analysis(
                    recording_id=recording_id, file_path=file_path, analysis_type=body.get("analysis_type", "basic")
                )

            elif task_type == "similarity":
                result = await self.processor.process_similarity_analysis(
                    recording_id=recording_id, limit=body.get("limit", 10)
                )

            elif task_type == "tracklist":
                result = await self.processor.process_tracklist_extraction(
                    recording_id=recording_id, file_path=file_path, source=body.get("source", "automatic")
                )

            else:
                logger.warning(f"Unknown task type: {task_type}")
                return

            logger.info(
                f"Successfully processed {task_type} task for {recording_id}",
                extra={"correlation_id": correlation_id, "result": result},
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error processing analysis message: {e}")
            raise


async def main() -> None:
    """Main entry point for analysis message consumer."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    consumer = AsyncAnalysisMessageConsumer()

    try:
        await consumer.connect()
        await consumer.start_consuming()

    except KeyboardInterrupt:
        logger.info("Shutting down analysis consumer")
    finally:
        await consumer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
