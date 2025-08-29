"""
Integration of async audio processing with RabbitMQ message queue.

This module provides async message consumption, batch processing,
and result publishing with the new async audio analysis infrastructure.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aio_pika
from aio_pika import ExchangeType, IncomingMessage

from services.analysis_service.src.async_audio_analysis import (
    AsyncAudioAnalyzer,
    AudioAnalysisResult,
)
from services.analysis_service.src.async_audio_processor import (
    AsyncAudioProcessor,
    AudioAnalysisScheduler,
    TaskPriority,
)
from services.analysis_service.src.async_error_handler import (
    AsyncErrorHandler,
)
from services.analysis_service.src.async_progress_tracker import (
    AsyncProgressTracker,
    BatchProgressAggregator,
)
from services.analysis_service.src.async_resource_manager import AsyncResourceManager

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRequest:
    """Audio analysis request from message queue."""

    recording_id: str
    file_path: str
    analysis_types: List[str]
    priority: TaskPriority
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None


class AsyncMessageQueueIntegration:
    """
    Integrates async audio processing with RabbitMQ message queue.
    """

    def __init__(
        self,
        rabbitmq_url: str,
        processor: AsyncAudioProcessor,
        analyzer: AsyncAudioAnalyzer,
        tracker: AsyncProgressTracker,
        resource_manager: AsyncResourceManager,
        error_handler: AsyncErrorHandler,
        queue_name: str = "audio_analysis",
        exchange_name: str = "analysis",
        enable_batch_processing: bool = True,
        batch_size: int = 10,
        batch_timeout_seconds: float = 5.0,
    ):
        """
        Initialize message queue integration.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            processor: AsyncAudioProcessor instance
            analyzer: AsyncAudioAnalyzer instance
            tracker: AsyncProgressTracker instance
            resource_manager: AsyncResourceManager instance
            error_handler: AsyncErrorHandler instance
            queue_name: Queue name for consuming messages
            exchange_name: Exchange name for publishing results
            enable_batch_processing: Enable batch message processing
            batch_size: Maximum batch size
            batch_timeout_seconds: Batch timeout
        """
        self.rabbitmq_url = rabbitmq_url
        self.processor = processor
        self.analyzer = analyzer
        self.tracker = tracker
        self.resource_manager = resource_manager
        self.error_handler = error_handler
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds

        # Connection state
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.queue: Optional[aio_pika.Queue] = None
        self.exchange: Optional[aio_pika.Exchange] = None

        # Batch processing
        self.batch_buffer: List[AnalysisRequest] = []
        self.batch_messages: List[IncomingMessage] = []
        self.batch_lock = asyncio.Lock()
        self.batch_timer_task: Optional[asyncio.Task] = None

        # Scheduler for prioritized processing
        self.scheduler = AudioAnalysisScheduler(processor)

        logger.info(f"AsyncMessageQueueIntegration initialized for queue: {queue_name}")

    async def connect(self) -> None:
        """Connect to RabbitMQ and set up queue."""
        try:
            # Create connection
            self.connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                client_properties={"connection_name": "analysis_service_async"},
            )

            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.batch_size)

            # Declare exchange
            self.exchange = await self.channel.declare_exchange(self.exchange_name, ExchangeType.TOPIC, durable=True)

            # Declare queue
            self.queue = await self.channel.declare_queue(
                self.queue_name, durable=True, arguments={"x-max-priority": 10}
            )

            # Bind queue to exchange
            await self.queue.bind(self.exchange, routing_key="analysis.*")

            logger.info("Connected to RabbitMQ successfully")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        if not self.queue:
            await self.connect()

        # Start scheduler
        await self.scheduler.start()

        # Start consuming
        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    await self._handle_message(message)
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await message.nack(requeue=True)

    async def _handle_message(self, message: IncomingMessage) -> None:
        """
        Handle incoming message.

        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Parse message
            request = await self._parse_message(message)

            if self.enable_batch_processing:
                # Add to batch
                await self._add_to_batch(request, message)
            else:
                # Process immediately
                await self._process_single_request(request, message)

        except Exception as e:
            logger.error(f"Failed to handle message: {str(e)}")
            await message.nack(requeue=True)

    async def _parse_message(self, message: IncomingMessage) -> AnalysisRequest:
        """
        Parse incoming message to AnalysisRequest.

        Args:
            message: RabbitMQ message

        Returns:
            AnalysisRequest object
        """
        try:
            data = json.loads(message.body.decode())

            # Determine priority from message
            priority_value = message.priority or 5
            if priority_value >= 8:
                priority = TaskPriority.CRITICAL
            elif priority_value >= 6:
                priority = TaskPriority.HIGH
            elif priority_value >= 3:
                priority = TaskPriority.NORMAL
            else:
                priority = TaskPriority.LOW

            return AnalysisRequest(
                recording_id=data["recording_id"],
                file_path=data["file_path"],
                analysis_types=data.get("analysis_types", ["bpm", "key", "mood"]),
                priority=priority,
                metadata=data.get("metadata", {}),
                correlation_id=message.correlation_id,
            )

        except Exception as e:
            logger.error(f"Failed to parse message: {str(e)}")
            raise

    async def _add_to_batch(self, request: AnalysisRequest, message: IncomingMessage) -> None:
        """
        Add request to batch buffer.

        Args:
            request: Analysis request
            message: Original message
        """
        async with self.batch_lock:
            self.batch_buffer.append(request)
            self.batch_messages.append(message)

            # Process if batch is full
            if len(self.batch_buffer) >= self.batch_size:
                await self._process_batch()
            elif not self.batch_timer_task or self.batch_timer_task.done():
                # Start batch timer
                self.batch_timer_task = asyncio.create_task(self._batch_timer())

    async def _batch_timer(self) -> None:
        """Batch timeout timer."""
        await asyncio.sleep(self.batch_timeout)
        async with self.batch_lock:
            if self.batch_buffer:
                await self._process_batch()

    async def _process_batch(self) -> None:
        """Process buffered batch of requests."""
        if not self.batch_buffer:
            return

        # Get current batch
        requests = self.batch_buffer.copy()
        messages = self.batch_messages.copy()
        self.batch_buffer.clear()
        self.batch_messages.clear()

        logger.info(f"Processing batch of {len(requests)} analysis requests")

        # Create batch progress aggregator
        batch_id = f"batch_{requests[0].recording_id[:8]}"
        aggregator = BatchProgressAggregator(self.tracker)

        # Start batch tracking
        task_ids = [req.recording_id for req in requests]
        await aggregator.start_batch(batch_id, task_ids)

        # Process requests in parallel
        tasks = []
        for i, request in enumerate(requests):
            task = asyncio.create_task(self._process_request_with_tracking(request, messages[i], aggregator, batch_id))
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {requests[i].recording_id} failed: {str(result)}")
                await messages[i].nack(requeue=True)
            else:
                await messages[i].ack()

        # Complete batch
        await aggregator.complete_batch(batch_id)

    async def _process_request_with_tracking(
        self,
        request: AnalysisRequest,
        message: IncomingMessage,
        aggregator: Optional[BatchProgressAggregator],
        batch_id: Optional[str],
    ) -> AudioAnalysisResult:
        """
        Process request with progress tracking.

        Args:
            request: Analysis request
            message: Original message
            aggregator: Batch aggregator (optional)
            batch_id: Batch ID (optional)

        Returns:
            Analysis result
        """
        task_id = request.recording_id

        try:
            # Start task tracking
            await self.tracker.start_task(
                task_id,
                total_stages=len(request.analysis_types),
                metadata={"file": request.file_path},
            )

            # Perform analysis directly
            result = await self._perform_analysis(request.file_path, request)

            # Update batch progress if applicable
            if aggregator and batch_id:
                await aggregator.update_batch_task(batch_id, task_id, 100.0)

            # Publish results if successful
            if result:
                await self._publish_results(result, request)
                return result
            else:
                raise RuntimeError(f"Analysis failed for {request.file_path}")

        except Exception as e:
            await self.tracker.fail_task(task_id, str(e))
            raise

    async def _perform_analysis(self, audio_file: str, request: AnalysisRequest) -> Optional[AudioAnalysisResult]:
        """
        Perform audio analysis with error handling.

        Args:
            audio_file: Audio file path
            request: Analysis request

        Returns:
            Analysis result
        """
        # Perform analysis with retry logic
        result = await self.error_handler.handle_with_retry(
            self.analyzer.analyze_audio_complete,
            audio_file,
            task_id=request.recording_id,
            audio_file=audio_file,
            timeout=30.0,
            enable_bpm="bpm" in request.analysis_types,
            enable_key="key" in request.analysis_types,
            enable_mood="mood" in request.analysis_types,
            priority=request.priority,
        )

        return result

    async def _analysis_complete_callback(
        self,
        audio_file: str,
        result: Optional[AudioAnalysisResult],
        error: Optional[Exception],
    ) -> None:
        """
        Callback when analysis completes.

        Args:
            audio_file: Audio file path
            result: Analysis result (if successful)
            error: Error (if failed)
        """
        if result:
            logger.info(f"Analysis completed for {audio_file}")
        else:
            logger.error(f"Analysis failed for {audio_file}: {str(error)}")

    async def _process_single_request(self, request: AnalysisRequest, message: IncomingMessage) -> None:
        """
        Process a single request immediately.

        Args:
            request: Analysis request
            message: Original message
        """
        try:
            await self._process_request_with_tracking(request, message, None, None)
            await message.ack()
        except Exception as e:
            logger.error(f"Failed to process request: {str(e)}")
            await message.nack(requeue=True)

    async def _publish_results(self, result: AudioAnalysisResult, request: AnalysisRequest) -> None:
        """
        Publish analysis results to result queue.

        Args:
            result: Analysis result
            request: Original request
        """
        if not self.exchange:
            logger.warning("Exchange not available for publishing results")
            return

        try:
            # Prepare result message
            result_data = {
                "recording_id": request.recording_id,
                "file_path": result.file_path,
                "bpm": result.bpm,
                "key": result.key,
                "mood": result.mood,
                "metadata": result.metadata,
                "processing_time_ms": result.processing_time_ms,
                "errors": result.errors,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Publish to result exchange
            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(result_data).encode(),
                    correlation_id=request.correlation_id,
                    content_type="application/json",
                ),
                routing_key="analysis.result",
            )

            logger.debug(f"Published results for {request.recording_id}")

        except Exception as e:
            logger.error(f"Failed to publish results: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown message queue integration."""
        logger.info("Shutting down message queue integration")

        # Stop scheduler
        await self.scheduler.stop()

        # Process remaining batch
        async with self.batch_lock:
            if self.batch_buffer:
                await self._process_batch()

        # Close connections
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()

        logger.info("Message queue integration shutdown complete")
