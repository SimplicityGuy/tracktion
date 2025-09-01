"""
Message handler for CUE generation operations.
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Callable

from aio_pika.abc import AbstractIncomingMessage

from services.tracklist_service.src.messaging.message_schemas import (
    BaseMessage,
    BatchCueGenerationCompleteMessage,
    BatchCueGenerationMessage,
    CueConversionMessage,
    CueGenerationCompleteMessage,
    CueGenerationMessage,
    CueValidationMessage,
    MessageType,
)
from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQClient
from services.tracklist_service.src.models.cue_file import (
    BatchCueGenerationResponse,
    BatchGenerateCueRequest,
    CueFormat,
    CueGenerationResponse,
    GenerateCueRequest,
)
from services.tracklist_service.src.services.cue_generation_service import (
    CueGenerationService,
)
from services.tracklist_service.src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class CueGenerationMessageHandler:
    """Handler for CUE generation message processing."""

    def __init__(
        self,
        cue_generation_service: CueGenerationService,
        storage_service: StorageService,
        rabbitmq_client: RabbitMQClient,
    ):
        """
        Initialize message handler.

        Args:
            cue_generation_service: CUE generation service instance
            storage_service: Storage service instance
            rabbitmq_client: RabbitMQ client instance
        """
        self.cue_generation_service = cue_generation_service
        self.storage_service = storage_service
        self.rabbitmq_client = rabbitmq_client

    async def handle_cue_generation(
        self, message: CueGenerationMessage, rabbitmq_message: AbstractIncomingMessage
    ) -> None:
        """
        Handle single CUE file generation requests.

        Args:
            message: CUE generation message
            rabbitmq_message: RabbitMQ message for acknowledgment
        """
        start_time = datetime.now(UTC)
        logger.info(f"Processing CUE generation request {message.job_id} for tracklist {message.tracklist_id}")

        try:
            # Get tracklist (placeholder - implement actual retrieval)
            tracklist = await self._get_tracklist_by_id(message.tracklist_id)
            if not tracklist:
                await self._send_completion_message(
                    message,
                    success=False,
                    error=f"Tracklist {message.tracklist_id} not found",
                    error_code="TRACKLIST_NOT_FOUND",
                    start_time=start_time,
                )
                await rabbitmq_message.ack()
                return

            # Create generation request
            request = GenerateCueRequest(
                format=CueFormat(message.format),
                options=message.options,
                validate_audio=message.validate_audio,
                audio_file_path=message.audio_file_path,
            )

            # Generate CUE file
            response: CueGenerationResponse = await self.cue_generation_service.generate_cue_file(tracklist, request)

            # Send completion message
            await self._send_completion_message(
                message,
                success=response.success,
                cue_file_id=str(response.cue_file_id) if response.cue_file_id else None,
                file_path=response.file_path,
                validation_report=(response.validation_report.model_dump() if response.validation_report else None),
                error=response.error,
                processing_time_ms=response.processing_time_ms,
                start_time=start_time,
            )

            await rabbitmq_message.ack()
            logger.info(
                f"Completed CUE generation request {message.job_id}: {'success' if response.success else 'failed'}"
            )

        except Exception as e:
            logger.error(
                f"Error processing CUE generation request {message.job_id}: {e}",
                exc_info=True,
            )

            await self._send_completion_message(
                message,
                success=False,
                error=f"Internal error: {e!s}",
                error_code="INTERNAL_ERROR",
                start_time=start_time,
            )

            # Check retry count before rejecting
            retry_count = message.retry_count
            if retry_count < 3:  # Max 3 retries
                # Increment retry count and republish
                message.retry_count += 1
                await self.rabbitmq_client.publish_message(message, delay_seconds=retry_count * 30)
                await rabbitmq_message.ack()
                logger.info(f"Retrying CUE generation request {message.job_id} (attempt {retry_count + 1})")
            else:
                await rabbitmq_message.reject(requeue=False)
                logger.error(f"Max retries exceeded for CUE generation request {message.job_id}")

    async def handle_batch_cue_generation(
        self,
        message: BatchCueGenerationMessage,
        rabbitmq_message: AbstractIncomingMessage,
    ) -> None:
        """
        Handle batch CUE file generation requests.

        Args:
            message: Batch CUE generation message
            rabbitmq_message: RabbitMQ message for acknowledgment
        """
        start_time = datetime.now(UTC)
        logger.info(
            f"Processing batch CUE generation request {message.batch_job_id} "
            f"for tracklist {message.tracklist_id} ({len(message.formats)} formats)"
        )

        try:
            # Get tracklist
            tracklist = await self._get_tracklist_by_id(message.tracklist_id)
            if not tracklist:
                await self._send_batch_completion_message(
                    message,
                    success=False,
                    error=f"Tracklist {message.tracklist_id} not found",
                    start_time=start_time,
                )
                await rabbitmq_message.ack()
                return

            # Create batch generation request
            request = BatchGenerateCueRequest(
                formats=[CueFormat(fmt) for fmt in message.formats],
                options=message.options,
                validate_audio=message.validate_audio,
                audio_file_path=message.audio_file_path,
            )

            # Generate CUE files
            response: BatchCueGenerationResponse = await self.cue_generation_service.generate_multiple_formats(
                tracklist, request
            )

            # Send completion message
            await self._send_batch_completion_message(
                message,
                success=response.success,
                total_files=response.total_files,
                successful_files=response.successful_files,
                failed_files=response.failed_files,
                results=[result.model_dump() for result in response.results],
                start_time=start_time,
            )

            await rabbitmq_message.ack()
            logger.info(
                f"Completed batch CUE generation request {message.batch_job_id}: "
                f"{response.successful_files}/{response.total_files} successful"
            )

        except Exception as e:
            logger.error(
                f"Error processing batch CUE generation request {message.batch_job_id}: {e}",
                exc_info=True,
            )

            await self._send_batch_completion_message(
                message,
                success=False,
                error=f"Internal error: {e!s}",
                start_time=start_time,
            )

            # Retry logic similar to single generation
            retry_count = message.retry_count
            if retry_count < 2:  # Fewer retries for batch operations
                message.retry_count += 1
                await self.rabbitmq_client.publish_message(message, delay_seconds=retry_count * 60)
                await rabbitmq_message.ack()
                logger.info(f"Retrying batch CUE generation request {message.batch_job_id} (attempt {retry_count + 1})")
            else:
                await rabbitmq_message.reject(requeue=False)
                logger.error(f"Max retries exceeded for batch CUE generation request {message.batch_job_id}")

    async def handle_cue_validation(
        self,
        message: CueValidationMessage,
        rabbitmq_message: AbstractIncomingMessage,
    ) -> None:
        """
        Handle CUE file validation requests.

        Args:
            message: CUE validation message
            rabbitmq_message: RabbitMQ message for acknowledgment
        """
        logger.info(f"Processing CUE validation request {message.validation_job_id} for file {message.cue_file_id}")

        try:
            # TODO: Implement validation logic
            # For now, just acknowledge the message
            logger.info(f"CUE validation request {message.validation_job_id} completed (placeholder)")
            await rabbitmq_message.ack()

        except Exception as e:
            logger.error(
                f"Error processing CUE validation request {message.validation_job_id}: {e}",
                exc_info=True,
            )
            await rabbitmq_message.reject(requeue=True)

    async def handle_cue_conversion(
        self,
        message: CueConversionMessage,
        rabbitmq_message: AbstractIncomingMessage,
    ) -> None:
        """
        Handle CUE file conversion requests.

        Args:
            message: CUE conversion message
            rabbitmq_message: RabbitMQ message for acknowledgment
        """
        logger.info(
            f"Processing CUE conversion request {message.conversion_job_id} "
            f"for file {message.source_cue_file_id} to {message.target_format}"
        )

        try:
            # TODO: Implement conversion logic
            # For now, just acknowledge the message
            logger.info(f"CUE conversion request {message.conversion_job_id} completed (placeholder)")
            await rabbitmq_message.ack()

        except Exception as e:
            logger.error(
                f"Error processing CUE conversion request {message.conversion_job_id}: {e}",
                exc_info=True,
            )
            await rabbitmq_message.reject(requeue=True)

    async def _get_tracklist_by_id(self, tracklist_id: UUID) -> Any | None:
        """Get tracklist by ID (placeholder implementation)."""
        # TODO: Implement actual tracklist retrieval from database
        # This would interface with the tracklist repository/service
        logger.debug(f"Retrieving tracklist {tracklist_id} (placeholder)")

        # Return mock tracklist for testing until database integration is complete
        # In production, this would query the actual tracklist database

        class MockTrack:
            def __init__(self, idx: int):
                self.title = f"Track {idx}"
                self.artist = f"Artist {idx}"
                self.start_time = f"{(idx - 1) * 5:02d}:00:00"
                self.end_time = f"{idx * 5:02d}:00:00" if idx < 10 else None
                self.bpm = 120 + idx
                self.key = "Am" if idx % 2 else "C"

        class MockTracklist:
            def __init__(self, tracklist_id: UUID):
                self.id = tracklist_id
                self.title = "Test Mix"
                self.artist = "Test DJ"
                self.audio_file_path = "audio.wav"
                self.created_at = datetime.now(UTC)
                self.tracks = [MockTrack(i) for i in range(1, 6)]
                self.genre = "Electronic"
                self.source = "test"

        # Return mock data for testing
        # This ensures the service can function until proper DB integration
        return MockTracklist(tracklist_id)

    async def _send_completion_message(
        self,
        original_message: CueGenerationMessage,
        success: bool,
        cue_file_id: str | None = None,
        file_path: str | None = None,
        validation_report: dict[str, Any] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        processing_time_ms: float | None = None,
        start_time: datetime | None = None,
    ) -> None:
        """Send CUE generation completion message."""
        queue_time_ms = None
        if start_time and original_message.timestamp:
            queue_time_ms = (start_time - original_message.timestamp).total_seconds() * 1000

        completion_message = CueGenerationCompleteMessage(
            message_id=uuid4(),
            message_type=MessageType.CUE_GENERATION_COMPLETE,
            correlation_id=original_message.correlation_id,
            retry_count=0,
            priority=5,
            original_message_id=original_message.message_id,
            job_id=original_message.job_id,
            tracklist_id=original_message.tracklist_id,
            success=success,
            cue_file_id=UUID(cue_file_id) if cue_file_id else None,
            file_path=file_path,
            file_size=None,
            checksum=None,
            validation_report=validation_report,
            error=error,
            error_code=error_code,
            processing_time_ms=processing_time_ms,
            queue_time_ms=queue_time_ms,
        )

        await self.rabbitmq_client.publish_message(completion_message)

    async def _send_batch_completion_message(
        self,
        original_message: BatchCueGenerationMessage,
        success: bool,
        total_files: int = 0,
        successful_files: int = 0,
        failed_files: int = 0,
        results: list[dict[str, Any]] | None = None,
        error: str | None = None,
        start_time: datetime | None = None,
    ) -> None:
        """Send batch CUE generation completion message."""
        total_processing_time_ms = None
        if start_time:
            total_processing_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        average_processing_time_ms = None
        if total_processing_time_ms and total_files > 0:
            average_processing_time_ms = total_processing_time_ms / total_files

        completion_message = BatchCueGenerationCompleteMessage(
            message_id=uuid4(),
            message_type=MessageType.BATCH_CUE_GENERATION_COMPLETE,
            correlation_id=original_message.correlation_id,
            retry_count=0,
            priority=5,
            original_message_id=original_message.message_id,
            batch_job_id=original_message.batch_job_id,
            tracklist_id=original_message.tracklist_id,
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            results=results or [],
            success=success,
            error=error,
            total_processing_time_ms=total_processing_time_ms,
            average_processing_time_ms=average_processing_time_ms,
        )

        await self.rabbitmq_client.publish_message(completion_message)

    async def start_consuming(self) -> None:
        """Start consuming messages from all relevant queues."""

        try:
            # Start consuming different message types
            await asyncio.gather(
                self.rabbitmq_client.consume_messages(
                    message_type=MessageType.CUE_GENERATION,
                    handler=cast("Callable[[BaseMessage, AbstractIncomingMessage], Any]", self.handle_cue_generation),
                ),
                self.rabbitmq_client.consume_messages(
                    message_type=MessageType.BATCH_CUE_GENERATION,
                    handler=cast(
                        "Callable[[BaseMessage, AbstractIncomingMessage], Any]",
                        self.handle_batch_cue_generation,
                    ),
                ),
                self.rabbitmq_client.consume_messages(
                    message_type=MessageType.CUE_VALIDATION,
                    handler=cast("Callable[[BaseMessage, AbstractIncomingMessage], Any]", self.handle_cue_validation),
                ),
                self.rabbitmq_client.consume_messages(
                    message_type=MessageType.CUE_CONVERSION,
                    handler=cast("Callable[[BaseMessage, AbstractIncomingMessage], Any]", self.handle_cue_conversion),
                ),
            )

        except Exception as e:
            logger.error(f"Error starting message consumption: {e}", exc_info=True)
            raise
