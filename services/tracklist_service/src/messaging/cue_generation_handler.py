"""
Message handler for CUE generation operations.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from aio_pika.abc import AbstractIncomingMessage
from services.analysis_service.src.cue_handler.converter import ConversionMode, CueConverter
from services.analysis_service.src.cue_handler.generator import CueFormat
from services.analysis_service.src.cue_handler.validator import CueValidator
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
    CueFileDB,
    CueGenerationResponse,
    GenerateCueRequest,
)
from services.tracklist_service.src.models.cue_file import (
    CueFormat as LocalCueFormat,
)
from services.tracklist_service.src.models.tracklist import Tracklist, TracklistDB
from services.tracklist_service.src.services.cue_generation_service import CueGenerationService
from services.tracklist_service.src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class CueGenerationMessageHandler:
    """Handler for CUE generation message processing."""

    def __init__(
        self,
        cue_generation_service: CueGenerationService,
        storage_service: StorageService,
        rabbitmq_client: RabbitMQClient,
        session_factory: Callable[[], Any],
    ):
        """
        Initialize message handler.

        Args:
            cue_generation_service: CUE generation service instance
            storage_service: Storage service instance
            rabbitmq_client: RabbitMQ client instance
            session_factory: Database session factory
        """
        self.cue_generation_service = cue_generation_service
        self.storage_service = storage_service
        self.rabbitmq_client = rabbitmq_client
        self.session_factory = session_factory

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
            # Get tracklist from database
            tracklist = await self.get_tracklist(message.tracklist_id)
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
                format=LocalCueFormat(message.format),
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
            # Get tracklist from database
            tracklist = await self.get_tracklist(message.tracklist_id)
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
                formats=[LocalCueFormat(fmt) for fmt in message.formats],
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
            # Get the CUE file path from the repository
            async with self.session_factory() as session:
                # Get the CUE file from database
                cue_file = await session.get(CueFileDB, message.cue_file_id)
                if not cue_file:
                    logger.error(f"CUE file {message.cue_file_id} not found")
                    await rabbitmq_message.reject(requeue=False)
                    return

                # Initialize validator
                validator = CueValidator()

                # Perform validation
                validation_result = validator.validate(cue_file.file_path)

                # Store validation results in the database
                cue_file.validation_status = "valid" if validation_result.is_valid else "invalid"
                cue_file.validation_errors = [
                    {"line": e.line_number, "message": e.message, "severity": str(e.severity)}
                    for e in validation_result.errors
                ]
                cue_file.validation_warnings = [
                    {"line": w.line_number, "message": w.message, "severity": str(w.severity)}
                    for w in validation_result.warnings
                ]
                cue_file.last_validated_at = datetime.now(UTC)

                await session.commit()

                logger.info(
                    f"CUE validation request {message.validation_job_id} completed. "
                    f"Valid: {validation_result.is_valid}, "
                    f"Errors: {len(validation_result.errors)}, "
                    f"Warnings: {len(validation_result.warnings)}"
                )

                # Log validation completion (message publishing can be added later when schema is created)

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
            # Get the source CUE file from the repository
            async with self.session_factory() as session:
                # Get the source CUE file from database
                source_cue = await session.get(CueFileDB, message.source_cue_file_id)
                if not source_cue:
                    logger.error(f"Source CUE file {message.source_cue_file_id} not found")
                    await rabbitmq_message.reject(requeue=False)
                    return

                # Initialize converter
                converter = CueConverter(mode=ConversionMode.STANDARD, validate_output=True, verbose=False)

                # Determine target format
                try:
                    target_format = CueFormat[message.target_format.upper()]
                except KeyError:
                    logger.error(f"Invalid target format: {message.target_format}")
                    await rabbitmq_message.reject(requeue=False)
                    return

                # Generate output file path
                source_path = Path(source_cue.file_path)
                output_path = source_path.with_suffix(f".{message.target_format.lower()}.cue")

                # Perform conversion
                conversion_report = converter.convert(
                    source_file=source_cue.file_path,
                    target_format=target_format,
                    output_file=str(output_path),
                )

                # Create new CUE file record for the converted file
                if conversion_report.success:
                    converted_cue = CueFileDB(
                        id=uuid4(),
                        tracklist_id=source_cue.tracklist_id,
                        file_path=str(output_path),
                        file_format=message.target_format.lower(),
                        file_size=output_path.stat().st_size if output_path.exists() else 0,
                        created_at=datetime.now(UTC),
                        is_primary=False,
                        source_type="conversion",
                        metadata={
                            "source_file_id": str(message.source_cue_file_id),
                            "conversion_job_id": str(message.conversion_job_id),
                            "changes": [
                                {"type": c.change_type, "command": c.command, "reason": c.reason}
                                for c in conversion_report.changes
                            ],
                            "warnings": conversion_report.warnings,
                        },
                    )
                    session.add(converted_cue)
                    await session.commit()

                    logger.info(
                        f"CUE conversion request {message.conversion_job_id} completed successfully. "
                        f"Converted {source_cue.file_path} to {output_path}"
                    )
                else:
                    logger.error(
                        f"CUE conversion request {message.conversion_job_id} failed. Errors: {conversion_report.errors}"
                    )

            await rabbitmq_message.ack()

        except Exception as e:
            logger.error(
                f"Error processing CUE conversion request {message.conversion_job_id}: {e}",
                exc_info=True,
            )
            await rabbitmq_message.reject(requeue=True)

    async def get_tracklist(self, tracklist_id: UUID) -> Tracklist | None:
        """
        Get tracklist by ID from database.

        Args:
            tracklist_id: UUID of the tracklist to retrieve

        Returns:
            Tracklist model if found, None otherwise

        Raises:
            Exception: If database query fails
        """
        try:
            async with self.session_factory() as session:
                # Query the tracklist from database
                tracklist_db = await session.get(TracklistDB, tracklist_id)

                if not tracklist_db:
                    logger.warning(f"Tracklist {tracklist_id} not found in database")
                    return None

                # Convert to Pydantic model for use by CUE generation service
                tracklist_model: Tracklist = tracklist_db.to_model()

                logger.debug(f"Retrieved tracklist {tracklist_id} with {len(tracklist_model.tracks)} tracks")
                return tracklist_model

        except Exception as e:
            logger.error(f"Error retrieving tracklist {tracklist_id} from database: {e}", exc_info=True)
            raise

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
