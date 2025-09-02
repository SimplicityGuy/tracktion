"""Main entry point for the analysis service."""

import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from dotenv import load_dotenv

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from bpm_detector import BPMDetector
from config import BPMConfig
from exceptions import InvalidAudioFileError, MetadataExtractionError, RetryableError, StorageError
from file_rename_proposal.config import FileRenameProposalConfig
from file_rename_proposal.integration import FileRenameProposalIntegration
from key_detector import KeyDetector
from message_consumer import MessageConsumer
from metadata_extractor import MetadataExtractor
from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import RecordingRepository
from storage_handler import StorageHandler

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get structured logger
logger = structlog.get_logger()


class AnalysisService:
    """Main analysis service class."""

    def __init__(self) -> None:
        """Initialize the analysis service."""
        self.running = False
        self.consumer: MessageConsumer | None = None
        self.extractor: MetadataExtractor | None = None
        self.storage: StorageHandler | None = None
        self.rename_integration: FileRenameProposalIntegration | None = None
        self.bpm_detector: BPMDetector | None = None
        self.key_detector: KeyDetector | None = None
        self.messaging_service: MessageConsumer | None = None
        self._shutdown_requested = False

        # Configuration
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self.queue_name = os.getenv("ANALYSIS_QUEUE", "analysis_queue")
        self.exchange_name = os.getenv("EXCHANGE_NAME", "tracktion_exchange")
        self.routing_key = os.getenv("ANALYSIS_ROUTING_KEY", "file.analyze")

        # Retry configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "5.0"))

        # Audio analysis configuration
        self.enable_audio_analysis = os.getenv("ENABLE_AUDIO_ANALYSIS", "true").lower() == "true"

    def initialize(self) -> None:
        """Initialize service components."""
        logger.info("Initializing analysis service")

        try:
            # Initialize metadata extractor
            self.extractor = MetadataExtractor()
            logger.info("Metadata extractor initialized")

            # Initialize storage handler
            self.storage = StorageHandler()
            logger.info("Storage handler initialized")

            # Initialize audio analysis components
            if self.enable_audio_analysis:
                try:
                    bpm_config = BPMConfig()
                    self.bpm_detector = BPMDetector(config=bpm_config)
                    logger.info("BPM detector initialized")

                    self.key_detector = KeyDetector()
                    logger.info("Key detector initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize audio analysis components: {e}")
                    self.enable_audio_analysis = False

            # Initialize message consumer
            self.consumer = MessageConsumer(
                rabbitmq_url=self.rabbitmq_url,
                queue_name=self.queue_name,
                exchange_name=self.exchange_name,
                routing_key=self.routing_key,
            )
            logger.info("Message consumer initialized")

            # Initialize file rename proposal integration (optional)
            try:
                db_manager = DatabaseManager()
                proposal_repo = RenameProposalRepository(db_manager)
                recording_repo = RecordingRepository(db_manager)
                rename_config = FileRenameProposalConfig.from_env()

                self.rename_integration = FileRenameProposalIntegration(
                    proposal_repo=proposal_repo,
                    recording_repo=recording_repo,
                    config=rename_config,
                )
                logger.info("File rename proposal integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize file rename proposal integration: {e}")
                self.rename_integration = None

            # Setup signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)

            logger.info("Analysis service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    async def process_message(self, message: dict[str, Any], correlation_id: str) -> None:
        """Process a single analysis message.

        Args:
            message: Message from RabbitMQ
            correlation_id: Correlation ID for tracing
        """
        logger.info(
            "Processing analysis message",
            correlation_id=correlation_id,
            message_keys=list(message.keys()),
        )

        # Extract required fields
        recording_id = message.get("recording_id")
        file_path = message.get("file_path")

        if not recording_id or not file_path:
            logger.error(
                "Invalid message: missing required fields",
                correlation_id=correlation_id,
                has_recording_id=bool(recording_id),
                has_file_path=bool(file_path),
            )
            return

        try:
            # Convert recording_id to UUID
            recording_uuid = UUID(recording_id)

            # Process the file
            self._process_file(recording_uuid, file_path, correlation_id)

            # Send success notification (if configured)
            await self._send_notification(
                recording_id=recording_uuid,
                status="completed",
                correlation_id=correlation_id,
            )

        except (ValueError, TypeError) as e:
            logger.error(
                f"Invalid recording ID format: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id,
            )

        except RetryableError as e:
            # These errors should trigger a retry
            logger.warning(
                f"Retryable error occurred: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id,
            )
            raise  # Let the consumer handle retry

        except Exception as e:
            logger.error(
                f"Failed to process message: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id,
                exc_info=True,
            )

            # Update recording status to failed
            if recording_id:
                try:
                    if self.storage:
                        self.storage.update_recording_status(UUID(recording_id), "failed", str(e), correlation_id)
                except Exception as update_error:
                    logger.error(
                        f"Failed to update recording status: {update_error}",
                        correlation_id=correlation_id,
                    )

    def _process_file(self, recording_id: UUID, file_path: str, correlation_id: str) -> None:
        """Process a single audio file.

        Args:
            recording_id: UUID of the recording
            file_path: Path to the audio file
            correlation_id: Correlation ID for tracing

        Raises:
            Various exceptions based on processing errors
        """
        logger.info(
            f"Processing file: {file_path}",
            correlation_id=correlation_id,
            recording_id=str(recording_id),
        )

        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # Extract metadata
                if not self.extractor:
                    raise RuntimeError("Extractor not initialized")
                metadata = self.extractor.extract(file_path)
                logger.info(
                    f"Extracted {len(metadata)} metadata fields",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    fields=list(metadata.keys()),
                )

                # Perform audio analysis if enabled and file is supported
                if self.enable_audio_analysis and self._is_audio_format_supported(file_path):
                    self._perform_audio_analysis(file_path, metadata, correlation_id, recording_id)

                # Store metadata
                if not self.storage:
                    raise RuntimeError("Storage not initialized")
                self.storage.store_metadata(recording_id, metadata, correlation_id)

                # Generate rename proposal if integration is available
                if self.rename_integration:
                    try:
                        proposal_id = self.rename_integration.process_recording_metadata(
                            recording_id, metadata, correlation_id
                        )
                        if proposal_id:
                            logger.info(
                                f"Generated rename proposal {proposal_id}",
                                correlation_id=correlation_id,
                                recording_id=str(recording_id),
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate rename proposal: {e}",
                            correlation_id=correlation_id,
                            recording_id=str(recording_id),
                        )

                # Update recording status
                self.storage.update_recording_status(recording_id, "processed", None, correlation_id)

                logger.info(
                    f"Successfully processed file: {file_path}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                )
                return

            except InvalidAudioFileError as e:
                # Don't retry for invalid files
                logger.error(
                    f"Invalid audio file: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    file_path=file_path,
                )
                if self.storage:
                    self.storage.update_recording_status(recording_id, "invalid", str(e), correlation_id)
                raise

            except (MetadataExtractionError, StorageError) as e:
                # Retry these errors
                retry_count += 1

                if retry_count <= self.max_retries:
                    logger.warning(
                        f"Processing failed (attempt {retry_count}/{self.max_retries}), retrying: {e}",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id),
                    )
                    time.sleep(self.retry_delay * retry_count)  # Exponential backoff
                else:
                    logger.error(
                        f"Processing failed after {self.max_retries} retries: {e}",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id),
                    )
                    if self.storage:
                        self.storage.update_recording_status(recording_id, "failed", str(e), correlation_id)
                    raise

            except Exception as e:
                # Unexpected error - don't retry
                logger.error(
                    f"Unexpected error processing file: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    exc_info=True,
                )
                if self.storage:
                    self.storage.update_recording_status(recording_id, "error", str(e), correlation_id)
                raise

    def _is_audio_format_supported(self, file_path: str) -> bool:
        """Check if the audio format is supported for analysis.

        Args:
            file_path: Path to the audio file

        Returns:
            True if format is supported
        """
        if not self.bpm_detector:
            return False

        # Get file extension
        ext = Path(file_path).suffix.lower()

        # Check against BPM detector's supported formats
        # Note: We should reuse the existing config from the detector rather than creating a new one
        if hasattr(self.bpm_detector, "config") and hasattr(self.bpm_detector.config, "supported_formats"):
            return ext in self.bpm_detector.config.supported_formats

        # Fallback to creating new config if detector doesn't expose its config
        bpm_config = BPMConfig()
        return ext in bpm_config.supported_formats

    def _perform_audio_analysis(
        self,
        file_path: str,
        metadata: dict[str, Any],
        correlation_id: str,
        recording_id: UUID,
    ) -> None:
        """Perform BPM and key detection on the audio file.

        Args:
            file_path: Path to the audio file
            metadata: Metadata dictionary to update
            correlation_id: Correlation ID for tracing
            recording_id: UUID of the recording
        """
        # BPM Detection
        if self.bpm_detector:
            try:
                bpm_result = self.bpm_detector.detect_bpm(file_path)

                # Add BPM results to metadata
                metadata["bpm"] = str(round(bpm_result["bpm"], 1))
                metadata["bpm_confidence"] = str(round(bpm_result["confidence"], 2))
                metadata["bpm_algorithm"] = bpm_result["algorithm"]

                if bpm_result.get("needs_review"):
                    metadata["bpm_needs_review"] = "true"

                logger.info(
                    f"BPM detected: {metadata['bpm']} (confidence: {metadata['bpm_confidence']})",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    algorithm=bpm_result["algorithm"],
                )
            except Exception as e:
                logger.warning(
                    f"BPM detection failed: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    file_path=file_path,
                )

        # Key Detection
        if self.key_detector:
            try:
                key_result = self.key_detector.detect_key(file_path)

                if key_result:
                    # Add key results to metadata
                    metadata["key"] = f"{key_result.key} {key_result.scale}"
                    metadata["key_confidence"] = str(round(key_result.confidence, 2))

                    if key_result.alternative_key:
                        metadata["key_alternative"] = f"{key_result.alternative_key} {key_result.alternative_scale}"

                    if key_result.needs_review:
                        metadata["key_needs_review"] = "true"

                    logger.info(
                        f"Key detected: {metadata['key']} (confidence: {metadata['key_confidence']})",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id),
                    )
                else:
                    logger.warning(
                        "Key detection returned no result",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id),
                    )
            except Exception as e:
                logger.warning(
                    f"Key detection failed: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    file_path=file_path,
                )

        # Add analysis version for future compatibility
        metadata["audio_analysis_version"] = "1.0"

    async def _send_notification(
        self,
        recording_id: UUID,
        status: str,
        correlation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send notification about processing completion.

        This is a placeholder for future implementation to notify
        other services about processing completion.

        Args:
            recording_id: UUID of the recording
            status: Processing status
            correlation_id: Correlation ID for tracing
            metadata: Optional metadata to include
        """
        # Send notification via the notification service
        if self.messaging_service:
            try:
                # Create notification message based on status
                notification_data: dict[str, Any] = {
                    "recording_id": str(recording_id),
                    "status": status,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "service": "analysis_service",
                }

                # Add metadata if provided
                if metadata:
                    notification_data["metadata"] = metadata

                # Determine notification type based on status
                if status == "completed":
                    notification_type = "analysis_completed"
                    notification_data["message"] = f"Analysis completed for recording {recording_id}"
                elif status == "failed":
                    notification_type = "analysis_failed"
                    notification_data["message"] = f"Analysis failed for recording {recording_id}"
                    notification_data["alert_type"] = "error"
                elif status == "processing":
                    notification_type = "analysis_started"
                    notification_data["message"] = f"Analysis started for recording {recording_id}"
                else:
                    notification_type = "analysis_status_update"
                    notification_data["message"] = f"Analysis status update for recording {recording_id}: {status}"

                # Send the notification message
                await self.messaging_service.publish_message(
                    exchange_name="notifications",
                    routing_key=notification_type,
                    message=notification_data,
                    correlation_id=correlation_id,
                )

                logger.info(
                    f"Notification sent: recording {recording_id} status={status}",
                    correlation_id=correlation_id,
                )
            except Exception as e:
                logger.error(
                    f"Failed to send notification for recording {recording_id}: {e}",
                    correlation_id=correlation_id,
                    exc_info=True,
                )
        else:
            logger.debug(
                f"Messaging service not available, notification not sent: recording {recording_id} status={status}",
                correlation_id=correlation_id,
            )

    def run(self) -> None:
        """Run the analysis service."""
        logger.info("Starting analysis service")

        try:
            # Initialize components
            self.initialize()

            # Mark service as running
            self.running = True

            # Start consuming messages
            logger.info("Starting message consumption")
            if self.consumer:
                self.consumer.consume(self.process_message)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")

        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)

        finally:
            self.shutdown()

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, initiating shutdown")
        self._shutdown_requested = True
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        if not self.running:
            return

        logger.info("Shutting down analysis service")
        self.running = False

        try:
            # Stop message consumer
            if self.consumer:
                self.consumer.stop()

            # Close storage connections
            if self.storage:
                self.storage.close()

            logger.info("Analysis service shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        health: dict[str, Any] = {
            "service": "analysis_service",
            "status": "healthy" if self.running else "not_running",
            "timestamp": time.time(),
            "components": {},
        }

        # Check RabbitMQ connection
        if self.consumer and self.consumer.connection:
            health["components"]["rabbitmq"] = {
                "status": ("connected" if not self.consumer.connection.is_closed else "disconnected")
            }
        else:
            health["components"]["rabbitmq"] = {"status": "not_initialized"}

        # Check storage
        if self.storage:
            health["components"]["storage"] = {"status": "initialized"}
        else:
            health["components"]["storage"] = {"status": "not_initialized"}

        # Overall health
        components = health["components"]
        if isinstance(components, dict):
            all_healthy = all(
                isinstance(comp, dict) and comp.get("status") in ["connected", "initialized"]
                for comp in components.values()
            )
            health["healthy"] = all_healthy and self.running
        else:
            health["healthy"] = False

        return health


def main() -> None:
    """Main entry point."""
    service = AnalysisService()
    service.run()


if __name__ == "__main__":
    main()
