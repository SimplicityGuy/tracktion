"""Main entry point for the analysis service."""

import os
import sys
import json
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import UUID
from dotenv import load_dotenv
import structlog

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from message_consumer import MessageConsumer
from metadata_extractor import MetadataExtractor
from storage_handler import StorageHandler
from exceptions import (
    AnalysisServiceError,
    InvalidAudioFileError,
    MetadataExtractionError,
    StorageError,
    RetryableError
)

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
        structlog.processors.JSONRenderer()
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
        self.consumer = None
        self.extractor = None
        self.storage = None
        self._shutdown_requested = False
        
        # Configuration
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        self.queue_name = os.getenv("ANALYSIS_QUEUE", "analysis_queue")
        self.exchange_name = os.getenv("EXCHANGE_NAME", "tracktion_exchange")
        self.routing_key = os.getenv("ANALYSIS_ROUTING_KEY", "file.analyze")
        
        # Retry configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "5.0"))

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
            
            # Initialize message consumer
            self.consumer = MessageConsumer(
                rabbitmq_url=self.rabbitmq_url,
                queue_name=self.queue_name,
                exchange_name=self.exchange_name,
                routing_key=self.routing_key
            )
            logger.info("Message consumer initialized")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
            
            logger.info("Analysis service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    def process_message(self, message: Dict[str, Any], correlation_id: str) -> None:
        """Process a single analysis message.
        
        Args:
            message: Message from RabbitMQ
            correlation_id: Correlation ID for tracing
        """
        logger.info(
            "Processing analysis message",
            correlation_id=correlation_id,
            message_keys=list(message.keys())
        )
        
        # Extract required fields
        recording_id = message.get("recording_id")
        file_path = message.get("file_path")
        
        if not recording_id or not file_path:
            logger.error(
                "Invalid message: missing required fields",
                correlation_id=correlation_id,
                has_recording_id=bool(recording_id),
                has_file_path=bool(file_path)
            )
            return
        
        try:
            # Convert recording_id to UUID
            recording_uuid = UUID(recording_id)
            
            # Process the file
            self._process_file(recording_uuid, file_path, correlation_id)
            
            # Send success notification (if configured)
            self._send_notification(
                recording_id=recording_uuid,
                status="completed",
                correlation_id=correlation_id
            )
            
        except (ValueError, TypeError) as e:
            logger.error(
                f"Invalid recording ID format: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id
            )
            
        except RetryableError as e:
            # These errors should trigger a retry
            logger.warning(
                f"Retryable error occurred: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id
            )
            raise  # Let the consumer handle retry
            
        except Exception as e:
            logger.error(
                f"Failed to process message: {e}",
                correlation_id=correlation_id,
                recording_id=recording_id,
                exc_info=True
            )
            
            # Update recording status to failed
            if recording_id:
                try:
                    self.storage.update_recording_status(
                        UUID(recording_id),
                        "failed",
                        str(e),
                        correlation_id
                    )
                except Exception as update_error:
                    logger.error(
                        f"Failed to update recording status: {update_error}",
                        correlation_id=correlation_id
                    )

    def _process_file(
        self,
        recording_id: UUID,
        file_path: str,
        correlation_id: str
    ) -> None:
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
            recording_id=str(recording_id)
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Extract metadata
                metadata = self.extractor.extract(file_path)
                logger.info(
                    f"Extracted {len(metadata)} metadata fields",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    fields=list(metadata.keys())
                )
                
                # Store metadata
                self.storage.store_metadata(recording_id, metadata, correlation_id)
                
                # Update recording status
                self.storage.update_recording_status(
                    recording_id,
                    "processed",
                    None,
                    correlation_id
                )
                
                logger.info(
                    f"Successfully processed file: {file_path}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id)
                )
                return
                
            except InvalidAudioFileError as e:
                # Don't retry for invalid files
                logger.error(
                    f"Invalid audio file: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    file_path=file_path
                )
                self.storage.update_recording_status(
                    recording_id,
                    "invalid",
                    str(e),
                    correlation_id
                )
                raise
                
            except (MetadataExtractionError, StorageError) as e:
                # Retry these errors
                retry_count += 1
                last_error = e
                
                if retry_count <= self.max_retries:
                    logger.warning(
                        f"Processing failed (attempt {retry_count}/{self.max_retries}), retrying: {e}",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id)
                    )
                    time.sleep(self.retry_delay * retry_count)  # Exponential backoff
                else:
                    logger.error(
                        f"Processing failed after {self.max_retries} retries: {e}",
                        correlation_id=correlation_id,
                        recording_id=str(recording_id)
                    )
                    self.storage.update_recording_status(
                        recording_id,
                        "failed",
                        str(e),
                        correlation_id
                    )
                    raise
                    
            except Exception as e:
                # Unexpected error - don't retry
                logger.error(
                    f"Unexpected error processing file: {e}",
                    correlation_id=correlation_id,
                    recording_id=str(recording_id),
                    exc_info=True
                )
                self.storage.update_recording_status(
                    recording_id,
                    "error",
                    str(e),
                    correlation_id
                )
                raise

    def _send_notification(
        self,
        recording_id: UUID,
        status: str,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None
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
        # TODO: Implement notification sending via RabbitMQ
        logger.debug(
            f"Notification would be sent: recording {recording_id} status={status}",
            correlation_id=correlation_id
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

    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        health = {
            "service": "analysis_service",
            "status": "healthy" if self.running else "not_running",
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check RabbitMQ connection
        if self.consumer and self.consumer.connection:
            health["components"]["rabbitmq"] = {
                "status": "connected" if not self.consumer.connection.is_closed else "disconnected"
            }
        else:
            health["components"]["rabbitmq"] = {"status": "not_initialized"}
        
        # Check storage
        if self.storage:
            health["components"]["storage"] = {"status": "initialized"}
        else:
            health["components"]["storage"] = {"status": "not_initialized"}
        
        # Overall health
        all_healthy = all(
            comp.get("status") in ["connected", "initialized"]
            for comp in health["components"].values()
        )
        health["healthy"] = all_healthy and self.running
        
        return health


def main() -> None:
    """Main entry point."""
    service = AnalysisService()
    service.run()


if __name__ == "__main__":
    main()