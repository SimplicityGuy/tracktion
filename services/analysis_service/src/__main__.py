"""Main entry point for the analysis service with structured logging."""

import os
import sys
import signal
import logging
from typing import Optional

from .message_consumer import MessageConsumer
from .structured_logging import configure_structured_logging

logger = logging.getLogger(__name__)


def setup_signal_handlers(consumer: MessageConsumer) -> None:
    """Set up signal handlers for graceful shutdown.

    Args:
        consumer: MessageConsumer instance to stop
    """

    def signal_handler(signum: int, frame: Optional[object]) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(
            f"Received {signal_name} signal, initiating graceful shutdown",
            extra={"signal": signal_name, "signum": signum},
        )
        consumer.stop()
        sys.exit(0)

    # Register handlers for SIGTERM and SIGINT
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main() -> None:
    """Main function to run the analysis service."""
    # Configure structured logging
    configure_structured_logging(
        service_name="analysis_service",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        include_console=True,
        include_hostname=True,
        include_function=True,
        include_thread=True,
    )

    logger.info(
        "Starting analysis service",
        extra={
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
        },
    )

    # Get configuration from environment
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    # Create message consumer
    try:
        consumer = MessageConsumer(
            rabbitmq_url=rabbitmq_url,
            redis_host=redis_host,
            redis_port=redis_port,
            enable_cache=True,
            enable_temporal_analysis=True,
            enable_key_detection=True,
            enable_mood_analysis=True,
            enable_batch_processing=os.getenv("ENABLE_BATCH_PROCESSING", "false").lower() == "true",
        )

        # Set up signal handlers for graceful shutdown
        setup_signal_handlers(consumer)

        # Start consuming messages
        logger.info("Starting message consumption")
        consumer.consume()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    except Exception as e:
        logger.error(f"Fatal error in analysis service: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Analysis service stopped")


if __name__ == "__main__":
    main()
