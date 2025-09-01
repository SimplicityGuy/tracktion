"""Main entry point for the File Watcher service."""

import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from watchdog.observers import Observer

from .message_publisher import MessagePublisher, RabbitMQConfig
from .watchdog_handler import TracktionEventHandler

load_dotenv()

# Generate unique instance ID for this service instance
INSTANCE_ID = os.environ.get("INSTANCE_ID") or str(uuid.uuid4())[:8]


# Configure structured logging with instance ID
def add_instance_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add instance ID to all log messages."""
    event_dict["instance_id"] = INSTANCE_ID
    return event_dict


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
        add_instance_id,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class FileWatcherService:
    """Service for monitoring directories for new audio files."""

    def __init__(self) -> None:
        """Initialize the File Watcher service."""
        # Get data directory from environment variable with fallback to old env var for compatibility
        data_dir = os.getenv("DATA_DIR", os.getenv("FILE_WATCHER_SCAN_PATH", "/data/music"))
        self.scan_path = Path(data_dir)
        self.instance_id = INSTANCE_ID
        self.running = False
        self.observer: Observer | None = None
        self.publisher: MessagePublisher | None = None
        self.shutdown_timeout = 10  # seconds to wait for graceful shutdown

        # RabbitMQ configuration
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")

        logger.info(
            "File Watcher initialized",
            instance_id=self.instance_id,
            scan_path=str(self.scan_path),
            supported_formats=list(TracktionEventHandler.SUPPORTED_EXTENSIONS),
        )

    def start(self) -> None:
        """Start the file watching service."""
        self.running = True
        logger.info("File Watcher service starting...", watched_directory=str(self.scan_path))

        # Log the directory being watched
        logger.info("Monitoring directory", path=str(self.scan_path), instance=self.instance_id)

        # Validate directory exists and is readable
        if not self.scan_path.exists():
            logger.error(f"ERROR: Data directory {self.scan_path} does not exist")
            sys.exit(1)

        if not os.access(self.scan_path, os.R_OK):
            logger.error(f"ERROR: No read permission for {self.scan_path}")
            sys.exit(1)

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Initialize message publisher with instance ID and watched directory
        try:
            config = RabbitMQConfig(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_pass,
            )
            self.publisher = MessagePublisher(
                config=config,
                instance_id=self.instance_id,
                watched_directory=str(self.scan_path),
            )
            self.publisher.connect()
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            # Continue without publishing for local testing
            self.publisher = None

        # Create event handler and observer
        event_handler = TracktionEventHandler(self.publisher)
        self.observer = Observer()

        # Schedule observer for target directory (recursive)
        try:
            self.observer.schedule(event_handler, str(self.scan_path), recursive=True)
            self.observer.start()
            logger.info("Watchdog observer started", path=str(self.scan_path), recursive=True)
        except Exception as e:
            logger.error("Failed to start watchdog observer", error=str(e), exc_info=True)
            self.running = False
            return

        # Keep service running while observer watches
        try:
            while self.running:
                time.sleep(1)
                # Check observer health
                if not self.observer.is_alive():
                    logger.error("Observer thread died unexpectedly")
                    # Attempt to restart observer
                    try:
                        self.observer = Observer()
                        self.observer.schedule(event_handler, str(self.scan_path), recursive=True)
                        self.observer.start()
                        logger.info("Observer restarted successfully")
                    except Exception as restart_error:
                        logger.error("Failed to restart observer", error=str(restart_error))
                        self.running = False
        except Exception as e:
            logger.error("Error in main service loop", error=str(e), exc_info=True)
        finally:
            # Cleanup
            self._cleanup()

        logger.info("File Watcher service stopped")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received", signal=signum)
        self.running = False

    def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        # Stop and join observer if it exists
        if self.observer and self.observer.is_alive():
            logger.info("Stopping watchdog observer...")
            self.observer.stop()
            self.observer.join(timeout=self.shutdown_timeout)
            if self.observer.is_alive():
                logger.warning(
                    "Observer did not stop within timeout",
                    timeout=self.shutdown_timeout,
                )
            else:
                logger.info("Watchdog observer stopped successfully")

        # Disconnect from RabbitMQ
        if self.publisher:
            self.publisher.disconnect()


def main() -> None:
    """Main entry point."""
    logger.info("Starting File Watcher Service")
    service = FileWatcherService()
    service.start()
    sys.exit(0)


if __name__ == "__main__":
    main()
