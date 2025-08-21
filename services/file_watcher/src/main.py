"""Main entry point for the File Watcher service."""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

try:
    from .file_scanner import FileScanner
    from .message_publisher import MessagePublisher
except ImportError:
    # For direct execution
    from file_scanner import FileScanner  # type: ignore[no-redef]
    from message_publisher import MessagePublisher  # type: ignore[no-redef]

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
        self.scan_path = Path(os.getenv("FILE_WATCHER_SCAN_PATH", "/data/music"))
        self.scan_interval = int(os.getenv("FILE_WATCHER_SCAN_INTERVAL", "60"))
        self.running = False

        # Initialize components
        self.scanner = FileScanner()
        self.publisher: MessagePublisher | None = None

        # RabbitMQ configuration
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_pass = os.getenv("RABBITMQ_PASS", "guest")

        logger.info(
            "File Watcher initialized",
            scan_path=str(self.scan_path),
            scan_interval=self.scan_interval,
            supported_formats=list(FileScanner.SUPPORTED_EXTENSIONS),
        )

    def start(self) -> None:
        """Start the file watching service."""
        self.running = True
        logger.info("File Watcher service starting...")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Initialize message publisher
        try:
            self.publisher = MessagePublisher(
                host=self.rabbitmq_host,
                port=self.rabbitmq_port,
                username=self.rabbitmq_user,
                password=self.rabbitmq_pass,
            )
            self.publisher.connect()
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            # Continue without publishing for local testing
            self.publisher = None

        # Main service loop
        while self.running:
            try:
                logger.debug("Scanning for new files", path=str(self.scan_path))

                # Scan for new audio files
                new_files = self.scanner.scan_directory(self.scan_path)

                # Publish discovery events for new files
                if new_files and self.publisher:
                    for file_info in new_files:
                        success = self.publisher.publish_file_discovered(file_info)
                        if not success:
                            logger.warning("Failed to publish file discovery", file_path=file_info.get("path"))

                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error("Error during file scan", error=str(e), exc_info=True)
                time.sleep(5)  # Wait before retrying

        # Cleanup
        if self.publisher:
            self.publisher.disconnect()

        logger.info("File Watcher service stopped")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received", signal=signum)
        self.running = False


def main() -> None:
    """Main entry point."""
    logger.info("Starting File Watcher Service")
    service = FileWatcherService()
    service.start()
    sys.exit(0)


if __name__ == "__main__":
    main()
