"""Main entry point for the File Watcher service."""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv

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
        logger.info(
            "File Watcher initialized",
            scan_path=str(self.scan_path),
            scan_interval=self.scan_interval,
        )

    def start(self) -> None:
        """Start the file watching service."""
        self.running = True
        logger.info("File Watcher service starting...")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Main service loop
        while self.running:
            try:
                logger.debug("Scanning for new files", path=str(self.scan_path))
                # TODO: Implement actual file scanning and message queue publishing
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error("Error during file scan", error=str(e), exc_info=True)
                time.sleep(5)  # Wait before retrying

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
