"""Main entry point for the cataloging service."""

import asyncio
import logging
import signal
import sys
from typing import Any

from .config import get_config
from .message_consumer import CatalogingMessageConsumer

logger = logging.getLogger(__name__)


class CatalogingService:
    """Main cataloging service class."""

    def __init__(self) -> None:
        """Initialize the cataloging service."""
        self.config = get_config()
        self.consumer = CatalogingMessageConsumer()
        self.cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the cataloging service."""
        try:
            # Connect to RabbitMQ
            await self.consumer.connect()
            logger.info("Cataloging service started")

            # Start periodic cleanup task
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup())

            # Start consuming messages
            consumer_task = asyncio.create_task(self.consumer.start_consuming())

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Cancel tasks
            consumer_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(consumer_task, self.cleanup_task, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in cataloging service: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the cataloging service."""
        logger.info("Stopping cataloging service...")
        await self.consumer.disconnect()
        logger.info("Cataloging service stopped")

    async def _periodic_cleanup(self) -> None:
        """Run periodic cleanup of old soft-deleted records."""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                await self.consumer.cleanup_old_deletes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                # Continue running even if cleanup fails
                await asyncio.sleep(3600)  # Retry in an hour

    def signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main() -> None:
    """Main entry point."""
    config = get_config()
    setup_logging(config.service.log_level)

    service = CatalogingService()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, service.signal_handler)
    signal.signal(signal.SIGINT, service.signal_handler)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
