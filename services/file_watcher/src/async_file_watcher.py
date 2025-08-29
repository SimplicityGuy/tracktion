"""Async file watcher implementation for high-throughput file monitoring."""

import asyncio
import hashlib
import os
import signal
import sys
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore
import structlog
import xxhash
from dotenv import load_dotenv
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .async_message_publisher import AsyncMessagePublisher
from .async_metadata_extractor import AsyncMetadataExtractor

load_dotenv()

# Generate unique instance ID for this service instance
INSTANCE_ID = os.environ.get("INSTANCE_ID") or str(uuid.uuid4())[:8]

logger = structlog.get_logger()


class AsyncFileEventHandler(FileSystemEventHandler):
    """Async event handler for file system events."""

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus", ".oga"}

    def __init__(
        self,
        publisher: AsyncMessagePublisher,
        instance_id: str,
        loop: asyncio.AbstractEventLoop,
        semaphore: asyncio.Semaphore,
        metadata_extractor: AsyncMetadataExtractor | None = None,
    ) -> None:
        """Initialize the async event handler.

        Args:
            publisher: Async message publisher for RabbitMQ
            instance_id: Unique identifier for this watcher instance
            loop: Event loop for async operations
            semaphore: Semaphore for limiting concurrent operations
            metadata_extractor: Optional async metadata extractor
        """
        super().__init__()
        self.publisher = publisher
        self.instance_id = instance_id
        self.loop = loop
        self.semaphore = semaphore
        self.processing_tasks: set[Future[Any]] = set()
        self.metadata_extractor = metadata_extractor

    def is_audio_file(self, path: str) -> bool:
        """Check if the file has a supported audio extension."""
        return Path(path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and self.is_audio_file(str(event.src_path)):
            # Schedule async processing
            task = asyncio.run_coroutine_threadsafe(self._process_file_async(str(event.src_path), "created"), self.loop)
            self.processing_tasks.add(task)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and self.is_audio_file(str(event.src_path)):
            task = asyncio.run_coroutine_threadsafe(
                self._process_file_async(str(event.src_path), "modified"), self.loop
            )
            self.processing_tasks.add(task)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory and self.is_audio_file(str(event.src_path)):
            task = asyncio.run_coroutine_threadsafe(self._process_file_async(str(event.src_path), "deleted"), self.loop)
            self.processing_tasks.add(task)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events."""
        if hasattr(event, "dest_path") and not event.is_directory:
            if self.is_audio_file(str(event.dest_path)):
                task = asyncio.run_coroutine_threadsafe(
                    self._process_move_async(str(event.src_path), str(event.dest_path)), self.loop
                )
                self.processing_tasks.add(task)

    async def _process_file_async(self, file_path: str, event_type: str) -> None:
        """Process file events asynchronously with concurrency control.

        Args:
            file_path: Path to the file
            event_type: Type of event (created, modified, deleted)
        """
        async with self.semaphore:  # Limit concurrent operations
            try:
                logger.info(
                    f"Processing {event_type} event",
                    file_path=file_path,
                    instance_id=self.instance_id,
                )

                # Calculate hashes for non-deleted files
                sha256_hash = None
                xxh128_hash = None
                metadata = None

                if event_type != "deleted" and os.path.exists(file_path):
                    sha256_hash, xxh128_hash = await self._calculate_hashes_async(file_path)

                    # Extract metadata if extractor is available
                    if self.metadata_extractor:
                        try:
                            metadata = await self.metadata_extractor.extract_metadata(file_path)
                        except Exception as e:
                            logger.warning("Failed to extract metadata", file_path=file_path, error=str(e))

                # Publish event to message queue
                await self.publisher.publish_file_event(
                    event_type=event_type,
                    file_path=file_path,
                    instance_id=self.instance_id,
                    sha256_hash=sha256_hash,
                    xxh128_hash=xxh128_hash,
                    metadata=metadata,
                )

                logger.info(
                    f"Successfully processed {event_type} event",
                    file_path=file_path,
                    sha256_hash=sha256_hash,
                )

            except Exception as e:
                logger.error(
                    f"Error processing {event_type} event",
                    file_path=file_path,
                    error=str(e),
                    exc_info=True,
                )

    async def _process_move_async(self, old_path: str, new_path: str) -> None:
        """Process file move/rename events asynchronously.

        Args:
            old_path: Original file path
            new_path: New file path
        """
        async with self.semaphore:
            try:
                # Determine if it's a rename or move
                old_dir = os.path.dirname(old_path)
                new_dir = os.path.dirname(new_path)
                event_type = "renamed" if old_dir == new_dir else "moved"

                logger.info(
                    f"Processing {event_type} event",
                    old_path=old_path,
                    new_path=new_path,
                    instance_id=self.instance_id,
                )

                # Calculate hashes for the file at new location
                sha256_hash, xxh128_hash = await self._calculate_hashes_async(new_path)

                # Publish move/rename event
                await self.publisher.publish_file_event(
                    event_type=event_type,
                    file_path=new_path,
                    old_path=old_path,
                    instance_id=self.instance_id,
                    sha256_hash=sha256_hash,
                    xxh128_hash=xxh128_hash,
                )

                logger.info(
                    f"Successfully processed {event_type} event",
                    old_path=old_path,
                    new_path=new_path,
                )

            except Exception as e:
                logger.error(
                    "Error processing move event",
                    old_path=old_path,
                    new_path=new_path,
                    error=str(e),
                    exc_info=True,
                )

    async def _calculate_hashes_async(self, file_path: str) -> tuple[str, str]:
        """Calculate SHA256 and XXH128 hashes asynchronously.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (sha256_hash, xxh128_hash)
        """
        sha256 = hashlib.sha256()
        xxh128 = xxhash.xxh128()

        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):  # Read in 8KB chunks
                sha256.update(chunk)
                xxh128.update(chunk)

        return sha256.hexdigest(), xxh128.hexdigest()

    async def wait_for_tasks(self) -> None:
        """Wait for all processing tasks to complete."""
        if self.processing_tasks:
            # Convert Future objects to awaitable tasks
            tasks = [asyncio.wrap_future(future) for future in self.processing_tasks]
            await asyncio.gather(*tasks, return_exceptions=True)


class AsyncFileWatcherService:
    """Async service for monitoring directories for audio files."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize the async file watcher service.

        Args:
            loop: Event loop to use (creates new one if not provided)
        """
        # Get data directory from environment
        data_dir = os.getenv("DATA_DIR", os.getenv("FILE_WATCHER_SCAN_PATH", "/data/music"))
        self.scan_path = Path(data_dir)
        self.instance_id = INSTANCE_ID
        self.running = False
        self.observer: Any = None  # Type: Observer | None
        self.publisher: AsyncMessagePublisher | None = None
        self.event_handler: AsyncFileEventHandler | None = None
        self.metadata_extractor: AsyncMetadataExtractor | None = None
        self.loop = loop or asyncio.get_event_loop()

        # Semaphore for limiting concurrent operations (default: 100)
        max_concurrent = int(os.getenv("MAX_CONCURRENT_FILES", "100"))
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # RabbitMQ configuration
        self.rabbitmq_url = os.getenv(
            "RABBITMQ_URL",
            f"amqp://{os.getenv('RABBITMQ_USER', 'guest')}:"
            f"{os.getenv('RABBITMQ_PASS', 'guest')}@"
            f"{os.getenv('RABBITMQ_HOST', 'localhost')}:"
            f"{os.getenv('RABBITMQ_PORT', '5672')}/",
        )

        logger.info(
            "Async File Watcher initialized",
            instance_id=self.instance_id,
            scan_path=str(self.scan_path),
            max_concurrent=max_concurrent,
        )

    async def start(self) -> None:
        """Start the async file watching service."""
        self.running = True
        logger.info(
            "Async File Watcher service starting...",
            watched_directory=str(self.scan_path),
            instance=self.instance_id,
        )

        # Validate directory
        if not self.scan_path.exists():
            logger.error(f"ERROR: Data directory {self.scan_path} does not exist")
            sys.exit(1)

        if not os.access(self.scan_path, os.R_OK):
            logger.error(f"ERROR: No read permission for {self.scan_path}")
            sys.exit(1)

        # Initialize async message publisher
        self.publisher = AsyncMessagePublisher(self.rabbitmq_url, self.instance_id)
        await self.publisher.connect()

        # Initialize metadata extractor
        self.metadata_extractor = AsyncMetadataExtractor(max_workers=4)

        # Initialize event handler with async support
        self.event_handler = AsyncFileEventHandler(
            self.publisher, self.instance_id, self.loop, self.semaphore, self.metadata_extractor
        )

        # Set up file system observer (still uses watchdog for events)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(self.scan_path), recursive=True)
        self.observer.start()

        logger.info(
            "Async File Watcher service started",
            monitoring=str(self.scan_path),
            instance=self.instance_id,
        )

        # Perform initial async scan
        await self.scan_existing_files()

    async def scan_existing_files(self) -> None:
        """Perform async scan of existing files in the directory."""
        logger.info("Starting async scan of existing files", path=str(self.scan_path))

        tasks = []
        file_count = 0

        # Walk through directory tree asynchronously
        for root, _, files in os.walk(self.scan_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in AsyncFileEventHandler.SUPPORTED_EXTENSIONS:
                    # Create task for each file
                    task = self._process_existing_file(str(file_path))
                    tasks.append(task)
                    file_count += 1

                    # Process in batches to avoid overwhelming the system
                    if len(tasks) >= 100:
                        await asyncio.gather(*tasks, return_exceptions=True)
                        tasks = []

        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            "Completed async scan of existing files",
            total_files=file_count,
            instance=self.instance_id,
        )

    async def _process_existing_file(self, file_path: str) -> None:
        """Process an existing file found during initial scan.

        Args:
            file_path: Path to the existing file
        """
        if self.event_handler:
            await self.event_handler._process_file_async(file_path, "created")

    async def stop(self) -> None:
        """Stop the async file watching service."""
        logger.info("Stopping async file watcher service...")
        self.running = False

        # Stop the observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=10)

        # Wait for pending tasks
        if self.event_handler:
            await self.event_handler.wait_for_tasks()

        # Shutdown metadata extractor
        if self.metadata_extractor:
            self.metadata_extractor.shutdown()

        # Disconnect from RabbitMQ
        if self.publisher:
            await self.publisher.disconnect()

        logger.info("Async file watcher service stopped")

    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            """Handle shutdown signals."""
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main() -> None:
    """Main entry point for async file watcher."""
    service = AsyncFileWatcherService()
    service.setup_signal_handlers()

    try:
        await service.start()
        # Keep running until stopped
        while service.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
