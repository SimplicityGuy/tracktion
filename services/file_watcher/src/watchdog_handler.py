"""Watchdog event handler for monitoring file system changes."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from .file_scanner import FileScanner

logger = structlog.get_logger()


class TracktionEventHandler(FileSystemEventHandler):
    """Event handler for file system changes using watchdog library."""

    # Supported audio file extensions
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".mp3",
        ".flac",
        ".wav",
        ".wave",
        ".m4a",
        ".mp4",
        ".m4b",
        ".m4p",
        ".m4v",
        ".m4r",
        ".ogg",
        ".oga",
    }

    def __init__(self, message_publisher: Any | None = None) -> None:
        """Initialize the event handler.

        Args:
            message_publisher: MessagePublisher instance for sending events to RabbitMQ

        """
        super().__init__()
        self.publisher = message_publisher

    def is_audio_file(self, path: str | bytes) -> bool:
        """Check if a file is a supported audio file.

        Args:
            path: File path to check

        Returns:
            True if file has a supported audio extension

        """
        path_str = path if isinstance(path, str) else path.decode("utf-8")
        return Path(path_str).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: FileSystemEvent object containing event details

        """
        if not event.is_directory and self.is_audio_file(event.src_path):
            logger.info("File created", path=event.src_path)
            self._send_event("created", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: FileSystemEvent object containing event details

        """
        if not event.is_directory and self.is_audio_file(event.src_path):
            logger.debug("File modified", path=event.src_path)
            self._send_event("modified", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events.

        Args:
            event: FileSystemEvent object containing event details

        """
        if not event.is_directory and self.is_audio_file(event.src_path):
            logger.info("File deleted", path=event.src_path)
            self._send_event("deleted", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events.

        Args:
            event: FileSystemEvent object containing event details

        """
        if event.is_directory:
            return

        # Check if either source or destination is an audio file
        if not (self.is_audio_file(event.src_path) or self.is_audio_file(event.dest_path)):
            return

        # Determine if it's a rename or move
        src_path_str = event.src_path if isinstance(event.src_path, str) else event.src_path.decode("utf-8")
        dest_path_str = event.dest_path if isinstance(event.dest_path, str) else event.dest_path.decode("utf-8")
        src_dir = Path(src_path_str).parent
        dest_dir = Path(dest_path_str).parent

        if src_dir == dest_dir:
            logger.info("File renamed", old_path=event.src_path, new_path=event.dest_path)
            self._send_event("renamed", event.dest_path, old_path=event.src_path)
        else:
            logger.info("File moved", old_path=event.src_path, new_path=event.dest_path)
            self._send_event("moved", event.dest_path, old_path=event.src_path)

    def _send_event(self, event_type: str, file_path: str | bytes, **kwargs: Any) -> None:
        """Send file event to message publisher.

        Args:
            event_type: Type of event (created, modified, deleted, moved, renamed)
            file_path: Path to the file
            **kwargs: Additional event data (e.g., old_path for moves/renames)

        """
        # Convert bytes to str if needed
        path_str = file_path if isinstance(file_path, str) else file_path.decode("utf-8")

        if not self.publisher:
            logger.warning(
                "No message publisher configured, event not sent",
                event_type=event_type,
                path=path_str,
            )
            return

        try:
            # Get file info
            file_path_obj = Path(path_str)
            file_info = {
                "path": str(file_path_obj.absolute()),
                "name": file_path_obj.name,
                "extension": file_path_obj.suffix.lower(),
                "event_type": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Add file stats and hashes if file still exists (not for deleted events)
            if file_path_obj.exists() and event_type != "deleted":
                try:
                    stat = file_path_obj.stat()
                    file_info.update(
                        {
                            "size_bytes": str(stat.st_size),
                            "modified_time": str(stat.st_mtime),
                        },
                    )

                    # Calculate dual hashes for non-deleted files
                    scanner = FileScanner()
                    sha256_hash, xxh128_hash = scanner._calculate_dual_hashes(file_path_obj)
                    file_info.update(
                        {
                            "sha256_hash": sha256_hash,
                            "xxh128_hash": xxh128_hash,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Could not get file stats or hashes",
                        path=path_str,
                        error=str(e),
                    )

            # Add any additional kwargs (like old_path)
            # Convert any bytes paths in kwargs to strings
            processed_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, bytes):
                    processed_kwargs[key] = value.decode("utf-8")
                else:
                    processed_kwargs[key] = value
            file_info.update(processed_kwargs)

            # Publish the event
            success = self.publisher.publish_file_event(file_info, event_type)
            if not success:
                logger.warning("Failed to publish file event", event_type=event_type, path=path_str)

        except Exception as e:
            logger.error(
                "Error sending event",
                event_type=event_type,
                path=path_str,
                error=str(e),
                exc_info=True,
            )
