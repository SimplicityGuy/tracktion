"""File scanner module for detecting audio files."""

import hashlib
from pathlib import Path

import structlog

logger = structlog.get_logger()


class FileScanner:
    """Scans directories for audio files."""

    # Supported audio file extensions
    SUPPORTED_EXTENSIONS: set[str] = {
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
        ".oga",  # OGG Vorbis support
    }

    def __init__(self, tracked_files: set[str] | None = None) -> None:
        """Initialize the file scanner.

        Args:
            tracked_files: Set of already tracked file paths
        """
        self.tracked_files = tracked_files or set()

    def scan_directory(self, directory: Path) -> list[dict[str, str]]:
        """Scan a directory for new audio files.

        Args:
            directory: Directory path to scan

        Returns:
            List of new audio file information dictionaries
        """
        new_files: list[dict[str, str]] = []

        if not directory.exists():
            logger.warning("Scan directory does not exist", path=str(directory))
            return new_files

        if not directory.is_dir():
            logger.error("Scan path is not a directory", path=str(directory))
            return new_files

        # Scan recursively for audio files
        for extension in self.SUPPORTED_EXTENSIONS:
            pattern = f"**/*{extension}"
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_str = str(file_path.absolute())

                    # Skip if already tracked
                    if file_str in self.tracked_files:
                        continue

                    # Get file info
                    try:
                        file_info = self._get_file_info(file_path)
                        new_files.append(file_info)
                        self.tracked_files.add(file_str)

                        # Log OGG file detection specifically
                        if extension in {".ogg", ".oga"}:
                            logger.info(
                                "OGG Vorbis file detected",
                                path=file_str,
                                size=file_info["size_bytes"],
                                extension=extension,
                            )
                        else:
                            logger.debug("Audio file detected", path=file_str, extension=extension)
                    except Exception as e:
                        logger.error("Failed to get file info", path=file_str, error=str(e))

        if new_files:
            logger.info(
                "Scan completed",
                directory=str(directory),
                new_files_count=len(new_files),
                total_tracked=len(self.tracked_files),
            )

        return new_files

    def _get_file_info(self, file_path: Path) -> dict[str, str]:
        """Get information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file information
        """
        stat = file_path.stat()

        return {
            "path": str(file_path.absolute()),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": str(stat.st_size),
            "modified_time": str(stat.st_mtime),
            "hash": self._calculate_file_hash(file_path),
        }

    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read

        Returns:
            Hexadecimal hash string
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read only first and last chunks for performance
                # This is sufficient for duplicate detection
                f.seek(0)
                first_chunk = f.read(chunk_size)
                sha256.update(first_chunk)

                # Get file size and read last chunk if file is large enough
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)
                    last_chunk = f.read(chunk_size)
                    sha256.update(last_chunk)

                # Include file size in hash
                sha256.update(str(file_size).encode())

        except Exception as e:
            logger.warning("Failed to calculate file hash", path=str(file_path), error=str(e))
            # Fallback to simple hash based on path and size
            return hashlib.sha256(f"{file_path}{file_path.stat().st_size}".encode()).hexdigest()

        return sha256.hexdigest()

    def is_audio_file(self, file_path: Path) -> bool:
        """Check if a file is a supported audio file.

        Args:
            file_path: Path to check

        Returns:
            True if file has a supported audio extension
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
