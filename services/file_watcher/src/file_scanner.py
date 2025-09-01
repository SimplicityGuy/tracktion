"""File scanner module for detecting audio files."""

import hashlib
from pathlib import Path
from typing import ClassVar

import structlog
import xxhash

logger = structlog.get_logger()


class FileScanner:
    """Scans directories for audio files."""

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
                            logger.debug(
                                "Audio file detected",
                                path=file_str,
                                extension=extension,
                            )
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
        sha256_hash, xxh128_hash = self._calculate_dual_hashes(file_path)

        return {
            "path": str(file_path.absolute()),
            "name": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": str(stat.st_size),
            "modified_time": str(stat.st_mtime),
            "sha256_hash": sha256_hash,
            "xxh128_hash": xxh128_hash,
        }

    def _calculate_dual_hashes(self, file_path: Path, chunk_size: int = 8192) -> tuple[str, str]:
        """Calculate both SHA256 and XXH128 hashes for a file.

        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read

        Returns:
            Tuple of (sha256_hash, xxh128_hash) as hexadecimal strings

        """
        sha256_hasher = hashlib.sha256()
        xxh128_hasher = xxhash.xxh128()

        try:
            with Path(file_path).open("rb") as f:
                # Single pass through file for both hashes
                while chunk := f.read(chunk_size):
                    sha256_hasher.update(chunk)
                    xxh128_hasher.update(chunk)

        except Exception as e:
            logger.warning("Failed to calculate file hashes", path=str(file_path), error=str(e))
            # Fallback to simple hash based on path and size if available
            try:
                file_size = file_path.stat().st_size
                fallback_data = f"{file_path}{file_size}".encode()
            except Exception:
                # If we can't even stat the file, use just the path
                fallback_data = str(file_path).encode()

            sha256_fallback = hashlib.sha256(fallback_data).hexdigest()
            xxh128_fallback = xxhash.xxh128(fallback_data).hexdigest()
            return sha256_fallback, xxh128_fallback

        return sha256_hasher.hexdigest(), xxh128_hasher.hexdigest()

    def _calculate_sha256_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA256 hash string

        """
        sha256_hash, _ = self._calculate_dual_hashes(file_path)
        return sha256_hash

    def _calculate_xxh128_hash(self, file_path: Path) -> str:
        """Calculate XXH128 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal XXH128 hash string

        """
        _, xxh128_hash = self._calculate_dual_hashes(file_path)
        return xxh128_hash

    def is_audio_file(self, file_path: Path) -> bool:
        """Check if a file is a supported audio file.

        Args:
            file_path: Path to check

        Returns:
            True if file has a supported audio extension

        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
