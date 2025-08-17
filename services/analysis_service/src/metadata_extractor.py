"""Audio metadata extraction module."""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

from mutagen import File
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE

logger = logging.getLogger(__name__)


class InvalidAudioFileError(Exception):
    """Raised when an audio file cannot be processed."""

    pass


class MetadataExtractionError(Exception):
    """Raised when metadata extraction fails."""

    pass


class MetadataExtractor:
    """Extracts metadata from audio files."""

    # Supported audio formats
    SUPPORTED_FORMATS = {".mp3", ".flac", ".wav", ".wave", ".m4a", ".mp4", ".m4b", ".m4p", ".m4v", ".m4r"}

    def __init__(self) -> None:
        """Initialize the metadata extractor."""
        self._format_handlers = {
            ".mp3": self._extract_mp3,
            ".flac": self._extract_flac,
            ".wav": self._extract_wav,
            ".wave": self._extract_wav,
            ".m4a": self._extract_mp4,
            ".mp4": self._extract_mp4,
            ".m4b": self._extract_mp4,
            ".m4p": self._extract_mp4,
            ".m4v": self._extract_mp4,
            ".m4r": self._extract_mp4,
        }

    def extract(self, file_path: str) -> Dict[str, Optional[str]]:
        """Extract metadata from an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary of metadata key-value pairs

        Raises:
            InvalidAudioFileError: If file doesn't exist or is unsupported
            MetadataExtractionError: If metadata extraction fails
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise InvalidAudioFileError(f"File not found: {file_path}")

        if not path.is_file():
            raise InvalidAudioFileError(f"Not a file: {file_path}")

        # Check format support
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise InvalidAudioFileError(
                f"Unsupported format: {extension}. Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        try:
            # Get format-specific handler
            handler = self._format_handlers.get(extension)
            if handler:
                metadata = handler(file_path)
            else:
                # Fallback to generic extraction
                metadata = self._extract_generic(file_path)

            # Add file format to metadata
            metadata["format"] = extension.lstrip(".")

            # Ensure all values are strings or None
            metadata = {k: str(v) if v is not None else None for k, v in metadata.items()}

            logger.info(
                f"Successfully extracted metadata from {file_path}",
                extra={"format": extension, "keys": list(metadata.keys())},
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            raise MetadataExtractionError(f"Failed to extract metadata: {e}")

    def _extract_generic(self, file_path: str) -> Dict[str, Any]:
        """Generic metadata extraction using mutagen.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary of metadata
        """
        audio_file = File(file_path)

        if audio_file is None:
            raise InvalidAudioFileError(f"Cannot read file: {file_path}")

        metadata = {}

        # Extract basic tags
        if audio_file.tags:
            metadata["title"] = self._get_tag_value(audio_file.tags, ["TIT2", "Title", "title", "\xa9nam"])
            metadata["artist"] = self._get_tag_value(audio_file.tags, ["TPE1", "Artist", "artist", "\xa9ART"])
            metadata["album"] = self._get_tag_value(audio_file.tags, ["TALB", "Album", "album", "\xa9alb"])
            metadata["date"] = self._get_tag_value(audio_file.tags, ["TDRC", "Date", "date", "\xa9day"])
            metadata["genre"] = self._get_tag_value(audio_file.tags, ["TCON", "Genre", "genre", "\xa9gen"])
            metadata["track"] = self._get_tag_value(audio_file.tags, ["TRCK", "Track", "track", "trkn"])

        # Extract technical metadata
        if audio_file.info:
            info = audio_file.info
            metadata["duration"] = self._format_duration(getattr(info, "length", None))
            metadata["bitrate"] = getattr(info, "bitrate", None)
            metadata["sample_rate"] = getattr(info, "sample_rate", None)
            metadata["channels"] = getattr(info, "channels", None)

        return metadata

    def _extract_mp3(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from MP3 file.

        Args:
            file_path: Path to the MP3 file

        Returns:
            Dictionary of metadata
        """
        audio = MP3(file_path)
        metadata = {}

        # Try EasyID3 for common tags
        try:
            easy = EasyID3(file_path)
            metadata["title"] = easy.get("title", [None])[0]
            metadata["artist"] = easy.get("artist", [None])[0]
            metadata["album"] = easy.get("album", [None])[0]
            metadata["date"] = easy.get("date", [None])[0]
            metadata["genre"] = easy.get("genre", [None])[0]
            metadata["track"] = easy.get("tracknumber", [None])[0]
        except Exception as e:
            logger.debug(f"EasyID3 extraction failed, using standard tags: {e}")
            # Fall back to standard ID3 tags
            if audio.tags:
                metadata["title"] = self._get_tag_value(audio.tags, ["TIT2"])
                metadata["artist"] = self._get_tag_value(audio.tags, ["TPE1"])
                metadata["album"] = self._get_tag_value(audio.tags, ["TALB"])
                metadata["date"] = self._get_tag_value(audio.tags, ["TDRC"])
                metadata["genre"] = self._get_tag_value(audio.tags, ["TCON"])
                metadata["track"] = self._get_tag_value(audio.tags, ["TRCK"])

        # Technical metadata
        if audio.info:
            metadata["duration"] = self._format_duration(audio.info.length)
            metadata["bitrate"] = audio.info.bitrate
            metadata["sample_rate"] = audio.info.sample_rate
            metadata["channels"] = audio.info.channels
            metadata["version"] = f"MPEG {audio.info.version}"
            metadata["layer"] = audio.info.layer

        return metadata

    def _extract_flac(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from FLAC file.

        Args:
            file_path: Path to the FLAC file

        Returns:
            Dictionary of metadata
        """
        audio = FLAC(file_path)
        metadata = {}

        # FLAC uses Vorbis comments
        if audio.tags:
            metadata["title"] = audio.get("title", [None])[0]
            metadata["artist"] = audio.get("artist", [None])[0]
            metadata["album"] = audio.get("album", [None])[0]
            metadata["date"] = audio.get("date", [None])[0]
            metadata["genre"] = audio.get("genre", [None])[0]
            metadata["track"] = audio.get("tracknumber", [None])[0]
            metadata["albumartist"] = audio.get("albumartist", [None])[0]
            metadata["comment"] = audio.get("comment", [None])[0]

        # Technical metadata
        if audio.info:
            metadata["duration"] = self._format_duration(audio.info.length)
            metadata["bitrate"] = self._calculate_bitrate(audio.info)
            metadata["sample_rate"] = audio.info.sample_rate
            metadata["channels"] = audio.info.channels
            metadata["bits_per_sample"] = audio.info.bits_per_sample

        return metadata

    def _extract_wav(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from WAV file.

        Args:
            file_path: Path to the WAV file

        Returns:
            Dictionary of metadata
        """
        audio = WAVE(file_path)
        metadata = {}

        # WAV files may have ID3 tags
        if audio.tags:
            metadata["title"] = self._get_tag_value(audio.tags, ["TIT2"])
            metadata["artist"] = self._get_tag_value(audio.tags, ["TPE1"])
            metadata["album"] = self._get_tag_value(audio.tags, ["TALB"])
            metadata["date"] = self._get_tag_value(audio.tags, ["TDRC"])
            metadata["genre"] = self._get_tag_value(audio.tags, ["TCON"])
            metadata["track"] = self._get_tag_value(audio.tags, ["TRCK"])

        # Technical metadata
        if audio.info:
            metadata["duration"] = self._format_duration(audio.info.length)
            metadata["bitrate"] = audio.info.bitrate
            metadata["sample_rate"] = audio.info.sample_rate
            metadata["channels"] = audio.info.channels
            metadata["bits_per_sample"] = audio.info.bits_per_sample

        return metadata

    def _extract_mp4(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from MP4/M4A file.

        Args:
            file_path: Path to the MP4/M4A file

        Returns:
            Dictionary of metadata
        """
        audio = MP4(file_path)
        metadata = {}

        # MP4 uses different tag names
        if audio.tags:
            metadata["title"] = self._get_mp4_tag(audio.tags, "\xa9nam")
            metadata["artist"] = self._get_mp4_tag(audio.tags, "\xa9ART")
            metadata["album"] = self._get_mp4_tag(audio.tags, "\xa9alb")
            metadata["date"] = self._get_mp4_tag(audio.tags, "\xa9day")
            metadata["genre"] = self._get_mp4_tag(audio.tags, "\xa9gen")
            metadata["albumartist"] = self._get_mp4_tag(audio.tags, "aART")
            metadata["comment"] = self._get_mp4_tag(audio.tags, "\xa9cmt")

            # Track number is stored differently
            if "trkn" in audio.tags:
                track_info = audio.tags["trkn"][0]
                if isinstance(track_info, tuple) and len(track_info) > 0:
                    metadata["track"] = str(track_info[0])

        # Technical metadata
        if audio.info:
            metadata["duration"] = self._format_duration(audio.info.length)
            metadata["bitrate"] = audio.info.bitrate
            metadata["sample_rate"] = audio.info.sample_rate
            metadata["channels"] = audio.info.channels
            metadata["codec"] = audio.info.codec
            metadata["codec_description"] = audio.info.codec_description

        return metadata

    def _get_tag_value(self, tags: Any, keys: list) -> Optional[str]:
        """Get the first available tag value from a list of possible keys.

        Args:
            tags: Tag object to search
            keys: List of possible tag keys

        Returns:
            Tag value or None
        """
        for key in keys:
            if key in tags:
                value = tags[key]
                if hasattr(value, "text"):
                    # ID3 frame
                    return str(value.text[0]) if value.text else None
                elif isinstance(value, list):
                    return str(value[0]) if value else None
                else:
                    return str(value)
        return None

    def _get_mp4_tag(self, tags: Any, key: str) -> Optional[str]:
        """Get MP4 tag value.

        Args:
            tags: MP4 tags object
            key: Tag key

        Returns:
            Tag value or None
        """
        if key in tags:
            value = tags[key]
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)
        return None

    def _format_duration(self, duration: Optional[float]) -> Optional[str]:
        """Format duration in seconds to a readable string.

        Args:
            duration: Duration in seconds

        Returns:
            Formatted duration string or None
        """
        if duration is None:
            return None

        # Store as seconds with 3 decimal places
        return str(round(duration, 3))

    def _calculate_bitrate(self, info: Any) -> Optional[int]:
        """Calculate bitrate for formats that don't provide it directly.

        Args:
            info: Audio info object

        Returns:
            Bitrate in bps or None
        """
        if hasattr(info, "bitrate") and info.bitrate:
            return int(info.bitrate)

        # Try to calculate from file size and duration
        if hasattr(info, "length") and info.length and hasattr(info, "filesize"):
            # bitrate = (file_size_in_bytes * 8) / duration_in_seconds
            return int((info.filesize * 8) / info.length)

        return None

    def get_supported_formats(self) -> set:
        """Get the set of supported audio formats.

        Returns:
            Set of supported file extensions
        """
        return self.SUPPORTED_FORMATS.copy()
