"""Audio metadata extraction module."""

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

from mutagen import File
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

logger = logging.getLogger(__name__)


class InvalidAudioFileError(Exception):
    """Raised when an audio file cannot be processed."""


class MetadataExtractionError(Exception):
    """Raised when metadata extraction fails."""


class MetadataExtractor:
    """Extracts metadata from audio files using the Mutagen library.

    This class provides comprehensive metadata extraction capabilities for various
    audio formats including MP3, FLAC, WAV, MP4/M4A, and OGG Vorbis. It handles
    format-specific tag structures and normalizes output to a consistent format.

    Attributes:
        SUPPORTED_FORMATS: Set of supported file extensions.
        VORBIS_STANDARD_FIELDS: Mapping of Vorbis comment fields to normalized names.

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract("song.mp3")
        >>> print(metadata["title"])  # Song title
        >>> supported = extractor.get_supported_formats()
    """

    # Supported audio formats
    SUPPORTED_FORMATS: ClassVar[set[str]] = {
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

    # Standard and extended Vorbis comment fields mapping
    VORBIS_STANDARD_FIELDS: ClassVar[dict[str, str]] = {
        # Standard fields
        "title": "title",
        "version": "version",
        "album": "album",
        "tracknumber": "track",
        "artist": "artist",
        "performer": "performer",
        "copyright": "copyright",
        "license": "license",
        "organization": "organization",
        "description": "description",
        "genre": "genre",
        "date": "date",
        "location": "location",
        "contact": "contact",
        "isrc": "isrc",
        # Extended fields
        "albumartist": "albumartist",
        "composer": "composer",
        "conductor": "conductor",
        "discnumber": "discnumber",
        "disctotal": "disctotal",
        "totaltracks": "totaltracks",
        "publisher": "publisher",
        "label": "label",
        "compilation": "compilation",
        "lyrics": "lyrics",
        "language": "language",
        "mood": "mood",
        "bpm": "bpm",
        "key": "key",
        "comment": "comment",
        "encoder": "encoder",
        # ReplayGain tags
        "replaygain_track_gain": "replaygain_track_gain",
        "replaygain_track_peak": "replaygain_track_peak",
        "replaygain_album_gain": "replaygain_album_gain",
        "replaygain_album_peak": "replaygain_album_peak",
    }

    def __init__(self) -> None:
        """Initialize the metadata extractor.

        Sets up format-specific handlers for each supported audio format.
        The handlers are optimized for each format's specific tag structure
        and technical metadata characteristics.
        """
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
            ".ogg": self._extract_ogg,
            ".oga": self._extract_ogg,
        }

    def extract(self, file_path: str) -> dict[str, str | None]:
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
            metadata = handler(file_path) if handler else self._extract_generic(file_path)

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
            raise MetadataExtractionError(f"Failed to extract metadata: {e}") from e

    def _extract_generic(self, file_path: str) -> dict[str, Any]:
        """Generic metadata extraction using mutagen.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary of metadata
        """
        audio_file = File(file_path)

        if audio_file is None:
            raise InvalidAudioFileError(f"Cannot read file: {file_path}")

        metadata: dict[str, Any] = {}

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

    def _extract_mp3(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from MP3 file.

        Args:
            file_path: Path to the MP3 file

        Returns:
            Dictionary of metadata
        """
        audio = MP3(file_path)
        metadata: dict[str, Any] = {}

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

    def _extract_flac(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from FLAC file.

        Args:
            file_path: Path to the FLAC file

        Returns:
            Dictionary of metadata
        """
        audio = FLAC(file_path)
        metadata: dict[str, Any] = {}

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

    def _extract_wav(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from WAV file.

        Args:
            file_path: Path to the WAV file

        Returns:
            Dictionary of metadata
        """
        audio = WAVE(file_path)
        metadata: dict[str, Any] = {}

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

    def _extract_mp4(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from MP4/M4A file.

        Args:
            file_path: Path to the MP4/M4A file

        Returns:
            Dictionary of metadata
        """
        audio = MP4(file_path)
        metadata: dict[str, Any] = {}

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

    def _sanitize_tag_value(self, value: Any, max_length: int = 5000) -> str | None:
        """Sanitize a tag value for safe storage.

        Args:
            value: The tag value to sanitize
            max_length: Maximum allowed length for the value

        Returns:
            Sanitized string value or None
        """
        if value is None:
            return None

        # Convert to string
        str_value = str(value)

        # Truncate if too long
        if len(str_value) > max_length:
            logger.warning(f"Tag value truncated from {len(str_value)} to {max_length} characters")
            str_value = str_value[:max_length]

        # Remove any null bytes or control characters
        str_value = "".join(char for char in str_value if ord(char) >= 32 or char in "\n\r\t")

        return str_value if str_value else None

    def _validate_date_format(self, date_str: str | None) -> str | None:
        """Validate and normalize date format.

        Args:
            date_str: Date string to validate

        Returns:
            Normalized date string or original if valid
        """
        if not date_str:
            return None

        # First sanitize the value
        sanitized = self._sanitize_tag_value(date_str, max_length=100)
        if not sanitized:
            return None

        # Common date formats in Vorbis comments:
        # YYYY, YYYY-MM, YYYY-MM-DD
        # For now, we'll accept any of these formats without strict parsing
        # This maintains flexibility while ensuring the value is safe

        # Basic validation - check if it starts with a 4-digit year

        if re.match(r"^\d{4}", sanitized):
            return sanitized

        # If it doesn't match expected format, still return it but log
        logger.debug(f"Unusual date format in Vorbis comment: {sanitized}")
        return sanitized

    def _handle_multiple_values(self, values: list | Any) -> str | None:
        """Handle fields with multiple values.

        Args:
            values: Single value or list of values

        Returns:
            Combined string value or None
        """
        if values is None:
            return None

        if isinstance(values, list):
            if len(values) == 0:
                return None
            if len(values) == 1:
                return self._sanitize_tag_value(values[0])
            # Join multiple values with semicolon
            non_null_values = [str(v) for v in values if v is not None]
            if non_null_values:
                combined = "; ".join(non_null_values)
                return self._sanitize_tag_value(combined)
            return None
        return self._sanitize_tag_value(values)

    def _extract_ogg(self, file_path: str) -> dict[str, Any]:
        """Extract metadata from OGG Vorbis file.

        Args:
            file_path: Path to the OGG file

        Returns:
            Dictionary of metadata including standard fields and custom tags
        """
        try:
            audio = OggVorbis(file_path)
        except Exception as e:
            logger.warning(f"Failed to parse OGG file {file_path}: {e}")
            raise MetadataExtractionError(f"Invalid OGG file: {e}") from e

        metadata: dict[str, Any] = {}
        custom_tags: dict[str, Any] = {}

        # Extract all tags from the file
        if audio.tags:
            # Limit total number of custom tags for safety
            max_custom_tags = 100
            custom_tag_count = 0

            # Process all tags in the file
            for key, values in audio.tags.items():
                key_lower = key.lower()

                # Handle multiple values per key
                value = self._handle_multiple_values(values)

                if value is None:
                    continue

                # Special handling for date field
                if key_lower == "date":
                    value = self._validate_date_format(value)

                # Check if it's a standard field
                if key_lower in self.VORBIS_STANDARD_FIELDS:
                    metadata[self.VORBIS_STANDARD_FIELDS[key_lower]] = value
                # Store as custom tag with original case (up to limit)
                elif custom_tag_count < max_custom_tags:
                    custom_tags[key] = value
                    custom_tag_count += 1
                else:
                    logger.warning(f"Maximum custom tags ({max_custom_tags}) reached, skipping: {key}")

            # Add custom tags to metadata if any exist
            if custom_tags:
                # Store custom tags as JSON string for type consistency
                metadata["custom_tags"] = json.dumps(custom_tags)

        # Technical metadata
        if audio.info:
            metadata["duration"] = self._format_duration(audio.info.length)
            metadata["bitrate"] = audio.info.bitrate
            metadata["sample_rate"] = audio.info.sample_rate
            metadata["channels"] = audio.info.channels

            # OGG-specific technical info
            metadata["bitrate_nominal"] = getattr(audio.info, "bitrate_nominal", None)
            metadata["bitrate_lower"] = getattr(audio.info, "bitrate_lower", None)
            metadata["bitrate_upper"] = getattr(audio.info, "bitrate_upper", None)

            # Try to determine if VBR or CBR
            if (
                hasattr(audio.info, "bitrate_nominal")
                and audio.info.bitrate_nominal
                and (
                    hasattr(audio.info, "bitrate_lower")
                    and audio.info.bitrate_lower
                    and hasattr(audio.info, "bitrate_upper")
                    and audio.info.bitrate_upper
                )
            ):
                # If lower and upper bounds differ, it's VBR
                if audio.info.bitrate_lower != audio.info.bitrate_upper:
                    metadata["bitrate_mode"] = "VBR"
                else:
                    metadata["bitrate_mode"] = "CBR"

            # Get file size
            try:
                metadata["file_size"] = str(Path(file_path).stat().st_size)
            except Exception as e:
                logger.debug(f"Could not get file size for {file_path}: {e}")

        return metadata

    def _get_tag_value(self, tags: Any, keys: list) -> str | None:
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
                if isinstance(value, list):
                    return str(value[0]) if value else None
                return str(value)
        return None

    def _get_mp4_tag(self, tags: Any, key: str) -> str | None:
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

    def _format_duration(self, duration: float | None) -> str | None:
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

    def _calculate_bitrate(self, info: Any) -> int | None:
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

        Returns a copy of the supported formats set to prevent external
        modification of the class-level constant.

        Returns:
            Set of supported file extensions (e.g., {'.mp3', '.flac', '.wav'}).

        Example:
            >>> extractor = MetadataExtractor()
            >>> formats = extractor.get_supported_formats()
            >>> '.mp3' in formats
            True
        """
        return self.SUPPORTED_FORMATS.copy()
