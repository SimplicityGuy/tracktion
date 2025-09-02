"""
CUE Handler integration service for tracklist service.

This module provides a wrapper around the analysis service's CUE handler
to integrate CUE file generation and validation with tracklist data.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, ClassVar

# Third-party imports (after path setup)
from cue_handler import (
    CueConverter,
    CueGenerator,
    CueHandler,
    CueTrack,
    CueValidator,
)
from cue_handler import CueFormat as CueHandlerFormat
from cue_handler.format_mappings import (
    get_format_capabilities as handler_get_format_capabilities,
)
from cue_handler.format_mappings import (
    get_lossy_warnings as handler_get_lossy_warnings,
)
from cue_handler.models import CueTime

# Local imports
from services.tracklist_service.src.models.cue_file import CueFormat
from services.tracklist_service.src.models.tracklist import Tracklist
from services.tracklist_service.src.utils.time_utils import timedelta_to_milliseconds

# Add the analysis service to the path so we can import from it
analysis_service_path = Path(__file__).parent.parent.parent.parent / "analysis_service" / "src"
sys.path.insert(0, str(analysis_service_path))

# Optional audio analysis imports - check availability without importing
try:
    MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
except ImportError:
    MUTAGEN_AVAILABLE = False

try:
    TINYTAG_AVAILABLE = importlib.util.find_spec("tinytag") is not None
except ImportError:
    TINYTAG_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_format_capabilities(cue_format: CueHandlerFormat) -> dict[str, Any]:
    """Get capabilities for a CUE format.

    Args:
        cue_format: The CUE format to get capabilities for

    Returns:
        Dictionary containing format capabilities including:
        - max_tracks: Maximum number of tracks supported (None for unlimited)
        - supports_isrc: Whether ISRC codes are supported
        - supports_flags: Whether track FLAGS are supported
        - supports_rem: Level of REM field support
        - supports_pregap: Whether PREGAP is supported
        - supports_postgap: Whether POSTGAP is supported
        - encoding: Text encoding used
        - multi_file: Whether multi-file references are supported
        - char_limit: Maximum character limit for text fields
        - bpm_storage: How BPM data is stored
        - color_coding: Whether color coding is supported
        - loop_points: Whether loop points are supported
        - beat_grid: Whether beat grid is supported

    Raises:
        ValueError: If the format is not recognized
    """
    try:
        capabilities = handler_get_format_capabilities(cue_format)

        # Convert internal capability format to our service format
        return {
            "max_tracks": capabilities.get("max_tracks", 99),
            "supports_isrc": capabilities.get("isrc_support", False),
            "supports_flags": capabilities.get("flags") not in [None, False],
            "supports_rem": capabilities.get("rem_fields", "none") != "none",
            "supports_pregap": capabilities.get("pregap_postgap", False),
            "supports_postgap": capabilities.get("pregap_postgap", False),
            "encoding": "UTF-8",  # All formats use UTF-8 in our implementation
            "multi_file": capabilities.get("multi_file", False),
            "char_limit": capabilities.get("char_limit", 80),
            "bpm_storage": capabilities.get("bpm_storage", "none"),
            "color_coding": capabilities.get("color_coding", False),
            "loop_points": capabilities.get("loop_points", False),
            "beat_grid": capabilities.get("beat_grid", False),
            "rem_fields_level": capabilities.get("rem_fields", "none"),
            "flags_level": capabilities.get("flags", "none"),
        }
    except Exception as e:
        raise ValueError(f"Cannot get capabilities for format {cue_format}: {e}") from e


def get_lossy_warnings(source_format: CueHandlerFormat, target_format: CueHandlerFormat) -> list[str]:
    """Get conversion warnings when converting between formats.

    Args:
        source_format: The source CUE format
        target_format: The target CUE format

    Returns:
        List of warning messages about potential data loss during conversion

    Examples:
        >>> warnings = get_lossy_warnings(CueHandlerFormat.STANDARD, CueHandlerFormat.CDJ)
        >>> print(warnings)
        ['PREGAP/POSTGAP commands will be removed', 'Multi-file references will be consolidated']
    """
    if source_format == target_format:
        return []  # No conversion needed, no warnings

    try:
        warnings = handler_get_lossy_warnings(source_format, target_format)

        # Add general conversion warning if no specific warnings exist
        if not warnings:
            source_caps = get_format_capabilities(source_format)
            target_caps = get_format_capabilities(target_format)

            general_warnings = []

            # Check for feature loss
            if source_caps.get("supports_isrc") and not target_caps.get("supports_isrc"):
                general_warnings.append("ISRC codes will be lost")

            if source_caps.get("supports_flags") and not target_caps.get("supports_flags"):
                general_warnings.append("Track FLAGS will be lost")

            if source_caps.get("supports_pregap") and not target_caps.get("supports_pregap"):
                general_warnings.append("PREGAP timing will be lost")

            if source_caps.get("supports_postgap") and not target_caps.get("supports_postgap"):
                general_warnings.append("POSTGAP timing will be lost")

            if source_caps.get("multi_file") and not target_caps.get("multi_file"):
                general_warnings.append("Multi-file structure will be consolidated")

            if source_caps.get("color_coding") and not target_caps.get("color_coding"):
                general_warnings.append("Color coding will be lost")

            if source_caps.get("loop_points") and not target_caps.get("loop_points"):
                general_warnings.append("Loop points will be lost")

            if source_caps.get("beat_grid") and not target_caps.get("beat_grid"):
                general_warnings.append("Beat grid information will be lost")

            # Check character limit reduction
            source_limit = source_caps.get("char_limit", 255)
            target_limit = target_caps.get("char_limit", 255)
            if source_limit > target_limit:
                general_warnings.append(
                    f"Text fields may be truncated (limit reduced from {source_limit} to {target_limit} characters)"
                )

            # Check track limit reduction
            source_tracks = source_caps.get("max_tracks")
            target_tracks = target_caps.get("max_tracks")
            if source_tracks is None and target_tracks is not None:
                general_warnings.append(f"Track count limited to {target_tracks} tracks")
            elif source_tracks and target_tracks and source_tracks > target_tracks:
                general_warnings.append(f"Track count limited to {target_tracks} tracks (reduced from {source_tracks})")

            if general_warnings:
                return general_warnings
            return [
                f"Converting from {source_format.value} to {target_format.value} "
                f"may result in format-specific data loss"
            ]

        return list(warnings)

    except Exception as e:
        # Fallback to generic warning if format mapping fails
        return [f"Converting from {source_format.value} to {target_format.value} may result in data loss: {e}"]


class CueFormatMapper:
    """Maps between tracklist service and CUE handler format enums."""

    FORMAT_MAPPING: ClassVar[dict[CueFormat, Any]] = {
        CueFormat.STANDARD: CueHandlerFormat.STANDARD,
        CueFormat.CDJ: CueHandlerFormat.CDJ,
        CueFormat.TRAKTOR: CueHandlerFormat.TRAKTOR,
        CueFormat.SERATO: CueHandlerFormat.SERATO,
        CueFormat.REKORDBOX: CueHandlerFormat.REKORDBOX,
        CueFormat.KODI: CueHandlerFormat.KODI,
    }

    REVERSE_MAPPING: ClassVar[dict[Any, CueFormat]] = {v: k for k, v in FORMAT_MAPPING.items()}

    @classmethod
    def to_cue_handler_format(cls, format_val: CueFormat) -> CueHandlerFormat:
        """Convert tracklist service format to CUE handler format."""
        if format_val not in cls.FORMAT_MAPPING:
            raise ValueError(f"Unsupported format: {format_val}")
        return cls.FORMAT_MAPPING[format_val]

    @classmethod
    def from_cue_handler_format(cls, format_val: CueHandlerFormat) -> CueFormat:
        """Convert CUE handler format to tracklist service format."""
        if format_val not in cls.REVERSE_MAPPING:
            raise ValueError(f"Unknown CUE handler format: {format_val}")
        return cls.REVERSE_MAPPING[format_val]


class TracklistToCueMapper:
    """Maps tracklist data to CUE handler data structures."""

    # Note: timedelta_to_milliseconds has been moved to utils.time_utils
    # and is imported at the top of this file

    @staticmethod
    def milliseconds_to_cue_time(ms: int) -> CueTime:
        """Convert milliseconds to CUE time format."""
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        frames = int((ms % 1000) * 75 / 1000)  # Convert milliseconds to frames (75fps)

        return CueTime(minutes=minutes, seconds=seconds, frames=frames)

    @classmethod
    def tracklist_to_cue_tracks(cls, tracklist: Tracklist) -> list[CueTrack]:
        """Convert tracklist tracks to CUE tracks."""
        cue_tracks = []

        for track in tracklist.tracks:
            start_ms = timedelta_to_milliseconds(track.start_time)

            # Build performer and title
            performer = track.artist
            title = track.title
            if track.remix:
                title = f"{title} ({track.remix})"

            # Create CUE track
            cue_track = CueTrack(
                number=track.position,
                title=title,
                performer=performer,
                start_time_ms=start_ms,
            )

            # Add track to indices (INDEX 01 is the main index)
            cue_track.indices[1] = start_ms

            # Add REM fields for additional metadata
            if track.label:
                cue_track.rem_fields["LABEL"] = track.label
            if track.catalog_track_id:
                cue_track.rem_fields["CATALOG_ID"] = str(track.catalog_track_id)
            if track.confidence < 1.0:
                cue_track.rem_fields["CONFIDENCE"] = f"{track.confidence:.2f}"
            if track.transition_type:
                cue_track.rem_fields["TRANSITION"] = track.transition_type
            if track.is_manual_entry:
                cue_track.rem_fields["MANUAL_ENTRY"] = "true"

            cue_tracks.append(cue_track)

        return cue_tracks


class CueIntegrationService:
    """Main service for integrating CUE handler with tracklist service."""

    def __init__(self) -> None:
        self.generator = CueGenerator()
        self.validator = CueValidator()
        self.converter = CueConverter()
        self.format_mapper = CueFormatMapper()
        self.tracklist_mapper = TracklistToCueMapper()

        # Check audio analysis capabilities on initialization
        self._check_audio_analysis_capabilities()

    def _check_audio_analysis_capabilities(self) -> None:
        """Check and log available audio analysis capabilities."""
        capabilities = []
        if MUTAGEN_AVAILABLE:
            capabilities.append("Mutagen (comprehensive metadata)")
        if TINYTAG_AVAILABLE:
            capabilities.append("TinyTag (basic metadata)")

        if capabilities:
            print(f"Audio analysis available: {', '.join(capabilities)}")
        else:
            print(
                "Warning: No audio analysis libraries available. Install mutagen or tinytag for enhanced capabilities."
            )

    def get_format_capabilities(self, cue_format: CueFormat) -> dict[str, Any]:
        """Get capabilities for a specific CUE format.

        Args:
            cue_format: CUE format to query

        Returns:
            Dictionary of format capabilities

        Raises:
            ValueError: If the format is not supported
        """
        try:
            if cue_format not in CueFormat:
                raise ValueError(f"Unsupported CUE format: {cue_format}")

            handler_format = self.format_mapper.to_cue_handler_format(cue_format)
            capabilities = get_format_capabilities(handler_format)

            return {
                "max_tracks": capabilities.get("max_tracks", 99),
                "supports_isrc": capabilities.get("supports_isrc", False),
                "supports_flags": capabilities.get("supports_flags", False),
                "supports_rem": capabilities.get("supports_rem", False),
                "supports_pregap": capabilities.get("supports_pregap", False),
                "supports_postgap": capabilities.get("supports_postgap", False),
                "encoding": capabilities.get("encoding", "UTF-8"),
                "line_ending": capabilities.get("line_ending", "CRLF"),
                "multi_file": capabilities.get("multi_file", False),
                "char_limit": capabilities.get("char_limit", 80),
                "bmp_storage": capabilities.get("bmp_storage", "none"),
                "color_coding": capabilities.get("color_coding", False),
                "loop_points": capabilities.get("loop_points", False),
                "beat_grid": capabilities.get("beat_grid", False),
            }

        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise RuntimeError(f"Failed to get capabilities for format {cue_format}: {e}") from e

    def get_conversion_warnings(self, source_format: CueFormat, target_format: CueFormat) -> list[str]:
        """Get potential warnings for format conversion.

        Args:
            source_format: Source format
            target_format: Target format

        Returns:
            List of warning messages

        Raises:
            ValueError: If either format is not supported
        """
        try:
            # Validate formats
            if source_format not in CueFormat:
                raise ValueError(f"Unsupported source format: {source_format}")
            if target_format not in CueFormat:
                raise ValueError(f"Unsupported target format: {target_format}")

            handler_source = self.format_mapper.to_cue_handler_format(source_format)
            handler_target = self.format_mapper.to_cue_handler_format(target_format)

            warnings = get_lossy_warnings(handler_source, handler_target)
            return warnings or []

        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise RuntimeError(f"Failed to get conversion warnings from {source_format} to {target_format}: {e}") from e

    def get_supported_formats(self) -> list[CueFormat]:
        """Get list of supported CUE formats."""
        return list(CueFormat)

    def get_supported_audio_formats(self) -> list[str]:
        """Get list of supported audio formats for analysis."""
        basic_formats = ["WAV", "MP3", "FLAC", "OGG", "AAC", "M4A", "AIFF"]

        if MUTAGEN_AVAILABLE:
            # Mutagen supports many more formats
            extended_formats = ["WMA", "APE", "WAVPACK", "OPUS", "MP4"]
            return sorted(basic_formats + extended_formats)

        return sorted(basic_formats)

    def get_audio_analysis_capabilities(self) -> dict[str, Any]:
        """Get information about available audio analysis capabilities."""
        return {
            "mutagen_available": MUTAGEN_AVAILABLE,
            "tinytag_available": TINYTAG_AVAILABLE,
            "supported_formats": self.get_supported_audio_formats(),
            "can_extract_metadata": MUTAGEN_AVAILABLE or TINYTAG_AVAILABLE,
            "can_detect_duration": MUTAGEN_AVAILABLE or TINYTAG_AVAILABLE,
            "can_detect_quality": MUTAGEN_AVAILABLE or TINYTAG_AVAILABLE,
            "recommended_library": ("mutagen" if MUTAGEN_AVAILABLE else "tinytag" if TINYTAG_AVAILABLE else None),
        }

    def generate_cue_content(
        self,
        tracklist: Any,
        cue_format: CueFormat = CueFormat.STANDARD,
        audio_filename: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """Generate CUE content from a tracklist.

        Args:
            tracklist: Tracklist object with tracks
            cue_format: Target CUE format
            audio_filename: Optional audio filename

        Returns:
            Tuple of (success, content, error_message)
        """
        try:
            # Map tracklist to CUE format
            mapper = TracklistToCueMapper()
            cue_tracks = mapper.tracklist_to_cue_tracks(tracklist.tracks)

            # Create CUE data structure
            cue_data = {
                "title": getattr(tracklist, "title", "Unknown Mix"),
                "performer": getattr(tracklist, "performer", "Unknown Artist"),
                "file": audio_filename or "audio.mp3",
                "tracks": cue_tracks,
            }

            # Generate CUE content using the handler
            handler_format = self.format_mapper.to_cue_handler_format(cue_format)
            handler = CueHandler(format=handler_format)

            # Generate the content
            content = handler.generate(cue_data)

            if not content:
                return False, None, "Failed to generate CUE content"

            return True, content, None

        except Exception as e:
            logger.error(f"Error generating CUE content: {e}")
            return False, None, str(e)


# Create a singleton instance for use across the service
cue_integration = CueIntegrationService()

# Export enhanced functions with audio analysis capabilities
__all__ = [
    "CueFormatMapper",
    "CueIntegrationService",
    "TracklistToCueMapper",
    "cue_integration",
    "get_format_capabilities",
    "get_lossy_warnings",
]
