"""Data models for CUE sheet representation."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re


class InvalidTimeFormatError(Exception):
    """Raised when a time string cannot be parsed."""

    pass


@dataclass
class CueTime:
    """Represents a time in CUE format (MM:SS:FF where FF is frames at 75fps)."""

    # Cached regex pattern for performance
    _TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")

    minutes: int
    seconds: int
    frames: int

    def __post_init__(self) -> None:
        """Validate time values."""
        if self.minutes < 0:
            raise InvalidTimeFormatError(f"Invalid minutes: {self.minutes}")
        if self.seconds < 0 or self.seconds >= 60:
            raise InvalidTimeFormatError(f"Invalid seconds: {self.seconds}")
        if self.frames < 0 or self.frames >= 75:
            raise InvalidTimeFormatError(f"Invalid frames: {self.frames} (must be 0-74)")

    @classmethod
    def from_string(cls, time_str: str) -> "CueTime":
        """Parse time from MM:SS:FF format.

        Args:
            time_str: Time string in MM:SS:FF format

        Returns:
            CueTime instance

        Raises:
            InvalidTimeFormatError: If format is invalid
        """
        # Match MM:SS:FF format (allow single digit minutes)
        match = cls._TIME_PATTERN.match(time_str.strip())
        if not match:
            raise InvalidTimeFormatError(f"Invalid time format: {time_str} (expected MM:SS:FF)")

        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frames = int(match.group(3))

        # Validate ranges
        if seconds >= 60:
            raise InvalidTimeFormatError(f"Invalid seconds: {seconds} (must be 0-59)")
        if frames >= 75:
            raise InvalidTimeFormatError(f"Invalid frames: {frames} (must be 0-74)")

        return cls(minutes=minutes, seconds=seconds, frames=frames)

    @classmethod
    def from_milliseconds(cls, ms: int) -> "CueTime":
        """Create CueTime from milliseconds.

        Args:
            ms: Time in milliseconds

        Returns:
            CueTime instance
        """
        # Convert to frames (75 frames per second)
        total_frames = (ms * 75) // 1000

        minutes = total_frames // (75 * 60)
        remaining_frames = total_frames % (75 * 60)
        seconds = remaining_frames // 75
        frames = remaining_frames % 75

        return cls(minutes=minutes, seconds=seconds, frames=frames)

    @classmethod
    def from_frames(cls, total_frames: int) -> "CueTime":
        """Create CueTime from total frames.

        Args:
            total_frames: Total number of frames

        Returns:
            CueTime instance
        """
        minutes = total_frames // (75 * 60)
        remaining_frames = total_frames % (75 * 60)
        seconds = remaining_frames // 75
        frames = remaining_frames % 75

        return cls(minutes=minutes, seconds=seconds, frames=frames)

    def to_frames(self) -> int:
        """Convert to total frames.

        Returns:
            Total number of frames
        """
        return self.minutes * 75 * 60 + self.seconds * 75 + self.frames

    def to_milliseconds(self) -> int:
        """Convert to milliseconds.

        Returns:
            Time in milliseconds
        """
        total_frames = self.to_frames()
        return (total_frames * 1000) // 75

    def __str__(self) -> str:
        """String representation in MM:SS:FF format."""
        return f"{self.minutes:02d}:{self.seconds:02d}:{self.frames:02d}"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"CueTime({self.minutes}:{self.seconds:02d}:{self.frames:02d})"

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, CueTime):
            return False
        return self.to_frames() == other.to_frames()

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if not isinstance(other, CueTime):
            return NotImplemented
        return self.to_frames() < other.to_frames()

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, CueTime):
            return NotImplemented
        return self.to_frames() <= other.to_frames()


@dataclass
class Track:
    """Represents a single track in a CUE sheet."""

    number: int
    track_type: str = "AUDIO"
    title: Optional[str] = None
    performer: Optional[str] = None
    songwriter: Optional[str] = None
    isrc: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    indices: Dict[int, CueTime] = field(default_factory=dict)
    pregap: Optional[CueTime] = None
    postgap: Optional[CueTime] = None
    rem_fields: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate track data."""
        if self.number < 1 or self.number > 99:
            raise ValueError(f"Track number must be 01-99, got {self.number}")

        # Validate character limits
        if self.title and len(self.title) > 80:
            self.title = self.title[:80]  # Truncate to limit

        if self.performer and len(self.performer) > 80:
            self.performer = self.performer[:80]  # Truncate to limit

        # Validate ISRC format
        if self.isrc and len(self.isrc) != 12:
            raise ValueError(f"ISRC must be 12 characters, got {len(self.isrc)}")

    def get_start_time(self) -> Optional[CueTime]:
        """Get the track start time (INDEX 01).

        Returns:
            CueTime of INDEX 01, or None if not set
        """
        return self.indices.get(1)

    def get_pregap_start(self) -> Optional[CueTime]:
        """Get the pregap start time (INDEX 00).

        Returns:
            CueTime of INDEX 00, or None if not set
        """
        return self.indices.get(0)

    def validate(self) -> List[str]:
        """Validate track structure.

        Returns:
            List of validation errors
        """
        errors = []

        # Check for required INDEX 01
        if 1 not in self.indices:
            errors.append(f"Track {self.number}: Missing required INDEX 01")

        # Check INDEX ordering
        if 0 in self.indices and 1 in self.indices:
            if self.indices[0] >= self.indices[1]:
                errors.append(f"Track {self.number}: INDEX 00 must be before INDEX 01")

        # Validate FLAGS
        valid_flags = {"DCP", "4CH", "PRE", "SCMS"}
        for flag in self.flags:
            if flag not in valid_flags:
                errors.append(f"Track {self.number}: Invalid flag {flag}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary for serialization.

        Returns:
            Dictionary representation of track
        """
        return {
            "number": self.number,
            "type": self.track_type,
            "title": self.title,
            "performer": self.performer,
            "songwriter": self.songwriter,
            "isrc": self.isrc,
            "flags": self.flags,
            "indices": {str(num): str(time) for num, time in self.indices.items()},
            "pregap": str(self.pregap) if self.pregap else None,
            "postgap": str(self.postgap) if self.postgap else None,
            "rem_fields": self.rem_fields,
        }


@dataclass
class FileReference:
    """Represents a FILE entry in a CUE sheet."""

    filename: str
    file_type: str
    tracks: List[Track] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate file reference."""
        valid_types = {"BINARY", "MOTOROLA", "AIFF", "WAVE", "MP3"}
        if self.file_type not in valid_types:
            # Store but warn about unknown type
            pass  # Warning handled in parser

    def validate(self) -> List[str]:
        """Validate file reference structure.

        Returns:
            List of validation errors
        """
        errors = []

        # Check for tracks
        if not self.tracks:
            errors.append(f"FILE {self.filename}: No tracks defined")

        # Validate track numbering
        prev_num = 0
        for track in self.tracks:
            if track.number != prev_num + 1:
                errors.append(f"FILE {self.filename}: Non-sequential track {track.number}")
            prev_num = track.number

            # Validate individual tracks
            track_errors = track.validate()
            errors.extend(track_errors)

        return errors


@dataclass
class CueSheet:
    """Represents a complete CUE sheet."""

    # Disc-level metadata
    title: Optional[str] = None
    performer: Optional[str] = None
    catalog: Optional[str] = None
    cdtextfile: Optional[str] = None

    # Files and tracks
    files: List[FileReference] = field(default_factory=list)

    # REM fields
    rem_fields: Dict[str, str] = field(default_factory=dict)

    # Parsing information
    parsing_errors: List[str] = field(default_factory=list)
    parsing_warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate CUE sheet data."""
        # Validate character limits
        if self.title and len(self.title) > 80:
            self.title = self.title[:80]

        if self.performer and len(self.performer) > 80:
            self.performer = self.performer[:80]

        # Validate catalog (UPC/EAN)
        if self.catalog and not re.match(r"^\d{13}$", self.catalog):
            # Store but mark as invalid format
            pass  # Warning handled in parser

    def get_all_tracks(self) -> List[Track]:
        """Get all tracks from all files.

        Returns:
            List of all tracks in order
        """
        tracks = []
        for file_ref in self.files:
            tracks.extend(file_ref.tracks)
        return tracks

    def get_track_count(self) -> int:
        """Get total number of tracks.

        Returns:
            Total track count
        """
        return len(self.get_all_tracks())

    def validate(self) -> List[str]:
        """Validate complete CUE sheet structure.

        Returns:
            List of validation errors
        """
        errors = []

        # Must have at least one file
        if not self.files:
            errors.append("No FILE entries found")

        # Validate each file
        for file_ref in self.files:
            file_errors = file_ref.validate()
            errors.extend(file_errors)

        # Check for duplicate track numbers across files
        track_numbers = []
        for track in self.get_all_tracks():
            if track.number in track_numbers:
                errors.append(f"Duplicate track number: {track.number}")
            track_numbers.append(track.number)

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert CUE sheet to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "performer": self.performer,
            "catalog": self.catalog,
            "cdtextfile": self.cdtextfile,
            "files": [
                {"filename": f.filename, "type": f.file_type, "tracks": [t.to_dict() for t in f.tracks]}
                for f in self.files
            ],
            "rem_fields": self.rem_fields,
            "parsing_errors": self.parsing_errors,
            "parsing_warnings": self.parsing_warnings,
        }
