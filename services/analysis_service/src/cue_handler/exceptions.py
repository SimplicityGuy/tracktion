"""Exception classes for CUE file handling."""


class CueError(Exception):
    """Base exception for all CUE-related errors."""


class CueParsingError(CueError):
    """Raised when a CUE file cannot be parsed."""


class CueValidationError(CueError):
    """Raised when a CUE sheet fails validation."""


class InvalidTimeFormatError(CueError):
    """Raised when a time string is in invalid format."""


class InvalidCommandError(CueError):
    """Raised when a CUE command is invalid or malformed."""


class FileNotFoundError(CueError):
    """Raised when a referenced audio file is not found."""


class EncodingError(CueError):
    """Raised when there are encoding/decoding issues."""
