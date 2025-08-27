"""Exception classes for CUE file handling."""


class CueException(Exception):
    """Base exception for all CUE-related errors."""

    pass


class CueParsingError(CueException):
    """Raised when a CUE file cannot be parsed."""

    pass


class CueValidationError(CueException):
    """Raised when a CUE sheet fails validation."""

    pass


class InvalidTimeFormatError(CueException):
    """Raised when a time string is in invalid format."""

    pass


class InvalidCommandError(CueException):
    """Raised when a CUE command is invalid or malformed."""

    pass


class FileNotFoundError(CueException):
    """Raised when a referenced audio file is not found."""

    pass


class EncodingError(CueException):
    """Raised when there are encoding/decoding issues."""

    pass
