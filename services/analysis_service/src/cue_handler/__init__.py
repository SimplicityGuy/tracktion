"""CUE file handler module for parsing, generating, and manipulating CUE sheets."""

__version__ = "1.0.0"

# Parser exports
from .parser import CueParser

# Model exports
from .models import CueSheet, Track, CueTime, FileReference

# Exception exports
from .exceptions import (
    CueParsingError,
    CueValidationError,
    InvalidTimeFormatError,
    InvalidCommandError,
)

__all__ = [
    # Parser
    "CueParser",
    # Models
    "CueSheet",
    "Track",
    "CueTime",
    "FileReference",
    # Exceptions
    "CueParsingError",
    "CueValidationError",
    "InvalidTimeFormatError",
    "InvalidCommandError",
]
