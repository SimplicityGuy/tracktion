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

# Generator exports
from .generator import (
    CueGenerator,
    CueFormat,
    CueDisc,
    CueFile,
    CueTrack,
)
from .formats import (
    CDJGenerator,
    TraktorGenerator,
    SeratoGenerator,
    RekordboxGenerator,
    KodiGenerator,
    get_generator,
)

# Editor exports
from .editor import CueEditor
from .backup import BackupManager

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
    # Generator
    "CueGenerator",
    "CueFormat",
    "CueDisc",
    "CueFile",
    "CueTrack",
    # Format generators
    "CDJGenerator",
    "TraktorGenerator",
    "SeratoGenerator",
    "RekordboxGenerator",
    "KodiGenerator",
    "get_generator",
    # Editor
    "CueEditor",
    "BackupManager",
]
