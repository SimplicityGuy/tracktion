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

# Validator exports
from .validator import (
    CueValidator,
    ValidationResult,
)
from .validation_rules import (
    Severity,
    ValidationIssue,
)

# Converter exports
from .converter import (
    CueConverter,
    ConversionMode,
    ConversionChange,
    ConversionReport,
    BatchConversionReport,
)
from .compatibility import (
    CompatibilityChecker,
    CompatibilityLevel,
    CompatibilityIssue,
    CompatibilityReport,
)
from .format_mappings import (
    FORMAT_CAPABILITIES,
    CONVERSION_RULES,
    LOSSY_CONVERSIONS,
    get_format_from_string,
    get_format_capabilities,
    get_conversion_rules,
    get_lossy_warnings,
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
    # Validator
    "CueValidator",
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    # Converter
    "CueConverter",
    "ConversionMode",
    "ConversionChange",
    "ConversionReport",
    "BatchConversionReport",
    # Compatibility
    "CompatibilityChecker",
    "CompatibilityLevel",
    "CompatibilityIssue",
    "CompatibilityReport",
    # Format mappings
    "FORMAT_CAPABILITIES",
    "CONVERSION_RULES",
    "LOSSY_CONVERSIONS",
    "get_format_from_string",
    "get_format_capabilities",
    "get_conversion_rules",
    "get_lossy_warnings",
]
