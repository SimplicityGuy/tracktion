"""CUE file handler module for parsing, generating, and manipulating CUE sheets."""

__version__ = "1.0.0"

# Parser exports
from .backup import BackupManager
from .compatibility import CompatibilityChecker, CompatibilityIssue, CompatibilityLevel, CompatibilityReport

# Converter exports
from .converter import BatchConversionReport, ConversionChange, ConversionMode, ConversionReport, CueConverter

# Editor exports
from .editor import CueEditor

# Exception exports
from .exceptions import CueParsingError, CueValidationError, InvalidCommandError, InvalidTimeFormatError
from .format_mappings import (
    CONVERSION_RULES,
    FORMAT_CAPABILITIES,
    LOSSY_CONVERSIONS,
    get_conversion_rules,
    get_format_capabilities,
    get_format_from_string,
    get_lossy_warnings,
)
from .formats import CDJGenerator, KodiGenerator, RekordboxGenerator, SeratoGenerator, TraktorGenerator, get_generator

# Generator exports
from .generator import CueDisc, CueFile, CueFormat, CueGenerator, CueTrack

# Model exports
from .models import CueSheet, CueTime, FileReference, Track
from .parser import CueParser
from .validation_rules import Severity, ValidationIssue

# Validator exports
from .validator import CueValidator, ValidationResult

__all__ = [
    # Format mappings
    "CONVERSION_RULES",
    # Format mappings
    "FORMAT_CAPABILITIES",
    # Format mappings
    "LOSSY_CONVERSIONS",
    # Editor
    "BackupManager",
    # Converter
    "BatchConversionReport",
    # Format generators
    "CDJGenerator",
    # Compatibility
    "CompatibilityChecker",
    "CompatibilityIssue",
    "CompatibilityLevel",
    "CompatibilityReport",
    "ConversionChange",
    "ConversionMode",
    "ConversionReport",
    "CueConverter",
    # Models
    "CueDisc",
    # Editor
    "CueEditor",
    # Models
    "CueFile",
    # Generator
    "CueFormat",
    # Generator
    "CueGenerator",
    # Parser
    "CueParser",
    # Exceptions
    "CueParsingError",
    # Models
    "CueSheet",
    # Models
    "CueTime",
    # Models
    "CueTrack",
    # Exceptions
    "CueValidationError",
    # Validator
    "CueValidator",
    # Models
    "FileReference",
    # Exceptions
    "InvalidCommandError",
    # Exceptions
    "InvalidTimeFormatError",
    # Format generators
    "KodiGenerator",
    # Format generators
    "RekordboxGenerator",
    # Format generators
    "SeratoGenerator",
    # Validator
    "Severity",
    # Models
    "Track",
    # Format generators
    "TraktorGenerator",
    # Validator
    "ValidationIssue",
    # Validator
    "ValidationResult",
    # Format mappings
    "get_conversion_rules",
    # Format mappings
    "get_format_capabilities",
    # Format mappings
    "get_format_from_string",
    # Format generators
    "get_generator",
    # Format mappings
    "get_lossy_warnings",
]
