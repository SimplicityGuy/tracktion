# Tracktion Code Style Guide

## Overview

This guide establishes consistent coding standards across all Tracktion services. All code must pass pre-commit hooks including ruff (linting), mypy (type checking), and automated formatting.

## Code Quality Standards

### Zero Tolerance Policy
- **ZERO ruff violations**: All linting issues must be resolved
- **ZERO mypy errors**: All type checking errors must be fixed
- **ALL pre-commit hooks must pass**: No commits with failing checks
- **No `--no-verify` flag usage**: Pre-commit hooks cannot be bypassed

### Pre-commit Workflow
```bash
# MANDATORY before every commit
pre-commit run --all-files

# Fix ALL issues reported
# Re-run to verify all fixes
pre-commit run --all-files

# Only commit when output shows ALL PASSED
```

## Python Code Standards

### File Organization and Imports

#### Import Ordering (isort configuration)
```python
# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

# Third-party imports
import numpy as np
import pika
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local application imports
from services.analysis_service.src.config import Config
from services.analysis_service.src.exceptions import AnalysisServiceError
```

#### Module Structure
```python
"""
Module docstring describing purpose and usage.

Longer description if needed explaining the module's role
in the overall system architecture.
"""

import statements...

# Module-level constants
DEFAULT_TIMEOUT = 30
MAX_RETRY_ATTEMPTS = 3

# Type aliases
AudioMetadata = Dict[str, Union[str, float, int]]
FilePath = str

# Classes and functions...
```

### Naming Conventions

#### Variables and Functions
```python
# Variables: snake_case
audio_file_path = "/path/to/file.mp3"
confidence_threshold = 0.7
analysis_results = {}

# Functions: snake_case with descriptive names
def calculate_bpm_with_fallback(audio_path: str) -> Dict[str, Any]:
    """Calculate BPM using primary algorithm with fallback."""
    pass

def extract_metadata_from_audio_file(file_path: Path) -> AudioMetadata:
    """Extract comprehensive metadata from audio file."""
    pass
```

#### Classes
```python
# Classes: PascalCase
class BPMDetector:
    """Detects BPM from audio files using multiple algorithms."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold

    def detect_bpm(self, audio_path: str) -> BPMDetectionResult:
        """Detect BPM from audio file."""
        pass

# Exception classes
class AnalysisServiceError(Exception):
    """Base exception for analysis service errors."""
    pass

class UnsupportedFormatError(AnalysisServiceError):
    """Raised when audio format is not supported."""
    pass
```

#### Constants
```python
# Module-level constants: UPPER_SNAKE_CASE
DEFAULT_SAMPLE_RATE = 44100
MAX_FILE_SIZE_MB = 500
SUPPORTED_AUDIO_FORMATS = ["mp3", "flac", "wav", "ogg", "m4a"]

# Class constants
class AudioProcessor:
    DEFAULT_CHUNK_SIZE = 8192
    MAX_PROCESSING_TIME = 300

    # Enums for constants with related values
    class ProcessingStatus(Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
```

### Type Annotations

#### Function Signatures
```python
# Always include type hints for parameters and return values
def match_tracklist_to_audio(
    scraped_tracklist: ScrapedTracklist,
    audio_metadata: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Match a scraped tracklist to audio file metadata.

    Args:
        scraped_tracklist: Scraped tracklist from 1001tracklists
        audio_metadata: Metadata from the audio file

    Returns:
        Tuple of (confidence_score, match_details)
    """
    pass

# Optional and Union types
def process_audio_file(
    file_path: str,
    output_format: Optional[str] = None,
    options: Dict[str, Any] = None
) -> Union[AudioAnalysisResult, ProcessingError]:
    """Process audio file with optional format conversion."""
    if options is None:
        options = {}
    pass
```

#### Class Type Annotations
```python
from __future__ import annotations  # For forward references

class AudioAnalyzer:
    """Analyzes audio files using multiple detection algorithms."""

    def __init__(
        self,
        config: AnalysisConfig,
        cache: Optional[CacheInterface] = None
    ) -> None:
        self.config = config
        self.cache = cache or NullCache()
        self._results: Dict[str, AnalysisResult] = {}

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return self.config.supported_formats.copy()

    async def analyze_async(self, file_path: str) -> AnalysisResult:
        """Asynchronously analyze audio file."""
        pass
```

### Docstrings

#### Google Style Docstrings
```python
def calculate_weighted_confidence(self, scores: List[Tuple[str, float]]) -> float:
    """
    Calculate composite confidence score using weighted scoring algorithm.

    This method implements a sophisticated weighted scoring system that combines
    multiple matching criteria to produce a single confidence score representing
    the likelihood that a tracklist matches an audio file.

    Args:
        scores: List of (category, score) tuples where:
            category: String identifier matching confidence_weights keys
            score: Normalized score between 0.0-1.0 for that matching aspect

    Returns:
        Weighted confidence score between 0.0 and 1.0, where higher values
        indicate stronger likelihood of correct tracklist-audio correlation

    Raises:
        ValueError: If scores list is empty or contains invalid values

    Example:
        >>> scores = [("title", 0.85), ("artist", 0.92), ("duration", 0.78)]
        >>> confidence = self.calculate_weighted_confidence(scores)
        >>> print(f"Confidence: {confidence:.2f}")  # Output: ~0.84
    """
    if not scores:
        return 0.0

    # Implementation...
```

#### Class Docstrings
```python
class KeyDetector:
    """
    Musical key detection using Essentia algorithms.

    This class provides comprehensive key detection capabilities using multiple
    algorithms for validation and confidence scoring. It implements both
    primary detection with KeyExtractor and validation using HPCP-based analysis.

    The detector uses a sophisticated confidence scoring system that:
    1. Compares results from multiple algorithms
    2. Applies confidence adjustments based on agreement
    3. Determines if manual review is recommended

    Attributes:
        confidence_threshold: Minimum confidence for reliable detection
        agreement_boost: Confidence multiplier when algorithms agree
        disagreement_penalty: Confidence multiplier when algorithms disagree
        needs_review_threshold: Threshold below which manual review is suggested

    Example:
        >>> detector = KeyDetector(confidence_threshold=0.7)
        >>> result = detector.detect_key("path/to/audio.mp3")
        >>> print(f"Key: {result.key} {result.scale}, Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        agreement_boost: float = 1.2,
        disagreement_penalty: float = 0.8,
        needs_review_threshold: float = 0.7,
    ):
        """Initialize the key detector with configuration parameters."""
        # Implementation...
```

### Code Formatting

#### Line Length and Wrapping
```python
# Maximum line length: 100 characters (configured in ruff.toml)

# Function calls - align parameters
result = self.some_long_method_name(
    first_parameter="value",
    second_parameter=42,
    third_parameter=True,
    optional_parameter=None
)

# Long strings
error_message = (
    "This is a very long error message that needs to be split "
    "across multiple lines to maintain readability and comply "
    "with the line length limit."
)

# List/dict comprehensions - split for clarity
filtered_results = [
    result for result in analysis_results
    if result.confidence > confidence_threshold
    and result.needs_review is False
]
```

#### Spacing and Indentation
```python
# Use 4 spaces for indentation (no tabs)

# Function definitions - two blank lines before
class AudioProcessor:
    """Process audio files."""


    def __init__(self, config: Config) -> None:
        """Initialize processor."""
        self.config = config

    def process_file(self, file_path: str) -> ProcessingResult:
        """Process single audio file."""
        # One blank line between methods
        pass


# Module level - two blank lines before classes/functions
def standalone_function() -> None:
    """Standalone function."""
    pass
```

### Error Handling Style

#### Exception Usage
```python
# Specific exception types
try:
    audio_data = load_audio_file(file_path)
except FileNotFoundError:
    logger.error(f"Audio file not found: {file_path}")
    raise AudioFileError(f"File not found: {file_path}") from None
except PermissionError as e:
    logger.error(f"Permission denied accessing {file_path}: {e}")
    raise AudioFileError(f"Access denied: {file_path}") from e
except Exception as e:
    logger.error(f"Unexpected error loading {file_path}: {e}", exc_info=True)
    raise AudioFileError(f"Failed to load audio file: {e}") from e

# Using custom exceptions with context
raise AnalysisServiceError(
    "BPM detection failed",
    error_code="BPM_DETECTION_FAILED",
    file_path=audio_path,
    algorithm="RhythmExtractor2013",
    confidence=result.confidence
)
```

#### Logging Style
```python
import logging

logger = logging.getLogger(__name__)

# Structured logging with context
logger.info(
    "Starting BPM analysis",
    file_path=audio_path,
    algorithm="primary",
    confidence_threshold=self.confidence_threshold
)

# Error logging with full context
logger.error(
    "Analysis failed",
    correlation_id=correlation_id,
    file_path=audio_path,
    error=str(e),
    exc_info=True  # Include stack trace
)

# Use lazy formatting for performance
logger.debug("Processing file %s with %d tracks", file_path, len(tracks))
```

### Async Code Style

#### Async Function Definitions
```python
async def analyze_audio_batch(
    self,
    file_paths: List[str],
    max_concurrent: int = 4
) -> List[AnalysisResult]:
    """
    Analyze multiple audio files concurrently.

    Args:
        file_paths: List of paths to audio files
        max_concurrent: Maximum concurrent analyses

    Returns:
        List of analysis results in same order as input
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_with_semaphore(path: str) -> AnalysisResult:
        async with semaphore:
            return await self.analyze_single(path)

    tasks = [analyze_with_semaphore(path) for path in file_paths]
    return await asyncio.gather(*tasks)
```

#### Context Managers
```python
# Async context managers
class AsyncAudioProcessor:
    async def __aenter__(self) -> AsyncAudioProcessor:
        await self.initialize_resources()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Optional[TracebackType]
    ) -> None:
        await self.cleanup_resources()

# Usage
async def process_files(file_paths: List[str]) -> List[ProcessingResult]:
    async with AsyncAudioProcessor() as processor:
        results = []
        for file_path in file_paths:
            result = await processor.process(file_path)
            results.append(result)
        return results
```

### Configuration and Constants

#### Configuration Classes
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BPMConfig:
    """Configuration for BPM detection algorithms."""

    # Required fields first
    confidence_threshold: float
    agreement_tolerance: float

    # Optional fields with defaults
    sample_rate: int = 44100
    enable_fallback: bool = True
    max_processing_time: int = 300

    # Complex defaults using field()
    supported_formats: List[str] = field(
        default_factory=lambda: ["mp3", "flac", "wav", "ogg"]
    )

    # Validation
    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.agreement_tolerance < 0:
            raise ValueError("agreement_tolerance must be non-negative")
```

#### Environment Variable Handling
```python
import os
from typing import Optional

class Config:
    """Service configuration from environment variables."""

    def __init__(self) -> None:
        # Required configuration - fail fast if missing
        self.service_name = self._get_required_env("SERVICE_NAME")
        self.database_url = self._get_required_env("DATABASE_URL")

        # Optional configuration with defaults
        self.debug = self._get_bool_env("DEBUG", default=False)
        self.max_workers = self._get_int_env("MAX_WORKERS", default=4)
        self.log_level = self._get_env("LOG_LEVEL", default="INFO")

    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.getenv(key)
        if value is None:
            raise ConfigurationError(f"Required environment variable {key} not set")
        return value

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get optional environment variable."""
        return os.getenv(key, default)

    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _get_int_env(self, key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError as e:
            raise ConfigurationError(f"Invalid integer for {key}: {os.getenv(key)}") from e
```

## Testing Standards

### Test File Organization
```python
# tests/unit/analysis_service/test_bmp_detector.py
"""Unit tests for BPM detection functionality."""

import pytest
from unittest.mock import Mock, patch

from services.analysis_service.src.bmp_detector import BPMDetector, BPMDetectionResult
from services.analysis_service.src.exceptions import InvalidAudioFileError


class TestBPMDetector:
    """Test cases for BPMDetector class."""

    @pytest.fixture
    def detector(self) -> BPMDetector:
        """Create BPMDetector instance for testing."""
        return BPMDetector(
            confidence_threshold=0.7,
            agreement_tolerance=5.0
        )

    @pytest.fixture
    def sample_audio_path(self) -> str:
        """Sample audio file path for testing."""
        return "/path/to/test/audio.mp3"
```

### Test Method Naming
```python
class TestMatchingService:
    def test_calculate_weighted_confidence_with_valid_scores(self):
        """Test weighted confidence calculation with valid input scores."""
        pass

    def test_calculate_weighted_confidence_with_empty_scores_returns_zero(self):
        """Test that empty scores list returns confidence of 0.0."""
        pass

    def test_fuzzy_match_identical_strings_returns_one(self):
        """Test fuzzy matching with identical strings returns perfect score."""
        pass

    def test_fuzzy_match_with_substring_bonus_applied(self):
        """Test fuzzy matching applies substring bonus when appropriate."""
        pass
```

### Assertion Style
```python
def test_bmp_detection_result_structure(detector, sample_audio_path):
    """Test BPM detection returns properly structured result."""
    result = detector.detect_bpm(sample_audio_path)

    # Test result structure
    assert isinstance(result, dict)
    assert "bmp" in result
    assert "confidence" in result
    assert "algorithm" in result

    # Test value types and ranges
    assert isinstance(result["bmp"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["algorithm"] in ["primary", "fallback", "consensus"]

    # Test boolean flags
    assert isinstance(result["needs_review"], bool)
```

## Performance Guidelines

### Memory Management
```python
# Use context managers for resource cleanup
def process_large_audio_file(file_path: str) -> ProcessingResult:
    """Process large audio file with memory management."""

    # Use context manager for automatic cleanup
    with AudioFileReader(file_path) as reader:
        # Process in chunks to manage memory
        chunk_size = 8192
        results = []

        while True:
            chunk = reader.read_chunk(chunk_size)
            if not chunk:
                break

            # Process chunk
            chunk_result = process_audio_chunk(chunk)
            results.append(chunk_result)

            # Explicit cleanup if needed
            del chunk

    return combine_results(results)
```

### Async Best Practices
```python
# Use semaphores to limit concurrency
async def process_files_with_limit(file_paths: List[str], max_concurrent: int = 4):
    """Process files with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(path: str):
        async with semaphore:
            return await process_single_file(path)

    tasks = [process_with_semaphore(path) for path in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## Documentation Standards

### API Documentation
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

app = FastAPI(
    title="Analysis Service API",
    description="Audio analysis service with BPM and key detection",
    version="1.0.0"
)

class AnalysisRequest(BaseModel):
    """Request model for audio analysis."""

    file_path: str = Field(..., description="Path to the audio file to analyze")
    include_bmp: bool = Field(True, description="Include BPM detection in analysis")
    include_key: bool = Field(True, description="Include key detection in analysis")

    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/audio.mp3",
                "include_bmp": True,
                "include_key": True
            }
        }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """
    Analyze audio file for BPM and musical key.

    This endpoint performs comprehensive audio analysis including:
    - BPM detection using RhythmExtractor2013 with Percival fallback
    - Musical key detection using KeyExtractor with HPCP validation
    - Confidence scoring and quality assessment

    The analysis may take 30-60 seconds for typical audio files.
    """
    try:
        result = await service.analyze_file(
            request.file_path,
            include_bmp=request.include_bmp,
            include_key=request.include_key
        )
        return AnalysisResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Audio file not found")
    except UnsupportedFormatError as e:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {e}")
```

## Security Guidelines

### Input Validation
```python
from pathlib import Path
import re

def validate_file_path(file_path: str) -> Path:
    """
    Validate and sanitize file path input.

    Args:
        file_path: User-provided file path

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or potentially dangerous
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Convert to Path for validation
    path = Path(file_path)

    # Check for path traversal attempts
    if ".." in path.parts:
        raise ValueError("Path traversal not allowed")

    # Validate file extension
    allowed_extensions = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # Ensure path is absolute and normalized
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}") from e

    return resolved_path
```

### Secrets Management
```python
# Never log sensitive information
def connect_to_database(database_url: str) -> Connection:
    """Connect to database with URL validation."""
    # Validate URL format without exposing credentials
    if not database_url.startswith(("postgresql://", "mysql://", "sqlite://")):
        raise ValueError("Invalid database URL format")

    try:
        connection = create_connection(database_url)
        # Log success without credentials
        logger.info("Database connection established",
                   host=connection.host,
                   database=connection.database)
        return connection
    except Exception as e:
        # Log error without exposing credentials
        logger.error("Database connection failed", error=str(e))
        raise
```

## Tool Configuration

### ruff Configuration (ruff.toml)
```toml
line-length = 100
target-version = "py311"

[lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # Pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
]

ignore = [
    "E501",  # Line too long (handled by formatter)
    "B008",  # Do not perform function calls in argument defaults
]

[lint.per-file-ignores]
"tests/*" = ["N806"]  # Allow non-lowercase variable names in tests

[lint.isort]
force-single-line = false
lines-after-imports = 2
```

### mypy Configuration (pyproject.toml)
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = "essentia.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "xxhash"
ignore_missing_imports = true
```

This code style guide ensures consistency, readability, and maintainability across all Tracktion services while adhering to Python best practices and the project's quality standards.
