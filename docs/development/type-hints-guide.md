# Type Hints Guide for Tracktion Services

## Overview

This guide documents the type hinting conventions and patterns used throughout the Tracktion codebase. All services use Python's type system extensively for better code clarity, IDE support, and runtime type checking with mypy.

## Type Hinting Philosophy

### Core Principles
1. **Explicit over implicit**: Always specify types for public APIs
2. **Gradual typing**: Start with basic types and add complexity as needed
3. **Practical over perfect**: Use `Any` when appropriate rather than overly complex types
4. **Documentation through types**: Types should clarify intent and usage
5. **Consistency**: Use the same patterns across all services

### Type Checking Tools
- **mypy**: Primary static type checker (configured in pyproject.toml)
- **ruff**: Linting with type-aware rules
- **IDE support**: Enhanced autocomplete and error detection

## Common Type Patterns

### Basic Types
```python
# Primitive types
def calculate_bpm(audio_path: str, sample_rate: int = 44100) -> float:
    """Calculate BPM from audio file."""
    pass

# Boolean with default
def enable_caching(enabled: bool = True) -> None:
    """Enable or disable result caching."""
    pass
```

### Collection Types
```python
from typing import Dict, List, Set, Tuple, Optional

# Generic collections
def get_supported_formats() -> List[str]:
    """Get list of supported audio formats."""
    return ["mp3", "flac", "wav"]

# Typed dictionaries for structured data
def get_analysis_result() -> Dict[str, Any]:
    """Get analysis results as dictionary."""
    return {
        "bpm": 128.0,
        "key": "C major",
        "confidence": 0.85
    }

# Tuples with specific types
def parse_audio_metadata(file_path: str) -> Tuple[str, str, int]:
    """Parse metadata returning (title, artist, duration)."""
    pass
```

### Optional Types
```python
from typing import Optional

# Optional parameters
def detect_key(audio_file: str, algorithm: Optional[str] = None) -> KeyResult:
    """Detect musical key with optional algorithm selection."""
    pass

# Optional return values
def find_recording(recording_id: str) -> Optional[Recording]:
    """Find recording by ID, return None if not found."""
    pass
```

### Union Types
```python
from typing import Union

# Multiple possible input types
def load_config(source: Union[str, Dict[str, Any]]) -> Config:
    """Load configuration from file path or dictionary."""
    pass

# Modern union syntax (Python 3.10+)
def process_duration(duration: int | float | str) -> float:
    """Process duration from various input formats."""
    pass
```

## Advanced Type Patterns

### Generic Types
```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class Repository(Generic[T]):
    """Generic repository pattern."""

    def get_all(self) -> List[T]:
        """Get all entities."""
        pass

    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

# Usage
class RecordingRepository(Repository[Recording]):
    """Repository for Recording entities."""
    pass
```

### Protocol Types (Structural Typing)
```python
from typing import Protocol

class AudioAnalyzer(Protocol):
    """Protocol for audio analysis components."""

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio file and return results."""
        ...

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold for analysis."""
        ...

# Any class implementing these methods satisfies the protocol
class BPMDetector:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        return {"bpm": 128.0, "confidence": 0.9}

    @property
    def confidence_threshold(self) -> float:
        return 0.7
```

### Callable Types
```python
from typing import Callable

# Function type signatures
FilterFunction = Callable[[str], bool]
TransformFunction = Callable[[Dict[str, Any]], Dict[str, Any]]

def process_files(files: List[str], filter_fn: FilterFunction) -> List[str]:
    """Process files using filter function."""
    return [f for f in files if filter_fn(f)]

# Async callbacks
AsyncCallback = Callable[[str], Awaitable[None]]

async def process_async(callback: AsyncCallback) -> None:
    """Process with async callback."""
    await callback("processing complete")
```

### Literal Types
```python
from typing import Literal

# Constrained string values
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
AudioFormat = Literal["mp3", "flac", "wav", "ogg", "m4a"]
AnalysisStatus = Literal["pending", "processing", "completed", "failed"]

def set_log_level(level: LogLevel) -> None:
    """Set logging level to specific value."""
    pass

def analyze_audio(file_path: str, format: AudioFormat) -> Dict[str, Any]:
    """Analyze audio file of specific format."""
    pass
```

## Service-Specific Type Patterns

### Analysis Service Types
```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class BPMDetectionResult:
    """Result from BPM detection analysis."""
    bpm: float
    confidence: float
    beats: List[float]
    algorithm: str
    needs_review: bool

@dataclass
class KeyDetectionResult:
    """Result from key detection analysis."""
    key: str
    scale: str  # "major" or "minor"
    confidence: float
    alternative_key: Optional[str] = None
    alternative_scale: Optional[str] = None
    agreement: bool = False
    needs_review: bool = False

# Complex analysis configuration
class AnalysisConfig:
    """Configuration for audio analysis."""

    def __init__(
        self,
        bpm_confidence_threshold: float = 0.7,
        key_confidence_threshold: float = 0.7,
        enable_fallback_algorithms: bool = True,
        max_processing_time: int = 300,
        supported_formats: List[AudioFormat] = None
    ) -> None:
        self.bmp_confidence_threshold = bmp_confidence_threshold
        # ... other initialization
```

### Tracklist Service Types
```python
from datetime import datetime
from uuid import UUID

@dataclass
class MatchingResult:
    """Result from tracklist matching."""
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TracklistMetadata:
    """Metadata for a scraped tracklist."""
    title: Optional[str]
    artist: Optional[str]
    duration_minutes: Optional[float]
    date: Optional[datetime]
    source_url: str

# Service method signatures
class MatchingService:
    def match_tracklist_to_audio(
        self,
        scraped_tracklist: ScrapedTracklist,
        audio_metadata: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Match tracklist to audio metadata."""
        pass

    def validate_audio_file(
        self,
        audio_file_path: str,
        expected_duration: Optional[int] = None
    ) -> bool:
        """Validate audio file exists and is accessible."""
        pass
```

### File Watcher Service Types
```python
from pathlib import Path
from typing import Set

# File system event types
EventType = Literal["created", "modified", "deleted", "moved"]

@dataclass
class FileEvent:
    """File system event data."""
    event_type: EventType
    file_path: Path
    timestamp: datetime
    file_size: Optional[int] = None
    is_directory: bool = False

class FileScanner:
    """Scanner for audio files."""

    def __init__(
        self,
        supported_extensions: Set[str],
        scan_path: Path,
        chunk_size: int = 8192
    ) -> None:
        self.supported_extensions = supported_extensions
        self.scan_path = scan_path
        self.chunk_size = chunk_size

    def scan_files(self) -> List[Dict[str, Any]]:
        """Scan directory for audio files."""
        pass

    def calculate_file_hash(self, file_path: Path) -> Tuple[str, str]:
        """Calculate dual hashes for file."""
        pass
```

## Type Annotations for Async Code

### Async Functions
```python
import asyncio
from typing import Awaitable, AsyncIterator, AsyncGenerator

# Basic async function
async def analyze_audio_async(file_path: str) -> Dict[str, Any]:
    """Asynchronously analyze audio file."""
    await asyncio.sleep(1)  # Simulate processing
    return {"bpm": 128.0}

# Async generator
async def process_files_async(
    file_paths: List[str]
) -> AsyncIterator[Dict[str, Any]]:
    """Process files and yield results asynchronously."""
    for file_path in file_paths:
        result = await analyze_audio_async(file_path)
        yield result

# Async context manager
from typing import AsyncContextManager

async def get_analysis_session() -> AsyncContextManager[AnalysisSession]:
    """Get async analysis session context manager."""
    pass
```

### Async Service Methods
```python
class AsyncAnalysisService:
    """Async audio analysis service."""

    async def analyze_batch(
        self,
        file_paths: List[str],
        max_concurrent: int = 4
    ) -> List[Dict[str, Any]]:
        """Analyze multiple files concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(path: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.analyze_single(path)

        tasks = [analyze_with_semaphore(path) for path in file_paths]
        return await asyncio.gather(*tasks)

    async def analyze_single(self, file_path: str) -> Dict[str, Any]:
        """Analyze single audio file."""
        pass
```

## Error Handling Types

### Custom Exception Types
```python
from typing import Optional, Dict, Any

class AnalysisServiceError(Exception):
    """Base exception for analysis service errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

class UnsupportedFormatError(AnalysisServiceError):
    """Raised when audio format is not supported."""
    pass

class CorruptedFileError(AnalysisServiceError):
    """Raised when audio file is corrupted."""
    pass
```

### Result Types with Error Handling
```python
from typing import Union, Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E', bound=Exception)

@dataclass
class Ok(Generic[T]):
    """Successful result."""
    value: T

@dataclass
class Err(Generic[E]):
    """Error result."""
    error: E

# Result type combining success and error cases
Result = Union[Ok[T], Err[E]]

def safe_analyze_audio(file_path: str) -> Result[BPMDetectionResult, AnalysisServiceError]:
    """Safely analyze audio, returning result or error."""
    try:
        result = analyze_audio(file_path)
        return Ok(result)
    except AnalysisServiceError as e:
        return Err(e)
```

## Configuration Types

### Service Configuration Classes
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "tracktion"
    username: str = "tracktion_user"
    password: str = field(repr=False)  # Don't print password
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = field(default=None, repr=False)
    socket_timeout: float = 5.0
    max_connections: int = 50

@dataclass
class AnalysisServiceConfig:
    """Complete configuration for analysis service."""
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Processing settings
    max_concurrent_analyses: int = 4
    audio_sample_rate: int = 44100
    supported_formats: List[str] = field(
        default_factory=lambda: ["mp3", "flac", "wav", "ogg"]
    )

    # Algorithm settings
    bmp_confidence_threshold: float = 0.7
    key_confidence_threshold: float = 0.7

    # External service configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
```

## Type Checking with mypy

### mypy Configuration
```toml
# pyproject.toml
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
```

### Handling Third-Party Libraries
```python
# For libraries without type stubs
import essentia.standard as es  # type: ignore[import]
import xxhash  # type: ignore[import]

# For partial typing of complex objects
from typing import Any

def process_essentia_result(result: Any) -> Dict[str, float]:
    """Process Essentia result (untyped library)."""
    # Extract typed values from untyped result
    bpm: float = float(result[0])
    confidence: float = float(result[1])
    return {"bpm": bpm, "confidence": confidence}
```

### Type Ignores with Documentation
```python
# When type ignore is necessary, always document why
def load_audio_file(path: str) -> Any:
    """Load audio file using Essentia."""
    loader = es.MonoLoader(filename=path)  # type: ignore[attr-defined]  # Essentia adds attributes at runtime
    return loader()

# For complex numpy array operations
def calculate_hpcp(audio: np.ndarray) -> np.ndarray:
    """Calculate HPCP from audio."""
    spectrum = es.Spectrum()(audio)  # type: ignore[operator]  # Essentia callable objects
    return es.HPCP()(spectrum)  # type: ignore[operator]
```

## Best Practices

### 1. Start Simple, Add Complexity Gradually
```python
# Start with basic types
def analyze_file(path: str) -> dict:
    pass

# Add more specific types as code matures
def analyze_file(path: str) -> Dict[str, Union[float, str, bool]]:
    pass

# Eventually use custom types for clarity
def analyze_file(path: str) -> AnalysisResult:
    pass
```

### 2. Use Type Aliases for Clarity
```python
# Define aliases for complex types
FilePath = str
Confidence = float
Timestamp = float
AudioMetadata = Dict[str, Union[str, float, int]]

def extract_metadata(file_path: FilePath) -> AudioMetadata:
    """Extract metadata from audio file."""
    pass
```

### 3. Leverage Dataclasses for Structured Data
```python
@dataclass
class AudioFile:
    """Represents an audio file with metadata."""
    path: Path
    format: AudioFormat
    duration_seconds: float
    file_size_bytes: int
    hash_sha256: str
    hash_xxh128: str
    metadata: Optional[AudioMetadata] = None

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)
```

### 4. Use Protocols for Flexible Interfaces
```python
class Cacheable(Protocol):
    """Protocol for objects that can be cached."""

    def get_cache_key(self) -> str:
        """Get unique cache key for this object."""
        ...

    def serialize(self) -> Dict[str, Any]:
        """Serialize object for caching."""
        ...

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Cacheable':
        """Deserialize object from cache data."""
        ...
```

### 5. Document Complex Types
```python
# Use docstrings to explain complex type relationships
def match_tracklist_to_audio(
    tracklist_data: Dict[str, Any],
    audio_metadata: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Match tracklist to audio file metadata.

    Args:
        tracklist_data: Dictionary containing:
            - 'title': Optional[str] - Mix/set title
            - 'artist': Optional[str] - DJ name
            - 'duration': Optional[float] - Duration in minutes
            - 'tracks': List[Dict] - Individual track data
        audio_metadata: Dictionary containing:
            - 'title': str - Audio file title tag
            - 'artist': str - Audio file artist tag
            - 'duration_seconds': int - Audio duration
            - 'date': Optional[str] - Recording date

    Returns:
        Tuple of:
            - float: Confidence score (0.0-1.0)
            - Dict[str, Any]: Detailed matching metadata
    """
    pass
```

This type system provides excellent developer experience with clear APIs, better IDE support, and catches many errors before runtime while maintaining Python's flexibility and expressiveness.
