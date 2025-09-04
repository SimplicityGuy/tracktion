# Best Practices

This document provides comprehensive best practices for developing, deploying, and maintaining Tracktion services, covering code quality, architecture patterns, security, performance, and operational excellence.

## Table of Contents

1. [Code Quality and Development](#code-quality-and-development)
2. [Architecture and Design Patterns](#architecture-and-design-patterns)
3. [Security Best Practices](#security-best-practices)
4. [Performance Optimization](#performance-optimization)
5. [Database Best Practices](#database-best-practices)
6. [API Design Best Practices](#api-design-best-practices)
7. [Testing Strategies](#testing-strategies)
8. [Error Handling and Logging](#error-handling-and-logging)
9. [Deployment and Operations](#deployment-and-operations)
10. [Monitoring and Observability](#monitoring-and-observability)

## Code Quality and Development

### 1. Python Code Standards

```python
"""
Best practices for Python code in Tracktion services.
"""

from typing import Optional, List, Dict, Any, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import asyncio
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

# Use dataclasses for simple data structures
@dataclass
class TrackMetadata:
    """Immutable track metadata with validation."""

    title: str
    artist: str
    duration_seconds: float
    file_path: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate data after initialization."""
        if self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")

        if not self.file_path:
            raise ValueError("File path cannot be empty")

# Use protocols for interface definitions
class AudioAnalyzer(Protocol):
    """Protocol for audio analysis implementations."""

    async def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio file and return results."""
        ...

# Use abstract base classes for complex interfaces
class BaseProcessor(ABC):
    """Base class for audio processors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data."""
        ...

    async def validate_input(self, input_data: Any) -> bool:
        """Validate input data before processing."""
        return input_data is not None

# Use context managers for resource management
@asynccontextmanager
async def audio_file_processor(file_path: str):
    """Context manager for audio file processing."""
    processor = None
    try:
        processor = await initialize_audio_processor(file_path)
        yield processor
    except Exception as e:
        logging.error(f"Audio processing failed for {file_path}: {e}")
        raise
    finally:
        if processor:
            await cleanup_processor(processor)

# Example usage of best practices
class BPMAnalyzer(BaseProcessor):
    """BPM analyzer following best practices."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)

    async def process(self, input_data: str) -> Dict[str, Any]:
        """Process audio file for BPM detection."""

        # Input validation
        if not await self.validate_input(input_data):
            raise ValueError("Invalid input data")

        # Log processing start
        self.logger.info(f"Starting BPM analysis for {input_data}")

        try:
            async with audio_file_processor(input_data) as processor:
                # Actual BPM analysis logic
                bpm_result = await self._analyze_bpm(processor)

                # Validate result confidence
                if bpm_result['confidence'] < self.confidence_threshold:
                    self.logger.warning(
                        f"Low confidence BPM result: {bpm_result['confidence']:.2f}"
                    )

                return {
                    'bpm': bpm_result['bpm'],
                    'confidence': bpm_result['confidence'],
                    'algorithm_version': self.config.get('version', '1.0'),
                    'processed_at': datetime.now(timezone.utc).isoformat()
                }

        except Exception as e:
            self.logger.error(f"BPM analysis failed: {e}")
            raise

    async def _analyze_bpm(self, processor) -> Dict[str, float]:
        """Internal BPM analysis method."""
        # Implementation details
        await asyncio.sleep(0.1)  # Simulate processing
        return {'bpm': 128.0, 'confidence': 0.95}

# Function naming and documentation best practices
async def calculate_audio_similarity(
    track1_features: Dict[str, float],
    track2_features: Dict[str, float],
    similarity_algorithm: str = 'cosine'
) -> float:
    """
    Calculate similarity between two audio tracks.

    Args:
        track1_features: Audio features for first track
        track2_features: Audio features for second track
        similarity_algorithm: Algorithm to use ('cosine', 'euclidean')

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        ValueError: If features are incompatible or algorithm unknown

    Examples:
        >>> features1 = {'energy': 0.8, 'valence': 0.6}
        >>> features2 = {'energy': 0.7, 'valence': 0.5}
        >>> similarity = await calculate_audio_similarity(features1, features2)
        >>> print(f"Similarity: {similarity:.2f}")
    """

    # Validate inputs
    if not track1_features or not track2_features:
        raise ValueError("Both track features must be provided")

    # Check feature compatibility
    common_features = set(track1_features.keys()) & set(track2_features.keys())
    if not common_features:
        raise ValueError("No common features found between tracks")

    # Calculate similarity based on algorithm
    if similarity_algorithm == 'cosine':
        return await _calculate_cosine_similarity(track1_features, track2_features)
    elif similarity_algorithm == 'euclidean':
        return await _calculate_euclidean_similarity(track1_features, track2_features)
    else:
        raise ValueError(f"Unknown similarity algorithm: {similarity_algorithm}")

# Error handling best practices
class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""

    def __init__(self, message: str, file_path: str = None, error_code: str = None):
        self.file_path = file_path
        self.error_code = error_code
        super().__init__(message)

async def robust_audio_processing(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Process audio file with comprehensive error handling.

    Returns None if processing fails, logs all errors appropriately.
    """

    logger = logging.getLogger("audio_processing")

    try:
        # Validate file exists and is readable
        if not await validate_audio_file(file_path):
            raise AudioProcessingError(
                f"Invalid audio file: {file_path}",
                file_path=file_path,
                error_code="INVALID_FILE"
            )

        # Process with timeout
        result = await asyncio.wait_for(
            process_audio_file(file_path),
            timeout=300.0  # 5 minute timeout
        )

        return result

    except asyncio.TimeoutError:
        logger.error(f"Audio processing timeout for {file_path}")
        return None

    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {e} (Code: {e.error_code})")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error processing {file_path}: {e}")
        return None
```

### 2. Code Organization and Structure

```python
# services/analysis_service/src/analyzers/base.py
"""
Base analyzer classes and interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

@dataclass
class AnalysisResult:
    """Standard analysis result format."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: int
    algorithm_version: str
    error_message: Optional[str] = None

class BaseAnalyzer(ABC):
    """Base class for all audio analyzers."""

    def __init__(self, name: str, version: str, config: Dict[str, Any]):
        self.name = name
        self.version = version
        self.config = config
        self.logger = logging.getLogger(f"analyzer.{name}")

    @abstractmethod
    async def analyze(self, file_path: str) -> AnalysisResult:
        """Perform analysis on audio file."""
        pass

    def validate_config(self) -> bool:
        """Validate analyzer configuration."""
        required_keys = self.get_required_config_keys()
        return all(key in self.config for key in required_keys)

    @abstractmethod
    def get_required_config_keys(self) -> List[str]:
        """Return list of required configuration keys."""
        pass

# services/analysis_service/src/analyzers/bpm.py
"""
BPM analyzer implementation.
"""

import time
from typing import List
from .base import BaseAnalyzer, AnalysisResult

class BPMAnalyzer(BaseAnalyzer):
    """BPM detection analyzer."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("bpm_analyzer", "2.1.0", config)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.algorithm = config.get('algorithm', 'autocorrelation')

    async def analyze(self, file_path: str) -> AnalysisResult:
        """Analyze audio file for BPM."""
        start_time = time.time()

        try:
            # Load and analyze audio
            bpm_data = await self._detect_bpm(file_path)

            processing_time = int((time.time() - start_time) * 1000)

            return AnalysisResult(
                success=True,
                data={
                    'bpm': bpm_data['bpm'],
                    'confidence': bpm_data['confidence'],
                    'tempo_stability': bpm_data.get('stability', 0.8)
                },
                confidence=bpm_data['confidence'],
                processing_time_ms=processing_time,
                algorithm_version=self.version
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"BPM analysis failed: {e}")

            return AnalysisResult(
                success=False,
                data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                algorithm_version=self.version,
                error_message=str(e)
            )

    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys for BPM analyzer."""
        return ['confidence_threshold', 'algorithm']

    async def _detect_bpm(self, file_path: str) -> Dict[str, float]:
        """Internal BPM detection logic."""
        # Implementation would go here
        return {
            'bpm': 128.0,
            'confidence': 0.9,
            'stability': 0.85
        }

# services/analysis_service/src/factory.py
"""
Analyzer factory for creating analyzer instances.
"""

from typing import Dict, Any, Type
from .analyzers.base import BaseAnalyzer
from .analyzers.bpm import BPMAnalyzer
from .analyzers.key import KeyAnalyzer
from .analyzers.mood import MoodAnalyzer

class AnalyzerFactory:
    """Factory for creating analyzer instances."""

    _analyzers: Dict[str, Type[BaseAnalyzer]] = {
        'bpm': BPMAnalyzer,
        'key': KeyAnalyzer,
        'mood': MoodAnalyzer,
    }

    @classmethod
    def create_analyzer(cls, analyzer_type: str, config: Dict[str, Any]) -> BaseAnalyzer:
        """Create analyzer instance."""

        if analyzer_type not in cls._analyzers:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

        analyzer_class = cls._analyzers[analyzer_type]
        analyzer = analyzer_class(config)

        # Validate configuration
        if not analyzer.validate_config():
            raise ValueError(f"Invalid configuration for {analyzer_type} analyzer")

        return analyzer

    @classmethod
    def get_available_analyzers(cls) -> List[str]:
        """Get list of available analyzer types."""
        return list(cls._analyzers.keys())
```

## Architecture and Design Patterns

### 1. Service Layer Architecture

```python
# services/shared/patterns/service_layer.py
"""
Service layer pattern implementation for clean architecture.
"""

from typing import Generic, TypeVar, Protocol, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import asynccontextmanager

T = TypeVar('T')
ID = TypeVar('ID')

# Domain models
class DomainEntity(ABC):
    """Base class for domain entities."""

    def __init__(self, id: Optional[ID] = None):
        self.id = id

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

@dataclass
class Track(DomainEntity):
    """Track domain entity."""
    title: str
    artist: str
    duration: float
    file_path: str
    bpm: Optional[float] = None
    key: Optional[str] = None

# Repository pattern
class Repository(Generic[T, ID], Protocol):
    """Repository interface for data access."""

    async def get_by_id(self, id: ID) -> Optional[T]:
        """Get entity by ID."""
        ...

    async def save(self, entity: T) -> T:
        """Save entity."""
        ...

    async def delete(self, id: ID) -> bool:
        """Delete entity by ID."""
        ...

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Find all entities with pagination."""
        ...

class TrackRepository(Repository[Track, int]):
    """Track repository implementation."""

    def __init__(self, db_session):
        self.db = db_session

    async def get_by_id(self, id: int) -> Optional[Track]:
        """Get track by ID."""
        # Database query implementation
        result = await self.db.execute(
            "SELECT * FROM tracks WHERE id = :id",
            {"id": id}
        )
        row = result.fetchone()

        if row:
            return Track(
                id=row.id,
                title=row.title,
                artist=row.artist,
                duration=row.duration,
                file_path=row.file_path,
                bpm=row.bpm,
                key=row.key
            )
        return None

    async def save(self, track: Track) -> Track:
        """Save track to database."""
        if track.id is None:
            # Insert new track
            result = await self.db.execute(
                """
                INSERT INTO tracks (title, artist, duration, file_path, bpm, key)
                VALUES (:title, :artist, :duration, :file_path, :bpm, :key)
                RETURNING id
                """,
                {
                    "title": track.title,
                    "artist": track.artist,
                    "duration": track.duration,
                    "file_path": track.file_path,
                    "bpm": track.bpm,
                    "key": track.key
                }
            )
            track.id = result.scalar()
        else:
            # Update existing track
            await self.db.execute(
                """
                UPDATE tracks
                SET title = :title, artist = :artist, duration = :duration,
                    file_path = :file_path, bpm = :bpm, key = :key
                WHERE id = :id
                """,
                {
                    "id": track.id,
                    "title": track.title,
                    "artist": track.artist,
                    "duration": track.duration,
                    "file_path": track.file_path,
                    "bpm": track.bpm,
                    "key": track.key
                }
            )

        await self.db.commit()
        return track

# Service layer
class TrackService:
    """Track service implementing business logic."""

    def __init__(self, track_repository: TrackRepository, analyzer_factory):
        self.track_repo = track_repository
        self.analyzer_factory = analyzer_factory
        self.logger = logging.getLogger("track_service")

    async def create_track(self, track_data: Dict[str, Any]) -> Track:
        """Create new track with validation."""

        # Validate input
        self._validate_track_data(track_data)

        # Create domain entity
        track = Track(
            title=track_data['title'],
            artist=track_data['artist'],
            duration=track_data['duration'],
            file_path=track_data['file_path']
        )

        # Save to repository
        saved_track = await self.track_repo.save(track)

        # Trigger asynchronous analysis
        await self._schedule_track_analysis(saved_track.id)

        self.logger.info(f"Created track: {saved_track.title} by {saved_track.artist}")
        return saved_track

    async def analyze_track(self, track_id: int, analysis_types: List[str]) -> Dict[str, Any]:
        """Analyze track with specified analysis types."""

        # Get track from repository
        track = await self.track_repo.get_by_id(track_id)
        if not track:
            raise ValueError(f"Track not found: {track_id}")

        # Perform analyses
        analysis_results = {}

        for analysis_type in analysis_types:
            try:
                analyzer = self.analyzer_factory.create_analyzer(
                    analysis_type,
                    self._get_analyzer_config(analysis_type)
                )

                result = await analyzer.analyze(track.file_path)
                analysis_results[analysis_type] = result

                # Update track with analysis results
                if result.success:
                    await self._update_track_analysis(track, analysis_type, result)

            except Exception as e:
                self.logger.error(f"Analysis {analysis_type} failed for track {track_id}: {e}")
                analysis_results[analysis_type] = {'error': str(e)}

        return analysis_results

    def _validate_track_data(self, track_data: Dict[str, Any]):
        """Validate track data."""
        required_fields = ['title', 'artist', 'duration', 'file_path']

        for field in required_fields:
            if field not in track_data or not track_data[field]:
                raise ValueError(f"Missing required field: {field}")

        if track_data['duration'] <= 0:
            raise ValueError("Duration must be positive")

    async def _update_track_analysis(self, track: Track, analysis_type: str, result):
        """Update track with analysis results."""

        if analysis_type == 'bpm' and 'bpm' in result.data:
            track.bpm = result.data['bpm']
        elif analysis_type == 'key' and 'key' in result.data:
            track.key = result.data['key']

        await self.track_repo.save(track)
```

### 2. Event-Driven Architecture

```python
# services/shared/patterns/events.py
"""
Event-driven architecture patterns.
"""

from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import asyncio
import json
import uuid

@dataclass
class DomainEvent:
    """Base domain event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(init=False)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggregate_id: Optional[str] = None
    version: int = 1
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = self.__class__.__name__

@dataclass
class TrackCreatedEvent(DomainEvent):
    """Event fired when a track is created."""
    event_type: str = "TrackCreated"

@dataclass
class TrackAnalyzedEvent(DomainEvent):
    """Event fired when track analysis completes."""
    event_type: str = "TrackAnalyzed"

# Event handler interface
class EventHandler(ABC):
    """Base event handler."""

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle domain event."""
        pass

class AnalysisRequestHandler(EventHandler):
    """Handler for track creation events - requests analysis."""

    def __init__(self, analysis_service):
        self.analysis_service = analysis_service
        self.logger = logging.getLogger("analysis_request_handler")

    async def handle(self, event: DomainEvent) -> None:
        """Handle track created event by requesting analysis."""

        if not isinstance(event, TrackCreatedEvent):
            return

        try:
            # Request full analysis for new track
            await self.analysis_service.analyze_track(
                track_id=event.aggregate_id,
                analysis_types=['bpm', 'key', 'mood']
            )

            self.logger.info(f"Requested analysis for track {event.aggregate_id}")

        except Exception as e:
            self.logger.error(f"Failed to request analysis: {e}")

# Event bus implementation
class EventBus:
    """Simple in-memory event bus."""

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._logger = logging.getLogger("event_bus")

    def subscribe(self, event_type: str, handler: EventHandler):
        """Subscribe handler to event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        self._logger.info(f"Subscribed {handler.__class__.__name__} to {event_type}")

    async def publish(self, event: DomainEvent):
        """Publish event to all subscribed handlers."""

        self._logger.info(f"Publishing event: {event.event_type} ({event.event_id})")

        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            self._logger.warning(f"No handlers for event type: {event.event_type}")
            return

        # Execute handlers concurrently
        tasks = [handler.handle(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any handler failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handler = handlers[i]
                self._logger.error(
                    f"Handler {handler.__class__.__name__} failed for event {event.event_id}: {result}"
                )

# Event store for persistence
class EventStore:
    """Event store for persisting domain events."""

    def __init__(self, db_session):
        self.db = db_session

    async def save_event(self, event: DomainEvent):
        """Save event to store."""

        await self.db.execute(
            """
            INSERT INTO domain_events
            (event_id, event_type, aggregate_id, version, data, occurred_at)
            VALUES (:event_id, :event_type, :aggregate_id, :version, :data, :occurred_at)
            """,
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "aggregate_id": event.aggregate_id,
                "version": event.version,
                "data": json.dumps(event.data),
                "occurred_at": event.occurred_at
            }
        )

        await self.db.commit()

    async def get_events_for_aggregate(self, aggregate_id: str) -> List[DomainEvent]:
        """Get all events for an aggregate."""

        result = await self.db.execute(
            """
            SELECT event_id, event_type, aggregate_id, version, data, occurred_at
            FROM domain_events
            WHERE aggregate_id = :aggregate_id
            ORDER BY occurred_at
            """,
            {"aggregate_id": aggregate_id}
        )

        events = []
        for row in result.fetchall():
            # Reconstruct event (simplified)
            event_data = {
                'event_id': row.event_id,
                'event_type': row.event_type,
                'aggregate_id': row.aggregate_id,
                'version': row.version,
                'data': json.loads(row.data),
                'occurred_at': row.occurred_at
            }

            # Create appropriate event type
            if row.event_type == "TrackCreated":
                event = TrackCreatedEvent(**event_data)
            elif row.event_type == "TrackAnalyzed":
                event = TrackAnalyzedEvent(**event_data)
            else:
                event = DomainEvent(**event_data)

            events.append(event)

        return events

# Enhanced service with event publishing
class EventDrivenTrackService(TrackService):
    """Track service with event publishing."""

    def __init__(self, track_repository, analyzer_factory, event_bus, event_store):
        super().__init__(track_repository, analyzer_factory)
        self.event_bus = event_bus
        self.event_store = event_store

    async def create_track(self, track_data: Dict[str, Any]) -> Track:
        """Create track and publish event."""

        # Create track using parent method
        track = await super().create_track(track_data)

        # Create and publish event
        event = TrackCreatedEvent(
            aggregate_id=str(track.id),
            data={
                'title': track.title,
                'artist': track.artist,
                'file_path': track.file_path
            }
        )

        # Save event to store
        await self.event_store.save_event(event)

        # Publish event
        await self.event_bus.publish(event)

        return track

    async def complete_analysis(self, track_id: int, analysis_results: Dict[str, Any]):
        """Complete analysis and publish event."""

        # Update track with results
        track = await self.track_repo.get_by_id(track_id)
        if track:
            # Create and publish analysis completed event
            event = TrackAnalyzedEvent(
                aggregate_id=str(track_id),
                data={
                    'analysis_results': analysis_results,
                    'track_title': track.title
                }
            )

            await self.event_store.save_event(event)
            await self.event_bus.publish(event)
```

## Security Best Practices

### 1. Input Validation and Sanitization

```python
# services/shared/security/validation.py
"""
Security-focused input validation.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import mimetypes

class SecurityValidator:
    """Security-focused input validation."""

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.scr', '.com', '.pif', '.jar', '.js', '.vbs',
        '.ps1', '.sh', '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl'
    }

    # Allowed audio file extensions
    ALLOWED_AUDIO_EXTENSIONS = {
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'
    }

    @classmethod
    def validate_file_upload(cls, filename: str, file_size: int,
                           content_type: str = None) -> Dict[str, Any]:
        """Validate file upload security."""

        issues = []

        # Check filename
        if not cls._is_safe_filename(filename):
            issues.append("Unsafe filename characters")

        # Check extension
        path = Path(filename)
        extension = path.suffix.lower()

        if extension in cls.DANGEROUS_EXTENSIONS:
            issues.append(f"Dangerous file extension: {extension}")

        if extension not in cls.ALLOWED_AUDIO_EXTENSIONS:
            issues.append(f"Audio file extension required, got: {extension}")

        # Check file size (100MB limit)
        if file_size > 100 * 1024 * 1024:
            issues.append("File too large (>100MB)")

        # Check MIME type if provided
        if content_type:
            if not content_type.startswith('audio/'):
                issues.append(f"Invalid MIME type: {content_type}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'sanitized_filename': cls._sanitize_filename(filename)
        }

    @classmethod
    def _is_safe_filename(cls, filename: str) -> bool:
        """Check if filename is safe."""

        # Reject empty or too long filenames
        if not filename or len(filename) > 255:
            return False

        # Reject directory traversal attempts
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            return False

        # Reject control characters and dangerous patterns
        dangerous_patterns = [
            r'[\x00-\x1f]',  # Control characters
            r'[<>:"|?*]',    # Windows reserved characters
            r'^\.',          # Hidden files
            r'\.{2,}',       # Multiple dots
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, filename):
                return False

        return True

    @classmethod
    def _sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe storage."""

        # Remove dangerous characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)

        # Limit length
        if len(sanitized) > 255:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            sanitized = name[:255-len(ext)] + ext

        # Ensure it doesn't start with dot
        if sanitized.startswith('.'):
            sanitized = 'file_' + sanitized

        return sanitized

    @classmethod
    def validate_api_input(cls, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate API input against schema."""

        issues = []
        sanitized_data = {}

        for field, rules in schema.items():
            value = data.get(field)

            # Check required fields
            if rules.get('required', False) and not value:
                issues.append(f"Required field missing: {field}")
                continue

            if value is None:
                continue

            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                issues.append(f"Invalid type for {field}: expected {expected_type.__name__}")
                continue

            # Length validation for strings
            if isinstance(value, str):
                min_len = rules.get('min_length', 0)
                max_len = rules.get('max_length', 10000)

                if len(value) < min_len:
                    issues.append(f"{field} too short (min {min_len} chars)")
                    continue

                if len(value) > max_len:
                    issues.append(f"{field} too long (max {max_len} chars)")
                    continue

                # Sanitize string
                sanitized_value = cls._sanitize_text_input(value)

            # Range validation for numbers
            elif isinstance(value, (int, float)):
                min_val = rules.get('min_value')
                max_val = rules.get('max_value')

                if min_val is not None and value < min_val:
                    issues.append(f"{field} below minimum: {min_val}")
                    continue

                if max_val is not None and value > max_val:
                    issues.append(f"{field} above maximum: {max_val}")
                    continue

                sanitized_value = value

            else:
                sanitized_value = value

            sanitized_data[field] = sanitized_value

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'sanitized_data': sanitized_data
        }

    @classmethod
    def _sanitize_text_input(cls, text: str) -> str:
        """Sanitize text input."""

        # Remove control characters except newlines and tabs
        sanitized = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\n\t'
        )

        # Remove potential XSS patterns
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
        ]

        for pattern in xss_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)

        return sanitized.strip()

# Usage in FastAPI
from fastapi import HTTPException, UploadFile

async def secure_file_upload(file: UploadFile) -> Dict[str, Any]:
    """Secure file upload with validation."""

    # Validate file
    validation_result = SecurityValidator.validate_file_upload(
        filename=file.filename,
        file_size=file.size,
        content_type=file.content_type
    )

    if not validation_result['is_valid']:
        raise HTTPException(
            status_code=400,
            detail=f"File validation failed: {', '.join(validation_result['issues'])}"
        )

    # Use sanitized filename
    safe_filename = validation_result['sanitized_filename']

    # Additional content validation
    content = await file.read(1024)  # Read first 1KB
    file.file.seek(0)  # Reset file pointer

    if not cls._validate_audio_content(content):
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file content"
        )

    return {
        'original_filename': file.filename,
        'safe_filename': safe_filename,
        'content_type': file.content_type,
        'size': file.size
    }
```

### 2. Rate Limiting and DDoS Protection

```python
# services/shared/security/rate_limiting.py
"""
Advanced rate limiting and DDoS protection.
"""

import time
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import asyncio
import redis

class RateLimitType(Enum):
    """Rate limit types."""
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""

    requests: int  # Number of requests
    window_seconds: int  # Time window
    limit_type: RateLimitType
    endpoint_pattern: Optional[str] = None
    burst_allowance: int = 0  # Additional burst capacity

    def __post_init__(self):
        self.key_prefix = f"rate_limit:{self.limit_type.value}"

class AdvancedRateLimiter:
    """Advanced rate limiting with Redis backend."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.rules: List[RateLimitRule] = []
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time

    def add_rule(self, rule: RateLimitRule):
        """Add rate limiting rule."""
        self.rules.append(rule)

    async def check_rate_limit(self, identifier: str, endpoint: str,
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if request should be rate limited."""

        current_time = time.time()

        # Check if IP is temporarily blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                return {
                    'allowed': False,
                    'reason': 'IP temporarily blocked',
                    'retry_after': int(self.blocked_ips[identifier] - current_time),
                    'remaining': 0
                }
            else:
                # Remove expired block
                del self.blocked_ips[identifier]

        # Check each rule
        for rule in self.rules:
            # Skip rules that don't apply to this endpoint
            if rule.endpoint_pattern and not self._matches_endpoint(endpoint, rule.endpoint_pattern):
                continue

            # Generate rate limit key
            key = self._generate_key(rule, identifier, endpoint, user_id)

            # Check rate limit
            result = await self._check_rule(rule, key, current_time)

            if not result['allowed']:
                # Check for abuse patterns
                await self._check_abuse_patterns(identifier, endpoint)

                return result

        return {'allowed': True, 'remaining': float('inf')}

    async def _check_rule(self, rule: RateLimitRule, key: str,
                         current_time: float) -> Dict[str, Any]:
        """Check specific rate limiting rule."""

        # Use sliding window with Redis
        window_start = current_time - rule.window_seconds

        # Remove old entries and count current requests
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, rule.window_seconds + 1)

        results = await pipe.execute()
        current_count = results[1]

        # Check if limit exceeded
        total_allowed = rule.requests + rule.burst_allowance

        if current_count >= total_allowed:
            return {
                'allowed': False,
                'reason': f'Rate limit exceeded: {rule.limit_type.value}',
                'retry_after': rule.window_seconds,
                'remaining': 0,
                'rule': rule
            }

        # Add current request
        request_id = f"{current_time}:{hash(key) % 10000}"
        await self.redis.zadd(key, {request_id: current_time})

        return {
            'allowed': True,
            'remaining': total_allowed - current_count - 1,
            'rule': rule
        }

    def _generate_key(self, rule: RateLimitRule, identifier: str,
                     endpoint: str, user_id: Optional[str] = None) -> str:
        """Generate Redis key for rate limiting."""

        if rule.limit_type == RateLimitType.PER_IP:
            return f"{rule.key_prefix}:ip:{identifier}"
        elif rule.limit_type == RateLimitType.PER_USER and user_id:
            return f"{rule.key_prefix}:user:{user_id}"
        elif rule.limit_type == RateLimitType.PER_ENDPOINT:
            return f"{rule.key_prefix}:endpoint:{endpoint}:{identifier}"
        else:
            return f"{rule.key_prefix}:global"

    def _matches_endpoint(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(endpoint, pattern)

    async def _check_abuse_patterns(self, identifier: str, endpoint: str):
        """Check for abuse patterns and implement temporary blocks."""

        # Count recent violations
        violation_key = f"violations:ip:{identifier}"
        current_time = time.time()

        # Add current violation
        await self.redis.zadd(violation_key, {str(current_time): current_time})

        # Count violations in last hour
        hour_ago = current_time - 3600
        await self.redis.zremrangebyscore(violation_key, 0, hour_ago)
        violation_count = await self.redis.zcard(violation_key)

        # Implement progressive blocking
        if violation_count >= 50:  # Severe abuse
            block_duration = 3600  # 1 hour
            self.blocked_ips[identifier] = current_time + block_duration

        elif violation_count >= 20:  # Moderate abuse
            block_duration = 600  # 10 minutes
            self.blocked_ips[identifier] = current_time + block_duration

        elif violation_count >= 10:  # Light abuse
            block_duration = 60  # 1 minute
            self.blocked_ips[identifier] = current_time + block_duration

# FastAPI middleware integration
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: AdvancedRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

        # Define rate limiting rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default rate limiting rules."""

        # Authentication endpoints - strict limits
        self.rate_limiter.add_rule(RateLimitRule(
            requests=5,
            window_seconds=300,  # 5 requests per 5 minutes
            limit_type=RateLimitType.PER_IP,
            endpoint_pattern="/auth/*"
        ))

        # API endpoints - moderate limits
        self.rate_limiter.add_rule(RateLimitRule(
            requests=100,
            window_seconds=60,  # 100 requests per minute
            limit_type=RateLimitType.PER_IP,
            endpoint_pattern="/api/*"
        ))

        # File upload - very strict limits
        self.rate_limiter.add_rule(RateLimitRule(
            requests=5,
            window_seconds=60,  # 5 uploads per minute
            limit_type=RateLimitType.PER_IP,
            endpoint_pattern="*/upload"
        ))

        # Global fallback
        self.rate_limiter.add_rule(RateLimitRule(
            requests=1000,
            window_seconds=60,  # 1000 requests per minute globally
            limit_type=RateLimitType.GLOBAL
        ))

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""

        # Get client identifier
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        user_id = None  # Would extract from JWT token

        # Check rate limits
        result = await self.rate_limiter.check_rate_limit(
            identifier=client_ip,
            endpoint=endpoint,
            user_id=user_id
        )

        if not result['allowed']:
            # Return rate limit response
            response = Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": str(result['retry_after']),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reason": result['reason']
                }
            )
            return response

        # Continue with request
        response = await call_next(request)

        # Add rate limit headers
        if 'remaining' in result:
            response.headers["X-RateLimit-Remaining"] = str(result['remaining'])

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        return request.client.host
```

This comprehensive best practices guide covers essential aspects of building robust, secure, and maintainable Tracktion services. Each section provides practical, production-ready examples that can be adapted to your specific requirements and infrastructure.
