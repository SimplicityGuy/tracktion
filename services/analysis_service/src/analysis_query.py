"""Query interface for retrieving analysis results."""

import contextlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result for a recording."""

    recording_id: UUID
    file_path: str

    # BPM Analysis
    bpm_data: dict[str, Any] | None = None
    temporal_data: dict[str, Any] | None = None

    # Key Detection
    key_data: dict[str, Any] | None = None

    # Mood Analysis
    mood_data: dict[str, Any] | None = None

    # Metadata
    analysis_timestamp: datetime | None = None
    from_cache: bool = False
    confidence_threshold: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["recording_id"] = str(self.recording_id)
        if self.analysis_timestamp:
            data["analysis_timestamp"] = self.analysis_timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def needs_review(self) -> bool:
        """Check if any analysis needs review."""
        needs_review = False

        # Check BPM confidence
        if self.bpm_data and self.bpm_data.get("confidence", 1.0) < self.confidence_threshold:
            needs_review = True

        # Check key detection
        if self.key_data and self.key_data.get("needs_review", False):
            needs_review = True

        # Check mood analysis
        if self.mood_data and self.mood_data.get("needs_review", False):
            needs_review = True

        return needs_review

    @property
    def summary(self) -> dict[str, Any]:
        """Get a summary of key analysis results."""
        summary = {
            "recording_id": str(self.recording_id),
            "file_path": self.file_path,
            "needs_review": self.needs_review,
        }

        # Add BPM summary
        if self.bpm_data and "bpm" in self.bpm_data:
            summary["bpm"] = {
                "value": self.bpm_data["bpm"],
                "confidence": self.bpm_data.get("confidence", 0),
            }
            if self.temporal_data:
                summary["bpm"]["stability"] = self.temporal_data.get("stability_score", 0)  # type: ignore[index]

        # Add key summary
        if self.key_data and "key" in self.key_data:
            summary["key"] = {
                "notation": f"{self.key_data['key']} {self.key_data['scale']}",
                "confidence": self.key_data.get("confidence", 0),
            }

        # Add mood summary
        if self.mood_data:
            summary["mood"] = {
                "genre": self.mood_data.get("primary_genre"),
                "danceability": self.mood_data.get("danceability", 0),
                "energy": self.mood_data.get("energy", 0),
                "valence": self.mood_data.get("valence", 0),
            }

        return summary


class AnalysisQuery:
    """Query interface for analysis results."""

    def __init__(self, storage_handler: Any = None, cache: Any = None) -> None:
        """Initialize query interface.

        Args:
            storage_handler: StorageHandler instance for database queries
            cache: AudioCache instance for cache queries
        """
        self.storage = storage_handler
        self.cache = cache

    def get_analysis_result(
        self,
        recording_id: str | UUID,
        include_cached: bool = True,
        confidence_threshold: float = 0.6,
    ) -> AnalysisResult | None:
        """Get complete analysis results for a recording.

        Args:
            recording_id: Recording UUID
            include_cached: Whether to check cache first
            confidence_threshold: Minimum confidence for reliable results

        Returns:
            AnalysisResult or None if not found
        """
        if isinstance(recording_id, str):
            recording_id = UUID(recording_id)

        result = AnalysisResult(
            recording_id=recording_id,
            file_path="",  # Will be populated from database
            confidence_threshold=confidence_threshold,
        )

        # Try to get from cache first if enabled
        if include_cached and self.cache:
            # This would need the file path, which we'd get from storage
            pass  # Cache requires file path, not recording ID

        # Get from database
        if self.storage:
            metadata = self._get_metadata_from_storage(recording_id)
            if metadata:
                result = self._parse_metadata_to_result(recording_id, metadata, confidence_threshold)

        return result if result.file_path else None

    def _get_metadata_from_storage(self, recording_id: UUID) -> dict[str, str] | None:
        """Get all metadata for a recording from storage.

        Args:
            recording_id: Recording UUID

        Returns:
            Dictionary of metadata key-value pairs
        """
        if not self.storage or not self.storage.metadata_repo:
            return None

        try:
            # Get recording info
            recording = self.storage.recording_repo.get(recording_id)
            if not recording:
                return None

            # Get all metadata
            metadata_items = self.storage.metadata_repo.get_by_recording(recording_id)

            # Convert to dictionary
            metadata = {"file_path": recording.file_path}
            for item in metadata_items:
                metadata[item.key] = item.value

            return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for recording {recording_id}: {e}")
            return None

    def _parse_metadata_to_result(
        self, recording_id: UUID, metadata: dict[str, str], confidence_threshold: float
    ) -> AnalysisResult:
        """Parse stored metadata into AnalysisResult.

        Args:
            recording_id: Recording UUID
            metadata: Dictionary of metadata
            confidence_threshold: Confidence threshold

        Returns:
            Parsed AnalysisResult
        """
        result = AnalysisResult(
            recording_id=recording_id,
            file_path=metadata.get("file_path", ""),
            confidence_threshold=confidence_threshold,
        )

        self._parse_bpm_data(metadata, result)
        self._parse_temporal_data(metadata, result)
        self._parse_key_data(metadata, result)
        self._parse_mood_data(metadata, result)

        return result

    def _parse_bpm_data(self, metadata: dict[str, str], result: AnalysisResult) -> None:
        """Parse BPM data from metadata."""
        if "bpm_average" in metadata:
            result.bpm_data = {
                "bpm": float(metadata.get("bpm_average", 0)),
                "confidence": float(metadata.get("bpm_confidence", 0)),
                "algorithm": metadata.get("bpm_algorithm", ""),
                "status": metadata.get("bpm_status", ""),
            }

    def _parse_temporal_data(self, metadata: dict[str, str], result: AnalysisResult) -> None:
        """Parse temporal data from metadata."""
        if "bpm_stability" in metadata:
            result.temporal_data = {
                "stability_score": float(metadata.get("bpm_stability", 0)),
                "start_bpm": (float(metadata.get("bpm_start", 0)) if "bpm_start" in metadata else None),
                "end_bpm": (float(metadata.get("bpm_end", 0)) if "bpm_end" in metadata else None),
            }

    def _parse_key_data(self, metadata: dict[str, str], result: AnalysisResult) -> None:
        """Parse key data from metadata."""
        if "musical_key" in metadata:
            result.key_data = {
                "key": metadata.get("musical_key", ""),
                "scale": metadata.get("musical_scale", ""),
                "confidence": float(metadata.get("key_confidence", 0)),
                "agreement": metadata.get("key_agreement", "").lower() == "true",
                "needs_review": metadata.get("key_needs_review", "").lower() == "true",
            }
            if "key_alternative" in metadata:
                result.key_data["alternative"] = {
                    "key": metadata.get("key_alternative", ""),
                    "scale": metadata.get("key_alternative_scale", ""),
                }

    def _parse_mood_data(self, metadata: dict[str, str], result: AnalysisResult) -> None:
        """Parse mood data from metadata."""
        if "genre_primary" in metadata:
            mood_data = {
                "primary_genre": metadata.get("genre_primary", ""),
                "genre_confidence": float(metadata.get("genre_confidence", 0)),
                "genres": [],
                "mood_scores": {},
            }

            # Add top genres
            for i in range(1, 4):
                genre_key = f"genre_{i}"
                if genre_key in metadata:
                    mood_data["genres"].append(  # type: ignore[attr-defined]
                        {
                            "genre": metadata.get(genre_key, ""),
                            "confidence": float(metadata.get(f"{genre_key}_confidence", 0)),
                        }
                    )

            # Add mood scores
            for key, value in metadata.items():
                if key.startswith("mood_") and not key.endswith("_status"):
                    mood_name = key.replace("mood_", "")
                    if mood_name not in ["confidence", "needs_review", "error"]:
                        with contextlib.suppress(ValueError):
                            mood_data["mood_scores"][mood_name] = float(value)  # type: ignore[index]

            # Add other attributes
            mood_data["danceability"] = float(metadata.get("danceability", 0))
            mood_data["energy"] = float(metadata.get("energy", 0))
            mood_data["valence"] = float(metadata.get("valence", 0))
            mood_data["arousal"] = float(metadata.get("arousal", 0))
            mood_data["voice_instrumental"] = metadata.get("voice_instrumental", "unknown")
            mood_data["overall_confidence"] = float(metadata.get("mood_confidence", 0))
            mood_data["needs_review"] = metadata.get("mood_needs_review", "").lower() == "true"

            result.mood_data = mood_data

    def query_by_criteria(
        self,
        bpm_range: tuple[float, float] | None = None,
        key: str | None = None,
        genre: str | None = None,
        danceability_min: float | None = None,
        energy_min: float | None = None,
        valence_range: tuple[float, float] | None = None,
        limit: int = 100,
    ) -> list[AnalysisResult]:
        """Query recordings by analysis criteria.

        Args:
            bpm_range: Tuple of (min_bpm, max_bpm)
            key: Musical key notation (e.g., "C major")
            genre: Genre name
            danceability_min: Minimum danceability score
            energy_min: Minimum energy score
            valence_range: Tuple of (min_valence, max_valence)
            limit: Maximum number of results

        Returns:
            List of matching AnalysisResult objects
        """
        results: list[AnalysisResult] = []

        # This would require complex database queries
        # For now, this is a placeholder for the interface

        logger.info(
            f"Querying with criteria: BPM={bpm_range}, Key={key}, Genre={genre}, Danceability>={danceability_min}"
        )

        return results

    def get_compatible_recordings(
        self, recording_id: str | UUID, compatibility_type: str = "harmonic"
    ) -> list[dict[str, Any]]:
        """Find recordings compatible with the given recording.

        Args:
            recording_id: Recording UUID
            compatibility_type: Type of compatibility ("harmonic", "tempo", "mood")

        Returns:
            List of compatible recordings with compatibility scores
        """
        if isinstance(recording_id, str):
            recording_id = UUID(recording_id)

        compatible: list[dict[str, Any]] = []

        # Get the source recording's analysis
        source = self.get_analysis_result(recording_id)
        if not source:
            return compatible

        if compatibility_type == "harmonic" and source.key_data:
            # Find harmonically compatible keys
            # This would query for recordings in relative keys, circle of fifths, etc.
            pass

        elif compatibility_type == "tempo" and source.bpm_data:
            # Find recordings with similar BPM (±5%)
            pass

        elif compatibility_type == "mood" and source.mood_data:
            # Find recordings with similar mood/energy
            pass

        return compatible


def create_analysis_message(
    recording_id: str,
    file_path: str,
    analysis_type: str = "full",
    priority: str = "normal",
) -> dict[str, Any]:
    """Create a message for analysis queue.

    Args:
        recording_id: Recording UUID
        file_path: Path to audio file
        analysis_type: Type of analysis ("full", "bpm", "key", "mood")
        priority: Message priority

    Returns:
        Message dictionary for RabbitMQ
    """
    return {
        "recording_id": recording_id,
        "file_path": file_path,
        "analysis_type": analysis_type,
        "priority": priority,
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0",
    }


def parse_analysis_message(message: str | bytes | dict) -> dict[str, Any]:
    """Parse an analysis message.

    Args:
        message: Raw message from queue

    Returns:
        Parsed message dictionary
    """
    if isinstance(message, bytes):
        message = message.decode("utf-8")

    if isinstance(message, str):
        message = json.loads(message)

    # Validate required fields
    required_fields = ["recording_id", "file_path"]
    for field in required_fields:
        if field not in message:
            raise ValueError(f"Missing required field: {field}")

    # Add defaults for optional fields
    if isinstance(message, dict):
        message.setdefault("analysis_type", "full")
        message.setdefault("priority", "normal")
        return message
    raise ValueError("Message must be a dictionary after parsing")
