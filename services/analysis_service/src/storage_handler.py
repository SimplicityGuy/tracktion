"""Database storage handler for extracted metadata."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
from uuid import UUID
from datetime import datetime

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from core_types.src.database import get_db_session
from core_types.src.repositories import RecordingRepository, MetadataRepository
from core_types.src.neo4j_repository import Neo4jRepository

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Raised when storage operations fail."""

    pass


class StorageHandler:
    """Handles storage of extracted metadata to databases."""

    def __init__(self) -> None:
        """Initialize storage handler with database connections."""
        self.recording_repo: Optional[RecordingRepository] = None
        self.metadata_repo: Optional[MetadataRepository] = None
        self.neo4j_repo: Optional[Neo4jRepository] = None
        self._initialize_repositories()

    def _initialize_repositories(self) -> None:
        """Initialize database repositories."""
        try:
            # Initialize PostgreSQL repositories
            db_session = get_db_session()
            self.recording_repo = RecordingRepository(db_session)
            self.metadata_repo = MetadataRepository(db_session)

            # Initialize Neo4j repository
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise StorageError("Neo4j connection environment variables not set")

            self.neo4j_repo = Neo4jRepository(neo4j_uri, neo4j_user, neo4j_password)

            logger.info("Successfully initialized database repositories")

        except Exception as e:
            logger.error(f"Failed to initialize repositories: {e}")
            raise StorageError(f"Database initialization failed: {e}")

    def store_metadata(self, recording_id: UUID, metadata: Dict[str, Optional[str]], correlation_id: str) -> bool:
        """Store extracted metadata in both databases.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing

        Returns:
            True if storage was successful

        Raises:
            StorageError: If storage operations fail
        """
        logger.info(
            f"Storing metadata for recording {recording_id}",
            extra={"correlation_id": correlation_id, "metadata_count": len(metadata)},
        )

        try:
            # Store in PostgreSQL
            self._store_postgresql_metadata(recording_id, metadata, correlation_id)

            # Store in Neo4j
            self._store_neo4j_metadata(recording_id, metadata, correlation_id)

            logger.info(
                f"Successfully stored metadata for recording {recording_id}", extra={"correlation_id": correlation_id}
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to store metadata for recording {recording_id}: {e}", extra={"correlation_id": correlation_id}
            )
            raise StorageError(f"Failed to store metadata: {e}")

    def _store_postgresql_metadata(
        self, recording_id: UUID, metadata: Dict[str, Optional[str]], correlation_id: str
    ) -> None:
        """Store metadata in PostgreSQL.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # First, verify the recording exists
            if not self.recording_repo:
                raise StorageError("Recording repository not initialized")
            recording = self.recording_repo.get(recording_id)
            if not recording:
                raise StorageError(f"Recording {recording_id} not found")

            # Delete existing metadata for this recording (if any)
            # This ensures we have fresh metadata
            if not self.metadata_repo:
                raise StorageError("Metadata repository not initialized")
            existing = self.metadata_repo.get_by_recording(recording_id)
            for item in existing:
                self.metadata_repo.delete(item.id)

            # Store each metadata item
            stored_count = 0
            for key, value in metadata.items():
                if value is not None:  # Only store non-null values
                    if self.metadata_repo:
                        self.metadata_repo.create(recording_id=recording_id, key=key, value=value)
                        stored_count += 1

            logger.info(
                f"Stored {stored_count} metadata items in PostgreSQL",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)},
            )

        except Exception as e:
            logger.error(f"PostgreSQL storage failed: {e}", extra={"correlation_id": correlation_id})
            raise

    def _store_neo4j_metadata(
        self, recording_id: UUID, metadata: Dict[str, Optional[str]], correlation_id: str
    ) -> None:
        """Store metadata in Neo4j graph database.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # First, ensure the recording node exists
            if not self.neo4j_repo:
                raise StorageError("Neo4j repository not initialized")
            recording_exists = self.neo4j_repo.recording_exists(recording_id)
            if not recording_exists:
                # Get recording details from PostgreSQL
                if not self.recording_repo:
                    raise StorageError("Recording repository not initialized")
                recording = self.recording_repo.get(recording_id)
                if recording:
                    # Create recording node in Neo4j
                    self.neo4j_repo.create_recording(
                        recording_id=recording_id,
                        file_path=recording.file_path,
                        file_hash=recording.file_hash,
                        properties={
                            "file_size": recording.file_size,
                            "created_at": recording.created_at.isoformat() if recording.created_at else None,
                            "updated_at": recording.updated_at.isoformat() if recording.updated_at else None,
                        },
                    )

            # Create metadata nodes and relationships
            stored_count = 0
            for key, value in metadata.items():
                if value is not None and self.neo4j_repo:
                    # Create metadata node
                    metadata_id = self.neo4j_repo.create_metadata(
                        key=key,
                        value=value,
                        properties={"extracted_at": datetime.utcnow().isoformat(), "correlation_id": correlation_id},
                    )

                    # Create HAS_METADATA relationship
                    self.neo4j_repo.create_has_metadata_relationship(
                        recording_id=recording_id,
                        metadata_id=metadata_id,
                        properties={"created_at": datetime.utcnow().isoformat()},
                    )
                    stored_count += 1

            # Update additional relationships based on metadata
            self._create_semantic_relationships(recording_id, metadata, correlation_id)

            # Create BPM-specific relationships if BPM data exists
            self._create_bpm_relationships(recording_id, metadata, correlation_id)

            logger.info(
                f"Stored {stored_count} metadata nodes in Neo4j",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)},
            )

        except Exception as e:
            logger.error(f"Neo4j storage failed: {e}", extra={"correlation_id": correlation_id})
            raise

    def _create_semantic_relationships(
        self, recording_id: UUID, metadata: Dict[str, Optional[str]], correlation_id: str
    ) -> None:
        """Create semantic relationships in Neo4j based on metadata.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # Create Artist node and relationship if artist exists
            if metadata.get("artist") and self.neo4j_repo:
                artist_id = self.neo4j_repo.create_or_get_artist(metadata["artist"])
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=artist_id,
                    relationship_type="PERFORMED_BY",
                    properties={"source": "metadata_extraction"},
                )

            # Create Album node and relationship if album exists
            if metadata.get("album") and self.neo4j_repo:
                album_id = self.neo4j_repo.create_or_get_album(metadata["album"], artist=metadata.get("artist"))
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=album_id,
                    relationship_type="PART_OF",
                    properties={"track_number": metadata.get("track"), "source": "metadata_extraction"},
                )

            # Create Genre node and relationship if genre exists
            if metadata.get("genre") and self.neo4j_repo:
                genre_id = self.neo4j_repo.create_or_get_genre(metadata["genre"])
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=genre_id,
                    relationship_type="HAS_GENRE",
                    properties={"source": "metadata_extraction"},
                )

            logger.debug(
                "Created semantic relationships in Neo4j",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)},
            )

        except Exception as e:
            # Log but don't fail - semantic relationships are supplementary
            logger.warning(
                f"Failed to create some semantic relationships: {e}", extra={"correlation_id": correlation_id}
            )

    def store_bpm_data(
        self,
        recording_id: UUID,
        bpm_data: Dict[str, Any],
        temporal_data: Optional[Dict[str, Any]] = None,
        correlation_id: str = "unknown",
    ) -> bool:
        """Store BPM analysis results in both databases.

        Args:
            recording_id: UUID of the recording
            bpm_data: BPM detection results
            temporal_data: Optional temporal analysis results
            correlation_id: Correlation ID for tracing

        Returns:
            True if storage was successful
        """
        try:
            # Prepare BPM metadata for storage
            bpm_metadata: Dict[str, Optional[str]] = {}

            # Store core BPM values
            if "bpm" in bpm_data:
                bpm_metadata["bpm_average"] = str(bpm_data["bpm"])

            if "confidence" in bpm_data:
                bpm_metadata["bpm_confidence"] = str(bpm_data["confidence"])

            if "algorithm" in bpm_data:
                bpm_metadata["bpm_algorithm"] = bpm_data["algorithm"]

            # Store temporal data if available
            if temporal_data:
                if "start_bpm" in temporal_data:
                    bpm_metadata["bpm_start"] = str(temporal_data["start_bpm"])

                if "end_bpm" in temporal_data:
                    bpm_metadata["bpm_end"] = str(temporal_data["end_bpm"])

                if "stability_score" in temporal_data:
                    bpm_metadata["bpm_stability"] = str(temporal_data["stability_score"])

                # Optionally store temporal array (based on configuration)
                if "temporal_bpm" in temporal_data and temporal_data["temporal_bpm"]:
                    # Store as JSON string for complex data
                    bpm_metadata["bpm_temporal"] = json.dumps(temporal_data["temporal_bpm"])

                if "average_bpm" in temporal_data:
                    # Override with temporal average if different
                    bpm_metadata["bpm_average"] = str(temporal_data["average_bpm"])

            # Store error if BPM detection failed
            if "error" in bpm_data:
                bpm_metadata["bpm_error"] = str(bpm_data["error"])
                bpm_metadata["bpm_status"] = "failed"
            else:
                bpm_metadata["bpm_status"] = "completed"

            # Store using existing metadata storage method
            success = self.store_metadata(recording_id, bpm_metadata, correlation_id)

            logger.info(
                f"Stored BPM data for recording {recording_id}",
                extra={
                    "correlation_id": correlation_id,
                    "bpm": bpm_metadata.get("bpm_average"),
                    "confidence": bpm_metadata.get("bpm_confidence"),
                    "stability": bpm_metadata.get("bpm_stability"),
                },
            )

            return success

        except Exception as e:
            logger.error(
                f"Failed to store BPM data for recording {recording_id}: {e}",
                extra={"correlation_id": correlation_id},
            )
            raise StorageError(f"Failed to store BPM data: {e}")

    def _create_bpm_relationships(
        self, recording_id: UUID, metadata: Dict[str, Optional[str]], correlation_id: str
    ) -> None:
        """Create BPM-based relationships in Neo4j.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary containing BPM metadata
            correlation_id: Correlation ID for tracing
        """
        try:
            if not self.neo4j_repo:
                return

            # Get BPM value
            bpm_str = metadata.get("bpm_average")
            if not bpm_str:
                return

            try:
                bpm = float(bpm_str)
            except (ValueError, TypeError):
                return

            # Create BPM range node and relationship
            bpm_range = self._get_bpm_range(bpm)
            if bpm_range:
                bpm_range_id = self.neo4j_repo.create_or_get_bpm_range(bpm_range)
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=bpm_range_id,
                    relationship_type="HAS_BPM_RANGE",
                    properties={
                        "bpm": bpm,
                        "confidence": metadata.get("bpm_confidence"),
                        "source": "bpm_detection",
                    },
                )

            # Create tempo stability relationship if available
            stability_str = metadata.get("bpm_stability")
            if stability_str:
                try:
                    stability = float(stability_str)
                    tempo_type = "constant" if stability > 0.8 else "variable"
                    tempo_id = self.neo4j_repo.create_or_get_tempo_type(tempo_type)
                    self.neo4j_repo.create_relationship(
                        from_id=recording_id,
                        to_id=tempo_id,
                        relationship_type="HAS_TEMPO_TYPE",
                        properties={"stability_score": stability, "source": "temporal_analysis"},
                    )
                except (ValueError, TypeError):
                    pass

            logger.debug(
                f"Created BPM relationships in Neo4j for recording {recording_id}",
                extra={"correlation_id": correlation_id, "bpm": bpm},
            )

        except Exception as e:
            # Log but don't fail - BPM relationships are supplementary
            logger.warning(
                f"Failed to create BPM relationships: {e}",
                extra={"correlation_id": correlation_id},
            )

    def _get_bpm_range(self, bpm: float) -> Optional[str]:
        """Categorize BPM into standard ranges.

        Args:
            bpm: The BPM value

        Returns:
            BPM range category or None
        """
        if bpm < 60:
            return "very_slow"  # Largo
        elif bpm < 76:
            return "slow"  # Adagio
        elif bpm < 108:
            return "moderate"  # Andante/Moderato
        elif bpm < 120:
            return "moderate_fast"  # Allegretto
        elif bpm < 140:
            return "fast"  # Allegro
        elif bpm < 168:
            return "very_fast"  # Vivace
        elif bpm < 200:
            return "extremely_fast"  # Presto
        else:
            return "ultra_fast"  # Prestissimo

    def get_bpm_statistics(self, recording_ids: Optional[List[UUID]] = None) -> Dict[str, Any]:
        """Get BPM statistics for recordings.

        Args:
            recording_ids: Optional list of recording IDs to filter

        Returns:
            Dictionary with BPM statistics
        """
        try:
            stats: Dict[str, Any] = {
                "total_analyzed": 0,
                "successful": 0,
                "failed": 0,
                "average_bpm": None,
                "average_confidence": None,
                "bpm_ranges": {},
            }

            if not self.metadata_repo:
                return stats

            # Get BPM metadata
            bpm_values = []
            confidence_values = []

            if recording_ids:
                # Get metadata for specific recordings
                for rec_id in recording_ids:
                    metadata_items = self.metadata_repo.get_by_recording(rec_id)
                    for item in metadata_items:
                        if item.key == "bpm_average" and item.value:
                            try:
                                bpm_values.append(float(item.value))
                                stats["successful"] += 1
                            except ValueError:
                                stats["failed"] += 1
                        elif item.key == "bpm_confidence" and item.value:
                            try:
                                confidence_values.append(float(item.value))
                            except ValueError:
                                pass

            stats["total_analyzed"] = stats["successful"] + stats["failed"]

            # Calculate averages
            if bpm_values:
                stats["average_bpm"] = sum(bpm_values) / len(bpm_values)

                # Count BPM ranges
                for bpm in bpm_values:
                    bpm_range = self._get_bpm_range(bpm)
                    if bpm_range:
                        stats["bpm_ranges"][bpm_range] = stats["bpm_ranges"].get(bpm_range, 0) + 1

            if confidence_values:
                stats["average_confidence"] = sum(confidence_values) / len(confidence_values)

            return stats

        except Exception as e:
            logger.error(f"Failed to get BPM statistics: {e}")
            return {"error": str(e)}

    def update_recording_status(
        self, recording_id: UUID, status: str, error_message: Optional[str] = None, correlation_id: Optional[str] = None
    ) -> bool:
        """Update the processing status of a recording.

        Args:
            recording_id: UUID of the recording
            status: New status (e.g., 'processed', 'failed')
            error_message: Optional error message if status is 'failed'
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if update was successful
        """
        try:
            if not self.recording_repo:
                raise StorageError("Recording repository not initialized")
            recording = self.recording_repo.get(recording_id)
            if not recording:
                logger.error(
                    f"Recording {recording_id} not found for status update", extra={"correlation_id": correlation_id}
                )
                return False

            # Update status in PostgreSQL
            updates = {"processing_status": status}
            if error_message:
                updates["processing_error"] = error_message

            updated = self.recording_repo.update(recording_id, **updates)

            # Also update in Neo4j
            if self.neo4j_repo and self.neo4j_repo.recording_exists(recording_id):
                self.neo4j_repo.update_recording_properties(
                    recording_id,
                    {
                        "processing_status": status,
                        "processing_error": error_message,
                        "last_processed": datetime.utcnow().isoformat(),
                    },
                )

            logger.info(
                f"Updated recording {recording_id} status to {status}", extra={"correlation_id": correlation_id}
            )

            return updated is not None

        except Exception as e:
            logger.error(f"Failed to update recording status: {e}", extra={"correlation_id": correlation_id})
            return False

    def close(self) -> None:
        """Close database connections."""
        try:
            if self.neo4j_repo:
                self.neo4j_repo.close()
            logger.info("Closed database connections")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def __enter__(self) -> "StorageHandler":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
