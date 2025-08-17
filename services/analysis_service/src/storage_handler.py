"""Database storage handler for extracted metadata."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from uuid import UUID
from datetime import datetime

# Add shared modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from core_types.src.database import get_db_session, get_neo4j_driver
from core_types.src.models import Recording, Metadata
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
        self.recording_repo = None
        self.metadata_repo = None
        self.neo4j_repo = None
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

    def store_metadata(
        self,
        recording_id: UUID,
        metadata: Dict[str, Optional[str]],
        correlation_id: str
    ) -> bool:
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
            extra={"correlation_id": correlation_id, "metadata_count": len(metadata)}
        )
        
        try:
            # Store in PostgreSQL
            self._store_postgresql_metadata(recording_id, metadata, correlation_id)
            
            # Store in Neo4j
            self._store_neo4j_metadata(recording_id, metadata, correlation_id)
            
            logger.info(
                f"Successfully stored metadata for recording {recording_id}",
                extra={"correlation_id": correlation_id}
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to store metadata for recording {recording_id}: {e}",
                extra={"correlation_id": correlation_id}
            )
            raise StorageError(f"Failed to store metadata: {e}")

    def _store_postgresql_metadata(
        self,
        recording_id: UUID,
        metadata: Dict[str, Optional[str]],
        correlation_id: str
    ) -> None:
        """Store metadata in PostgreSQL.
        
        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # First, verify the recording exists
            recording = self.recording_repo.get(recording_id)
            if not recording:
                raise StorageError(f"Recording {recording_id} not found")
            
            # Delete existing metadata for this recording (if any)
            # This ensures we have fresh metadata
            existing = self.metadata_repo.get_by_recording(recording_id)
            for item in existing:
                self.metadata_repo.delete(item.id)
            
            # Store each metadata item
            stored_count = 0
            for key, value in metadata.items():
                if value is not None:  # Only store non-null values
                    metadata_item = self.metadata_repo.create(
                        recording_id=recording_id,
                        key=key,
                        value=value
                    )
                    stored_count += 1
            
            logger.info(
                f"Stored {stored_count} metadata items in PostgreSQL",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)}
            )
            
        except Exception as e:
            logger.error(
                f"PostgreSQL storage failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            raise

    def _store_neo4j_metadata(
        self,
        recording_id: UUID,
        metadata: Dict[str, Optional[str]],
        correlation_id: str
    ) -> None:
        """Store metadata in Neo4j graph database.
        
        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # First, ensure the recording node exists
            recording_exists = self.neo4j_repo.recording_exists(recording_id)
            if not recording_exists:
                # Get recording details from PostgreSQL
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
                        }
                    )
            
            # Create metadata nodes and relationships
            stored_count = 0
            for key, value in metadata.items():
                if value is not None:
                    # Create metadata node
                    metadata_id = self.neo4j_repo.create_metadata(
                        key=key,
                        value=value,
                        properties={
                            "extracted_at": datetime.utcnow().isoformat(),
                            "correlation_id": correlation_id
                        }
                    )
                    
                    # Create HAS_METADATA relationship
                    self.neo4j_repo.create_has_metadata_relationship(
                        recording_id=recording_id,
                        metadata_id=metadata_id,
                        properties={
                            "created_at": datetime.utcnow().isoformat()
                        }
                    )
                    stored_count += 1
            
            # Update additional relationships based on metadata
            self._create_semantic_relationships(recording_id, metadata, correlation_id)
            
            logger.info(
                f"Stored {stored_count} metadata nodes in Neo4j",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)}
            )
            
        except Exception as e:
            logger.error(
                f"Neo4j storage failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            raise

    def _create_semantic_relationships(
        self,
        recording_id: UUID,
        metadata: Dict[str, Optional[str]],
        correlation_id: str
    ) -> None:
        """Create semantic relationships in Neo4j based on metadata.
        
        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs
            correlation_id: Correlation ID for tracing
        """
        try:
            # Create Artist node and relationship if artist exists
            if metadata.get('artist'):
                artist_id = self.neo4j_repo.create_or_get_artist(metadata['artist'])
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=artist_id,
                    relationship_type="PERFORMED_BY",
                    properties={"source": "metadata_extraction"}
                )
            
            # Create Album node and relationship if album exists
            if metadata.get('album'):
                album_id = self.neo4j_repo.create_or_get_album(
                    metadata['album'],
                    artist=metadata.get('artist')
                )
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=album_id,
                    relationship_type="PART_OF",
                    properties={
                        "track_number": metadata.get('track'),
                        "source": "metadata_extraction"
                    }
                )
            
            # Create Genre node and relationship if genre exists
            if metadata.get('genre'):
                genre_id = self.neo4j_repo.create_or_get_genre(metadata['genre'])
                self.neo4j_repo.create_relationship(
                    from_id=recording_id,
                    to_id=genre_id,
                    relationship_type="HAS_GENRE",
                    properties={"source": "metadata_extraction"}
                )
            
            logger.debug(
                "Created semantic relationships in Neo4j",
                extra={"correlation_id": correlation_id, "recording_id": str(recording_id)}
            )
            
        except Exception as e:
            # Log but don't fail - semantic relationships are supplementary
            logger.warning(
                f"Failed to create some semantic relationships: {e}",
                extra={"correlation_id": correlation_id}
            )

    def update_recording_status(
        self,
        recording_id: UUID,
        status: str,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None
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
            recording = self.recording_repo.get(recording_id)
            if not recording:
                logger.error(
                    f"Recording {recording_id} not found for status update",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Update status in PostgreSQL
            updates = {"processing_status": status}
            if error_message:
                updates["processing_error"] = error_message
            
            updated = self.recording_repo.update(recording_id, **updates)
            
            # Also update in Neo4j
            if self.neo4j_repo.recording_exists(recording_id):
                self.neo4j_repo.update_recording_properties(
                    recording_id,
                    {
                        "processing_status": status,
                        "processing_error": error_message,
                        "last_processed": datetime.utcnow().isoformat()
                    }
                )
            
            logger.info(
                f"Updated recording {recording_id} status to {status}",
                extra={"correlation_id": correlation_id}
            )
            
            return updated is not None
            
        except Exception as e:
            logger.error(
                f"Failed to update recording status: {e}",
                extra={"correlation_id": correlation_id}
            )
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